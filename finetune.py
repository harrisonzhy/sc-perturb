#!/usr/bin/env python3
import argparse as ap
import os, math
import numpy as np
from scipy import sparse

import torch
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
import mygene

# STATE utilities (same entry points as _train.py)
from cell_load.data_modules import PerturbationDataModule
from cell_load.utils.modules import get_datamodule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.precision import MixedPrecision

from state.tx.callbacks import (
    BatchSpeedMonitorCallback,
    CumulativeFLOPSCallback,
    GradNormCallback,
    ModelFLOPSUtilizationCallback,
)
from state.tx.utils import get_checkpoint_callbacks, get_lightning_module, get_loggers


def load_cfg_from_yaml(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config YAML not found: {path}")
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

    # minimal validation + defaults
    for top in ["output_dir", "name", "data", "model", "training"]:
        if top not in cfg:
            raise KeyError(f"Missing required top-level key: {top}")
    cfg.setdefault("overwrite", False)
    cfg.setdefault("use_wandb", False)
    cfg.setdefault("wandb", {"project": "", "entity": "", "local_wandb_dir": "./wandb"})
    cfg["data"].setdefault("kwargs", {})
    cfg["model"].setdefault("kwargs", {})
    cfg["training"].setdefault("batch_size", 1)
    cfg["training"].setdefault("max_steps", 2000)
    cfg["training"].setdefault("val_freq", 200)
    cfg["training"].setdefault("train_seed", 1337)
    cfg["training"].setdefault("gradient_clip_val", 1.0)
    cfg["training"].setdefault("devices", 1)
    cfg["training"].setdefault("strategy", "auto")
    cfg.setdefault("grn", {"path": None, "lambda": 1e-3})

    # sanity: checkpoint path should be a .ckpt
    init_from = cfg["model"]["kwargs"].get("init_from")
    if init_from and not init_from.endswith(".ckpt"):
        print(f"[warn] model.kwargs.init_from='{init_from}' does not look like a .ckpt")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg

# ----------------- GRN utils -----------------
def load_grn_laplacian(
    csv_path: str,
    gene_order: list | None = None,
    symmetric: str = "max",
):
    """
    Build L = I - D^{-1/2} A D^{-1/2} from a CSV edgelist with header:
        source,target,weight
    If gene_order is provided (list of gene IDs), we align to that order
    and drop any edges with genes not in the list. Otherwise we infer the
    node set from the CSV and return the order we used.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse

    df = pd.read_csv(csv_path)
    if not {"source", "target"}.issubset(df.columns):
        raise ValueError("CSV must have columns: source,target[,weight]")
    if "weight" not in df.columns:
        df["weight"] = 1.0

    src = df["source"].astype(str)
    dst = df["target"].astype(str)
    w   = df["weight"].astype(np.float32).to_numpy()

    if gene_order is None:
        # infer node set from edges
        cats = src.astype("category").cat.categories.union(
            dst.astype("category").cat.categories
        )
        src = src.astype("category").cat.set_categories(cats)
        dst = dst.astype("category").cat.set_categories(cats)
        iu  = src.cat.codes.to_numpy()
        iv  = dst.cat.codes.to_numpy()
        index_to_gene = list(cats)
        n = len(index_to_gene)
    else:
        # align to provided gene order
        gene_to_index = {g: i for i, g in enumerate(gene_order)}
        mask = src.isin(gene_to_index) & dst.isin(gene_to_index)
        if not mask.any():
            raise ValueError("After alignment, no GRN edges remain; check gene IDs.")
        src = src[mask].map(gene_to_index)
        dst = dst[mask].map(gene_to_index)
        w   = w[mask.to_numpy()]
        iu  = src.to_numpy(dtype=np.int64)
        iv  = dst.to_numpy(dtype=np.int64)
        index_to_gene = list(gene_order)
        n = len(index_to_gene)

    # build COO then coalesce duplicates by summing
    A = sparse.coo_matrix((w, (iu, iv)), shape=(n, n), dtype=np.float32).tocsr()

    # symmetrize
    if symmetric == "max":
        A = A.maximum(A.T)
    elif symmetric == "sum":
        A = A + A.T
        A.setdiag(0.0)
        A.eliminate_zeros()
    else:
        raise ValueError("symmetric must be 'max' or 'sum'")

    # normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    deg[deg == 0.0] = 1.0
    Dinv = sparse.diags((1.0 / np.sqrt(deg)).astype(np.float32))
    L = sparse.eye(n, dtype=np.float32) - (Dinv @ A @ Dinv)
    return L.tocsr(), index_to_gene

class GRNReg:
    """
    GRN regularizer: for each Q/K Linear, compute tr(F^T L F) with
    F ≈ (P^T @ W^T), where P is a fixed random projection (genes←latent-in).
    """
    def __init__(self, module: torch.nn.Module, L_csr: sparse.csr_matrix, grn_lambda: float = 1e-3, seed: int = 0):
        self.module = module
        self.grn_lambda = float(grn_lambda)
        self.L_shape = L_csr.shape
        idx = np.vstack(L_csr.nonzero()).astype(np.int64)
        self.L_indices_cpu = torch.tensor(idx)
        self.L_values_cpu = torch.tensor(L_csr.data, dtype=torch.float32)
        self._proj_cache = None
        self._seed = seed

    def _projection(self, device: torch.device, in_dim: int) -> torch.Tensor:
        if self._proj_cache is None or self._proj_cache.shape[0] != in_dim:
            g = torch.Generator(device=device)
            g.manual_seed(self._seed)
            self._proj_cache = torch.randn(in_dim, self.L_shape[0], generator=g, device=device) / math.sqrt(in_dim)
        return self._proj_cache  # [in_dim, N_genes]

    def penalty(self, device: torch.device) -> torch.Tensor:
        if self.grn_lambda <= 0:
            return torch.tensor(0.0, device=device)

        L = torch.sparse_coo_tensor(
            self.L_indices_cpu.to(device), self.L_values_cpu.to(device), self.L_shape, device=device
        )

        total = torch.tensor(0.0, device=device)
        for m in self.module.modules():
            if hasattr(m, "q_proj") and hasattr(m, "k_proj"):
                for W in (m.q_proj, m.k_proj):  # only Q/K
                    # W.weight: [out_dim, in_dim]
                    in_dim = W.weight.shape[1]
                    P = self._projection(device, in_dim)   # [in_dim, N]
                    F = (P.T @ W.weight.T)                 # [N, out_dim]
                    LF = torch.sparse.mm(L, F)             # [N, out_dim]
                    total = total + (F * LF).sum()

        return self.grn_lambda * total


# -------------- fine-tune helpers --------------
def freeze_all_but_attention(root: torch.nn.Module):
    """Freeze everything except q/k/v/o in attention blocks."""
    for p in root.parameters():
        p.requires_grad = False
    trainable = []
    for m in root.modules():
        if hasattr(m, "q_proj") and hasattr(m, "k_proj") and hasattr(m, "v_proj"):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(m, name):
                    for p in getattr(m, name).parameters():
                        p.requires_grad = True
                        trainable.append(p)
    return trainable


# -------------- main training flow (mirrors _train.py, then adds our GRN/attention changes) --------------
def run(cfg: DictConfig):
    import json
    import logging
    import pickle
    import shutil
    from pathlib import Path
    from os.path import exists, join

    logger = logging.getLogger(__name__)
    torch.set_float32_matmul_precision("medium")

    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # --- output dirs / wandb ---
    run_output_dir = join(cfg["output_dir"], cfg["name"])
    if os.path.exists(run_output_dir) and cfg.get("overwrite", False):
        print(f"Output dir {run_output_dir} already exists, overwriting")
        shutil.rmtree(run_output_dir)
    os.makedirs(run_output_dir, exist_ok=True)

    if cfg.get("use_wandb", False):
        os.makedirs(cfg["wandb"]["local_wandb_dir"], exist_ok=True)

    with open(join(run_output_dir, "config.yaml"), "w") as f:
        f.write(cfg_yaml)

    # --- seed ---
    pl.seed_everything(cfg["training"]["train_seed"])

    # --- special case param hacks (kept from _train.py style) ---
    if cfg["data"]["kwargs"].get("pert_col") == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

    # --- sentence length (same logic pattern) ---
    try:
        sentence_len = cfg["model"]["cell_set_len"]
    except KeyError:
        if cfg["model"]["name"].lower() in ["cpa", "scvi"] or cfg["model"]["name"].lower().startswith("scgpt"):
            if "cell_sentence_len" in cfg["model"]["kwargs"] and cfg["model"]["kwargs"]["cell_sentence_len"] > 1:
                sentence_len = cfg["model"]["kwargs"]["cell_sentence_len"]
                cfg["training"]["batch_size"] = 1
            else:
                sentence_len = 1
        else:
            try:
                sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["n_positions"]
            except Exception:
                sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["max_position_embeddings"]

    # --- datamodule ---
    data_module: PerturbationDataModule = get_datamodule(
        cfg["data"]["name"],
        cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=sentence_len,
    )
    print(f"[debug] data.name={cfg['data']['name']}")
    print(f"[debug] data.kwargs keys={list(cfg['data']['kwargs'].keys())}")
    with open(join(run_output_dir, "data_module.torch"), "wb") as f:
        data_module.save_state(f)

    data_module.setup(stage="fit")
    var_dims = data_module.get_var_dims()

    # --- decoder_cfg ---
    if cfg["data"]["kwargs"]["output_space"] == "gene":
        gene_dim = var_dims.get("hvg_dim", 2000)
    else:
        gene_dim = var_dims.get("gene_dim", 2000)
    decoder_cfg = dict(
        latent_dim=var_dims["output_dim"],
        gene_dim=gene_dim,
        hidden_dims=cfg["model"]["kwargs"].get("decoder_hidden_dims", [1024, 1024, 512]),
        dropout=cfg["model"]["kwargs"].get("decoder_dropout", 0.1),
        residual_decoder=cfg["model"]["kwargs"].get("residual_decoder", False),
    )
    cfg["model"]["kwargs"]["decoder_cfg"] = decoder_cfg

    # --- build model ---
    model = get_lightning_module(
        cfg["model"]["name"],
        cfg["data"]["kwargs"],
        cfg["model"]["kwargs"],
        cfg["training"],
        var_dims,
    )
    print(f"Model params ≈ {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3:.2f} GB")

    # --- loggers / callbacks (same style) ---
    loggers = get_loggers(
        output_dir=cfg["output_dir"],
        name=cfg["name"],
        wandb_project=cfg["wandb"]["project"],
        wandb_entity=cfg["wandb"]["entity"],
        local_wandb_dir=cfg["wandb"]["local_wandb_dir"],
        use_wandb=cfg["use_wandb"],
        cfg=cfg,
    )
    for lg in loggers:
        if isinstance(lg, WandbLogger):
            with open(os.path.join(run_output_dir, "wandb_path.txt"), "w") as f:
                f.write(lg.experiment.path)
            break

    ckpt_callbacks = get_checkpoint_callbacks(
        cfg["output_dir"],
        cfg["name"],
        cfg["training"]["val_freq"],
        cfg["training"].get("ckpt_every_n_steps", 4000),
    )
    callbacks = ckpt_callbacks + [BatchSpeedMonitorCallback()]
    if cfg["model"]["name"] == "state":
        callbacks.append(GradNormCallback())
    if cfg["training"].get("use_mfu", False) and cfg["model"]["name"] == "state":
        mfu_cb = ModelFLOPSUtilizationCallback(
            available_flops=cfg["training"]["mfu_kwargs"]["available_flops"],
            use_backward=cfg["training"]["mfu_kwargs"]["use_backward"],
            logging_interval=cfg["training"]["mfu_kwargs"]["logging_interval"],
            cell_set_len=cfg["model"]["kwargs"]["cell_set_len"],
            window_size=cfg["training"]["mfu_kwargs"]["window_size"],
        )
        callbacks.append(mfu_cb)
        callbacks.append(CumulativeFLOPSCallback(use_backward=cfg["training"]["cumulative_flops_use_backward"]))

    plugins = [MixedPrecision(precision="bf16-mixed", device="cuda")] if cfg["model"]["name"].lower().startswith("scgpt") else []
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=cfg["training"].get("devices", 1),
        strategy=cfg["training"].get("strategy", "auto"),
        max_steps=cfg["training"]["max_steps"],
        check_val_every_n_epoch=None,
        val_check_interval=cfg["training"]["val_freq"],
        logger=loggers,
        plugins=plugins,
        callbacks=callbacks,
        gradient_clip_val=cfg["training"]["gradient_clip_val"] if cfg["model"]["name"].lower() != "cpa" else None,
        use_distributed_sampler=False,
    )
    if "log_every_n_steps" in cfg["training"]:
        trainer_kwargs["log_every_n_steps"] = cfg["training"]["log_every_n_steps"]

    print(f"Building trainer with kwargs: {trainer_kwargs}")
    trainer = pl.Trainer(**trainer_kwargs)
    print("Trainer built successfully")

    # ------------------- checkpoint load -------------------
    #checkpoint_path = os.path.join("zhanghy/orcd/scratch/zhanghy/sc-perturb", "last.ckpt")
    #print("Checkpoint path:", checkpoint_path)
    #if not os.path.exists(checkpoint_path):
    #    checkpoint_path = None
    checkpoint_path = None
    manual_init = cfg["model"]["kwargs"].get("init_from", None)
    if checkpoint_path is None and manual_init is not None:
        print(f"Loading manual checkpoint from {manual_init}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(manual_init, map_location=device, weights_only=False)
        model_state = model.state_dict()
        checkpoint_state = checkpoint["state_dict"]

        # Handle output_space change by rebuilding decoder
        ckpt_output_space = checkpoint.get("hyper_parameters", {}).get("output_space", "gene")
        cur_output_space = cfg["data"]["kwargs"]["output_space"]
        if ckpt_output_space != cur_output_space:
            print(f"Output space mismatch: ckpt='{ckpt_output_space}' vs current='{cur_output_space}'. Rebuilding decoder.")
            if cfg["model"]["kwargs"].get("gene_decoder_bool", True) is not False:
                model.decoder_cfg = decoder_cfg
                model._build_decoder()
                model._decoder_externally_configured = True
                print(f"New decoder: output_space='{cur_output_space}', gene_dim={decoder_cfg['gene_dim']}")

        # Pert encoder input-dim mismatch → rebuild
        pert_key = "pert_encoder.0.weight"
        if pert_key in checkpoint_state:
            ckpt_pert_dim = checkpoint_state[pert_key].shape[1]
            if hasattr(model, "pert_dim") and ckpt_pert_dim != model.pert_dim:
                from state.tx.models.utils import build_mlp
                model.pert_encoder = build_mlp(
                    in_dim=model.pert_dim,
                    out_dim=model.hidden_dim,
                    hidden_dim=model.hidden_dim,
                    n_layers=model.n_encoder_layers,
                    dropout=model.dropout,
                    activation=model.activation_class,
                )
                print(f"Rebuilt pert_encoder: model.pert_dim={model.pert_dim}, ckpt expects {ckpt_pert_dim}")

        # Shape-filtered load
        filtered = {k: v for k, v in checkpoint_state.items() if k in model_state and v.shape == model_state[k].shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"Loaded checkpoint with shape filtering: missing={len(missing)}, unexpected={len(unexpected)}")
        ckpt_path = None  # start training fresh from the loaded weights

    # ------------------- our modifications: freeze attention + GRN penalty -------------------
    trainable = freeze_all_but_attention(model)
    print(f"Trainable attention params: {sum(p.numel() for p in trainable):,}")

    if "grn" in cfg and cfg["grn"].get("path"):

        def gene_to_ensembl(gene_ids, species="human"):
            mg = mygene.MyGeneInfo()
            hits = mg.querymany(gene_ids,
                        scopes="symbol,alias,old_symbol,ensembl.gene,ensembl.transcript",
                        fields="ensembl.gene", species=species, as_dataframe=False)

            manual = {
                "dvl1": "ENSG00000107404",
                "nbl1": "ENSG00000158747",
                "al391650.1": "ENSG00000236782",
                "trnp1": "ENSG00000253368",
            }

            out = []
            for h in hits:
                q = h.get("query")
                e = h.get("ensembl")
                gid = e[0]["gene"] if isinstance(e, list) else e.get("gene") if isinstance(e, dict) else None
                gid = gid or manual.get(q.lower())
                if not gid:
                    raise ValueError(f"No Ensembl ID found for '{q}' (species='{species}')")
                out.append(gid)
            return out

        ordering = gene_to_ensembl(data_module.get_var_names())
        L, used_order = load_grn_laplacian(csv_path=cfg["grn"]["path"], gene_order=ordering)
        grn_reg = GRNReg(model, L, grn_lambda=float(cfg["grn"].get("lambda", 1e-3)))
    else:
        grn_reg = None

    print("Done loading grn reg")

    # Wrap training_step to add GRN term (no edits to model class)
    orig_training_step = model.training_step
    def training_step_with_grn(*args, **kwargs):
        out = orig_training_step(*args, **kwargs)
        if grn_reg is not None:
            device = next(model.parameters()).device
            reg = grn_reg.penalty(device)
            if isinstance(out, dict):
                out["loss"] = out["loss"] + reg
            else:
                out = out + reg
            model.log("grn_reg", reg, prog_bar=True, on_step=True, on_epoch=False)
        return out
    model.training_step = training_step_with_grn

    # Optimizer on attention-only params
    params = [p for p in model.parameters() if p.requires_grad]
    lr = cfg["model"]["kwargs"].get("lr", cfg["training"].get("lr", 2e-5))
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=cfg["training"].get("weight_decay", 0.01))
    model.configure_optimizers = lambda: optimizer

    # ------------------- train -------------------
    print("Starting trainer.fit()...")
    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)
    print("trainer.fit() done.")

    # Save final checkpoint if not already present
    final_ckpt = os.path.join(ckpt_callbacks[0].dirpath, "final.ckpt")
    if not os.path.exists(final_ckpt):
        trainer.save_checkpoint(final_ckpt)
    print(f"Saved: {final_ckpt}")


def main():
    parser = ap.ArgumentParser("Attention-only fine-tuning with GRN prior (integrated with _train.py logic)")
    parser.add_argument("--hparams", default="./finetune.yaml", help="Path to YAML hparams file")
    parser.add_argument("hydra_overrides", nargs="*", help="Hydra-style overrides")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.hparams)
    cli_cfg = OmegaConf.from_dotlist(args.hydra_overrides)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    run(cfg)

if __name__ == "__main__":
    main()

