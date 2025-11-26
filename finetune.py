import io
import argparse as ap
import os, math, sys, warnings
from scipy import sparse
from pathlib import Path

import contextlib
from tqdm import tqdm

import anndata as ad
import numpy as np
import pandas as pd
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

from contextlib import redirect_stdout

def load_cfg_from_yaml(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config YAML not found: {path}")
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

    # Check hparams are specified
    for top in ["output_dir", "name", "data", "model", "training"]:
        if top not in cfg:
            raise KeyError(f"Missing required top-level key: {top}")

    init_from = cfg["model"]["kwargs"].get("init_from")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg

def print_full_model(root: torch.nn.Module):
    print("\n=== Model ===")
    for module_name, module in root.named_modules():
        indent = "  " * (module_name.count("."))
        print(f"{indent}{module_name or '<root>'}  ({type(module).__name__})")

        # Print parameters *directly owned by this module*
        for param_name, param in module.named_parameters(recurse=False):
            print(f"{indent}  └─ {param_name:20s} shape={tuple(param.shape)}  requires_grad={param.requires_grad}")

# ----------------- GRN utils -----------------
def load_grn_laplacian(
    csv_path: str,
    gene_order: list[str],
    symmetric: str = "max",
):
    """
    Build L = I - D^{-1/2} A D^{-1/2} from a CSV edgelist with header:
        source,target,weight
    If gene_order is provided, we first drop any edges with genes not in that list,
    then build the Laplacian ONLY over the subset of genes that actually appear in
    at least one remaining edge (so isolated genes are not penalized).
    Returns:
        (L, ordered_indices)
        - L: scipy.sparse.csr_matrix or None if no usable edges remain
        - ordered_indices: list[int] of genes corresponding to L's row/col order
    """

    df = pd.read_csv(csv_path)
    if not {"source", "target"}.issubset(df.columns):
        raise ValueError("CSV must have columns: source,target[,weight]")
    if "weight" not in df.columns:
        print("[WARN] Could not find column 'weight', setting all to 1.0")
        df["weight"] = 1.0

    src = df["source"].astype(str)
    dst = df["target"].astype(str)
    w   = df["weight"].astype(np.float32).to_numpy()

    gene_set = set(gene_order)
    mask = src.isin(gene_set) & dst.isin(gene_set)
    if not mask.any():
        print("[WARN] Nothing to penalize")
        return None, []
    src_kept = src[mask]
    dst_kept = dst[mask]
    w = w[mask.to_numpy()]

    used_set = set(src_kept) | set(dst_kept)
    used_order = [g for g in gene_order if g in used_set]
    ordered_indices = [i for i, g in enumerate(gene_order) if g in used_set]

    # map to compact indices
    gene_to_index = {g: i for i, g in enumerate(used_order)}
    iu = src_kept.map(gene_to_index).to_numpy(dtype=np.int64)
    iv = dst_kept.map(gene_to_index).to_numpy(dtype=np.int64)

    n = len(ordered_indices)
    
    print(f"Laplacian has shape ({n}, {n})")
    if n == 0:
        return None, []

    # build COO then coalesce duplicates by summing
    A = sparse.coo_matrix((w, (iu, iv)), shape=(n, n), dtype=np.float32).tocsr()

    # symmetrize
    if symmetric == "max":
        A = A.maximum(A.T)
    else:
        raise ValueError("symmetric must be 'max'")

    # normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    if n == 0 or (deg == 0).all():
        return None, []
    Dinv = sparse.diags((1.0 / np.sqrt(deg)).astype(np.float32))
    L = sparse.eye(n, dtype=np.float32) - (Dinv @ A @ Dinv)

    print("Finished computing full Laplacian")
    return L.tocsr(), ordered_indices

class GRNReg:
    """
    GRN-based regularizer for attention layers.

    - L_csr: scipy.sparse CSR Laplacian, shape [n_genes, n_genes]
    - gene_embed: torch.Tensor of shape [n_genes, in_dim]
      giving a fixed embedding / feature vector for each gene.
    """

    def __init__(self, module, L_csr: sparse.csr_matrix,
                ordered_indices, 
                grn_lambda: float = 1e-3):
        self.module = module
        self.grn_lambda = float(grn_lambda)

        # store Laplacian in COO form for PyTorch
        row, col = L_csr.nonzero()
        idx = np.vstack([row, col]).astype(np.int64)
        self.I = torch.tensor(idx, dtype=torch.long)                 # [2, nnz]
        self.V = torch.tensor(L_csr.data, dtype=torch.float32)       # [nnz]
        self.shape = L_csr.shape                                     # (n_genes, n_genes)

        # gene embeddings: [n_genes, in_dim]
        W = module.basal_encoder.weight.detach()
        self.gene_embed = W[:, ordered_indices].T.clone()
        self.gene_embed.requires_grad_(False)
        print(f"Gene embed weights shape: {self.gene_embed.shape}")

    def _L(self, device):
        return torch.sparse_coo_tensor(
            self.I.to(device),
            self.V.to(device),
            self.shape,
            device=device,
        )

    def penalty(self, device: torch.device):
        if self.grn_lambda <= 0.0:
            return torch.tensor(0.0, device=device)

        L = self._L(device)
        E = self.gene_embed.to(device)   # [n_genes, in_dim]
        n_genes, in_dim = E.shape

        # sanity check: Laplacian and embeddings must agree on n_genes
        if n_genes != self.shape[0]:
            raise ValueError(
                f"gene_embed has {n_genes} genes but Laplacian has {self.shape[0]}"
            )

        total = torch.tensor(0.0, device=device)

        for m in self.module.modules():
            if hasattr(m, "q_proj") and hasattr(m, "k_proj"):
                Wq = m.q_proj.weight      # [out_dim, in_dim]
                Wk = m.k_proj.weight

                if Wq.shape[1] != in_dim or Wk.shape[1] != in_dim:
                    raise ValueError(
                        f"in_dim mismatch: gene_embed has {in_dim}, "
                        f"but q_proj/k_proj expect {Wq.shape[1]}"
                    )

                # per-gene queries / keys
                Q = E @ Wq.T              # [n_genes, out_dim]
                K = E @ Wk.T              # [n_genes, out_dim]

                # Laplacian smoothness: tr(Q^T L Q) + tr(K^T L K)
                LQ = torch.sparse.mm(L, Q)   # [n_genes, out_dim]
                LK = torch.sparse.mm(L, K)

                total = total + (Q * LQ).sum() + (K * LK).sum()

        return self.grn_lambda * total

# -------------- fine-tune helpers --------------
import torch
from torch import nn


def freeze_all_but_last_n_layers(root: torch.nn.Module, n_layers: int):
    """
    Freeze all parameters except those in the last `n_layers`
    transformer decoder blocks in `root.transformer_backbone.layers`.

    Within each selected block, unfreeze:
      - self_attn.{q_proj,k_proj,v_proj,o_proj}
      - mlp (all params)
      - norm modules inside the block (LlamaRMSNorm / LayerNorm)

    Returns:
        List[nn.Parameter]: the list of trainable parameters.
    """
    # 1) Freeze transformer backbone and basal encoder
    for p in root.transformer_backbone.parameters():
        p.requires_grad = False
    for p in root.basal_encoder.parameters():
        p.requires_grad = False

    # 2) Get the transformer layers explicitly
    if not hasattr(root, "transformer_backbone"):
        raise ValueError("Model has no attribute 'transformer_backbone'")

    backbone = root.transformer_backbone

    if not hasattr(backbone, "layers"):
        raise ValueError("transformer_backbone has no attribute 'layers'")

    blocks = list(backbone.layers)
    if len(blocks) == 0:
        print("[WARN] No transformer blocks found in transformer_backbone.layers")
        return []

    if n_layers <= 0:
        print("[WARN] n_layers <= 0, all parameters remain frozen")
        return []

    # 3) Pick last n_layers blocks (clip if n_layers > #blocks)
    n_layers = min(n_layers, len(blocks))
    blocks_to_train = blocks[-n_layers:]

    trainable = []

    num_attn = 0
    num_mlp = 0
    num_norm = 0

    for block in blocks_to_train:
        # ---- attention projections ----
        if hasattr(block, "self_attn"):
            attn = block.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(attn, name):
                    mod = getattr(attn, name)
                    for p in mod.parameters():
                        p.requires_grad = True
                        trainable.append(p)
                        num_attn += 1

        # ---- MLP / feed-forward ----
        if hasattr(block, "mlp"):
            for p in block.mlp.parameters():
                p.requires_grad = True
                trainable.append(p)
                num_mlp += 1

        # ---- Norms inside the block (RMSNorm / LayerNorm) ----
        for sub in block.modules():
            cls_name = sub.__class__.__name__
            if isinstance(sub, torch.nn.LayerNorm) or cls_name in ("LlamaRMSNorm", "RMSNorm"):
                for p in sub.parameters():
                    p.requires_grad = True
                    trainable.append(p)
                    num_norm += 1
    
    for p in root.basal_encoder.parameters():
        p.requires_grad = False

    print("Trainable parameter counts:")
    print(f"Attention: {num_attn}, MLP: {num_mlp}, Norm: {num_norm}")

    if not trainable:
        print("[WARN] No trainable parameters found in selected blocks")
    
    return trainable

# -------------- run training --------------
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

    # ------------------- freeze last n layers + GRN loss -------------------
    trainable = freeze_all_but_last_n_layers(root=model, n_layers=2)
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print_full_model(root=model)

    import re, mygene
   
    def gene_to_ensembl(gene_ids, species="human", batch_size=1000, progress=True):
        with open("grn_out/symbols_dict.pkl", "rb") as f:
            symbols_map = pickle.load(f)
        return [symbols_map.get(gid) for gid in gene_ids] 
        
    if "grn" in cfg and cfg["grn"].get("path"):
        ens_list = gene_to_ensembl(data_module.get_var_names())
        print("Load GRN and compute Laplacian")
        L, ordered_indices = load_grn_laplacian(csv_path=cfg["grn"]["path"], gene_order=ens_list)
        print("Load GRN regularizer")
        grn_reg = GRNReg(model, L, grn_lambda=float(cfg["grn"].get("lambda", 1e-3)), ordered_indices=ordered_indices)
    else:
        print("[WARN] No GRN regularizer found")
        grn_reg = None

    # Wrap training_step to add GRN term
    print("Set up training")
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

    # Optimizer on trainable params
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
    parser = ap.ArgumentParser("Fine-tuning with GRN prior")
    parser.add_argument("--hparams", default="./finetune.yaml", help="Path to YAML hparams file")
    parser.add_argument("hydra_overrides", nargs="*", help="Hydra-style overrides")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.hparams)
    cli_cfg = OmegaConf.from_dotlist(args.hydra_overrides)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    run(cfg)

if __name__ == "__main__":
    main()

