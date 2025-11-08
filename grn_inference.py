#!/usr/bin/env python3
# infer.py

import argparse, os, sys, warnings
from pathlib import Path

# sqlite shim for lamindb on some clusters
try:
    import pysqlite3 as sqlite3  # type: ignore
    sys.modules["sqlite3"] = sqlite3
except Exception:
    pass

import lamindb as ln
import lamindb_setup
import bionty as bt
import anndata as ad
import numpy as np
import pandas as pd
import torch

LOCAL_DIR = Path("/tmp/zhanghy/lamindb").resolve()
print(f"Using local instance directory: {LOCAL_DIR}")
if not LOCAL_DIR.exists():
    print("-> Initializing local anonymous instance...")
    ln.setup.init(storage=str(LOCAL_DIR))
else:
    print("-> Local directory already exists, skipping init.")
ln.connect("anonymous/lamindb")
print("\n✅ Connected successfully")
print(f"lamindb version     : {ln.__version__}")
print(f"lamindb-setup version: {lamindb_setup.__version__}")

from huggingface_hub import hf_hub_download
from scprint import scPrint

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- args ----------------
p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="Path to .h5ad/.h5/.hdf5 or a dir containing one")
p.add_argument("--outdir", default="grn_out")
p.add_argument("--cell-type", default="H1-hESC")
p.add_argument("--organism-id", default="NCBITaxon:9606")
p.add_argument("--layer", default=None, help="AnnData layer; use 'X' or leave empty for .X")
p.add_argument("--max-cells", type=int, default=300)
p.add_argument("--num-genes", type=int, default=3000)
p.add_argument("--how", default="random expr")
p.add_argument("--preprocess", default="softmax")
p.add_argument("--head-agg", default="mean")
p.add_argument("--filtration", default="none")
p.add_argument("--forward-mode", default="none")
p.add_argument("--ckpt-repo", default="jkobject/scPRINT")
p.add_argument("--ckpt-file", default="v2-medium.ckpt")
args = p.parse_args()

# ---------- organism row present & normalized ----------
OID = args.organism_id
org = bt.Organism.filter(ontology_id=OID).first()
if org is None:
    o = bt.Organism.filter(name="Homo sapiens").first()
    if o is None:
        o = bt.Organism(name="Homo sapiens", ontology_id=OID)
        o.save()
        print("Created Organism: Homo sapiens with", OID)
    else:
        o.ontology_id = OID
        o.save()
        print("Updated Organism: Homo sapiens now has", OID)
else:
    print("Found Organism with", OID)

# ---------- load anndata ----------
def find_h5ad(pathlike: str | Path) -> Path:
    pth = Path(pathlike)
    if pth.is_file() and pth.suffix == ".h5ad":
        return pth
    if pth.is_dir():
        cands = sorted(pth.glob("*.h5ad"))
        if cands:
            return cands[0]
    raise FileNotFoundError(f"No .h5ad found at {pth}")

adata_path = find_h5ad(args.input)
print("Reading:", adata_path)
adata: ad.AnnData = ad.read_h5ad(adata_path)

# ---------- harmonize obs & subset to H1 ----------
if "organism_ontology_term_id" not in adata.obs.columns:
    adata.obs["organism_ontology_term_id"] = OID
else:
    adata.obs["organism_ontology_term_id"] = OID

ct_col = "cell_type"
if ct_col not in adata.obs.columns:
    raise RuntimeError(f"`{ct_col}` column not found in adata.obs")

aliases = [args.cell_type, "H1", "H1 hESC", "H1_ESC", "H1-hESC (unperturbed)", "H1-hESC"]
present = set(adata.obs[ct_col].astype(str).unique())
chosen = next((ct for ct in aliases if ct in present), None)
if chosen is None:
    chosen = next((v for v in present if str(v).upper().startswith("H1")), None)
if chosen is None:
    raise RuntimeError(f"Couldn’t find an H1-like cell_type. Seen: {sorted(list(present))[:12]}")

print("Subsetting to H1 cell line:", chosen)
adata = adata[adata.obs[ct_col] == chosen].copy()

# --- choose layer ---
layer = args.layer
if layer is None or str(layer).lower() in {"", "x", "none"}:
    layer = None
elif not (adata.layers and layer in adata.layers.keys()):
    avail = list(adata.layers.keys()) if adata.layers else []
    raise RuntimeError(f"Layer '{args.layer}' not present. Available: {avail or ['<none>']} (use --layer X to use .X)")

# ---------- lightweight prep ----------
# keep genes with some counts
X = adata.layers[layer] if layer else adata.X
if not isinstance(X, np.ndarray):
    X = X.A if hasattr(X, "A") else X.toarray()

gene_var = X.sum(axis=0)
keep_idx = np.argsort(gene_var)[::-1][: min(args.num_genes, X.shape[1])]
adata = adata[:, keep_idx].copy()

# cell downsample
if adata.n_obs > args.max_cells:
    sel = np.random.RandomState(0).choice(adata.n_obs, args.max_cells, replace=False)
    adata = adata[sel].copy()

print("adata_prep_obs:", adata.obs.columns)

# ---------- load scPRINT checkpoint ----------
def load_model():
    ckpt_path = "small.ckpt"
    model = scPrint.load_from_checkpoint(
        ckpt_path,
        precpt_gene_emb=None,
        transformer="normal",
    )
    return model

ckpt = load_model()

# ---------- build GRN task ----------
try:
    from scprint.tasks.grn import GNInfer
except Exception:
    try:
        from scprint.tasks.gn import GNInfer
    except Exception:
        try:
            from scprint.tasks import GNInfer
        except Exception as e:
            raise ImportError("Could not import GNInfer from scprint") from e

print("Organism in this instance:", bt.Organism.filter(ontology_id=OID).first() is not None)


# --- normalize organism columns so scdataloader.load_genes() finds the record
import pandas as pd
import bionty as bt
import scdataloader.utils as scu
import scdataloader.collator as scc

HUMAN_OID = "NCBITaxon:9606"
MOUSE_OID = "NCBITaxon:10090"
ADATA_FOR_GENES = adata  # fallback source

_orig_load_genes = scc.load_genes

def _norm(s):
    s = str(s).strip().lower()
    if s in {"human","homo sapiens","9606","ncbitaxon:9606"}: return "NCBITaxon:9606"
    if s in {"mouse","mus musculus","10090","ncbitaxon:10090"}: return "NCBITaxon:10090"
    return s

def load_genes_label_to_request(organisms):
    df = _orig_load_genes(organisms)
    req = organisms[0] if isinstance(organisms, (list, tuple)) else organisms
    req = str(req)
    req_canon = _norm(req)
    canon = df["organism"].map(_norm)
    df.loc[canon == req_canon, "organism"] = req
    return df

scc.load_genes = load_genes_label_to_request
# (optional, if some code calls utils.load_genes)
try:
    import scdataloader.utils as scu
    scu.load_genes = load_genes_label_to_request
except Exception:
    pass

# --- fix Collator._setup: treat empty valid_genes as None, guard start_idx ---
_orig_setup = scc.Collator._setup

import scdataloader.utils as _scu
import scdataloader.collator as _scc
import scprint.tasks.grn as _grn

# Build a gene table from the in-memory AnnData
def _genedf_from_adata_exact(A, organisms):
    idx = A.var_names.astype(str)
    base = pd.DataFrame(index=idx)
    base.index.name = "ensembl_gene_id"
    sym_col = next((c for c in ["gene_symbol","gene_name","symbol","feature_name"] if c in A.var.columns), None)
    base["symbol"] = A.var[sym_col].astype(str).values if sym_col else idx
    up = base["symbol"].str.upper(); idx_up = idx.str.upper()
    base["mt"]   = up.str.startswith("MT-") | idx_up.str.startswith("MT-")
    base["ribo"] = up.str.startswith(("RPS","RPL"))
    base["hb"]   = up.str.match(r"^HB(?!P)")
    base = base[~base.index.duplicated(keep="first")]
    blocks = []
    for org in [str(o) for o in organisms]:
        b = base.copy()
        b["organism"] = org  # exact string, so Collator == succeeds
        blocks.append(b)
    out = pd.concat(blocks, axis=0).sort_index()
    out["organism"] = out["organism"].astype(str)
    return out

# Replace load_genes everywhere to ignore any DB access
def load_genes_from_adata_exact(organisms):
    return _genedf_from_adata_exact(adata, organisms)

_scu.load_genes = load_genes_from_adata_exact
_scc.load_genes = load_genes_from_adata_exact

# Safe Collator that ignores empty/zero-overlap valid_genes and never crashes on indexing
class _SafeCollator(_scc.Collator):
    def _setup(self, *args, **kwargs):
        # Accept both call styles: (org_to_id, valid_genes, genelist) OR (genedf, org_to_id, valid_genes, genelist)
        genedf = None; org_to_id = None; valid_genes = None; genelist = []
        if len(args) == 3:
            org_to_id, valid_genes, genelist = args
        elif len(args) == 4:
            genedf, org_to_id, valid_genes, genelist = args
        else:
            genedf      = kwargs.get("genedf", None)
            org_to_id   = kwargs.get("org_to_id", None)
            valid_genes = kwargs.get("valid_genes", None)
            genelist    = kwargs.get("genelist", [])

        # Always build from current AnnData; ignore provided genedf
        genedf = load_genes_from_adata_exact(self.organisms).copy()
        genedf.index = genedf.index.astype(str)
        genedf["organism"] = genedf["organism"].astype(str)

        # Normalize inputs
        orgs = [str(o) for o in self.organisms]
        org_to_id = org_to_id if isinstance(org_to_id, dict) else None

        # Decide if we actually apply valid_genes (only if non-empty AND overlapping)
        use_valid = None
        if isinstance(valid_genes, (list, tuple, np.ndarray)) and len(valid_genes) > 0:
            vg = [str(x) for x in valid_genes]
            if len(set(vg) & set(genedf.index)) > 0:
                use_valid = vg

        tot = genedf if use_valid is None else genedf.loc[genedf.index.isin(use_valid)]

        self.org_to_id = org_to_id
        self.to_subset = {}
        self.accepted_genes = {}
        self.start_idx = {}
        self.organism_ids = set(org_to_id[k] for k in orgs if org_to_id and k in org_to_id) or set(orgs)

        for organism in orgs:
            org_key = org_to_id[organism] if (org_to_id and organism in org_to_id) else organism

            mask = (tot["organism"] == organism).values
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                # if filtering wiped this organism, fall back to unfiltered genedf
                fallback = (genedf["organism"] == organism).values
                if not fallback.any():
                    raise RuntimeError(f"No genes for organism '{organism}' in synthesized table.")
                self.start_idx[org_key] = int(np.where(fallback)[0][0])
                ogenedf = genedf[genedf["organism"] == organism]
                self.accepted_genes[org_key] = np.ones(len(ogenedf), dtype=bool)
                if genelist:
                    self.to_subset[org_key] = np.ones(len(ogenedf), dtype=bool)
                continue

            self.start_idx[org_key] = int(idxs[0])

            ogenedf = genedf[genedf["organism"] == organism]
            if use_valid is not None:
                self.accepted_genes[org_key] = ogenedf.index.isin(use_valid).values
            else:
                self.accepted_genes[org_key] = np.ones(len(ogenedf), dtype=bool)

            if genelist:
                base = ogenedf if use_valid is None else ogenedf.loc[ogenedf.index.isin(use_valid)]
                self.to_subset[org_key] = base.index.isin(genelist).values

# Patch both modules so GNInfer uses this exact class
_scc.Collator = _SafeCollator
_grn.Collator = _SafeCollator

# --- Patch GNInfer.save to fix var/GRN shape mismatch ---
import numpy as np, pandas as pd, anndata as ad
import scprint.tasks.grn as _grn


def _save_edges_only(self, grn, subadata):
    # to numpy
    if torch is not None and isinstance(grn, torch.Tensor):
        G = grn.detach().cpu().numpy()
    else:
        G = np.asarray(grn)

    # if it’s (n, n, H) collapse heads
    if G.ndim == 3 and G.shape[0] == G.shape[1]:
        G = G.mean(-1)

    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise RuntimeError(f"Expected square GRN, got {G.shape}")

    n = int(G.shape[0])

    # choose gene names that match the GRN size
    genes = None
    for cand in [
        list(getattr(self, "_model_genes", [])),
        (list(map(str, subadata.var_names)) if getattr(subadata, "n_vars", 0) == n else None),
        list(getattr(self, "_adata_var_names", [])),
    ]:
        if isinstance(cand, list) and len(cand) >= n:
            genes = cand[:n]
            break
    if genes is None:
        genes = [f"G{i}" for i in range(n)]

    # build edges dataframe (dense)
    idx = pd.MultiIndex.from_product([genes, genes], names=["source", "target"])
    df = pd.DataFrame({"weight": G.reshape(-1)}, index=idx).reset_index()
    return df

_grn.GNInfer.save = _save_edges_only

print("Create GNInfer...")
gn = GNInfer(
    how=args.how,
    preprocess=args.preprocess,
    head_agg=args.head_agg,
    filtration=args.filtration,
    forward_mode=args.forward_mode,
    num_genes=args.num_genes,
    max_cells=args.max_cells,
    dtype=torch.float32,
    doplot=False,
    #layer="X",
)
gn._model_genes = list(getattr(ckpt, "genes", []))
gn._adata_var_names = list(map(str, adata.var_names))

# ---------- run ----------
print("Start inference...")
try:
    edges_df = gn(ckpt, adata, cell_type="H1")
except RuntimeError as e:
    if "not attached to a Trainer" in str(e):
        warnings.warn(str(e))
        edges_df = gn(ckpt, adata)
    else:
        raise

# ---------- save ----------
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
edges_path = outdir / "edges.csv"
edges_df.to_csv(edges_path, index=False)
print(f"→ saved: {edges_path}")
print(edges_df.head(10))

