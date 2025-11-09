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
p.add_argument("--layer", default=None, help="AnnData layer; leave empty for .X")
p.add_argument("--max-cells", type=int, default=200000)
p.add_argument("--num-genes", type=int, default=18080)
p.add_argument("--how", default="given")
p.add_argument("--preprocess", default="softmax")
p.add_argument("--head-agg", default="mean")
p.add_argument("--filtration", default="none")
p.add_argument("--forward-mode", default="none")
p.add_argument("--ckpt-repo", default="jkobject/scPRINT")
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
        transformer="flash",
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
import scdataloader.utils as _scu
import scdataloader.collator as _scc

HUMAN_OID = "NCBITaxon:9606"
MOUSE_OID = "NCBITaxon:10090"
ADATA_FOR_GENES = adata  # fallback source

_orig_load_genes = _scc.load_genes

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

_scc.load_genes = load_genes_label_to_request
# (optional, if some code calls utils.load_genes)
try:
    import scdataloader.utils as scu
    _scu.load_genes = load_genes_label_to_request
except Exception:
    pass

# --- fix Collator._setup: treat empty valid_genes as None, guard start_idx ---
_orig_setup = _scc.Collator._setup

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

_RealCollator = _scc.Collator
def Collator(*args, **kwargs):
    # inject our final list; Collator supports 'genelist' (and often 'valid_genes')
    kwargs["genelist"] = list(final_genes)
    kwargs["valid_genes"] = list(final_genes)  # harmless if unused
    return _RealCollator(*args, **kwargs)

# Patch both modules so GNInfer uses this exact class
_scc.Collator = Collator
_grn.Collator = Collator

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

# --------- debug ---------
import re
import scdataloader.collator as scc
import pandas as pd
import re

# normalize raw var names (strip Ensembl version suffixes)
adata.var_names = adata.var_names.str.replace(r"\.\d+$", "", regex=True)
adata.var_names_make_unique()

# don't drop anything implicitly
if hasattr(_scc, "drop"):
    _scc.drop = set()

# intersection = final gene list
orgs = ["NCBITaxon:9606"]
vocab_df = _scc.load_genes(orgs)
vocab_ids = set(map(str, vocab_df.index))
adata_ids = set(map(str, adata.var_names))
final_genes = sorted(adata_ids & vocab_ids)       # this should be 18,080
print("[metrics] num final genes:", len(final_genes))

# Patch both modules so the call in scprint.tasks.grn uses our wrapper
_scc.Collator = Collator
_grn.Collator = Collator

print("Create GNInfer...")
gn = GNInfer(
    how=args.how,
    genes=final_genes,
    preprocess=args.preprocess,
    head_agg=args.head_agg,
    filtration=args.filtration,
    forward_mode=args.forward_mode,
    num_genes=args.num_genes,
    max_cells=args.max_cells,
    dtype=torch.float16,
    doplot=False,
)

# subset adata to exactly the usable genes and pin num_genes
adata = adata[:, final_genes]
if hasattr(gn, "num_genes"):
    gn.num_genes = len(final_genes)
    #gn.num_genes = len(adata)

gn._model_genes = list(getattr(ckpt, "genes", []))
gn._adata_var_names = list(map(str, adata.var_names))

# ---------- run ----------
print("Start inference...")
try:
    with torch.no_grad():
        edges_df = gn(ckpt, adata, cell_type="H1")
except RuntimeError as e:
    if "not attached to a Trainer" in str(e):
        warnings.warn(str(e))
        edges_df = gn(ckpt, adata)
    else:
        raise

print(f"[metrics] realized rows: {len(edges_df):,}  (expected {len(final_genes)**2:,})")


# ---------- save ----------
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
edges_path = outdir / "edges_all.csv"
edges_df.to_csv(edges_path, index=False)
print(f"Saved to: {edges_path}")
print(edges_df.head(10))

