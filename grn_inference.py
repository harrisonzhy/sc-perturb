import lamindb as ln
import os

import pysqlite3 as sqlite3
import sys; sys.modules['sqlite3'] = sqlite3

storage = "/orcd/home/002/zhanghy/orcd/scratch/zhanghy/lamindb"
os.makedirs(storage, exist_ok=True)
ln.setup.init(storage=storage, schema="bionty")
ln.connect()
print("LaminDB ready at", storage)

import scanpy as sc
import pandas as pd

from scprint import scPrint
from scdataloader import Preprocessor, utils
from scprint.tasks import GNInfer, Embedder, Denoiser, withknn
from scdataloader.utils import load_genes

import argparse

def preprocess_data(path):
    adata = sc.read(path)
    adata.obs_names_make_unique()

    adata.obs.drop(columns="is_primary_data", inplace=True)
    preprocessor = Preprocessor(do_postp=False)
    adata = preprocessor(adata)
    
    return adata

def load_model():
    # Load hf model
    model_checkpoint_file = hf_hub_download(
        repo_id="jkobject/scPRINT", filename=f"v2-medium.ckpt"
    )
    
    # make sure that you check if you have a GPU with flashattention or not (see README)
    try:
        m = torch.load(model_checkpoint_file)
    # if not use this instead since the model weights are by default mapped to GPU types
    except RuntimeError: 
        m = torch.load(model_checkpoint_file, map_location=torch.device('cpu'))
        
    # again here by default the model was trained with flash attention, so if you do not have a GPU you will need to replace the attention mechanism with regular attention 
    transformer = "flash" if torch.cuda.is_available() else "normal"

    # both are for compatibility issues with different versions of the pretrained model, so we need to load it with the correct transformer
    if "prenorm" in m['hyper_parameters']:
        m['hyper_parameters'].pop("prenorm")
        torch.save(m, model_checkpoint_file)
    if "label_counts" in m['hyper_parameters']:
        # you need to set precpt_gene_emb=None otherwise the model will look for its precomputed gene embeddings files although they were already converted into model weights, so you don't need this file for a pretrained model
        model = scPrint.load_from_checkpoint(model_checkpoint_file, precpt_gene_emb=None, classes=m['hyper_parameters']['label_counts'], transformer=transformer)
    else:
        model = scPrint.load_from_checkpoint(model_checkpoint_file, precpt_gene_emb=None, transformer=transformer)

    # this might happen if you have a model that was trained with a different set of genes than the one you are using in the ontology (e.g. newer ontologies), While having genes in the onlogy not in the model is fine. the opposite is not, so we need to remove the genes that are in the model but not in the ontology
    missing = set(model.genes) - set(load_genes(model.organisms).index)
    if len(missing) > 0:
        print(
            "Warning: some genes missmatch exist between model and ontology: solving...",
        )
        model._rm_genes(missing)

    # again if not on GPU you need to convert the model to float64
    if not torch.cuda.is_available():
        model = model.to(torch.float32)
        
    # you can perform your inference on float16 if you have a GPU, otherwise use float64
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # the models are often loaded with some parts still displayed as "cuda" and some as "cpu", so we need to make sure that the model is fully on the right device 
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    return model

def create_gnn_inferrer():
    grn_inferrer = GNInfer(
        how="most var across",
        preprocess="softmax",
        head_agg='none',
        filtration="none",
        forward_mode="none",
        num_genes=4000,
        max_cells=300,
        doplot=False,
        batch_size=16,
        cell_type_col='cell_type',
        dtype=dtype,
        layer=list(range(model.nlayers))[:]
    )
    return grn_inferrer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--adata_path", required=True, help="Path to real AnnData file")
    args = parser.parse_args()
    adata_path = args.adata_path
    adata = preprocess_data(adata_path)
    model = load_model()
    grn_inferrer = create_gnn_inferrer()
    grn = grn_inferrer(model, adata, cell_type="hepg2")
    grn.varp['all'] = grn.varp['GRN'].copy()
    grn.varp['GRN'] = grn.varp['GRN'].mean(-1)
    
    os.make_dirs("grn/results", exist_ok=True)
    grn.write_h5ad("grn/results/hepg2_grn_all.h5ad")

if __name__ == "__main__":
    main()
