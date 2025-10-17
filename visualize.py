import argparse
import anndata as ad
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="View contents of an .h5ad file.")
    parser.add_argument("--path", required=True, help="Path to the .h5ad file.")
    args = parser.parse_args()

    # Load the .h5ad file
    print(f"\nLoading file: {args.path}")
    adata = ad.read_h5ad(args.path)
    print("\nSuccessfully loaded AnnData object.")
    print(adata)

    # Print basic info
    print("\n=== Summary ===")
    print(f"Shape: {adata.shape}")
    print(f"Observations (obs): {list(adata.obs.columns)}")
    print(f"Variables (var): {list(adata.var.columns)}")
    print(f"Obsm keys: {list(adata.obsm.keys())}")
    print(f"Layers: {list(adata.layers.keys()) if hasattr(adata, 'layers') else 'None'}")

    # Show head of obs and var
    print("\n=== adata.obs (first 5 rows) ===")
    print(adata.obs.head())

    print("\n=== adata.var (first 5 rows) ===")
    print(adata.var.head())

    if adata.obsm:
        print("\n=== Embeddings (obsm) ===")
        for k in adata.obsm.keys():
            print(f"{k}: shape {adata.obsm[k].shape}")

if __name__ == "__main__":
    main()

