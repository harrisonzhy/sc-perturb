import h5py
import numpy as np

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
        "-p", "--path",
        required=True,
        help="Path to the .h5 file (e.g., /path/to/file.h5)",
    )
args = parser.parse_args()
path = Path(args.path).resolve()
print(f"File path: {path}")
        
def open_h5py():
    with h5py.File(path, "r") as f:
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                print(k, obj.shape, obj.dtype)
            else:
                print(k, "Group")

def read_obs_column(f, col):
    node = f["obs"][col]
    # Plain dataset (strings, numbers)
    if isinstance(node, h5py.Dataset):
        arr = node[:]
        if arr.dtype.kind in ("S", "O"):
            arr = arr.astype(str)
        return arr
    # Categorical encoding: obs/<col>/{codes,categories}
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = node["codes"][:]
        cats = node["categories"][:]
        if cats.dtype.kind in ("S", "O"):
            cats = cats.astype(str)
        return cats[codes]
    return None

with h5py.File(path, "r") as f:
    obs = f["obs"]
    cols = list(obs.keys())
    print(f"obs columns ({len(cols)}):")
    for c in cols:
        try:
            vals = read_obs_column(f, c)
            if vals is None:
                print(f"\n{c}: (unsupported or empty)")
                continue
            uniq = np.unique(vals.astype(str))
            print(f"\n=== {c} ===")
            print(f"n_unique={len(uniq)}")
            print("sample:", uniq[:10])  # first 10 unique values
        except Exception as e:
            print(f"\n{c}: error -> {e}")

