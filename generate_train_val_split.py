import anndata as ad
import numpy as np

a = ad.read_h5ad("competition_support_set/competition_train.h5")

pert_col = "target_gene"
train_idx, val_idx = [], []

np.random.seed(42)

MIN_VAL = 20         # minimum cells per perturbation in validation
VAL_FRAC = 0.1       # target fraction

for pert, df_group in a.obs.groupby(pert_col):
    idx = df_group.index.values
    n = len(idx)

    # shuffle indices
    np.random.shuffle(idx)

    # compute number of validation samples
    n_val = max(int(n * VAL_FRAC), MIN_VAL)

    # but ensure we don't take >50% of the perturbation
    n_val = min(n_val, n // 2)

    # ensure n_val < n
    if n_val == 0 or n_val >= n:
        print(f"Skipping splitting for {pert} (too few cells)")
        train_idx.extend(idx)
        continue

    val_idx.extend(idx[:n_val])
    train_idx.extend(idx[n_val:])

print("Train:", len(train_idx))
print("Val:", len(val_idx))

train_ad = a[train_idx].copy()
val_ad = a[val_idx].copy()

train_ad.write("competition_train_split_train.h5ad")
val_ad.write("competition_train_split_val.h5ad")

print("Saved train/val splits.")

