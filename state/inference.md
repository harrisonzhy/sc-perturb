Run inference using:
```
uv run state tx infer \
--output "../competition/prediction.h5ad" \
--model-dir "../competition/first_run" \
--checkpoint "../competition/first_run/checkpoints/step=8800.ckpt" \
--adata "../competition_support_set/competition_val_template.h5ad" \ 
--pert-col "target_gene"
```

To visualize the output h5ad file, run `visualize.py`:
```
python3 visualize.py
```

