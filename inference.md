Run inference using:
```
cd state
uv run state tx infer \
--output "../competition/prediction.h5ad" \
--model-dir "../competition/first_run" \
--checkpoint "../competition/first_run/checkpoints/step=8800.ckpt" \
--adata "../competition_support_set/competition_val_template.h5ad" \ 
--pert-col "target_gene"
```

To run scPRINT GRN inference, you must first have an active `lamindb` instance. The script below should set it up:
```
lamin settings set auto-connect false
python3 grn_inference.py --input competition_support_set/competition_val_template.h5ad --num-genes 18080 --max-cells 200000
```

To visualize the output h5ad file, run `visualize.py`:
```
python3 visualize.py --path <output path above>
```
E.g.
```
python3 visualize.py --path ../competition/prediction.h5ad
```

To package submission for vcc, run the following, substituting the appropriate output paths for your own:
```
uv tool run cell-eval prep -i ../competition/prediction.h5ad -g ../competition_support_set/gene_names.csv
```

To transfer vcc file to local machine for submission, use `rsync`, e.g.:
```
rsync -avP zhanghy@orcd-login003.mit.edu:/orcd/home/002/zhanghy/orcd/scratch/zhanghy/sc-perturb/competition/prediction.prep.vcc ~/Downloads/
```

