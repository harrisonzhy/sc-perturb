### GRN inference
To run scPRINT GRN inference, you must first download the model from HuggingFace:
```
python3 download_sc_model.py
```
Next, you must also have an active `lamindb` instance. The script below should set it up automatically:
```
lamin settings set auto-connect false
python3 grn_inference.py --input competition_support_set/competition_train.h5 --num-genes 18080 --max-cells 200000 --cell-type ARC_H1
```

### Perturbation model inference
Run inference using:
```
cd state
uv run state tx infer \
--output "../competition/prediction.h5ad" \
--model-dir "../competition/first_run" \
--checkpoint "../competition/first_run/checkpoints/step=8800.ckpt" \
--adata "../competition_support_set/competition_train_split_val.h5ad" \ 
--pert-col "target_gene"
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

### Evaluation
Run evaluation for the fine-tuned model by running inference to generate predictions:
```
uv run --active state tx infer \
  --output "/home/zhanghy/orcd/scratch/zhanghy/sc-perturb/ft-out/prediction.h5ad" \
  --model-dir "/home/zhanghy/orcd/scratch/zhanghy/sc-perturb/ft-out/ft_attn_grn" \
  --checkpoint "/home/zhanghy/orcd/scratch/zhanghy/sc-perturb/ft-out/ft_attn_grn/checkpoints/step=200.ckpt" \
  --adata "/home/zhanghy/orcd/scratch/zhanghy/sc-perturb/competition_support_set/competition_train_split_val.h5ad" \
  --pert-col target_gene
```
Then, run scoring to obtain evaluation metrics:
```
uv tool run cell-eval run \
  --num-threads 8 \
  --de-real competition_support_set/real_de_new.csv \
  --adata-pred ft-out/prediction.h5ad \
  --adata-real /home/zhanghy/orcd/scratch/zhanghy/sc-perturb/competition_support_set/competition_train_split_val.h5ad \
  --de-real ft-v1/H1_real_de.csv \
  --control-pert non-targeting \
  --pert-col target_gene \
  --celltype-col cell_type \
  -o ft-out/cell-eval-outdir
```
