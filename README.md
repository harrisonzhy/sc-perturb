# sc-perturb

To set up the environment, first request 1 or more gpus via:
```
salloc --mem=<nalloc>G --time=<time> -p mit_normal_gpu --gres=gpu:<ngpus>
```
E.g.:
```
salloc --mem=2G --time=01:00:00 -p mit_normal_gpu --gres=gpu:2
```

Then run the environment setup script:
```
bash env.sh
```
Create the `vcc-env` Python virtual environment and activate it:
```
python3 -m venv create vcc-env
source vcc-env/bin/activate
```
Download `uv` tool:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Run `uv` command in `state` directory to install Python dependencies:
```
uv run state --help
```
Set `wandb` config for training visualization and logging. (Ensure to change the script to use your `entity`):
```
cd state
bash set_wandb.sh
cd ..
```
Download the Replogle-Nadig training dataset:
```
python3 data.py
```
Modify the `.toml` files, e.g. `competition_support_set/starter.toml`, as necessary. Then run training, e.g.:
```
uv run state tx train \
  data.kwargs.toml_config_path="competition_support_set/starter.toml" \
  data.kwargs.num_workers=8 \
  data.kwargs.batch_col="batch_var" \
  data.kwargs.pert_col="target_gene" \
  data.kwargs.cell_type_key="cell_type" \
  data.kwargs.control_pert="non-targeting" \
  data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt" \
  training.max_steps=40000 \
  training.ckpt_every_n_steps=20000 \
  model=state_sm \
  wandb.tags="[first_run]" \
  wandb.project=vcc \
  wandb.entity=arcinstitute \
  output_dir="competition" \
  name="first_run"
```

