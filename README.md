# sc-perturb

## Initial environment setup
To set up the environment, first request 1 or more gpus via:
```
salloc --mem=<nalloc>G --time=<time> -p mit_normal_gpu --gres=gpu:<ngpus>
```
E.g.:
```
salloc --mem=40G --time=06:00:00 -p mit_normal_gpu --gres=gpu:2
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
Modify the `.toml` files, e.g. `competition_support_set/starter.toml`, as necessary. Then run training, e.g. the command below will checkpoint every `training.ckpt_every_n_steps=100` steps. 
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
  training.ckpt_every_n_steps=100 \
  model=state_sm \
  wandb.tags="[first_run]" \
  wandb.project=vcc \
  wandb.entity=arcinstitute \
  output_dir="competition" \
  name="first_run"
```

## Resume training from checkpoint
To resume training from a checkpoint, copy or move the given checkpoint like so (see `state/src/state/_cli/_tx/train.py` for more information):
```
cp competition/.../checkpoints/step\=2000.ckpt competition/.../checkpoints/last.ckpt
```

## Post-initial environment setup
After following the steps for the initial environment setup, all you have to do is request GPU(s), re-run the environment setup script, and activate the Python environment:
```
salloc --mem=40G --time=06:00:00 -p mit_normal_gpu --gres=gpu:2
cd sc-perturb
sh env.sh
source vcc-env/bin/activate
```

