# human2dex

Unified repository for the human2dex pipeline.

The repository joins the two maintained codebases on the remote device:

- data collection and processing from DexUMI
- training and real-robot inference from Dex_Data-Scaling-Laws-Infer

## Workflow

The supported path is:

PICO / cameras / robot
  -> teleop/ collection (PKL episodes)
  -> glove_aug_pipeline/ and tools/ processing
  -> dataset.zarr.zip
  -> train_scripts/ and train.py
  -> eval_* / scripts_real/ inference

### 1. Collect data

Use the collection environment configured for the hardware:

    python teleop/teleop_mujoco.py
    python teleop/teleop_real.py
    python teleop/collect_data.py --task-name pick_cube

For fused MVS capture, use teleop/collect_data_fused.py or
teleop/collect_l515_mvs.py with the corresponding YAML file.

### 2. Convert and process data

The canonical converter is tools/convert_pkl_to_training_zarr.py. It supports
hand_command, wuji_command, pts21_mano, wrist and fused sources.

    python tools/convert_pkl_to_training_zarr.py \
      --input /path/to/pkl_dataset \
      --output /path/to/dex_data/task \
      --overwrite \
      --gripper-source hand_command \
      --data-scaling-laws-root "$PWD"

For multi-stage glove processing, start with a matching pipeline script under
glove_aug_pipeline/, for example:

    bash glove_aug_pipeline/run_human2dex_pipeline.sh

### 3. Train

Training scripts are in train_scripts/. They expect the Zarr dataset path to
be supplied through the script environment or command line.

    bash train_scripts/glove_pick_bread_head_2_pico_only_wrist_o6_20hz.sh

For another task, use the matching glove_*.sh script. Configure accelerate
before multi-GPU training.

### 4. Run inference

Real robot entry points are:

    python eval_real_franka_pts21.py --help
    python eval_real_franka_o6.py --help
    python eval_real_dexumi.py --help

Camera calibration, action checks, recording, and video rendering live in
scripts_real/.

## Layout

- teleop/, o6_right_hand/, linker_o6/: collection, retargeting, and Linker O6 hardware.
- glove_aug_pipeline/, tools/, scripts/: PKL audit, augmentation, conversion, and Zarr preparation.
- diffusion_policy/, train.py, train_scripts/: policy training.
- umi/, eval_*.py, real_inference_*.py, scripts_real/: UMI and real-world inference.
- configs/, example/, eval_*_config.yaml: calibration and runtime configuration.
- wrist/, reactive_diffusion_policy/: wrist perception and auxiliary policy components.

Runtime data, logs, checkpoints, generated Zarr files, SAM3 weights, and Python
caches are intentionally excluded by .gitignore.

## Environments

The main training environment is described by conda_environment.yaml.
Hardware-specific collection environments are documented in
o6_right_hand/requirements.txt, wrist/requirements.txt, and the retargeting package metadata.

The configuration files still contain machine-specific dataset and calibration
paths where required by the hardware. Update those paths before running on a
different machine.
