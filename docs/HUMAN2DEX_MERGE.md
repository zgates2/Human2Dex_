# Repository merge notes

Source repositories:

- Dex_Data-Scaling-Laws-Infer at commit a803710 (remote origin/main b5d7ab4)
- DexUMI at commit 1255383 (remote origin/main c6a651b)

The repositories had only three tracked path collisions:

1. README.md: replaced by the human2dex workflow documentation.
2. .gitignore: replaced by a unified runtime/data ignore policy.
3. tools/convert_pkl_to_training_zarr.py: kept the DexUMI implementation because it
   supports all current gripper, wrist, fused, and PTS21 sources and is used by the
   active training scripts.

Pruned from the merged checkout:

- diffusion_policy/config/legacy/
- generic benchmark environments (block pushing, kitchen, PushT, robomimic)
- their unused runners, workspaces, task configs, and block-pushing generator

These paths are not used by the human2dex collection, Zarr preparation, policy
training, or real-robot inference entry points. Runtime outputs, data, logs,
checkpoints, SAM3 weights, and caches were not copied because they are ignored
artifacts rather than source code.

- scripts_real/latency_test_wsg.py (incomplete source; removed after full syntax check)
