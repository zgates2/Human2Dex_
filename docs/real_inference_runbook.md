# Real Inference Runbook

Last updated: 2026-05-13

This is the short test path for the current single-arm setup.

## 1. Sync And Config

Deploy the current repo to the robot PC if needed:

```bash
scripts/rsync_to_pc4090.sh
```

On the robot PC, check `example/eval_robots_config.yaml`:

- `robots[0].robot_ip` / `robot_port`
- `grippers[0].port1` / `motor1_id`
- `grippers[0].gripper_action_latency`
- `cameras.backend`, `serials`; for MVS keep `obs_latency: 0.0`
- `cameras.capture_fps`: use `30` first while USB/MVS stream stability is
  being checked; raise back to `60` only after the timestamp smoke test is
  clean.
- `force_sensors[0].port`, `cali_params_path`

## 2. Hardware Smoke Tests

Run only the tests for hardware that is enabled in the YAML.

```bash
python scripts_real/calibrate_gripper_latency.py
python scripts_real/calibrate_mvs_latency.py
python scripts_real/calibrate_force_zero.py
```

Expected checks:

- Gripper latency should stay positive and stable.
- MVS should fail fast if the backend or serial is wrong. The timestamp smoke
  test should report frame intervals near `1 / cameras.capture_fps`, with few
  or no `>1.5x expected` gaps. MVS uses the hardware/device timestamp path, so
  keep `cameras.obs_latency: 0.0`.
- Force script only reports frame rate, mean raw wrench, and noise; it does
  not produce runtime gravity-compensation config.

## 3. Choose Inference Mode

Use `eval_real.py` if the checkpoint runs locally on the robot PC:

```bash
python eval_real.py \
  -i /path/to/checkpoint_or_workspace \
  -o data_local/eval_test \
  -rc example/eval_robots_config.yaml \
  -f 5 \
  -mt 100 \
  -j
```

Use `umi_policy_client.py` if the model runs on another machine or cloud. The
server must expose `POST /dp` and return `{"action": ...}` with shape
`(horizon, 7 * num_robots)`, ordered as `xyz + rpy + gripper_rad`.

```bash
python umi_policy_client.py \
  -s http://SERVER_HOST:PORT \
  -o data_local/policy_client_test \
  -rc example/eval_robots_config.yaml \
  -f 5 \
  -mt 100 \
  --action-horizon 16 \
  --init-joints \
  --init-gripper-rad 0.0
```

## 4. First Policy Test

Before letting the policy move the arm freely:

- Use a short `-mt`, for example `50` or `100`.
- Keep `height_threshold: -1` while LK gripper values are in radians.
- Start with no object contact, then test light contact after force values look
  stable.
- After a run, verify that `videos/<episode_id>/0.mp4` and
  `replay_buffer.zarr` were created.
