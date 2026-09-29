# Real Inference Follow-ups

Last updated: 2026-05-13

This note records issues that are not blocking the current single-arm
inference run, but are worth revisiting if behavior drifts or the setup
changes.

## Current Runtime Assumptions

- Single arm, `robot0`.
- LK gripper position is in radians even though the UMI-compatible field name
  remains `robot0_gripper_width`.
- Current gripper timing/config:
  - `gripper_action_latency: 0.065`
  - `close_offset_rad: 0.10`
  - `close_offset_active_below_rad: 0.05`
- `close_offset_rad` is only applied for targets near closed position. It is
  intended to make `0.0 rad` reach the mechanical closed stop, matching the
  collection setup.
- Force sensor gravity compensation uses the collection-side formula, but
  replaces VIO orientation with robot TCP orientation.
- Runtime force calibration params are copied from the collection repo to
  `assets/cali_params.pkl`.
- Force sensor should not be tared in the online inference path when gravity
  compensation uses `cali_bias`.

## Gripper Timing

The 30-trial calibration showed two latency modes:

- Opening median: about `0.077 s`
- Closing median: about `0.055 s`
- Combined median across two runs: about `0.065 s`

For now, a single `gripper_action_latency = 0.065` is a reasonable compromise.
If tasks show visible open/close desynchronization with arm motion, split this
into separate config fields:

- `gripper_open_action_latency`
- `gripper_close_action_latency`

Direction can be inferred in `BimanualUmiEnv.exec_actions()` from target
position changes:

- `target_pos > previous_target_pos + deadband`: opening
- `target_pos < previous_target_pos - deadband`: closing
- otherwise: hold/default latency

Use a deadband around `0.02 rad` to avoid switching latency due to small model
output jitter.

## Gripper Closure Offset

The first latency test showed closed commands settling at about `0.067-0.073
rad`, so the gripper was not fully closed after startup. Enabling
`close_offset_rad = 0.10` fixed this in the calibration output: opening trials
now start from about `0.000 rad`.

Things to watch:

- If the gripper chatters or pushes too hard at closed position, reduce
  `close_offset_rad` to `0.05-0.08`.
- If it again fails to fully close, increase in small steps, for example
  `0.02 rad` at a time.
- Keep the offset active only near closed targets. Do not subtract it from all
  gripper positions, or policy open positions will be shifted.

## Force Compensation

Current implementation assumes the force sensor frame orientation is aligned
with the robot TCP frame. If measured compensated force changes with wrist
rotation while unloaded, add a fixed TCP-to-force rotation config and use:

```text
R_base_force = R_base_tcp @ R_tcp_force
```

Other force items to revisit:

- Confirm `cali_bias[:3]` was calibrated in a frame compatible with robot base.
  If old calibration was tied to VIO map and VIO map differs from robot base,
  recalibrate using robot poses or add the map-to-base transform.
- Confirm `force_tool_offset = [0, 0, 0.2165]` is still the force-frame to
  TCP/EE torque point offset for the current mounting.
- Replay episodes now store `robot*_wrench_raw`; when gravity compensation is
  configured they also store `robot*_wrench` computed on the episode time axis.

## Observation Schema

`umi_policy_client.py` currently builds the policy server payload manually.
Before changing model/server code, verify field names and shapes match what the
server expects:

- `fisheye_img`
- `left_robot_tcp_xyzrpy`
- `left_robot_gripper_width_for_action`
- `left_robot_tcp_wrench`
- `left_robot_tcp_wrench_raw`

If the server adopts a formal `shape_meta`, replace the hand-written payload
builder with a schema-driven mapper.

## Action Scheduling

Current scheduling anchors action timestamps to the observation timestamp that
generated the action chunk:

```text
action_timestamps = obs_timestamp + (i + 1) * dt
```

Things to revisit if timing still feels off:

- The server/client path uses onset latency, not time-to-target. This improves
  motion start alignment, but does not make gripper arrival simultaneous with
  arm arrival.
- `action_horizon_guess` in `umi_policy_client.py` is currently fixed. If the
  server returns a longer horizon, allocate temporal aggregation buffers after
  the first response or configure the horizon explicitly.
- If inference latency is often over budget, log the age of consumed actions
  and consider dropping more stale chunks instead of scheduling their tails.

## MVS Runtime

MVS cameras now start their video recorders and fail fast when the C++ backend,
serial, SDK, or camera connection is wrong. Runtime alignment uses the MVS
hardware/device timestamp path, so `cameras.obs_latency` should stay `0.0` for
the MVS backend. Use `cameras.capture_fps: 30` first while the USB/MVS stream is
being checked, then raise back to `60` only if the timestamp smoke test shows
stable frame intervals. If startup times out, check:

- `umi/real_world/_mvs_cpp` was built on the deployment host.
- `cameras.serials` matches `list_mvs_serials()`.
- The camera is on USB3, not USB2.

## Collision Logic

Table collision logic still interprets the gripper value as physical width in
meters. This is harmless while `height_threshold = -1`, but it is wrong for LK
rad values if table collision is re-enabled.

Before enabling table collision with LK gripper:

- Convert rad to approximate finger width, or
- Disable gripper-width dependent keypoints for LK mode.

For dual-arm runs, verify sphere collision uses the configured base transform
instead of hard-coded offsets.

## Hardware Entry Points

Before relying on YAML-only configuration for a new machine, check that the
entry point actually constructs the requested hardware:

- `gripper_type: lkmotor` should construct `LkGripperProxy`.
- `cameras.backend: mvs` should construct `MultiMvsCamera`.
- Franka `robot_port`, `Kx_scale`, and `Kxd_scale` should be passed through.

Keep the runtime config and calibration scripts in sync. A value used by
`calibrate_*` should be the same value used by `eval_real.py` and
`umi_policy_client.py`.
