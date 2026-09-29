# Human2Dex 相机骨架标定进度

## 执行约定

- 每次改代码或采集前先阅读本文件。
- 每次完成后只补充状态、结果和下一步，保持简短。
- 旧推理 episode 不进入当前外参拟合或验收。
- 物理相机一旦移动，必须升级 `camera_mount_id` 并重新拟合外参。

## 当前契约

- 相机：MVS `DA9057801`
- O6 安装版本：`mvs_DA9057801_o6_mount_v1`
- 用户确认相机已固定：2026-07-27
- 原始内参：`converted_data/params/fisheye_calib.yaml`
- 原始图像：`1440x1080`
- 策略图像链：中心裁剪 `(180,0,1080,1080)` → `224x224` → 旋转 180°
- 完整契约：`converted_data/params/camera_contract_mvs_DA9057801_o6_mount_v1.yaml`

## 阶段状态

- [x] 确认原始鱼眼内参与策略图像处理链
- [x] 固定 O6 相机安装版本
- [x] 阶段0：外参工具读取完整相机契约
- [x] 阶段2：新棋盘格数据验证内参并提升候选内参
- [ ] 阶段3：新采集 O6 RGB + q_meas 标定序列
- [ ] 阶段4：fit/holdout 外参拟合及完整 21 点验收
- [ ] Wuji 独立 FK、安装契约和外参
- [ ] 几何验收通过后再接入 DP 骨架输入

## 当前验收门槛

- 内参：新棋盘格留出数据无明显边缘系统误差。
- O6 外参 holdout：中位误差 ≤ 4 px，P90 ≤ 8 px。
- 不允许特定手指或闭合程度出现持续同向漂移。

## 进度日志

### 2026-07-27 初始化

- 建立持续更新文档。
- 旧推理录制明确排除。
- 下一步：完成阶段0契约适配、内参验证脚本和专用 O6 标定采集器审计。

### 2026-07-27 阶段0完成

- 外参工具默认绑定 `mvs_DA9057801_o6_mount_v1`，标注/拟合产物记录契约、内参、URDF 与图像链哈希。
- `py_compile`、CLI、契约检查和合成 holdout 自检通过（median 0.402 px，P90 0.614 px）。
- 下一步：用新棋盘格图像验证当前内参；未采集物理数据。

### 2026-07-27 阶段2工具就绪

- 新增 `tools/fisheye_validate.py`：固定现有 K/D，输出全局/边缘误差、覆盖度、overlay 和 JSON；合成自检通过。
- 当前仍待新拍原始 `1440x1080` 棋盘格图像；失败时复用 `tools/fisheye_calibrate.py` 重标后再验证。
- 下一步：准备 O6 专用 RGB + q_meas 标定采集入口。

### 2026-07-27 阶段3工具就绪

- 新增 `Dex_Data-Scaling-Laws-Infer/scripts_real/collect_o6_camera_calibration.py`，人工按键保存策略 RGB + O6 q_meas。
- 工具不连接 Franka、不加载 policy；默认只读。只有显式启用姿态文件后，每次由用户按 `N` 才发送一条 O6 姿态。
- 当前仍待用户贴可选指尖点并采 20 个静止姿态；旧 episode 不使用。

### 2026-07-27 阶段4工具就绪

- 标注固定为 20 姿态、每 4 帧一张 holdout；fit 后在 holdout 渲染完整 21 点。
- 报告增加按手指/按帧偏置及误差与 q 的相关性；全局和系统偏置门槛已写入结果。
- 采集器→标注器兼容性、外参拟合与诊断合成自检均通过；真实结果仍待用户采集/点击。

### 2026-07-27 阶段2首次实拍验证未通过

- 25/25 张检测到棋盘，但只覆盖图像上半部 2 个象限，边缘帧为 0；当前旧内参报告 median 6.415 px、P90 186.336 px，不能通过。
- 排查发现 `fisheye_calibrate.py` 缺少 `CALIB_RECOMPUTE_EXTRINSIC`：旧流程在本批图上 RMS 17.39 px，修正后临时拟合 RMS 0.90 px。
- 临时内参在同批图回代为 median 0.386 px、P90 1.041 px；因没有边缘覆盖且拟合/验证同源，不能作为最终验收结果。
- 已最小修复重标脚本；未覆盖 `converted_data/params/fisheye_calib.yaml`，未更新相机契约。
- 下一步：保留现有 25 张作为中心/上半部拟合集，另拍 8 张边缘拟合图和 8 张独立 holdout，再输出候选内参并只在 holdout 上验收。

### 2026-07-27 阶段2重新采集与候选内参

- 重新采集 37 张，28 张成功检测，覆盖四个象限；不再补拍。
- 标定器增加求解退化帧自动剔除，剔除 3 张后使用 24 张稳健拟合，RMS 1.45 px（224 输入约 0.30 px）。
- 候选内参位于 `data_calibration/mvs_DA9057801_o6_mount_v1/candidate_intrinsics_v1/params/fisheye_calib.yaml`；尚未覆盖正式内参和相机契约。
- 下一步：确认后提升候选内参为正式版本，随后直接进入阶段3。

### 2026-07-27 阶段2正式内参就绪

- 候选内参已提升为 `converted_data/params/fisheye_calib.yaml`，旧文件已备份为 `converted_data/params/fisheye_calib.yaml.before_candidate_v1_20260727`。
- 相机契约已同步 raw/policy K、D、sha256；状态为 `accepted_candidate_v1`。
- `py_compile`、契约哈希检查、`fisheye_validate.py self-test`、采集器 self-test 和外参工具 contract-check/self-test 均通过（fisheye self-test median 0.203 px，P90 0.377 px）。
- 下一步：进入阶段3，采集 O6 RGB + q_meas 标定序列。

### 2026-07-27 阶段3 episode_0003 采集完成

- `episode_0003` 保存 19 张策略 RGB + O6 `q_meas`；图像文件完整，帧号不重复，时间跨度约 79.48 s。
- O6 六维状态变化范围足够；缺少 `pinch` 一帧，但 19 帧仍可用于第一版外参拟合。
- 候选帧检查通过：19 eligible，15 fit，4 holdout；下一步直接点击标注。

### 2026-07-27 阶段4 O6 外参拟合通过

- 使用 `o6_clicks_ep0003.json` 拟合 `camera <- O6_base`，输出 `fit_ep0003_v1/camera_from_o6_base.json`。
- fit：74 点，median 1.673 px，P90 3.056 px；holdout：19 点，median 1.639 px，P90 3.139 px，max 4.736 px。
- 验收通过：`passed=True`，无系统性手指失败；holdout 骨架渲染已输出到 `fit_ep0003_v1/holdout_rendered` 和 `holdout_contact_sheet.jpg`。
- 当前结论：O6 部署侧 FK 21 点可以在固定腕部相机画面中稳定投影，下一步接入离线 skeleton/heatmap 生成。

### 2026-07-27 O6 普通推理 episode overlay 验证

- 使用外参 `fit_ep0003_v1/camera_from_o6_base.json` 在 `episode_0092` 和 `episode_0109` 上离线渲染 O6 FK overlay。
- `episode_0092`：58/58 帧渲染成功，5 帧 `rgbFrameRepeated`；`episode_0109`：33/33 帧渲染成功，3 帧 `rgbFrameRepeated`。
- 两个 episode 每帧 20 个手部点在图像内；缺失点均为 `wrist` root，原因是 hand base origin 在当前相机坐标下 depth 为负，不影响指尖/接触锚点。
- 采样 contact sheet 肉眼检查通过：骨架随 O6 开合变化稳定贴附，无跳到物体/桌面的明显失败。
- 输出目录：`data_local/o6_camera_calibration/mvs_DA9057801_o6_mount_v1/inference_overlay_ep0092_ep0109`。

### 2026-07-27 O6 overlay 可视化样式修正

- 按用户反馈重生成 solid overlay：保留每根手指固定颜色，不使用半透明混合；骨架连接使用细线，关键点使用实心小圆点。
- `episode_0092`：58/58 帧渲染成功；`episode_0109`：33/33 帧渲染成功；有效点统计仍为每帧 20 个手部点，缺失点仅 `wrist`。
- 新输出目录：`data_local/o6_camera_calibration/mvs_DA9057801_o6_mount_v1/inference_overlay_ep0092_ep0109_solid`。

## 2026-07-27 原下一次实际执行（已被 2026-07-29 稳定版 pipeline 更新替代）

1. 实现正式 O6 sidecar 生成：输出 20/21点 `uv`、valid/confidence、5指尖/功能锚点 heatmap；`wrist` root 默认允许 invalid。
2. 先对少量 O6 推理 episode 生成 sidecar 和 overlay 自检，再扩展到训练/部署数据。
3. 通过后讨论如何把 `RGB + skeleton/heatmap` 接入 DP；暂时不启动训练。

```bash
# O6 外参产物
cd /home/zjc/Desktop/human2dex
data_local/o6_camera_calibration/mvs_DA9057801_o6_mount_v1/fit_ep0003_v1/camera_from_o6_base.json
```

### 2026-07-29 稳定版 wrist/fusion/projection + camera-mount augmentation pipeline

- 目标：把原始人手采集 PKL 处理成可直接用于 DP 训练的增强数据，同时减少人手 RGB 与灵巧手/骨架显式关系之间的视觉 gap。
- 当前主路线：原始 RGB 只跑一次 wrist Stage1 + projection head；原始 PKL 写入/刷新 `wrist_*`、`fused_*`、O6/Wuji command；增强 episode 复制这些 generated fields；`wrist_uv21_rgb` / `wrist_anchor_uv_rgb` 按增强使用的同一 camera-mount affine 变换；骨架在 mask/颜色/相机扰动之后最后渲染到增强 RGB 上，避免被增强覆盖。
- 代码位置：`server-03:/home/zjc/Desktop/human2dex`。
- 已实现文件：
  - `wrist/wrist_pose/stage2_projection.py`
  - `glove_aug_pipeline/wrist_skeleton_overlay.py`
  - `glove_aug_pipeline/camera_mount_augmentation.py`
  - `tools/add_wrist_predictions_to_pkl.py`
  - `tools/add_wrist_fusion_predictions_to_pkl.py`
  - `glove_aug_pipeline/02_augment_dataset.py`
  - `glove_aug_pipeline/run_pick_6_pro_camera_mount_pipeline.sh`
  - `glove_aug_pipeline/README.md`
- 软件验证已完成：
  - 相关 Python 文件 `py_compile` 通过；
  - pipeline launcher `bash -n` 通过；
  - `train_stage2_projection_head.py --self-test` 通过；
  - wrist 1-frame dry-run 成功：`predicted=1, wrote_success=1, errors=0`；
  - `/tmp` 1 episode × 1 frame augmentation smoke 成功，增强 PKL 写入 `augmentation_prediction_sync` metadata。
- 当前运行状态：用户已在 server-03 启动全量处理：
  - 命令：`bash /home/zjc/Desktop/human2dex/glove_aug_pipeline/run_pick_6_pro_camera_mount_pipeline.sh`
  - 当前处于 `[1/3] Generate SAM3 masks`；
  - 数据规模：245 episodes，50173 frames，8 GPUs；
  - 日志中的 `timm.models.layers FutureWarning` 是依赖警告，不是失败。
- 尚未完成/不能宣称完成：
  - 全量 SAM3 mask 生成结果尚未回读确认；
  - wrist/fusion/projection 全量写入尚未确认；
  - 增强 RGB 上的骨架 overlay 尚未抽查；
  - 增强 PKL 中 `wrist_uv21_rgb` 是否处于增强图坐标系尚未抽查；
  - 尚未转换增强 Zarr、未训练 DP、未做实机验证。

### 2026-07-29 更新后的下一步

1. 等待 server-03 全量 pipeline 跑完，优先检查是否有异常退出。
2. 抽查 SAM3 mask QC，确认手部 mask 不大面积漏分或错分到物体/桌面。
3. 抽查原始 PKL 是否包含 `wrist_pts21_mano`、`wrist_uv21_rgb`、`fused_pts21_mano`、`fused_o6_command`、`fused_wuji_command`。
4. 抽查增强 RGB：骨架应在增强后的手上，颜色固定、实心点、细线，不被 mask/颜色增强覆盖。
5. 抽查增强 PKL：`wrist_uv21_rgb` / `wrist_anchor_uv_rgb` 应已经随 camera-mount affine 变换到增强图坐标系。
6. 以上通过后，再转换 Zarr 并训练 DP；训练前不直接用未验收的增强数据。
7. 部署侧仍需后续单独接入实时 skeleton-render 接口，建议在 `eval_real_franka_o6.py` 中用配置开关控制；该项不属于当前已完成内容。
