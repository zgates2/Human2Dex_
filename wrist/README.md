# Wrist RGB 到 3D 手部关键点 Baseline

这是 Stage 1 baseline：单帧 wrist RGB 图像输入，预测 wrist-centered / hand-local 的 3D 手部关键点。

模型只预测 MediaPipe 21 点拓扑中除 wrist 外的 20 个关节。第 0 个 wrist 点固定为 `[0, 0, 0]`，在评估和可视化时再拼回完整 `21 x 3`。

## 环境

建议使用已有的 `wuji310` 环境，或安装以下依赖：

```bash
pip install -r wrist/requirements.txt
```

DINOv3 本地模型文件默认位于：

```text
/home/zjc/Desktop/human2dex/wrist/dino
```

如果当前环境里的 `transformers` 版本不能直接加载 DINOv3，代码会自动使用本地 ViT-S/16 fallback loader 读取 `model.safetensors`。官方 `AutoModel` 加载路径仍然会优先尝试。

## 构建数据索引

默认数据来源：

```text
/home/zjc/Desktop/human2dex/data/wrist_test_1
```

每帧需要包含：

- `rgbImage`
- `pts21_mano`
- `timestamp`
- 可选 `valid_mask`

构建完整索引：

```bash
python wrist/scripts/build_index.py \
  --data-root /home/zjc/Desktop/human2dex/data/pick_2 \
  --output-dir /home/zjc/Desktop/human2dex/wrist/outputs/index
```

快速 smoke test：

```bash
python wrist/scripts/build_index.py --limit-episodes 3
```

索引按 episode 划分 train / val / test，避免同一个 episode 泄漏到不同 split。

当前完整索引统计：

```text
train: 80882 frames
val:   10362 frames
test:  10389 frames
total: 101633 frames
```

## 数据采集建议

这里的“条”建议按有效帧样本计算，也就是一条样本对应一帧 wrist RGB 和一组同步的 `pts21_mano`。实际采集时不要只按帧数看，还要按 episode / sequence 组织数据，因为训练、验证、测试必须按 episode 划分，同一个 episode 不能进入不同 split。

### 每条样本建议保存的内容

第一阶段必须保存：

- `rgbImage` 或稳定可访问的 `image_path`
- `pts21_mano`：`21 x 3`，MediaPipe 21 点顺序，单位 meter，wrist-centered
- `valid_mask`：`21`，每个 joint 是否有效
- `sequence_id`
- `frame_idx`
- `timestamp`

强烈建议同时保存：

- `raw26x7` 或 Pico 原始手部数据，便于后续重算 pseudo-GT
- `rgbCaptureNs`
- `picoReceiveNs` 或等价 Pico 时间戳
- `rgbToPicoReceiveDeltaNs`，用于检查 RGB 与 Pico 的同步误差
- `rgbFrameRepeated`
- `rgbFrameGap`
- `rgbAlignResidualNs`
- `episode_id`
- `camera_intrinsics`
- 采集元信息：左右手、被试编号、场景编号、光照编号、是否持物、物体类别

如果后续要做 2D projection 或真实 RGB overlay，还需要额外保存 wrist camera 到 hand-local 坐标系的外参，例如 `camera_T_hand`。只有相机内参不足以做可信的 3D 到 2D overlay。

### 第一版 baseline 的采集规模

建议分三档目标：

- 最小可跑通：`20k-30k` 条有效帧，约 `80-120` 个 episode。只能验证 pipeline 和 overfit sanity check，不建议作为最终 baseline。
- 推荐 baseline：`100k-200k` 条有效帧，约 `300-600` 个 episode。适合训练当前 DINOv3 + attention pooling baseline。
- 更稳的泛化版本：`300k-500k+` 条有效帧，约 `900-1500+` 个 episode。适合覆盖更多人、光照、背景、物体和遮挡。

如果按 30 FPS 采集，`100k` 条有效帧约等于 55 分钟有效视频。但不要连续采一个长视频，建议拆成大量短 episode，每个 episode `5-15` 秒，方便按 sequence 切分，也能降低 train / val / test 泄漏。

### 推荐采集内容和数量配比

以 `100k` 条有效帧作为第一版推荐目标，可以按下面比例采集：

| 内容类别 | 建议有效帧数 | 建议 episode 数 | 采集目的 |
| --- | ---: | ---: | --- |
| 基础静态手势 | `15k-20k` | `60-100` | 覆盖 open hand、fist、pinch、OK、point、peace、thumb up、flat hand 等常见姿态 |
| 单指关节运动 | `15k-20k` | `60-100` | 分别采集拇指、食指、中指、无名指、小指的屈伸、外展、内收，覆盖细粒度手指形变 |
| 连续手势过渡 | `15k-20k` | `60-100` | open 到 fist、open 到 pinch、pinch 到 release、多指连续弯曲，减少只会识别静态姿态的问题 |
| 腕部视角变化 | `10k-15k` | `40-80` | 覆盖手腕 roll / pitch / yaw、手掌朝上/朝下/侧向，提升 wrist camera 视角泛化 |
| 自遮挡和极端姿态 | `10k-15k` | `40-80` | 覆盖手指互相遮挡、拇指藏在掌心、半握拳、贴近镜头、远离镜头 |
| 持物和交互 | `20k-30k` | `80-150` | 覆盖捏小物、抓圆柱、抓盒子、拿工具、按压、旋转、放下等真实使用场景 |
| 光照 / 背景 / 服饰变化 | `10k-15k` | `40-80` | 覆盖亮光、暗光、侧光、杂乱背景、不同袖口和肤色，降低过拟合 |

这些类别可以有重叠，例如“持物”同时包含“自遮挡”和“腕部视角变化”。实际落地时优先保证总有效帧和 episode 多样性，不要机械追求每一类完全独立。

### 手势与动作清单

建议至少覆盖以下内容：

- 静态手势：open hand、fist、half fist、pinch、three-jaw pinch、OK、point、peace、thumb up、thumb opposition、flat hand
- 单指动作：每根手指单独弯曲、伸直、外展、内收；拇指 opposition 和 abduction 要单独多采
- 组合动作：多指同时弯曲、食指和拇指捏合、中指/无名指/小指逐个闭合
- 腕部姿态：手掌朝相机、手背朝相机、侧向、旋前、旋后、轻微上下左右移动
- 交互动作：捏小方块、捏细棒、抓杯状物、抓圆柱、抓盒子、按按钮、拨动开关、拿起和放下物体
- 失败和边界样本：手部分出画面、运动模糊、强遮挡、低光照，但这类样本不要超过总量的 `10-15%`

### 被试和划分建议

如果只采一个人的手，模型很容易学到个人手型、肤色、袖口和动作习惯。推荐：

- 最小 baseline：`1-2` 名被试，每人 `30k-80k` 条有效帧
- 推荐 baseline：`3-5` 名被试，每人 `30k-60k` 条有效帧
- 泛化版本：`8-15+` 名被试，每人 `20k-50k` 条有效帧

验证集和测试集最好包含未见过的 episode；如果目标是跨人泛化，测试集还应该包含未见过的被试。第一阶段可以先按 episode 划分，但后续建议额外做一版 leave-one-subject-out 测试。

### 采集质量要求

建议设置下面的最低质量门槛：

- `pts21_mano[0]` 应接近 `[0, 0, 0]`
- `pts21_mano` 单位必须是 meter，不要混入 mm
- MediaPipe 21 点 joint order 必须固定
- `valid_mask` 有效比例建议大于 `90%`
- RGB 与 Pico 的时间差建议小于 `20 ms`，最好小于 `10 ms`
- 重复 RGB 帧比例建议小于 `5%`
- 每个 split 都要覆盖主要手势、视角和持物类别

正式大规模采集前，先采 `2k-5k` 条做小规模试采，跑一次 overfit sanity check。如果 100-500 帧不能快速压低训练误差，优先检查时间同步、坐标系、joint order、单位、valid_mask 和图像标签是否对齐。

## 训练

先做 overfit sanity check：

```bash
python wrist/scripts/train.py \
  --config wrist/configs/baseline.yaml \
  --run-name overfit_256 \
  --overfit-frames 256 \
  --epochs-stage-a 2 \
  --epochs-stage-b 0
```

正式训练：

```bash
python wrist/scripts/train.py --config wrist/configs/baseline.yaml
```

如果使用 conda 环境：

```bash
conda run -n wuji310 python wrist/scripts/train.py --config wrist/configs/baseline.yaml
```

8 张 A100 推荐用 DDP 启动：

```bash
torchrun --standalone --nproc_per_node=8 \
  wrist/scripts/train.py \
  --config wrist/configs/a100_8gpu.yaml \
  --run-name dinov3_vits16_a100x8
```

也可以覆盖每卡 batch size：

```bash
torchrun --standalone --nproc_per_node=8 \
  wrist/scripts/train.py \
  --config wrist/configs/a100_8gpu.yaml \
  --run-name dinov3_vits16_a100x8_bs96 \
  --batch-size 96 \
  --num-workers 8
```

这里的 `batch_size` 是每张 GPU 的 batch，不是总 batch。有效 batch 为：

```text
effective_batch = batch_size * GPU数量 * grad_accum_steps
```

例如 `batch_size=64`、8 卡、`grad_accum_steps=1` 时，有效 batch 是 `512`。

训练分两阶段：

- Stage A：冻结 DINOv3 backbone，只训练 Attention Pooling 和 MLP Head。
- Stage B：解冻 DINOv3 最后 4 个 transformer blocks，继续 fine-tune。

checkpoint 输出到：

```text
wrist/outputs/runs/<run_name>/checkpoints/best.pt
wrist/outputs/runs/<run_name>/checkpoints/last.pt
```

`best.pt` 按 validation MPJPE 选择。

## 评估

```bash
python wrist/scripts/eval.py \
  --checkpoint wrist/outputs/runs/<run_name>/checkpoints/best.pt \
  --split test
```

输出指标：

- `mpjpe_mm`
- `pck_10mm`
- `pck_20mm`
- `bone_length_error_mm`
- `pa_mpjpe_mm`
- `per_joint_mpjpe_mm`

注意：`PA-MPJPE` 只作为辅助参考，不用于选择 `best.pt`，因为它会掩盖绝对尺度和姿态错误。

## 可视化

```bash
python wrist/scripts/visualize.py \
  --checkpoint wrist/outputs/runs/<run_name>/checkpoints/best.pt \
  --split val \
  --attention
```

可视化内容：

- RGB 图像
- GT 3D skeleton
- Pred 3D skeleton
- Pred vs GT 同坐标系对比
- 可选 attention heatmap

当前没有 wrist camera 到 hand-local 坐标系的准确外参 `camera_T_hand`，所以不会把 3D skeleton 强行投影回 RGB 图像上做准确 overlay。只有相机内参不足以做可信的 2D 投影。

## 关键实现约定

- 输入图像 resize/pad 到 `448 x 448`，保持宽高比。
- normalize 使用 ImageNet/DINO mean/std：
  - mean = `[0.485, 0.456, 0.406]`
  - std = `[0.229, 0.224, 0.225]`
- 模型输出为 `20 x 3`，评估前拼回 wrist 得到 `21 x 3`。
- `pts21_mano` 被视为 Pico/MANO pseudo-GT，不视为绝对真值。
- 所有 loss 和 metric 都使用 `valid_mask`。
- Stage 1 不包含 MANO、VQ-VAE、dex-retargeting、temporal model 或 2D projection loss。

## 多卡训练说明

训练脚本支持 `torchrun` / DDP：

- 每个 rank 只处理自己 shard 的训练数据。
- 验证集也按 rank 分片，不重复 padding，metric 通过 all-reduce 汇总。
- 只有 rank 0 写 `metrics.csv`、`metrics.json`、`best.pt`、`last.pt` 和可视化。
- 默认使用 bf16 / fp16 autocast；A100 配置中固定使用 bf16。
- 默认开启 TF32、channels-last、persistent workers 和 DataLoader prefetch。

如果 8 卡利用率不高，优先尝试：

```text
1. 把 --batch-size 从 64 提到 96 或 128
2. 把 --num-workers 调到 8-12
3. 确认数据在本地 NVMe/SSD，而不是慢速网络盘
4. 减少可视化频率，a100_8gpu.yaml 默认每 5 epoch 保存一次
```

## Stage 2：RGB 到 MANO Pose 再到 21 点

Stage 2 的目标是保持最终接口不变，仍然输出 wrist-centered 的
`pts21_mano / joints21`，但模型内部不再直接回归 21 个散点，而是：

```text
RGB -> DINOv3 -> MANO hand pose -> manotorch MANO Layer -> joints21
```

第一版不做 VQ、temporal、diffusion、contact，也不改任何 retargeting 代码。

### 依赖和 MANO 模型文件

Stage 2 默认使用 `manotorch`：

```bash
pip install git+https://github.com/lixiny/manotorch.git
```

MANO 模型文件不随仓库提交。需要从 MANO 官网按 license 下载，并放到：

```text
wrist/mano/models/MANO_RIGHT.pkl
wrist/mano/models/MANO_LEFT.pkl
```

当前默认训练右手，至少需要 `MANO_RIGHT.pkl`。如果缺少 `manotorch` 或 MANO
模型文件，Stage2 fitting / training 脚本会直接给出明确错误。

检查环境：

```bash
python wrist/scripts/fit_mano.py \
  --config wrist/configs/stage2_mano.yaml \
  --check-mano
```

### 1. 构建 Stage2 数据索引

```bash
python wrist/scripts/build_stage2_index.py \
  --data-root data/wrist_test_1 \
  --output-dir wrist/outputs/stage2/index
```

快速检查：

```bash
python wrist/scripts/build_stage2_index.py \
  --data-root data/wrist_test_1 \
  --output-dir wrist/outputs/stage2/index_debug \
  --limit-episodes 3
```

输出：

```text
wrist/outputs/stage2/index/stage2_index.jsonl
wrist/outputs/stage2/index/quality_report.json
```

索引会过滤 missing RGB、Pico inactive、同步误差过大、重复帧、frame gap 异常、
骨长异常和关键点速度 spike。阈值都可以通过命令行参数覆盖。

### 2. 离线 MANO Fitting

```bash
python wrist/scripts/fit_mano.py \
  --config wrist/configs/stage2_mano.yaml \
  --index wrist/outputs/stage2/index/stage2_index.jsonl \
  --output wrist/outputs/stage2/fits/stage2_mano_fits.jsonl
```

小样本 smoke test：

```bash
python wrist/scripts/fit_mano.py \
  --config wrist/configs/stage2_mano.yaml \
  --limit-frames 256
```

输出：

```text
wrist/outputs/stage2/fits/stage2_mano_fits.jsonl
wrist/outputs/stage2/fits/fit_report.json
```

每帧包含 `mano_pose`、`mano_beta`、`mano_joints21_fit`、`fit_error_mm` 和
`fit_valid`。`fit_error_mm > 20` 的帧默认不会进入 Stage2 训练。

### 3. 训练 Stage2

先做 overfit sanity check：

```bash
python wrist/scripts/train_stage2.py \
  --config wrist/configs/stage2_mano.yaml \
  --run-name stage2_overfit_128 \
  --overfit-frames 128 \
  --epochs-stage-a 2 \
  --epochs-stage-b 0
```

正式训练：

```bash
python wrist/scripts/train_stage2.py \
  --config wrist/configs/stage2_mano.yaml \
  --run-name stage2_mano_baseline
```

8 卡训练：

```bash
torchrun --standalone --nproc_per_node=8 \
  wrist/scripts/train_stage2.py \
  --config wrist/configs/stage2_mano.yaml \
  --run-name stage2_mano_a100x8 \
  --batch-size 64 \
  --num-workers 8
```

### 4. 评估和可视化

```bash
python wrist/scripts/eval_stage2.py \
  --checkpoint wrist/outputs/stage2/runs/<run_name>/checkpoints/best.pt \
  --split test
```

```bash
python wrist/scripts/visualize_stage2.py \
  --checkpoint wrist/outputs/stage2/runs/<run_name>/checkpoints/best.pt \
  --num-samples 16
```

Stage2 评估会报告：

- `mpjpe_mm`
- `pck_10mm`
- `pck_20mm`
- `fingertip_mpjpe_mm`
- `bone_length_error_mm`
- `per_joint_mpjpe_mm`

### Stage2 实现约定

- 网络预测 MANO full pose `[48]`，其中 `global_orient [3]` 只用于对齐 wrist-centered 目标手，`mano_pose [45]` 是手部关节姿态。
- `transl = 0`；输出 `joints21` 仍做 wrist-centered 处理，外部接口不需要关心 MANO global orient。
- MANO Layer 不训练，只作为 differentiable forward。
- `global_orient`、`mano_pose`、`mano_full_pose`、`mano_beta` 来自离线 fitting；推理时 checkpoint 中保存默认 beta。
- `manotorch` 输出按 meters 处理，最终 `joints21` 仍是 meter、wrist-centered。
