# 白色织物手套数据增强 Pipeline

这个 pipeline 只处理采集数据中的图片，并且只增强白色织物手套区域。输出数据集会尽量保持和原始数据相同的 episode 结构、图片文件名和 `.pkl` 元数据，方便现有训练代码继续读取。

推荐流程现在是：

```text
SAM3 文本提示自动生成手套 mask -> 用 mask 做手套外观增强
```

旧的“首帧人工 ROI + 白色颜色分割”路径仍然保留，作为没有 GPU 或 SAM3 效果不稳定时的 fallback。

## 输入和输出

默认输入数据：

```bash
/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge
```

默认增强输出数据：

```bash
/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_v1
```

每个输出 episode 保持如下结构：

```bash
episode_name/
├── images/
│   ├── original_filename_000000.jpg
│   └── ...
└── original_metadata.pkl
```

`.pkl` 文件会原样复制，图片文件名保持不变，只替换为增强后的图片。

## 环境

SAM3 mask 生成需要在 `sam3` conda 环境中运行，并且官方 SAM3 图像推理路径需要 CUDA：

```bash
conda activate sam3
```

2A100 上已使用如下资源：

```bash
SAM3 权重: /home/zjc/Desktop/human2dex/sam3/sam3_pt/sam3.pt
SAM3 仓库: /home/zjc/Desktop/human2dex/sam3/sam3-code
YAML 配置: /home/zjc/Desktop/human2dex/glove_aug_pipeline/sam3_glove_mask.yaml
```

如果 `torch.cuda.is_available()` 是 `False`，`generate_sam3_masks.py` 会直接报错；这种情况下需要到有 NVIDIA GPU/CUDA 的环境运行 mask 生成。

## 1. 用 SAM3 自动生成手套 mask

先跑一个 episode 的少量帧看效果：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python \
  /home/zjc/Desktop/human2dex/glove_aug_pipeline/generate_sam3_masks.py \
  --config /home/zjc/Desktop/human2dex/glove_aug_pipeline/sam3_glove_mask.yaml
```

参数都集中在：

```bash
/home/zjc/Desktop/human2dex/glove_aug_pipeline/sam3_glove_mask.yaml
```

输出结构：

```bash
pick_sponge_sam3_masks_test/
├── masks/<episode>/*_mask.png
├── overlays/<episode>/*_overlay.jpg
├── stats/<episode>.json
└── summary.json
```

重点检查 `overlays/`：左边是原图，右边是 SAM3 mask overlay 和检测框。红色区域应该覆盖手套，不应该大面积覆盖白墙、桌面或海绵。

常用调参：

```text
prompt                         文本提示列表，可以补充多个 prompt
candidate_confidence_threshold 内部候选召回阈值，低一些能找回不完整手套
confidence_threshold           最终置信度参考阈值
force_single_instance          强制每帧选 1 个最像手套的实例
glove_prior_weight             加入白色、低饱和、面积合理的手套外观先验
track_overlap_weight           偏向和上一帧重叠的候选，提高时序稳定性
dilate_iterations              mask 略微膨胀，补一点不完整边缘
reuse_previous_on_miss         某帧完全没有候选时复用上一帧 mask
min_mask_area                  过滤很小的噪声 mask
max_area_ratio                 过滤覆盖画面过大的误检
```

如果相机是从手腕看向手心，手套形状不像常见手套，优先使用 YAML 里当前的宽松单手套配置。overlay 框上的 `0.24/0.58*` 表示 SAM3 原始分数是 0.24，综合排序分数是 0.58，星号表示低于最终 `confidence_threshold` 但被强制保留。

确认小样本后跑全量：

把 `sam3_glove_mask.yaml` 中这几项改掉：

```yaml
output: /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_sam3_masks
limit_episodes: null
limit_frames: null
```

然后运行：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python \
  /home/zjc/Desktop/human2dex/glove_aug_pipeline/generate_sam3_masks.py \
  --config /home/zjc/Desktop/human2dex/glove_aug_pipeline/sam3_glove_mask.yaml
```

## 2. 用 SAM3 mask 做数据增强

先跑小样本测试：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python /home/zjc/Desktop/human2dex/glove_aug_pipeline/augment_dataset.py \
  --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge \
  --output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_sam3_test \
  --mask-root /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_sam3_masks_test/masks \
  --qc-output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_sam3_test_qc \
  --variants 1 \
  --glove-variation strong \
  --glove-material mixed \
  --background-variation light \
  --limit-episodes 1 \
  --limit-frames 20 \
  --overwrite
```

如果只想看手套变化，不想改变桌面/背景：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python /home/zjc/Desktop/human2dex/glove_aug_pipeline/augment_dataset.py \
  --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge \
  --output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_hand_only_test \
  --mask-root /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_sam3_masks_test/masks \
  --qc-output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_hand_only_test_qc \
  --variants 1 \
  --glove-variation extreme \
  --glove-material mixed \
  --background-variation off \
  --limit-episodes 1 \
  --limit-frames 20 \
  --overwrite
```

确认增强 QC 后跑全量：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python /home/zjc/Desktop/human2dex/glove_aug_pipeline/augment_dataset.py \
  --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge \
  --output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_sam3_v1 \
  --mask-root /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_sam3_masks/masks \
  --qc-output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_sam3_v1_qc \
  --variants 1 \
  --glove-variation strong \
  --glove-material mixed \
  --background-variation light
```

默认情况下，脚本会跳过已经存在的输出图片，方便中断后续跑。如果想重新生成已有图片，加：

```bash
--overwrite
```

如果想保存增强实际使用的二值 mask，加：

```bash
--save-masks
```

### 2.1 可选的二维手型增强

`02_augment_dataset.py` 支持保守的 finger-web geometry augmentation。它不是
MANO 三维骨长修改，而是从 SAM3 mask 的下轮廓寻找可见指尖和指缝，移动指缝来
改变手指长度观感。安全约束如下：

- 每个 episode/variant 只采样一次长度比例，避免逐帧随机手型。
- 默认要求连续 5 帧检测可靠后才启用，过滤接触/遮挡阶段短暂的误检小岛。
- 检测到的指尖位移默认限制在 2 px 内（以 640x360 为参考分辨率）。
- 只有至少 3 个干净指尖、2 个指缝时才应用；持物、严重遮挡帧自动跳过。
- 大于正常手指尺度的轮廓峰会被过滤，避免把物体接触缺口当作手指。
- mask 面积变化超过 3% 时回退到原图和原 mask。
- PKL 和动作标签仍原样复制，因此该增强只适合把手型视为视觉 nuisance 的任务。
  如果 action/observation 直接使用 `pts21_mano`，必须另行同步几何标签。

`pick_cube` 的独立小样本配置：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python \
  glove_aug_pipeline/02_augment_dataset.py \
  --config glove_aug_pipeline/02_augment_pick_cube_shape.yaml
```

该配置写入独立的 `pick_cube_shape_aug_test` 和
`pick_cube_shape_aug_test_qc`，不会覆盖现有 `pick_cube` 或 `pick_cube_qc`。
外观默认使用 `normal + natural + light`，用于先验证手型变化而不让过强的
颜色/纹理差异掩盖几何伪影。
先检查 QC 中的 `shape.applied`、`max_tip_displacement_px` 和
`mask_area_change_ratio`，再考虑取消 `episode_0001` 限制进行全量生成。

## 3. 人工 ROI fallback

每个 episode 在首帧上标注一次手套区域。主手套框尽量贴紧手套，不要把白墙、灯带、白桌布等背景大面积框进去。如果主框不可避免包含这些亮色背景，就额外画排除框。

运行标注：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python /home/zjc/Desktop/human2dex/glove_aug_pipeline/annotate_roi.py \
  --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge \
  --output /home/zjc/Desktop/human2dex/glove_aug_pipeline/roi_annotations.json
```

标注按键：

```text
g：切换到手套模式，然后按住鼠标左键拖拽，画主手套框
e：切换到排除模式，然后按住鼠标左键拖拽，画背景排除框
u：撤销最后一个排除框
Enter：保存当前 episode 的标注
r：重置当前 episode 的标注
s：跳过当前 episode
q：保存已有标注并退出
```

非 GUI 快速测试一个 episode：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python /home/zjc/Desktop/human2dex/glove_aug_pipeline/annotate_roi.py \
  --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge \
  --output /home/zjc/Desktop/human2dex/glove_aug_pipeline/roi_annotations.json \
  --episode demo_20260521_171105_ep0006 \
  --bbox 0,0,300,230 \
  --exclude-bbox 0,120,80,230
```

可以重复传多个 `--exclude-bbox`：

```bash
--exclude-bbox 0,120,80,230 --exclude-bbox 260,55,300,120
```

ROI 路径的小样本测试：

建议先处理 2 个 episode，每个 episode 只处理前 50 帧：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python /home/zjc/Desktop/human2dex/glove_aug_pipeline/augment_dataset.py \
  --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge \
  --output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_v1_test \
  --roi /home/zjc/Desktop/human2dex/glove_aug_pipeline/roi_annotations.json \
  --qc-output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_v1_test_qc \
  --variants 1 \
  --limit-episodes 2 \
  --limit-frames 50
```

质检图会输出到：

```bash
/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_v1_test_qc
```

每张质检图格式是：

```text
原图 | mask overlay | 增强图
```

其中：

```text
青色框：主手套 ROI
橙色框：排除区域
红色区域：实际分割出的手套 mask
```

如果红色 mask 覆盖到了白墙、灯带、桌布等背景，需要重新标注该 episode 的排除框，然后重新跑小样本测试。

ROI 路径跑全量：

确认小样本质检没问题后，再跑全量：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python /home/zjc/Desktop/human2dex/glove_aug_pipeline/augment_dataset.py \
  --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge \
  --output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_v1 \
  --roi /home/zjc/Desktop/human2dex/glove_aug_pipeline/roi_annotations.json \
  --qc-output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_v1_qc \
  --variants 1
```

## 4. 检查输出结构

全量生成后，可以检查新数据集和原数据集的 episode、图片数量、`.pkl` 对齐情况：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python /home/zjc/Desktop/human2dex/glove_aug_pipeline/inspect_outputs.py \
  --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge \
  --output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_glove_aug_v1
```

## 稳定版：mask + wrist/fusion/projection + 增强骨架渲染

现在推荐直接使用完整 launcher。它会按顺序执行：

1. 对原始数据生成 SAM3 hand mask；
2. 只在原始 RGB 上运行 wrist MANO21 + 2D projection head，并写入/刷新 `wrist_*`、`fused_*`、O6/Wuji command 字段；
3. 生成增强 RGB，把原始 episode 的 generated fields 同步到每个增强 variant，并把 `wrist_uv21_rgb` 经过同一个 camera-mount affine 后画到最终增强图上。

小样本 QC：

```bash
cd /home/zjc/Desktop/human2dex

LIMIT_EPISODES=1 LIMIT_FRAMES=80 AUG_WORKERS=1 MASK_OVERWRITE=1 AUG_OVERWRITE=1 \
bash glove_aug_pipeline/run_pick_6_pro_camera_mount_pipeline.sh
```

全量默认处理 `pick_6_mix_5/pick_6_mix_5`：

```bash
cd /home/zjc/Desktop/human2dex

bash glove_aug_pipeline/run_pick_6_pro_camera_mount_pipeline.sh
```

常用切换：

```bash
INPUT_ROOT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_pro \
MASK_OUTPUT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_pro_sam3_masks \
AUG_OUTPUT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_pro_cam_mount_aug \
QC_OUTPUT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_pro_cam_mount_aug_qc \
bash glove_aug_pipeline/run_pick_6_pro_camera_mount_pipeline.sh
```

关键约束：

- 不对增强 RGB 再跑 wrist 模型；
- `wrist_uv21_rgb` 在增强 PKL 中表示增强后 RGB 坐标系；
- `wrist_pts21_mano` / `fused_pts21_mano` 是 3D MANO 局部点，不做 2D affine；
- 骨架最后渲染，覆盖在颜色增强后的手上。

该脚本会生成：

```text
JSON 检查报告
原图/增强图左右对比预览
```

## 常用参数

`augment_dataset.py` 常用参数：

```text
--mask-root PATH      使用 SAM3 预生成 mask，推荐
--roi PATH            使用人工 ROI + 颜色分割 fallback
--variants 1          每张原图生成几张增强图，当前建议先用 1
--glove-variation     glove strength: off / light / normal / strong / extreme
                      中文：手套变化强度，默认 strong
--glove-material      material style: mixed / fabric / rubber / dots / stripe
                      中文：手套材质风格，默认 mixed
--background-variation background strength: off / light / normal / strong
                      中文：背景变化强度，默认 light
--limit-episodes N    只处理前 N 个 episode，适合测试
--limit-frames N      每个 episode 只处理前 N 帧，适合测试
--overwrite           覆盖已有增强图片
--save-masks          额外保存每帧 mask
--qc-frames N         每个 episode 保存 N 张质检 overlay
--no-progress         关闭进度条，适合把日志重定向到文件时使用
```

建议先不用数字范围，只改这三个英文预设：

```text
轻微保守：--glove-variation normal  --glove-material fabric --background-variation off
日常推荐：--glove-variation strong  --glove-material mixed  --background-variation light
强烈变化：--glove-variation extreme --glove-material mixed  --background-variation normal
只改手套：--glove-variation extreme --glove-material mixed  --background-variation off
```

## 注意事项

1. SAM3 mask 质量先看 `overlays/`，增强质量再看 `*_qc/`。
2. 如果 SAM3 把背景当成手套，优先调高 `--confidence-threshold` 或降低 `--max-area-ratio`。
3. 当前增强只改变手套区域的颜色、亮度、纹理风格，不改变几何形状和动作标签。
4. `.pkl` 不会被修改，训练代码仍然读取原来的元数据和图片文件名。
5. ROI fallback 中，主手套框要尽量贴近白色手套，白墙、灯带、白桌布优先用排除框扣掉。
