# Human2Dex 数据处理流水线

这套流水线将腕部鱼眼人手数据转换为双视角 G/L 训练数据。它只编排已有的 wrist、PICO 融合、SAM3、外观增强、鱼眼重投影和 Zarr 转换工具；不修改任何模型结构、已有增强预设或字段语义。

## 数据契约

```text
raw_root (永不写入)
  -> base_enriched_v1        PKL: wrist_* / fused_* / pocket
  -> sam3_masks_v1           仅 mask
  -> appearance_only_v1      外观增强后绘制彩色 21 点骨架，无几何扰动
  -> appearance_gl_v1        带骨架的增强 RGB -> G/L
  -> qc_segments_v1/o6|wuji  仅连续有效片段
  -> 两套独立 Zarr
```

`G` 的 pocket 固定在 `(0.5, 0.35)`，保留前方操作空间；`L` 固定在 `(0.5, 0.5)`。二者均由**已经绘制彩色 21 点骨架**的增强 RGB 和同一帧 `wrist_grasp_pocket_uv` 生成。因此最终策略只读取 G/L，但会同时看到与人手 21 点一同重投影的骨架；它不读取未经处理的原始 RGB。

## 阶段

| 脚本 | 作用 | 并行 |
|---|---|---|
| `01_prepare_base_enriched.py` | PKL 深拷贝、RGB 硬链接/回退拷贝 | 64 CPU 进程 |
| `02_backfill_wrist_fusion.py` | 既有 wrist 推理、PICO 融合、O6/Wuji 重定向 | 既有多 GPU + CPU fusion |
| `03_backfill_grasp_pocket.py` | RGB 直接 pocket 回填 | 每卡一个 PKL 分片 |
| `04_generate_sam3_masks.py` | 既有 SAM3 人手 mask | 8 GPU |
| `05_appearance_only_augment.py` | 既有 mask 外观增强，最后绘制彩色 21 点骨架；强制关闭相机几何扰动 | CPU 多进程 |
| `06_materialize_gl_views.py` | 既有 G/L CUDA 批量重投影 | 8 GPU |
| `07_qc_and_segment.py` | 字段/文件/标签检查，切出连续有效片段 | CPU 多进程 |
| `08_build_training_zarr.py` | O6/Wuji 分别转 Zarr | CPU 多线程/进程 |

GPU 阶段按顺序执行，因为每一阶段默认占满 8 卡；这避免“两个 8 卡程序同时争抢同一批 GPU”。

## 使用

先复制或修改 [00_pipeline_config.yaml](00_pipeline_config.yaml)：只需将 `paths` 整块替换为某一数据集的新版本输出路径，并确认三类权重与鱼眼内参。输出路径必须全新，不能指向 `raw_root`。

```bash
cd /home/zjc/Desktop/human2dex

# 只看完整命令和第一阶段输入，不写入数据
bash glove_aug_pipeline/run_human2dex_pipeline.sh --dry-run

# 从某个已完成阶段继续，例如已经有 G/L 后只做质检和 Zarr
bash glove_aug_pipeline/run_human2dex_pipeline.sh --from 7 --to 8

# 全链路；正式运行前需人工确认 config 中的输出路径均不存在
bash glove_aug_pipeline/run_human2dex_pipeline.sh
```

单独运行任一阶段也可以，例如：

```bash
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python \
  glove_aug_pipeline/06_materialize_gl_views.py \
  --config glove_aug_pipeline/00_pipeline_config.yaml
```

每阶段会在输出根的 `.human2dex_pipeline/` 写入命令、参数和权重哈希。`07` 还会写入 `reports/qc_summary.json` 和 `overlays/`；无效 pocket、缺 G/L、轨迹无效或相应手部动作无效的帧不会进入 Zarr。
