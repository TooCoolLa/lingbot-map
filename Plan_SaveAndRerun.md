# 计划：保存推理结果 + Rerun 可视化

## 背景

当前 `demo.py` 运行推理后立即用 viser 可视化，无法保存结果供后续查看，也没有 Rerun 作为替代可视化方案。本计划添加：
1. `--save_results <dir>` — 将后处理后的预测结果持久化到磁盘
2. `--viz_rerun` — 使用 Rerun 代替 viser 进行可视化
3. `vis_rerun.py` — 独立脚本，从已保存的结果文件夹加载并用 Rerun 可视化

## 新建文件

### 1. `lingbot_map/io.py` — 保存/加载模块

保存格式（逐帧 `.npz` + PNG 图片，紧凑高效）：
```
<save_dir>/
    metadata.json          # {version, num_frames, image_height, image_width, keys}
    frame_000000.npz       # extrinsic(3,4), intrinsic(3,3), depth(H,W,1), depth_conf(H,W), world_points(H,W,3), world_points_conf(H,W)
    frame_000001.npz
    ...
    images/
        image_000000.png   # uint8 RGB
        image_000001.png
        ...
```

核心函数：
- `save_results(save_dir, predictions, images)` — 逐帧保存 `.npz` + PNG
- `load_results(save_dir, frame_indices=None)` — 返回 `(predictions_dict, images_array, metadata)`

为什么用逐帧 `.npz` 而非单个大文件：支持流式加载（Rerun 逐帧发送），避免一次性加载所有密集点图到内存。图片存为 PNG（uint8）而非 float32，节省约 10 倍磁盘空间。

保存**后处理结果**（extrinsic/intrinsic 已计算好），而非原始 `pose_enc`。这样加载时不需要模型代码。

### 2. `lingbot_map/vis/rerun_viewer.py` — Rerun 可视化模块

`RerunViewer` 类：
- `__init__(predictions, images, conf_threshold, max_points_per_frame, point_radius, grpc_port, web_port, use_point_map, mask_sky, image_folder)`
- `_prepare_data()` — 将 torch/numpy 输入统一转为 numpy，从 extrinsic/intrinsic 提取相机参数
- `_filter_and_sample(frame_idx)` — 去除 NaN/Inf，应用置信度阈值，子采样至 ≤50K 点
- `_log_frame(frame_idx)` — 记录点云 (`rr.Points3D`)、相机变换 (`rr.Transform3D`)、内参 (`rr.Pinhole`)、图像 (`rr.Image`)，使用 `rr.set_time("frame", sequence=i)`
- `run()` — `rr.init()` → `rr.serve_grpc()` → `rr.serve_web_viewer()` → 逐帧记录 → 保持进程

Rerun API 调用顺序（来自 Rerun_Note.md）：`rr.init()` 必须在 `rr.serve_grpc()` 之前。

相机坐标系：后处理的 `extrinsic` 是 c2w (3,4)。构建 4x4 矩阵，传 rotation+translation 给 `rr.Transform3D(from_parent=False)`。内参传给 `rr.Pinhole`。

### 3. `vis_rerun.py` — 独立 CLI 脚本

薄封装：argparse → `load_results()` → `RerunViewer()` → `.run()`

参数：`--results_dir`, `--conf_threshold`, `--max_points_per_frame`, `--point_radius`, `--use_depth`, `--grpc_port`, `--web_port`, `--frame_range`, `--mask_sky`, `--image_folder`

## 修改文件

### 4. `demo.py` — 添加参数 + 保存逻辑 + Rerun 分支

新增参数：
```python
parser.add_argument("--save_results", type=str, default=None)
parser.add_argument("--viz_rerun", action="store_true")
parser.add_argument("--rerun_grpc_port", type=int, default=9876)
parser.add_argument("--rerun_web_port", type=int, default=9877)
```

在后处理之后（约第 384 行）插入：
```python
if args.save_results:
    from lingbot_map.io import save_results
    save_results(args.save_results, predictions, images_cpu)
```

替换可视化部分（约第 387 行）：根据 `args.viz_rerun` 分支 → `RerunViewer` 或原有 `PointCloudViewer`。

### 5. `lingbot_map/vis/__init__.py` — 添加 RerunViewer 导出

带保护导入：
```python
try:
    from lingbot_map.vis.rerun_viewer import RerunViewer
except ImportError:
    RerunViewer = None
```

### 6. `pyproject.toml` — 添加 rerun 可选依赖

```toml
[project.optional-dependencies]
vis = ["viser>=0.2.23", "trimesh", "matplotlib", "onnxruntime", "requests"]
rerun = ["rerun-sdk>=0.19"]
demo = ["lingbot-map[vis]"]
```

## 实现顺序

1. `lingbot_map/io.py`（无新代码依赖）
2. `lingbot_map/vis/rerun_viewer.py`（依赖现有 geometry 工具）
3. `vis_rerun.py`（依赖 io.py + rerun_viewer.py）
4. `demo.py` 修改（依赖 io.py + rerun_viewer.py）
5. `lingbot_map/vis/__init__.py` + `pyproject.toml`（收尾接线）

## 验证

1. **保存/加载往返测试**：对小图像集运行 `demo.py --save_results /tmp/test_results`，然后 `vis_rerun.py --results_dir /tmp/test_results`，验证 Rerun web 查看器中点云和相机正确显示
2. **推理后直接 Rerun**：运行 `demo.py --viz_rerun`（不保存），验证推理完成后 Rerun 直接启动
3. **保存 + Rerun**：运行 `demo.py --save_results /tmp/test --viz_rerun`，验证保存和 Rerun 都正常工作
4. **默认 viser 不受影响**：不带新参数运行 `demo.py`，验证 viser 行为不变
