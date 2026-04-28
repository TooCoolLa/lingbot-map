# LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction

LingBot-Map is a feed-forward 3D foundation model designed for streaming 3D reconstruction from image sequences or videos. It unifies coordinate grounding, dense geometric cues, and long-range drift correction within a single transformer framework.

## 🏗️ Project Architecture

The codebase is organized as follows:
- **`lingbot_map/`**: Core library.
    - **`models/`**: Implementation of `GCTStream` (streaming inference) and `GCTStreamWindow` (windowed inference).
    - **`aggregator/`**: Vision Transformer backbone based on DINOv2.
    - **`heads/`**: Task-specific heads for Camera (pose estimation), Depth, and Point cloud generation.
    - **`layers/`**: Custom transformer layers, including support for paged KV cache via `FlashInfer`.
    - **`vis/`**: Visualization backends for `viser` and `rerun`.
    - **`io.py`**: Result serialization (NPZ per frame + PNG images).
- **`demo.py`**: The primary interactive demo for reconstruction and visualization.
- **`vis_rerun.py`**: Tool for visualizing pre-saved inference results using Rerun.
- **`gct_profile.py`**: Utility for profiling model performance and memory usage.

## 🛠️ Setup & Environment

- **Python**: >= 3.10
- **PyTorch**: Recommended 2.9.1 with CUDA 12.8.
- **FlashInfer**: Required for efficient paged KV cache attention (highly recommended for performance).
- **Visualization**:
    - `viser`: Default for browser-based interactive 3D viewing.
    - `rerun-sdk`: Alternative high-performance viewer (install with `pip install ".[rerun]"`).

### Installation
```bash
pip install -e .
# For full visualization support
pip install -e ".[vis,rerun]"
```

## 🚀 Key Workflows

### 1. Interactive Demo
The `demo.py` script is the main entry point. It handles frame extraction, inference, and visualization.

```bash
# Streaming inference with sky masking
python demo.py --model_path weights/lingbot-map-long.pt --image_folder example/church --mask_sky

# Save results for later viewing
python demo.py --model_path ... --image_folder ... --save_results output/my_scene
```

### 2. Windowed Inference
For extremely long sequences (>3000 frames), use windowed mode to maintain memory stability.
```bash
python demo.py --mode windowed --window_size 128 --video_path video.mp4
```

### 3. Rerun Visualization
You can visualize live inference or saved results with Rerun.
```bash
# Live inference with Rerun
python demo.py --viz_rerun --image_folder ...

# Visualize saved results
python vis_rerun.py --results_dir output/my_scene
```

## 💡 Technical Notes for Development

- **Memory Management**: 
    - Use `--offload_to_cpu` to move per-frame predictions to RAM, saving GPU memory.
    - `--keyframe_interval` (streaming mode) reduces KV cache growth by only caching every N-th frame.
- **Performance Optimization**:
    - `--compile` enables `torch.compile` on hot modules, providing a ~5 FPS boost in streaming mode.
    - `--camera_num_iterations` defaults to 4; setting it to 1 speeds up inference at a slight cost to pose accuracy.
- **Rerun Port Mapping**: If running on a remote server, map both the gRPC port (default `9876`) and the Web port (default `9877`).
- **Sky Masking**: Uses an ONNX model (`skyseg.onnx`) to filter out sky points. Masks are cached in `<image_folder>_sky_masks/`.

## 🧪 Testing & Validation

- Validation is primarily done through interactive visualization of the reconstruction quality.
- Use `gct_profile.py` to check for regressions in latency or memory usage after architectural changes.
- Refer to `Plan_SaveAndRerun.md` for the design specifications of the I/O and Rerun modules.
