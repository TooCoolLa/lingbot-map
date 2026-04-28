"""Save and load inference results to/from disk.

Save format::

    <save_dir>/
        metadata.json          # {version, num_frames, image_height, image_width, keys}
        frame_000000.npz       # extrinsic, intrinsic, depth, depth_conf, world_points, world_points_conf
        frame_000001.npz
        ...
        images/
            image_000000.jpg   # uint8 RGB
            image_000001.jpg
            ...
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

_SAVE_VERSION = 1
_DEFAULT_SAVE_WORKERS = os.cpu_count() or 4

_PER_FRAME_TENSOR_KEYS = [
    "extrinsic",          # (3, 4) float32
    "intrinsic",          # (3, 3) float32
    "depth",              # (H, W, 1) float32
    "depth_conf",         # (H, W) float32
    "world_points",       # (H, W, 3) float32
    "world_points_conf",  # (H, W) float32
]


def _save_frame(save_dir, images_dir, predictions, available_keys, images, i):
    """Save a single frame's npz + image to disk."""
    frame_data = {}
    for key in available_keys:
        val = predictions[key]
        if isinstance(val, torch.Tensor):
            val = val[i].detach().cpu().numpy()
        else:
            val = np.asarray(val)[i]
        frame_data[key] = val.astype(np.float32)

    np.savez_compressed(
        os.path.join(save_dir, f"frame_{i:06d}.npz"),
        **frame_data,
    )

    img_uint8 = (images[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(
        os.path.join(images_dir, f"image_{i:06d}.jpg"),
        cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
    )


def save_results(
    save_dir: str,
    predictions: Dict[str, object],
    images: object,
    num_workers: int = _DEFAULT_SAVE_WORKERS,
) -> None:
    """Save postprocessed predictions and images to disk.

    Args:
        save_dir: Output directory path.
        predictions: Postprocessed predictions dict (CPU tensors or numpy, unbatched).
            Required keys: extrinsic, intrinsic, depth, depth_conf,
            world_points, world_points_conf.
        images: Images tensor/array (S, 3, H, W) float32 in [0,1], on CPU.
        num_workers: Number of parallel threads for saving.
            Defaults to CPU count. Set to 1 for sequential saving.
    """
    os.makedirs(save_dir, exist_ok=True)
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    images = np.asarray(images, dtype=np.float32)

    S = images.shape[0]
    H, W = images.shape[2], images.shape[3]

    available_keys = [k for k in _PER_FRAME_TENSOR_KEYS if k in predictions]

    if num_workers <= 1:
        for i in range(S):
            _save_frame(save_dir, images_dir, predictions, available_keys, images, i)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_save_frame, save_dir, images_dir, predictions, available_keys, images, i)
                for i in range(S)
            ]
            for future in as_completed(futures):
                future.result()

    # Write metadata after all frames are saved to avoid
    # inconsistent state if save is interrupted
    metadata = {
        "version": _SAVE_VERSION,
        "num_frames": S,
        "image_height": H,
        "image_width": W,
        "keys": available_keys,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {S} frames to {save_dir}")


def _load_frame(save_dir, keys, frame_idx, out_idx):
    """Load a single frame's npz + image from disk.

    Returns (out_idx, arrays, image) on success, or (out_idx, None, None) on failure.
    """
    try:
        frame_data = np.load(os.path.join(save_dir, f"frame_{frame_idx:06d}.npz"))
        arrays = {key: frame_data[key] for key in keys}

        img_path = os.path.join(save_dir, "images", f"image_{frame_idx:06d}.jpg")
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

        return out_idx, arrays, image
    except Exception as e:
        print(f"Warning: failed to load frame {frame_idx}: {e}")
        return out_idx, None, None


def load_results(
    save_dir: str,
    frame_indices: Optional[list] = None,
    num_workers: int = 8,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, dict]:
    """Load saved results from disk.

    Args:
        save_dir: Directory saved by save_results().
        frame_indices: Optional list of frame indices to load.
            None = load all frames.
        num_workers: Number of parallel threads for loading.
            Set to 1 for sequential loading.

    Returns:
        (predictions, images, metadata):
            predictions: dict of numpy arrays with the S dimension.
                e.g. predictions["extrinsic"] is (S, 3, 4).
            images: (S, 3, H, W) float32 in [0,1].
            metadata: The saved metadata dict.
    """
    with open(os.path.join(save_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    S = metadata["num_frames"]
    H = metadata["image_height"]
    W = metadata["image_width"]
    keys = metadata["keys"]

    if frame_indices is None:
        # Auto-detect available frames from disk (handles incomplete saves)
        available = sorted(
            int(f.split("_")[1].split(".")[0])
            for f in os.listdir(save_dir)
            if f.startswith("frame_") and f.endswith(".npz")
        )
        frame_indices = available
        if len(available) != S:
            print(f"Warning: metadata says {S} frames but found {len(available)} npz files; using {len(available)}")

    # Probe first valid frame to get shapes for pre-allocation
    first_arrays = None
    for fi in frame_indices:
        try:
            first_data = np.load(os.path.join(save_dir, f"frame_{fi:06d}.npz"))
            first_arrays = {key: first_data[key] for key in keys}
            break
        except Exception:
            continue
    if first_arrays is None:
        raise RuntimeError(f"No valid frames found in {save_dir}")

    predictions = {}
    for key in keys:
        predictions[key] = np.empty(
            (len(frame_indices),) + first_arrays[key].shape,
            dtype=np.float32,
        )

    images = np.empty((len(frame_indices), 3, H, W), dtype=np.float32)

    # Track which frames were successfully loaded
    valid_mask = np.zeros(len(frame_indices), dtype=bool)

    if num_workers <= 1:
        # Sequential fallback
        for out_idx, frame_idx in enumerate(frame_indices):
            _, arrays, image = _load_frame(save_dir, keys, frame_idx, out_idx)
            if arrays is not None:
                for key in keys:
                    predictions[key][out_idx] = arrays[key]
                images[out_idx] = image
                valid_mask[out_idx] = True
    else:
        # Parallel loading
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_load_frame, save_dir, keys, frame_idx, out_idx)
                for out_idx, frame_idx in enumerate(frame_indices)
            ]
            for future in as_completed(futures):
                out_idx, arrays, image = future.result()
                if arrays is not None:
                    for key in keys:
                        predictions[key][out_idx] = arrays[key]
                    images[out_idx] = image
                    valid_mask[out_idx] = True

    # Trim to only valid frames
    num_failed = int((~valid_mask).sum())
    if num_failed > 0:
        print(f"Warning: {num_failed} frame(s) failed to load and were skipped")
        predictions = {key: val[valid_mask] for key, val in predictions.items()}
        images = images[valid_mask]

    print(f"Loaded {int(valid_mask.sum())} frames from {save_dir}")
    return predictions, images, metadata
