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

_PER_FRAME_TENSOR_KEYS = [
    "extrinsic",          # (3, 4) float32
    "intrinsic",          # (3, 3) float32
    "depth",              # (H, W, 1) float32
    "depth_conf",         # (H, W) float32
    "world_points",       # (H, W, 3) float32
    "world_points_conf",  # (H, W) float32
]


def save_results(
    save_dir: str,
    predictions: Dict[str, object],
    images: object,
) -> None:
    """Save postprocessed predictions and images to disk.

    Args:
        save_dir: Output directory path.
        predictions: Postprocessed predictions dict (CPU tensors or numpy, unbatched).
            Required keys: extrinsic, intrinsic, depth, depth_conf,
            world_points, world_points_conf.
        images: Images tensor/array (S, 3, H, W) float32 in [0,1], on CPU.
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
    metadata = {
        "version": _SAVE_VERSION,
        "num_frames": S,
        "image_height": H,
        "image_width": W,
        "keys": available_keys,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    for i in range(S):
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

    print(f"Saved {S} frames to {save_dir}")


def _load_frame(save_dir, keys, frame_idx, out_idx):
    """Load a single frame's npz + image from disk."""
    frame_data = np.load(os.path.join(save_dir, f"frame_{frame_idx:06d}.npz"))
    arrays = {key: frame_data[key] for key in keys}

    img_path = os.path.join(save_dir, "images", f"image_{frame_idx:06d}.jpg")
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    return out_idx, arrays, image


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
        frame_indices = list(range(S))

    # Load first frame to get shapes for pre-allocation
    first = np.load(os.path.join(save_dir, f"frame_{frame_indices[0]:06d}.npz"))
    predictions = {}
    for key in keys:
        predictions[key] = np.empty(
            (len(frame_indices),) + first[key].shape,
            dtype=np.float32,
        )

    images = np.empty((len(frame_indices), 3, H, W), dtype=np.float32)

    if num_workers <= 1:
        # Sequential fallback
        for out_idx, frame_idx in enumerate(frame_indices):
            _, arrays, image = _load_frame(save_dir, keys, frame_idx, out_idx)
            for key in keys:
                predictions[key][out_idx] = arrays[key]
            images[out_idx] = image
    else:
        # Parallel loading
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_load_frame, save_dir, keys, frame_idx, out_idx)
                for out_idx, frame_idx in enumerate(frame_indices)
            ]
            for future in as_completed(futures):
                out_idx, arrays, image = future.result()
                for key in keys:
                    predictions[key][out_idx] = arrays[key]
                images[out_idx] = image

    print(f"Loaded {len(frame_indices)} frames from {save_dir}")
    return predictions, images, metadata
