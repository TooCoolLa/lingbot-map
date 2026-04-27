# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sky segmentation utilities for filtering sky points from point clouds.
"""

import glob
import os
import concurrent.futures
from typing import Optional, Tuple

import numpy as np
import cv2
from tqdm.auto import tqdm

try:
    import onnxruntime
except ImportError:
    onnxruntime = None
    print("onnxruntime not found. Sky segmentation may not work.")


_SKYSEG_INPUT_SIZE = (320, 320)
_SKYSEG_SOFT_THRESHOLD = 0.1
_SKYSEG_CACHE_VERSION = "imagenet_norm_softmap_inverted_v3"


def _get_cache_version_path(sky_mask_dir: str) -> str:
    return os.path.join(sky_mask_dir, ".skyseg_cache_version")


def _prepare_sky_mask_cache(sky_mask_dir: Optional[str]) -> None:
    """Ensure the sky mask cache directory exists and write the version stamp."""
    if sky_mask_dir is None:
        return
    os.makedirs(sky_mask_dir, exist_ok=True)
    version_path = _get_cache_version_path(sky_mask_dir)
    if not os.path.exists(version_path):
        with open(version_path, "w", encoding="utf-8") as f:
            f.write(_SKYSEG_CACHE_VERSION)


def run_skyseg(
    onnx_session,
    input_size: Tuple[int, int],
    image: np.ndarray,
) -> np.ndarray:
    """
    Run ONNX sky segmentation on a BGR image and return an 8-bit score map.
    """
    resize_image = cv2.resize(image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x / 255.0 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[1], input_size[0]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    denom = max(max_value - min_value, 1e-8)
    onnx_result = (onnx_result - min_value) / denom
    onnx_result *= 255.0
    return onnx_result.astype(np.uint8)


def _mask_to_float(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.float32)
    if mask.size == 0:
        return mask
    return np.clip(mask, 0.0, 1.0)


def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.dtype == np.uint8:
        return mask
    mask = mask.astype(np.float32)
    if mask.size > 0 and mask.max() <= 1.0:
        mask = mask * 255.0
    return np.clip(mask, 0.0, 255.0).astype(np.uint8)


def _result_map_to_non_sky_conf(result_map: np.ndarray) -> np.ndarray:
    # The raw skyseg map is higher on sky and lower on non-sky.
    return 1.0 - _mask_to_float(result_map)


def segment_sky_from_array(
    image: np.ndarray,
    skyseg_session,
    target_h: int,
    target_w: int
) -> np.ndarray:
    """
    Segment sky from an image array using ONNX model.
    """
    image_rgb = _image_to_rgb_uint8(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result_map = run_skyseg(skyseg_session, _SKYSEG_INPUT_SIZE, image_bgr)
    result_map = cv2.resize(result_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return _result_map_to_non_sky_conf(result_map)


def segment_sky(
    image_path: str,
    skyseg_session,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Segment sky from an image using ONNX model.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    result_map = run_skyseg(skyseg_session, _SKYSEG_INPUT_SIZE, image)
    result_map = cv2.resize(result_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = _result_map_to_non_sky_conf(result_map)

    if output_path is not None:
        cv2.imwrite(output_path, _mask_to_uint8(mask))

    return mask


def _list_image_files(image_folder: str) -> list[str]:
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return [f for f in image_files if os.path.splitext(f.lower())[1] in image_extensions]


def _image_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1.01 else image.astype(np.uint8)
    return image


def _get_mask_filename(image_paths: Optional[list[str]], index: int) -> str:
    if image_paths is not None and index < len(image_paths):
        return os.path.basename(image_paths[index])
    return f"frame_{index:06d}.png"


def _save_sky_mask_visualization(
    image: np.ndarray,
    sky_mask: np.ndarray,
    output_path: str,
) -> None:
    image_rgb = _image_to_rgb_uint8(image)
    if sky_mask.shape[:2] != image_rgb.shape[:2]:
        sky_mask = cv2.resize(sky_mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_uint8 = _mask_to_uint8(sky_mask)
    mask_rgb = np.repeat(mask_uint8[..., None], 3, axis=2)
    overlay = image_rgb.astype(np.float32).copy()
    sky_pixels = _mask_to_float(sky_mask) <= _SKYSEG_SOFT_THRESHOLD
    overlay[sky_pixels] = overlay[sky_pixels] * 0.35 + np.array([255, 64, 64], dtype=np.float32) * 0.65
    overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    panel = np.concatenate([image_rgb, mask_rgb, overlay], axis=1)
    cv2.imwrite(output_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def load_or_create_sky_masks(
    image_folder: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    images: Optional[np.ndarray] = None,
    skyseg_model_path: str = "skyseg.onnx",
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
    target_shape: Optional[Tuple[int, int]] = None,
    num_frames: Optional[int] = None,
) -> Optional[np.ndarray]:
    if onnxruntime is None:
        return None

    if not os.path.exists(skyseg_model_path):
        download_skyseg_model(skyseg_model_path)

    providers = onnxruntime.get_available_providers()
    selected_providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in providers]
    skyseg_session = onnxruntime.InferenceSession(skyseg_model_path, providers=selected_providers)
    sky_masks = []

    # Setup directories
    if image_paths is None and image_folder is not None:
        image_paths = _list_image_files(image_folder)
    
    if sky_mask_dir is None and image_folder is not None:
        sky_mask_dir = image_folder.rstrip("/") + "_sky_masks"
    _prepare_sky_mask_cache(sky_mask_dir)

    num_images = images.shape[0] if images is not None else len(image_paths)
    if num_frames is not None:
        num_images = min(num_images, num_frames)

    # Use ThreadPool for Async I/O
    io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

    print(f"Processing {num_images} sky masks (with async I/O)...")
    for i in tqdm(range(num_images)):
        image_name = _get_mask_filename(image_paths, i)
        mask_filepath = os.path.join(sky_mask_dir, image_name) if sky_mask_dir is not None else None
        
        sky_mask = None
        if mask_filepath and os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
            # Basic validation of cached mask
            if sky_mask is not None:
                # Resize if target_shape is provided
                if target_shape is not None and sky_mask.shape[:2] != target_shape:
                    sky_mask = cv2.resize(sky_mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                sky_masks.append(_mask_to_float(sky_mask))
                continue

        # Inference needed
        if images is not None:
            image_rgb = _image_to_rgb_uint8(images[i])
            image_h, image_w = image_rgb.shape[:2]
            sky_mask = segment_sky_from_array(image_rgb, skyseg_session, image_h, image_w)
        else:
            image_path = image_paths[i]
            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            sky_mask = segment_sky_from_array(image_rgb, skyseg_session, image_bgr.shape[0], image_bgr.shape[1])
        
        # Async Save
        if mask_filepath:
            io_executor.submit(cv2.imwrite, mask_filepath, _mask_to_uint8(sky_mask))
        
        if sky_mask_visualization_dir:
            viz_path = os.path.join(sky_mask_visualization_dir, image_name)
            io_executor.submit(_save_sky_mask_visualization, image_rgb, sky_mask, viz_path)

        if target_shape is not None and sky_mask.shape[:2] != target_shape:
            sky_mask = cv2.resize(sky_mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        
        sky_masks.append(_mask_to_float(sky_mask))

    io_executor.shutdown(wait=False)
    return np.stack(sky_masks) if sky_masks else None


def apply_sky_segmentation(
    conf: np.ndarray,
    image_folder: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    images: Optional[np.ndarray] = None,
    skyseg_model_path: str = "skyseg.onnx",
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
) -> np.ndarray:
    S, H, W = conf.shape
    sky_mask_array = load_or_create_sky_masks(
        image_folder=image_folder, image_paths=image_paths, images=images,
        skyseg_model_path=skyseg_model_path, sky_mask_dir=sky_mask_dir,
        sky_mask_visualization_dir=sky_mask_visualization_dir,
        target_shape=(H, W), num_frames=S,
    )
    if sky_mask_array is None: return conf
    sky_mask_array = sky_mask_array[:S]
    sky_mask_binary = (sky_mask_array > _SKYSEG_SOFT_THRESHOLD).astype(np.float32)
    return conf * sky_mask_binary


def download_skyseg_model(output_path: str = "skyseg.onnx") -> str:
    import requests
    url = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
    print(f"Downloading sky segmentation model from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return output_path
