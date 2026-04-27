#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lingbot_map.vis import PointCloudViewer

import concurrent.futures

def load_frame(f, results_dir):
    """Worker function to load a single frame's data."""
    data = np.load(f)
    frame_data = {
        "depth": data["depth"],
        "depth_conf": data["depth_conf"],
        "extrinsic": data["extrinsic"],
        "intrinsic": data["intrinsic"],
    }
    if "world_points" in data:
        frame_data["world_points"] = data["world_points"]
    
    # Extract frame index for path construction
    # Compatible with frame_000000.npz and similar formats
    basename = os.path.basename(f)
    frame_idx_str = "".join(filter(str.isdigit, basename))
    frame_idx = int(frame_idx_str) if frame_idx_str else 0
    
    # Try multiple possible image paths for compatibility
    possible_img_paths = [
        os.path.join(results_dir, f"frame_{frame_idx_str}.png"),
        os.path.join(results_dir, f"frame_{frame_idx:06d}.png"),
        os.path.join(results_dir, "images", f"image_{frame_idx:06d}.jpg"),
        os.path.join(results_dir, "images", f"image_{frame_idx:06d}.png"),
    ]
    
    img_path = None
    for p in possible_img_paths:
        if os.path.exists(p):
            img_path = p
            break
            
    if img_path:
        # Load as uint8 to save memory during parallel loading
        img = np.array(Image.open(img_path).convert("RGB"))
        frame_data["image_u8"] = img # (H, W, 3)
    else:
        H, W = data["depth"].shape[:2]
        frame_data["image_u8"] = np.zeros((H, W, 3), dtype=np.uint8)
    
    return frame_data

def load_saved_results(results_dir, stride=1, first_k=-1, max_workers=8):
    """Load results saved by demo.py --save_results using multi-threading."""
    print(f"Loading results from {results_dir} (stride={stride}, first_k={first_k}) using {max_workers} workers...")
    
    # Find all frame NPZ files
    frame_files = sorted(glob.glob(os.path.join(results_dir, "frame_*.npz")))
    if not frame_files:
        raise FileNotFoundError(f"No results found in {results_dir}")

    # Apply stride and first_k
    if stride > 1:
        frame_files = frame_files[::stride]
    if first_k > 0:
        frame_files = frame_files[:first_k]

    # Load data in parallel
    results = [None] * len(frame_files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map file paths to indices to maintain order
        future_to_idx = {executor.submit(load_frame, f, results_dir): i for i, f in enumerate(frame_files)}
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(frame_files), desc="Loading frames"):
            idx = future_to_idx[future]
            results[idx] = future.result()

    print("Stacking arrays...")
    # Reorganize into pred_dict, converting images to float32 at the last moment
    pred_dict = {
        "depth": np.stack([r["depth"] for r in results]),
        "depth_conf": np.stack([r["depth_conf"] for r in results]),
        "extrinsic": np.stack([r["extrinsic"] for r in results]),
        "intrinsic": np.stack([r["intrinsic"] for r in results]),
        # Convert (S, H, W, 3) u8 -> (S, 3, H, W) f32
        "images": np.stack([r["image_u8"] for r in results]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0,
    }
    
    if "world_points" in results[0]:
        pred_dict["world_points"] = np.stack([r["world_points"] for r in results])
        # Use world_points_conf if available, otherwise fallback to depth_conf
        if "world_points_conf" in results[0]:
             pred_dict["world_points_conf"] = np.stack([r["world_points_conf"] for r in results])
        else:
             pred_dict["world_points_conf"] = pred_dict["depth_conf"]

    return pred_dict

def main():
    parser = argparse.ArgumentParser(description="Visualize saved LingBot-Map results using Viser")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing saved results")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--conf_percentile", type=float, default=50.0, help="Initial confidence percentile filter (0-100)")
    parser.add_argument("--downsample_factor", type=int, default=1, 
                        help="Point cloud downsample factor (e.g., 10 means sample 1 point out of every 10 points. Default: 1, no downsampling)")
    parser.add_argument("--point_size", type=float, default=0.00001, help="Point size")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky masking (requires --image_folder)")
    parser.add_argument("--image_folder", type=str, default=None, help="Original image folder (required for sky masking)")
    parser.add_argument("--stride", type=int, default=1, help="Load every N-th frame from the results directory")
    parser.add_argument("--first_k", type=int, default=-1, help="Only load the first K frames")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers for loading data (default: all CPU cores)")
    
    args = parser.parse_args()

    pred_dict = load_saved_results(args.results_dir, stride=args.stride, first_k=args.first_k, max_workers=args.workers)
    
    # Note: PointCloudViewer expectations
    # pred_dict needs to be in a slightly different format sometimes? 
    # Actually PointCloudViewer._process_pred_dict handles the raw outputs.
    
    viewer = PointCloudViewer(
        pred_dict=pred_dict,
        port=args.port,
        vis_threshold=args.conf_percentile,
        downsample_factor=args.downsample_factor,
        point_size=args.point_size,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )
    
    print(f"Visualization started at http://localhost:{args.port}")
    viewer.run()

if __name__ == "__main__":
    main()
