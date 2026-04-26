#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lingbot_map.vis import PointCloudViewer

def load_saved_results(results_dir):
    """Load results saved by demo.py --save_results."""
    print(f"Loading results from {results_dir}...")
    
    # Find all frame NPZ files
    frame_files = sorted(glob.glob(os.path.join(results_dir, "frame_*.npz")))
    if not frame_files:
        # Try finding a single combined file if implemented that way
        combined_file = os.path.join(results_dir, "results.npz")
        if os.path.exists(combined_file):
            data = np.load(combined_file)
            # Reconstruct pred_dict logic would go here
            # But demo.py saves per-frame, so let's focus on that
            pass
        else:
            raise FileNotFoundError(f"No results found in {results_dir}")

    # Load per-frame data
    all_depths = []
    all_confs = []
    all_extrinsics = []
    all_intrinsics = []
    all_images = []
    all_world_points = []

    for f in tqdm(frame_files, desc="Loading frames"):
        data = np.load(f)
        all_depths.append(data["depth"])
        all_confs.append(data["depth_conf"])
        all_extrinsics.append(data["extrinsic"])
        all_intrinsics.append(data["intrinsic"])
        if "world_points" in data:
            all_world_points.append(data["world_points"])
        
        # Load corresponding image
        frame_idx = os.path.basename(f).replace("frame_", "").replace(".npz", "")
        img_path = os.path.join(results_dir, f"frame_{frame_idx}.png")
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
            all_images.append(img.transpose(2, 0, 1)) # (3, H, W)
        else:
            # Placeholder if image missing
            H, W = data["depth"].shape[:2]
            all_images.append(np.zeros((3, H, W), dtype=np.float32))

    pred_dict = {
        "depth": np.stack(all_depths),
        "depth_conf": np.stack(all_confs),
        "extrinsic": np.stack(all_extrinsics),
        "intrinsic": np.stack(all_intrinsics),
        "images": np.stack(all_images),
    }
    
    if all_world_points:
        pred_dict["world_points"] = np.stack(all_world_points)
        # Use existing conf if available
        if "world_points_conf" in data:
             pred_dict["world_points_conf"] = pred_dict["depth_conf"]

    return pred_dict

def main():
    parser = argparse.ArgumentParser(description="Visualize saved LingBot-Map results using Viser")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing saved results")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--conf_percentile", type=float, default=50.0, help="Initial confidence percentile filter (0-100)")
    parser.add_argument("--downsample_factor", type=int, default=10, help="Point cloud downsample factor")
    parser.add_argument("--point_size", type=float, default=0.00001, help="Point size")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky masking (requires --image_folder)")
    parser.add_argument("--image_folder", type=str, default=None, help="Original image folder (required for sky masking)")
    
    args = parser.parse_args()

    pred_dict = load_saved_results(args.results_dir)
    
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
