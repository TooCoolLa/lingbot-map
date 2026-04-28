"""Visualize saved LingBot-MAP results using Rerun.

Usage:
    python vis_rerun.py --results_dir /path/to/saved_results
    python vis_rerun.py --results_dir /path/to/saved_results --conf_threshold 2.0
    python vis_rerun.py --results_dir /path/to/saved_results --frame_range 0 50
"""

import argparse
import os

from lingbot_map.io import load_results
from lingbot_map.vis.rerun_viewer import RerunViewer


def main():
    parser = argparse.ArgumentParser(
        description="Visualize saved LingBot-MAP results with Rerun"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Directory containing saved results (from --save_results)",
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=1.5,
        help="Confidence threshold for point filtering",
    )
    parser.add_argument(
        "--max_points_per_frame", type=int, default=50000,
        help="Max points per frame (Rerun performance limit)",
    )
    parser.add_argument(
        "--point_radius", type=float, default=0.03,
        help="Point radius in Rerun viewer",
    )
    parser.add_argument(
        "--use_depth", action="store_true",
        help="Use depth-based projection instead of world_points",
    )
    parser.add_argument("--grpc_port", type=int, default=9876)
    parser.add_argument("--web_port", type=int, default=9877)
    parser.add_argument(
        "--frame_range", type=int, nargs=2, default=None,
        metavar=("START", "END"),
        help="Only load frames in [START, END) range",
    )
    parser.add_argument(
        "--mask_sky", action="store_true",
        help="Apply sky segmentation",
    )
    parser.add_argument(
        "--image_folder", type=str, default=None,
        help="Path to original images (for sky segmentation)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=os.cpu_count() or 4,
        help="Number of parallel threads for loading results and logging to Rerun (1=sequential)",
    )

    args = parser.parse_args()

    frame_indices = None
    if args.frame_range is not None:
        frame_indices = list(range(args.frame_range[0], args.frame_range[1]))

    predictions, images, metadata = load_results(
        args.results_dir, frame_indices=frame_indices,
        num_workers=args.num_workers,
    )

    viewer = RerunViewer(
        predictions=predictions,
        images=images,
        conf_threshold=args.conf_threshold,
        max_points_per_frame=args.max_points_per_frame,
        point_radius=args.point_radius,
        grpc_port=args.grpc_port,
        web_port=args.web_port,
        use_point_map=not args.use_depth,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
        num_workers=args.num_workers,
    )
    viewer.run()


if __name__ == "__main__":
    main()
