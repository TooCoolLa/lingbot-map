"""Rerun-based 3D visualization for LingBot-MAP inference results.

Features:
- 3D point cloud with per-point colors
- Camera frustums with gradient coloring
- Frame-by-frame time slider
- Confidence-based filtering
- Efficient: <=50K points per frame, random sampling for larger clouds

Usage from demo.py::

    from lingbot_map.vis.rerun_viewer import RerunViewer
    viewer = RerunViewer(predictions=predictions, images=images_cpu)
    viewer.run()

Usage from vis_rerun.py (standalone)::

    from lingbot_map.io import load_results
    from lingbot_map.vis.rerun_viewer import RerunViewer
    predictions, images, _ = load_results("/path/to/results")
    viewer = RerunViewer(predictions=predictions, images=images)
    viewer.run()
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import numpy as np
import torch

import rerun as rr
import rerun.blueprint as rrb


def _to_numpy(x):
    """Convert torch.Tensor or array-like to numpy float32."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class RerunViewer:
    """Visualize LingBot-MAP results using the Rerun viewer.

    Args:
        predictions: Dict of tensors (postprocessed, CPU).
            Required keys: extrinsic, intrinsic, world_points, world_points_conf,
            depth, depth_conf.
        images: (S, 3, H, W) float32 tensor or numpy array in [0,1].
        conf_threshold: Confidence threshold for point filtering.
        max_points_per_frame: Max points to send per frame (Rerun performance).
        point_radius: Point radius in Rerun.
        grpc_port: Port for Rerun gRPC server.
        web_port: Port for Rerun web viewer.
        use_point_map: Use world_points (True) or depth-based unprojection (False).
        mask_sky: Apply sky segmentation.
        image_folder: Path to images (for sky segmentation).
        num_workers: Number of parallel threads for logging frames.
            Defaults to CPU count.
        global_map: If True, also log a static global map of all points.
    """

    def __init__(
        self,
        predictions: Dict,
        images,
        conf_threshold: float = 1.5,
        max_points_per_frame: int = 50000,
        point_radius: float = 0.03,
        grpc_port: int = 9876,
        web_port: int = 9877,
        use_point_map: bool = True,
        mask_sky: bool = False,
        image_folder: Optional[str] = None,
        num_workers: Optional[int] = None,
        global_map: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.max_points_per_frame = max_points_per_frame
        self.point_radius = point_radius
        self.grpc_port = grpc_port
        self.web_port = web_port
        self.num_workers = num_workers or os.cpu_count() or 4
        self.global_map = global_map

        self._prepare_data(predictions, images, use_point_map)

        if mask_sky and image_folder is not None:
            from lingbot_map.vis.sky_segmentation import apply_sky_segmentation
            self.conf = apply_sky_segmentation(
                self.conf, image_folder=image_folder, images=self.images_nchw,
            )

    def _prepare_data(self, predictions, images, use_point_map):
        """Convert and extract all needed arrays from predictions + images."""
        # Images: (S, 3, H, W) -> numpy float32
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        self.images_nchw = np.asarray(images, dtype=np.float32)
        S, _, H, W = self.images_nchw.shape

        # Images in (S, H, W, 3) uint8 for Rerun color mapping
        self.images_hw3 = (
            (self.images_nchw.transpose(0, 2, 3, 1) * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )

        # World points
        if use_point_map and "world_points" in predictions:
            self.world_points = _to_numpy(predictions["world_points"])  # (S, H, W, 3)
            conf_key = "world_points_conf" if "world_points_conf" in predictions else "depth_conf"
            self.conf = _to_numpy(predictions[conf_key])
        else:
            from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
            depth = _to_numpy(predictions["depth"])
            ext = _to_numpy(predictions["extrinsic"])
            intr = _to_numpy(predictions["intrinsic"])
            self.world_points = unproject_depth_map_to_point_map(depth, ext, intr)
            self.conf = _to_numpy(predictions["depth_conf"])

        # Camera data: extrinsic is c2w (S, 3, 4) after postprocess
        extrinsic = _to_numpy(predictions["extrinsic"])
        intrinsic = _to_numpy(predictions["intrinsic"])

        self.S = S
        self.H, self.W = H, W

        # Build 4x4 c2w matrices
        self.cam_to_world_4x4 = np.zeros((S, 4, 4), dtype=np.float64)
        self.cam_to_world_4x4[:, :3, :4] = extrinsic
        self.cam_to_world_4x4[:, 3, 3] = 1.0

        self.intrinsic = intrinsic

    def _filter_and_sample(self, frame_idx: int, max_points: Optional[int] = None):
        """Filter by confidence and subsample points for a single frame.

        Returns:
            positions: (N, 3) float32
            colors: (N, 3) uint8
        """
        points = self.world_points[frame_idx].reshape(-1, 3)
        colors = self.images_hw3[frame_idx].reshape(-1, 3)
        conf = self.conf[frame_idx].reshape(-1)

        # Remove NaN/Inf
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        colors = colors[valid]
        conf = conf[valid]

        # Confidence filter
        mask = conf > self.conf_threshold
        points = points[mask]
        colors = colors[mask]

        if len(points) == 0:
            return points, colors

        # Subsample if too many points
        max_pts = max_points or self.max_points_per_frame
        if len(points) > max_pts:
            indices = np.random.choice(len(points), max_pts, replace=False)
            indices.sort()
            points = points[indices]
            colors = colors[indices]

        return points, colors

    def _log_frame(self, frame_idx: int):
        """Log a single frame's data to Rerun."""
        rr.set_time("frame", sequence=frame_idx)

        # Point cloud
        positions, colors = self._filter_and_sample(frame_idx)
        if len(positions) > 0:
            rr.log(
                "world/points",
                rr.Points3D(
                    positions=positions,
                    colors=colors,
                    radii=self.point_radius,
                ),
            )

        # Camera transform (c2w)
        c2w = self.cam_to_world_4x4[frame_idx]
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=c2w[:3, 3],
                mat3x3=c2w[:3, :3],
                from_parent=False,
            ),
        )

        # Camera intrinsics
        K = self.intrinsic[frame_idx]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        rr.log(
            "world/camera",
            rr.Pinhole(
                resolution=[self.W, self.H],
                image_from_camera=K.astype(np.float64),
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )

        # Camera image
        img = self.images_hw3[frame_idx]  # (H, W, 3) uint8
        rr.log("world/camera/image", rr.Image(img))

    def run(self):
        """Initialize Rerun, log all frames, and serve."""
        # CRITICAL: rr.init() must come before rr.serve_grpc()
        rr.init("lingbot_map")

        grpc_uri = rr.serve_grpc(grpc_port=self.grpc_port)
        rr.serve_web_viewer(
            connect_to=grpc_uri,
            web_port=self.web_port,
            open_browser=True,
        )

        # Set up global world coordinate system (Right-Down-Forward for OpenCV)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

        # Blueprint
        rr.send_blueprint(rrb.Vertical(
            rrb.Spatial3DView(name="3D Scene", origin="world"),
            rrb.Spatial2DView(name="Camera Image", origin="world/camera/image"),
        ))

        # Log all frames
        print(f"Logging {self.S} frames to Rerun (parallel with {self.num_workers} workers)...")
        if self.num_workers <= 1:
            for i in range(self.S):
                self._log_frame(i)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._log_frame, i) for i in range(self.S)]
                for future in as_completed(futures):
                    future.result()  # Raise exceptions if any occurred

        # Optional: Log the entire point cloud as a single static entity for global map view
        if self.global_map:
            pts_per_frame = max(1000, self.max_points_per_frame // 10)
            print(f"Generating global map (parallel with {self.num_workers} workers, {pts_per_frame} pts/frame)...")
            
            if self.num_workers <= 1:
                results = [self._filter_and_sample(i, pts_per_frame) for i in range(self.S)]
            else:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    results = list(executor.map(lambda i: self._filter_and_sample(i, pts_per_frame), range(self.S)))
            
            all_pts = [r[0] for r in results if len(r[0]) > 0]
            all_cols = [r[1] for r in results if len(r[0]) > 0]
            
            if len(all_pts) > 0:
                print(f"Logging {sum(len(p) for p in all_pts)} points to global map...")
                rr.log(
                    "world/map",
                    rr.Points3D(
                        positions=np.concatenate(all_pts),
                        colors=np.concatenate(all_cols),
                        radii=self.point_radius * 0.5,
                    ),
                    static=True,
                )

        print(
            f"All frames logged. Rerun viewer at "
            f"http://localhost:{self.web_port}/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A{self.grpc_port}%2Fproxy"
        )

        # Keep process alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Rerun viewer stopped.")
