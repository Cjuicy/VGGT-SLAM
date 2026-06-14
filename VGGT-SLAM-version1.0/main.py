import os
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver

from vggt.models.vggt import VGGT
from vggt_slam_pp.adapters.export_session import ExportSession
from vggt_slam_pp.contracts.runtime import RunIdentity
from vggt_slam_pp.io.checksums import sha256_file


BASELINE_ARCHIVE_SHA256 = (
    "f34897e5745c6380dfd819bf87c8a016aebb8e9ffe7a0025304015fa7b0f0411"
)


parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
parser.add_argument("--plot_focal_lengths", action="store_true", help="Plot focal lengths for the submaps")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to reduce image size by 1/N")
parser.add_argument("--max_loops", type=int, default=1, help="Maximum number of loop closures per submap")
parser.add_argument("--projective_solver", choices=("ransac", "ransac_irls"), default="ransac", help="SL(4) relative-transform estimator; ransac preserves the baseline path")
parser.add_argument("--projective_confidence_mode", choices=("legacy", "joint"), default="legacy", help="legacy preserves original point filtering; joint requires both observations to pass their own confidence thresholds")
parser.add_argument("--projective_threshold", type=float, default=0.01, help="3D correspondence inlier threshold used by RANSAC and Tukey IRLS")
parser.add_argument("--projective_seed", type=int, default=None, help="Optional random seed for reproducible projective RANSAC sampling")
parser.add_argument("--irls_max_iterations", type=int, default=10, help="Maximum IRLS refinement iterations when projective_solver=ransac_irls")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--vis_stride", type=int, default=1, help="Stride interval in the 3D point cloud image for visualization. Try increasing (such as 4) to reduce lag in visualizing large maps.")
parser.add_argument("--vis_point_size", type=float, default=0.003, help="Visualization point size")
parser.add_argument("--vggt_weight", type=str, default="weights/model.pt", help="Local VGGT state-dict path; runtime downloads are disabled")
parser.add_argument("--salad_checkpoint", type=str, default="weights/dino_salad.ckpt", help="Local SALAD checkpoint, required only when max_loops > 0")
parser.add_argument("--dinov2_source", type=str, default="external_sources/dinov2", help="Local official DINOv2 source directory; runtime downloads are disabled")
parser.add_argument("--dinov2_weight", type=str, default="weights/dinov2_vitb14_pretrain.pth", help="Local DINOv2 ViT-B/14 pretraining weight")
parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto", help="Inference device; auto prefers CUDA, then MPS, then CPU")
parser.add_argument("--export_submaps_dir", type=str, default=None, help="Optional M0 cache output directory; disabled by default")
parser.add_argument("--run_id", type=str, default="baseline-run", help="Stable identifier recorded in M0 artifacts")
parser.add_argument("--run_purpose", choices=("baseline_reference", "pp_frontend_bridge"), default="baseline_reference", help="Whether this is a baseline metric run or ++ front-end export")


def resolve_device(requested_device):
    """Resolve one explicit device and fail early when it is unavailable."""
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but MPS is unavailable")
    return torch.device(requested_device)

def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()
    use_optical_flow_downsample = True
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    run_identity = RunIdentity(
        run_id=args.run_id,
        solver_mode="baseline_sim3_compat" if args.use_sim3 else "baseline_sl4",
        run_purpose=args.run_purpose,
        max_loops=args.max_loops,
        submap_size=args.submap_size,
        min_disparity=args.min_disparity,
    )

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        gradio_mode=False,
        vis_stride = args.vis_stride,
        vis_point_size = args.vis_point_size,
        device=device,
        enable_loop_closure=args.max_loops > 0,
        salad_checkpoint=Path(args.salad_checkpoint),
        dinov2_source=Path(args.dinov2_source),
        dinov2_weight=Path(args.dinov2_weight),
        projective_solver=args.projective_solver,
        projective_confidence_mode=args.projective_confidence_mode,
        projective_threshold=args.projective_threshold,
        projective_seed=args.projective_seed,
        irls_max_iterations=args.irls_max_iterations,
    )

    print("Initializing and loading VGGT model...")
    vggt_weight = Path(args.vggt_weight)
    if not vggt_weight.is_file():
        raise FileNotFoundError(f"VGGT weight not found: {vggt_weight}")
    model = VGGT()
    model.load_state_dict(
        torch.load(vggt_weight, map_location="cpu", weights_only=True)
    )

    model.eval()
    model = model.to(device)

    export_session = None
    if args.export_submaps_dir is not None:
        export_session = ExportSession(
            output_root=Path(args.export_submaps_dir),
            run=run_identity,
            baseline_sha256=BASELINE_ARCHIVE_SHA256,
            weight_sha256=sha256_file(vggt_weight),
        )

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, args.downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    data = []
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
            if enough_disparity:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            print(image_names_subset)
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops)

            data.append(predictions["intrinsic"][:,0,0])

            solver.add_points(predictions)

            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            if export_session is not None:
                loop_sources = tuple(
                    loop.detected_submap_id
                    for loop in predictions["detected_loops"]
                )
                export_session.export_latest_submap(
                    solver.map.get_latest_submap(),
                    loop_sources=loop_sources,
                )
                export_session.export_graph_state(solver.map, solver.graph)

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            
            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())
    if export_session is not None:
        export_session.finalize(solver.map, solver.graph)

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path)

        # Log the full point cloud as one file, used for visualization.
        # solver.map.write_points_to_file(args.log_path.replace(".txt", "_points.pcd"))

        if not args.skip_dense_log:
            # Log the dense point cloud for each submap.
            solver.map.save_framewise_pointclouds(args.log_path.replace(".txt", "_logs"))

    if args.plot_focal_lengths:
        # Define a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        for i, values in enumerate(data):
            y = values  # Y-values from the list
            x = [i] * len(values)  # X-values (same for all points in the list)
            plt.scatter(x, y, color=colors[i], label=f'List {i+1}')

        plt.xlabel("poses")
        plt.ylabel("Focal lengths")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()
