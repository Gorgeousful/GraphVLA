import argparse
import os
from pathlib import Path
from time import time

import nvdiffrast.torch as dr
import numpy as np
import trimesh

from foundationpose import (
    FoundationPose,
    PoseRefinePredictor,
    ScorePredictor,
    YcbineoatReader,
)
from Utils import (
    draw_posed_3d_box,
    draw_xyz_axis,
    set_logging_format,
    set_seed,
)


def run_demo_data(
    root: Path,
    max_frames: int,
    est_refine_iter: int,
    track_refine_iter: int,
    debug: int,
    debug_dir: Path,
):
    mesh_file = root / "demo_data" / "mustard0" / "mesh" / "textured_simple.obj"
    scene_dir = root / "demo_data" / "mustard0"

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(str(mesh_file))
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "ob_in_cam").mkdir(parents=True, exist_ok=True)
    (debug_dir / "track_vis").mkdir(parents=True, exist_ok=True)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=debug,
        glctx=glctx,
    )

    reader = YcbineoatReader(video_dir=str(scene_dir), shorter_side=None, zfar=np.inf)
    frame_count = min(max_frames, len(reader.color_files))
    if frame_count <= 0:
        raise RuntimeError(f"No frames found under {scene_dir}")

    tic = time()
    last_pose = None
    for i in range(frame_count):
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = estimator.register(
                K=reader.K,
                rgb=color,
                depth=depth,
                ob_mask=mask,
                iteration=est_refine_iter,
            )
        else:
            pose = estimator.track_one(
                rgb=color,
                depth=depth,
                K=reader.K,
                iteration=track_refine_iter,
            )

        last_pose = pose.reshape(4, 4)
        np.savetxt(debug_dir / "ob_in_cam" / f"{reader.id_strs[i]}.txt", last_pose)

        if debug >= 1:
            center_pose = last_pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            _ = draw_xyz_axis(
                vis,
                ob_in_cam=center_pose,
                scale=0.1,
                K=reader.K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )

        print(f"frame={i} pose_translation={last_pose[:3, 3].tolist()}")

    elapsed = time() - tic
    print(f"processed_frames={frame_count} elapsed={elapsed:.3f}s")
    print(f"last_pose=\n{last_pose}")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--est-refine-iter", type=int, default=2)
    parser.add_argument("--track-refine-iter", type=int, default=1)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=root / "debug_package_smoke",
    )
    args = parser.parse_args()

    run_demo_data(
        root=root,
        max_frames=args.max_frames,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        debug=args.debug,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()
