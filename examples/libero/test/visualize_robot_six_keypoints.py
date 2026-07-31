from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from examples.libero.embodiment.robot import GeomFrankaPanda


POINT_NAMES = (
    "Root",
    "Left base",
    "Right base",
    "Left fingertip",
    "Right fingertip",
    "TCP",
)
COLORS = ("black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")


def main() -> None:
    geometry = GeomFrankaPanda()
    tcp_state = np.asarray([0.08, -0.04, 1.1, 0.2, -0.1, 0.3], dtype=np.float64)
    extrinsic = np.eye(4, dtype=np.float64)
    output = Path("tmp/gripper_six_keypoints.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(13, 6))
    axis = figure.add_subplot(121, projection="3d")
    points = geometry.project_gripper_to_xyz(tcp_state, extrinsic, gripper_width=0.03)
    for name, color, point in zip(POINT_NAMES, COLORS, points, strict=True):
        axis.scatter(*point, color=color, s=55)
        axis.text(*point, f"  {name}", color=color, fontsize=8)
    axis.plot(*points[[0, 1, 3, 4, 2, 0]].T, color="gray", alpha=0.6)
    axis.set_title("Six-point gripper representation (width = 0.03 m)")
    axis.set_xlabel("World X (m)")
    axis.set_ylabel("World Y (m)")
    axis.set_zlabel("World Z (m)")

    axis = figure.add_subplot(122, projection="3d")
    for width, color in ((0.0, "tab:gray"), (0.03, "tab:green"), (0.08, "tab:blue")):
        points = geometry.project_gripper_to_xyz(tcp_state, extrinsic, gripper_width=width)
        axis.scatter(*points[[3, 4]].T, color=color, s=45, label=f"Width {width:.2f} m")
        axis.plot(*points[[3, 5, 4]].T, color=color)
    axis.set_title("Dynamic fingertip positions")
    axis.set_xlabel("World X (m)")
    axis.set_ylabel("World Y (m)")
    axis.set_zlabel("World Z (m)")
    axis.legend()

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    print(output.resolve())


if __name__ == "__main__":
    main()
