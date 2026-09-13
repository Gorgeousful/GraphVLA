"""Render only gripper geometry and the six labeled actor points."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import mujoco
import numpy as np
import trimesh

OUT = Path(__file__).resolve().parent
NAMES = ["Root", "Left base", "Right base", "Left fingertip", "Right fingertip", "TCP"]
COLORS = ["#8041ba", "#2171b5", "#069b85", "#ef8a22", "#dd476c", "#d2a000"]


def render(model, data, frame, geom_names, points, horizontal_sign, filename):
    origin = data.xpos[model.body(frame).id]
    rotation = data.xmat[model.body(frame).id].reshape(3, 3)
    triangles = []
    for name in geom_names:
        gid = model.geom(name).id
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            v, f = model.mesh_vert[va:va + vn], model.mesh_face[fa:fa + fn]
        elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            box = trimesh.creation.box(extents=2 * model.geom_size[gid])
            v, f = box.vertices, box.faces
        else:
            continue
        v = (v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid] - origin) @ rotation
        triangles.append(v[f] * 1000)
    triangles = np.concatenate(triangles)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    shade = 0.55 + 0.28 * np.abs(normals[:, 0]) / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)
    colors = np.column_stack([shade, shade, shade, np.full(len(shade), 0.6)])
    order = np.argsort(triangles[:, :, 0].mean(axis=1) * horizontal_sign)
    projected = triangles[:, :, [1, 2]].copy()
    projected[:, :, 0] *= horizontal_sign
    p = np.asarray(points)[:, [1, 2]] * 1000
    p[:, 0] *= horizontal_sign
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_collection(PolyCollection(projected[order], facecolors=colors[order], linewidths=0))
    triangle = p[[0, 1, 2, 0]]
    ax.fill(triangle[:, 0], triangle[:, 1], color="#4578bd", alpha=0.09, zorder=5)
    ax.plot(triangle[:, 0], triangle[:, 1], "--", color="#4578bd", lw=1.6, zorder=6)
    for base, tip, color in ((1, 3, COLORS[1]), (2, 4, COLORS[2])):
        ax.plot(p[[base, tip], 0], p[[base, tip], 1], "--", color=color, lw=1.3, alpha=0.75, zorder=6)
    ax.plot(p[[3, 5, 4], 0], p[[3, 5, 4], 1], ":", color="#505968", lw=1.3, zorder=6)
    for i, (name, color) in enumerate(zip(NAMES, COLORS)):
        dx = -16 if p[i, 0] < -1 else 16
        dy = -12 if i < 3 else 15
        ax.scatter(*p[i], s=90, color=color, edgecolors="white", linewidths=1.2, zorder=10)
        ax.annotate(name, p[i], xytext=(dx, dy), textcoords="offset points", fontsize=12,
                    weight="bold", color=color, ha="left" if dx > 0 else "right", va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.9), zorder=11)
    extent = projected.reshape(-1, 2)
    ax.set_xlim(extent[:, 0].min() - 38, extent[:, 0].max() + 38)
    ax.set_ylim(min(0, extent[:, 1].min()) - 15, extent[:, 1].max() + 18)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / filename, dpi=240, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(OUT / filename)


model = mujoco.MjModel.from_xml_path(str(OUT.parent / "examples/libero/embodiment/franka_panda/robot.xml"))
data = mujoco.MjData(model)
for name, q in (("finger_joint1", 0.04), ("finger_joint2", -0.04)):
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = q
mujoco.mj_forward(model, data)
fid = model.body("right_hand").id
points = [np.zeros(3)] + [(data.xpos[model.body(name).id] - data.xpos[fid]) @ data.xmat[fid].reshape(3, 3)
                               for name in ("leftfinger", "rightfinger")]
# Exact definition from GeomFrankaPanda._gripper_keypoints_local at width 0.08 m.
points += [np.array([0, 0.04, 0.097]), np.array([0, -0.04, 0.097]), np.array([0, 0, 0.097])]
render(model, data, "right_hand", ["hand_visual", "finger1_visual", "finger2_visual"],
       points, -1, "panda_actor_clean.png")

source = Path("/data0/luokang/research/robosuite/robosuite/models/assets/grippers/robotiq_gripper_85.xml")
root = ET.parse(source).getroot()
for mesh in list(root.findall("asset/mesh")):
    if mesh.get("file").endswith("base_link_vis.stl"):
        root.find("asset").remove(mesh)
    else:
        mesh.set("file", str(source.parent / mesh.get("file")))
model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
data = mujoco.MjData(model)
snapshot = json.loads((OUT / "robotiq85_actor_points.json").read_text())["snapshots"][0]
data.qpos[:] = snapshot["qpos"]
mujoco.mj_forward(model, data)
names = [model.geom(gid).name for gid in range(model.ngeom) if model.geom_group[gid] == 1]
render(model, data, "robotiq_85_adapter_link", names, np.asarray(snapshot["points"]) / 1000,
       1, "robotiq85_actor_clean.png")
