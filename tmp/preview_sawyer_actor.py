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


# Use upstream geometry unchanged; only resolve assets and name unnamed visual geoms.
SOURCE = Path("/data0/luokang/research/robosuite/robosuite/models/assets/grippers/rethink_gripper.xml")
root = ET.parse(SOURCE).getroot()
for mesh in root.findall("asset/mesh"):
    mesh.set("file", str(SOURCE.parent / mesh.get("file")))
for i, geom in enumerate(root.findall(".//worldbody//geom")):
    if geom.get("name") is None:
        geom.set("name", f"preview_geom_{i}")
model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
data = mujoco.MjData(model)
fid = model.body("gripper_base").id
# Freeze the proximal ends of the finger shafts at the fully open reference.
# These remain virtual anchors in gripper_base, not moving finger landmarks.
for side, q in (("l", 0.020833), ("r", -0.020833)):
    data.qpos[model.jnt_qposadr[model.joint(f"{side}_finger_joint").id]] = q
mujoco.mj_forward(model, data)
anchors = [np.zeros(3)]
for side in ("l", "r"):
    gid = model.geom(f"{side}_finger_g0").id
    proximal = np.array([0., 0., -model.geom_size[gid, 2]])
    world = data.geom_xpos[gid] + data.geom_xmat[gid].reshape(3, 3) @ proximal
    anchors.append((world - data.xpos[fid]) @ data.xmat[fid].reshape(3, 3))
anchors = np.asarray(anchors)
geom_names = [model.geom(i).name for i in range(model.ngeom) if model.geom_group[i] == 1]
snapshots = []
for label, q in (("open", 0.020833), ("middle", 0.005), ("closed", -0.0115)):
    data.qpos[model.jnt_qposadr[model.joint("l_finger_joint").id]] = q
    data.qpos[model.jnt_qposadr[model.joint("r_finger_joint").id]] = -q
    mujoco.mj_forward(model, data)
    origin, rotation = data.xpos[fid], data.xmat[fid].reshape(3, 3)
    points = list(anchors.copy())
    for side, sign in (("l", -1), ("r", 1)):
        gid = model.geom(f"{side}_fingerpad_g0").id
        inward_face = np.array([0., sign * model.geom_size[gid, 1], 0.])
        world = data.geom_xpos[gid] + data.geom_xmat[gid].reshape(3, 3) @ inward_face
        points.append((world - origin) @ rotation)
    points.append((points[3] + points[4]) / 2)
    points = np.asarray(points)
    np.testing.assert_allclose(points[5], (points[3] + points[4]) / 2, atol=1e-12)
    snapshots.append(dict(opening=label, qpos=data.qpos.tolist(), points_mm=(points * 1000).tolist(),
                          pad_gap_mm=float(np.linalg.norm(points[3] - points[4]) * 1000)))
    filename = "sawyer_actor_clean.png" if label == "open" else f"sawyer_actor_clean_{label}.png"
    render(model, data, "gripper_base", geom_names, points, -1, filename)
fixed = np.asarray([s["points_mm"][:3] for s in snapshots])
assert np.max(np.abs(fixed - fixed[0])) == 0
report = dict(source_xml=str(SOURCE), frame="gripper_base", point_names=NAMES,
              base_definition="Proximal face centers of l/r_finger_g0 at full opening, frozen in gripper_base; never follow sliding fingers.",
              fingertip_definition="Inward contact-face centers of l/r_fingerpad_g0.",
              tcp_definition="Midpoint of the two fingertips.",
              validation="Kinematic qpos snapshots, not simulated grasp or dynamics validation.",
              fixed_anchor_drift_mm=float(np.max(np.abs(fixed - fixed[0]))), snapshots=snapshots)
(OUT / "sawyer_actor_points.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
