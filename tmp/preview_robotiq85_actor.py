"""Preview proposed actor anchors using the local robosuite XML; no source edits."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import mujoco
import numpy as np
import plotly.graph_objects as go
import trimesh

OUT = Path(__file__).resolve().parent
SOURCE = Path("/data0/luokang/research/robosuite/robosuite/models/assets/grippers/robotiq_gripper_85.xml")
NAMES = ["Root", "Left base", "Right base", "Left fingertip", "Right fingertip", "TCP"]
COLORS = ["#8041ba", "#2171b5", "#069b85", "#ef8a22", "#dd476c", "#d2a000"]
root = ET.parse(SOURCE).getroot()
for mesh in list(root.findall("asset/mesh")):
    if mesh.get("file").endswith("base_link_vis.stl"):
        # Duplicate name and missing file in upstream XML. Keep its existing base mesh.
        root.find("asset").remove(mesh)
    else:
        mesh.set("file", str(SOURCE.parent / mesh.get("file")))
model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
data = mujoco.MjData(model)
snapshots = []
for ctrl in (0.0, 0.4, 0.8):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = [-0.026, -0.267, -0.200] * 2  # Robotiq85GripperBase.init_qpos
    data.ctrl[:] = ctrl
    for _ in range(3000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    # The fixed adapter frame equals world in this isolated gripper model.
    points = [data.xpos[model.body(name).id].copy() for name in
              ("robotiq_85_adapter_link", "left_outer_knuckle", "right_outer_knuckle")]
    for side in ("left", "right"):
        gid = model.geom(f"{side}_fingerpad_collision").id
        # Inner contact face center, not box center. Both pad frames face inward along -y.
        local = np.array([0.0, -model.geom_size[gid, 1], 0.0])
        points.append(data.geom_xpos[gid] + data.geom_xmat[gid].reshape(3, 3) @ local)
    points.append((points[3] + points[4]) / 2)
    vertices, faces = [], []
    for gid in range(model.ngeom):
        if model.geom_group[gid] != 1:
            continue
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            v = model.mesh_vert[va:va + vn]
            f = model.mesh_face[fa:fa + fn]
        elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            box = trimesh.creation.box(extents=2 * model.geom_size[gid])
            v, f = box.vertices, box.faces
        else:
            continue
        faces.append(f + sum(len(x) for x in vertices))
        vertices.append(v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    points = np.asarray(points) * 1000
    snapshots.append(dict(control=ctrl, time_s=float(data.time), qpos=data.qpos.tolist(),
                          qvel_norm=float(np.linalg.norm(data.qvel)), points=points,
                          gap_mm=float(np.linalg.norm(points[3] - points[4])),
                          vertices=np.concatenate(vertices) * 1000, faces=np.concatenate(faces)))
anchors = np.asarray([s["points"][:3] for s in snapshots])
drift = float(np.max(np.linalg.norm(anchors - anchors[0], axis=-1)))
assert drift < 1e-9, drift

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 3, figsize=(17, 8), sharex=True, sharey=True)
interactive = go.Figure()
for col, (ax, s) in enumerate(zip(axes, snapshots)):
    p, v, f = s["points"], s["vertices"], s["faces"]
    triangles = v[f]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    shade = 0.62 + 0.28 * np.abs(normals[:, 0]) / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)
    facecolors = np.column_stack([shade, shade, shade, np.ones(len(shade)) * 0.55])
    order = np.argsort(triangles[:, :, 0].mean(axis=1))
    ax.add_collection(PolyCollection(triangles[order][:, :, [1, 2]], facecolors=facecolors[order], linewidths=0))
    tri = p[[0, 1, 2, 0]]
    ax.fill(tri[:, 1], tri[:, 2], color="#4578bd", alpha=0.09)
    ax.plot(tri[:, 1], tri[:, 2], "--", color="#4578bd", lw=1.6)
    ax.plot(p[3:5, 1], p[3:5, 2], ":", color="#505968", lw=1.3)
    offsets = [(13, -7), (-16, -14), (16, -14), (-14, 15), (14, 15), (13, 13)]
    for i, (name, color, offset) in enumerate(zip(NAMES, COLORS, offsets)):
        ax.scatter(p[i, 1], p[i, 2], s=85, c=color, edgecolors="white", linewidths=1, zorder=10)
        ax.annotate(name, (p[i, 1], p[i, 2]), xytext=offset, textcoords="offset points",
                    ha="right" if offset[0] < 0 else "left", va="center", color=color,
                    fontsize=10, weight="bold", zorder=11,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))
    ax.set(title=f"Actuator target: {s['control']:.1f} rad\nMeasured pad gap: {s['gap_mm']:.1f} mm",
           xlim=(-95, 95), ylim=(-18, 183), xlabel="Adapter Y (mm)", aspect="equal")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    visible = col == 0
    interactive.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                    color="#aeb8c5", opacity=0.35, name="Gripper geometry", visible=visible))
    interactive.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers+text",
                                       marker=dict(size=6, color=COLORS), text=NAMES, textposition="top center",
                                       name="Six actor points", visible=visible,
                                       hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f}) mm<extra></extra>"))
    interactive.add_trace(go.Scatter3d(x=tri[:, 0], y=tri[:, 1], z=tri[:, 2], mode="lines",
                                       line=dict(color="#4578bd", width=5, dash="dash"),
                                       name="Fixed anchor triangle", visible=visible))
axes[0].set_ylabel("Adapter Z (mm)")
fig.suptitle("Robotiq85 | Proposed six actor points", fontsize=23, weight="bold", y=0.98)
fig.text(0.5, 0.925, "Front projection along adapter X | Root and base anchors stay fixed in the adapter frame",
         ha="center", fontsize=12, color="#485366")
fig.legend(handles=[Line2D([], [], color="#4578bd", ls="--", label="Fixed anchor triangle")],
           loc="lower center", bbox_to_anchor=(0.5, 0.12), ncol=2, frameon=False)
fig.text(0.5, 0.085, f"Anchor drift across snapshots: {drift:.3e} mm | TCP = midpoint of the two fingertip contact centers",
         ha="center", fontsize=11)
fig.text(0.5, 0.045, "Original robosuite 1.4.1 geometry; existing base mesh replaces missing visual asset.\n"
         "Independent 6-second simulation snapshots; maximum close command does not fully close this model. XML left/right names retained.",
         ha="center", fontsize=10, color="#596579")
fig.subplots_adjust(left=0.06, right=0.985, top=0.83, bottom=0.21, wspace=0.15)
fig.savefig(OUT / "robotiq85_actor_points.png", dpi=180)
fig.savefig(OUT / "robotiq85_actor_points.pdf")
steps = []
for col, s in enumerate(snapshots):
    steps.append(dict(method="update", label=f"{s['control']:.1f}",
                      args=[{"visible": [i // 3 == col for i in range(9)]},
                            {"title": f"Robotiq85 actor points | Target {s['control']:.1f} rad | Pad gap {s['gap_mm']:.1f} mm"}]))
interactive.update_layout(title=f"Robotiq85 actor points | Target 0.0 rad | Pad gap {snapshots[0]['gap_mm']:.1f} mm",
                          template="plotly_white", height=850,
                          scene=dict(xaxis_title="Adapter X (mm)", yaxis_title="Adapter Y (mm)",
                                     zaxis_title="Adapter Z (mm)", aspectmode="data",
                                     camera=dict(eye=dict(x=1.7, y=0.7, z=0.6))),
                          sliders=[dict(active=0, currentvalue={"prefix": "Actuator target (rad): "}, steps=steps)],
                          annotations=[dict(text="Drag to rotate. Fixed blue triangle; TCP = fingertip midpoint; XML left/right names. "
                                                 "Command snapshots are not calibrated opening widths.",
                                            x=0, y=1.04, xref="paper", yref="paper", showarrow=False)])
interactive.write_html(OUT / "robotiq85_actor_points.html", include_plotlyjs=True)
report = {"source_xml": str(SOURCE), "mujoco_version": mujoco.__version__, "frame": "robotiq_85_adapter_link",
          "units": "mm", "point_names": NAMES, "fixed_anchor_drift_mm": drift,
          "tcp_definition": "Midpoint of the left and right fingertip contact-face centers.",
          "preview_xml_adjustment": "Removed duplicate base mesh declaration referencing a missing visual file; retained existing base mesh.",
          "snapshots": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()
                         if k not in ("vertices", "faces")} for s in snapshots]}
(OUT / "robotiq85_actor_points.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"fixed_anchor_drift_mm": drift, "pad_gaps_mm": [s["gap_mm"] for s in snapshots]}, indent=2))
