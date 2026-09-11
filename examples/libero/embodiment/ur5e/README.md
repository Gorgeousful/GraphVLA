# UR5e + Robotiq85

`GeomRobot(embodiment="ur5e")` provides the six actor points and gripper
mesh/point-cloud projections. The LIBERO evaluation client and GraphPoint server
both accept `--embodiment ur5e`; the default remains `franka_panda`.

## Assets and local corrections

Assets were copied from the local robosuite 1.4.1 checkout at
`/data0/luokang/research/robosuite`, commit
`b9d8d3de5e3dfd1724f4a0e6555246c460407daa`:

- `arm.xml`: `robosuite/models/assets/robots/ur5e/robot.xml`.
- `gripper.xml`: `robosuite/models/assets/grippers/robotiq_gripper_85.xml`.
- `robot.xml`: the arm and gripper assembled at the arm's `right_hand` body.
- `meshes/`: only the meshes referenced by these XML files; original scales,
  shapes, body transforms, inertias and actuator gains are retained.
- `LICENSE.robosuite`: source repository license.

The original gripper XML has a duplicate base mesh declaration referencing a
missing visual file. The local copy removes it and uses the existing base mesh.
Standalone XML files explicitly specify radians, as required by their limits.

The original soft tendons and internal linkage contacts do not produce a stable
parallel closing motion. The project copy uses joint equality constraints
(`outer=q`, `inner finger=-q`, `inner knuckle=q`, symmetric left/right), excludes
the six interfering internal body pairs, and adds joint damping `0.1` and
armature `0.01`. Equality settings are `solref="0.01 1"` and
`solimp="0.99 0.99 0.001"`. Contacts with scene objects and between opposing
fingerpads remain enabled. This is a locally corrected robosuite model, not an
unchanged upstream dynamics reproduction. No installed robosuite or LIBERO
files are modified.

## Six-point convention

Order: root, left base, right base, left fingertip, right fingertip, TCP.

- Root: adapter origin.
- Bases: the two outer-knuckle pivot centers. Their adapter-relative positions
  are fixed for every gripper opening.
- Fingertips: centers of the inward contact faces of `*_fingerpad_collision`.
- TCP position: the exact midpoint of the two fingertip centers.
- TCP orientation: the fixed `actor_frame` axes, rotated by pi about native
  adapter Z so semantic left is +Y, matching Panda.

The nominal geometric opening spans approximately 85.36 to 0.86 mm for joint
angles 0 to 0.8 rad. Scalar geometry APIs accept meters and clip to this range.
Inference instead uses all six measured joints, preserving asymmetry and
contact-induced linkage deviations. TCP offset is recomputed from this state;
it is not a constant translation from the native controller site.

## Inference contract

UR5e requests include `"embodiment": "ur5e"` and `observation.state` with 12
values: `[midpoint_xyz(3), actor_axis_angle(3), gripper_qpos(6)]`. Joint order:

```
finger_joint, left_inner_finger_joint, left_inner_knuckle_joint,
right_outer_knuckle_joint, right_inner_finger_joint, right_inner_knuckle_joint
```

The server builds actor geometry from those joints, normalizes closedness by
Robotiq's maximum opening, and fits predicted actor points using the current
measured gripper shape. Its seven-value output is an absolute canonical midpoint
pose plus gripper command. The client converts that pose to the native OSC
`grip_site` position AND orientation on every step, using the live gripper state.
UR5e currently requires GraphPoint and absolute actions; other policy families
are rejected explicitly. Training/data-preprocessing paths are not changed.

LIBERO's saved initial states contain Panda joint arrays. `UR5eEnv` reads their
scene-joint layout using a camera-free Panda environment and transfers only
matching object/fixture positions and velocities by joint name. UR5e retains
its own reset joint configuration. This preserves scene placement, but does
not guarantee identical reachability or initial end-effector pose across arms.
UR5e evaluation output directories have an `-ur5e` suffix to separate results.

## Run

Inside Docker `lk_cu128`, activate conda `robobrain` and use the repository root.
Keep the existing checkpoint, perception and device arguments, and add
`--embodiment ur5e` to the server. For example:

```bash
python -m script.server --example libero --embodiment ur5e \
  --ckpt-path examples/libero/result/0906-pointdropknn005-basetcpfinger-cls4-rolechain-current-progress-sam-custom0902-ep10-rel/checkpoints/step_30000.pt \
  --port 8001 --execute-chunk-len 5 --sam-only
```

In a second terminal in the same environment:

```bash
python -m examples.libero.eval.client --embodiment ur5e \
  --task-suite-name libero_custom_0902 --tasks 0 --num-trials-per-task 1 \
  --absolute-action --control-freq 20 --port 8001 --num-workers 1
```

Focused tests (geometry roundtrips, stable opening/closing, server adaptation,
scene-state transfer and live controller/geometry agreement):

```bash
python -m pytest -q examples/libero/test/test_ur5e_embodiment.py
```

These checks do not measure trained-policy task success or establish cross-arm
generalization. A full checkpoint-driven rollout must be evaluated separately.
