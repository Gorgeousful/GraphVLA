# Sawyer + RethinkGripper

`GeomRobot(embodiment="sawyer")` and `--embodiment sawyer` on the LIBERO
client and GraphPoint server select this embodiment. Panda remains the default.

## Assets

Copied from local robosuite 1.4.1 at `/data0/luokang/research/robosuite`,
commit `b9d8d3de5e3dfd1724f4a0e6555246c460407daa`:

- `arm.xml`: `robosuite/models/assets/robots/sawyer/robot.xml`.
- `gripper.xml`: `robosuite/models/assets/grippers/rethink_gripper.xml`.
- `robot.xml`: arm and gripper assembled at `right_hand`.
- `meshes/`: referenced mesh files copied without geometry changes.
- `LICENSE.robosuite`: upstream license.

Local changes: relative asset paths, explicit radians, visual geom names and
three empty fixed bodies (`actor_frame`, `left_base_anchor`, `right_base_anchor`).
The two upstream actuator control ranges are swapped relative to their associated
joint limits; each local actuator range is corrected to match its own joint.
Two internal collision exclusions prevent the coarse housing cylinder from
penetrating the sliding finger bodies by about 22-25 mm and resisting motion.
External object contacts and opposing finger contacts remain enabled. No other
inertial, friction, damping, actuator gain or mesh parameters are changed.
Installed robosuite and LIBERO files are untouched.

## Six actor points

Coordinates below are in the fixed `actor_frame`, aligned with `gripper_base`.
Semantic left is +Y. Order: root, left base, right base, left fingertip,
right fingertip, TCP.

- Root: `(0, 0, 0)`.
- Bases: `(0, +0.048083, 0.0469)` and `(0, -0.048083, 0.0469)` meters.
  These are the proximal face centers of the two finger-shaft collision boxes
  at full opening, frozen in the housing frame. They never follow finger motion.
- Fingertips: inward contact-face centers of `l_fingerpad_g0` and `r_fingerpad_g0`.
- TCP: exact midpoint of those two centers; orientation uses the fixed actor axes.

The nominal pad gap spans 14.1-78.766 mm. Scalar geometry APIs clip widths to this
range. The nonzero minimum is part of the original finger geometry and limits.
It is not forced to zero. Inference uses both actual joints, including asymmetry;
closedness follows the existing gap / maximum-gap normalization convention.
Preview references are `tmp/sawyer_actor_clean*.png` and `tmp/sawyer_actor_points.json`.

## Inference

Observation state has eight values:
`[midpoint_xyz(3), actor_axis_angle(3), l_finger_joint, r_finger_joint]`.
Although Panda also has an eight-value state, its joint/pose semantics differ;
client and server must select the same embodiment.

The server constructs actor points from measured joints and fits predicted points
back to an absolute midpoint pose. The client converts position AND orientation
to the native OSC `grip_site` frame. The existing UR5e scene-transfer and controller
code is reused with Sawyer robot/joint/pad definitions. Saved Panda initial states
transfer only named scene joints; Sawyer retains its own reset arm configuration.
Result directories receive the existing embodiment suffix (`-sawyer`).

Use a GraphPoint point-action checkpoint and absolute control. Direct action
policies and delta control are not adapted here. Training preprocessing is unchanged.

Inside Docker `lk_cu128`, activate conda `robobrain` and run from the repository root:

```bash
python -m script.server --example libero --embodiment sawyer \
  --ckpt-path YOUR_GRAPHPOINT_CHECKPOINT --port 8001 --execute-chunk-len 5 --sam-only
```

```bash
python -m examples.libero.eval.client --embodiment sawyer \
  --task-suite-name libero_custom_0902 --tasks 0 --num-trials-per-task 1 \
  --absolute-action --control-freq 20 --port 8001 --num-workers 1
```

## Validation

```bash
python -m pytest -q examples/libero/test/test_sawyer_embodiment.py
```

Covers fixed anchors, midpoint and pose roundtrips, asymmetric measured joints,
mesh motion, stable gripper dynamics, server preprocessing/action recovery,
CLI mismatch detection and live LIBERO scene/controller/geometry agreement.
These checks do not measure checkpoint-driven task success or cross-embodiment
generalization. Reachability and grasp success still require actual evaluation.
