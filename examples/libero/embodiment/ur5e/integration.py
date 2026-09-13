"""Project-local robosuite registration and LIBERO state / controller adaptation."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers.robotiq_85_gripper import Robotiq85Gripper
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.models.robots.manipulators.ur5e_robot import UR5e
from robosuite.robots import ROBOT_CLASS_MAPPING, SingleArm
from libero.libero.envs import OffScreenRenderEnv

ASSETS = Path(__file__).resolve().parent
JOINT_NAMES = ("finger_joint", "left_inner_finger_joint", "left_inner_knuckle_joint",
               "right_outer_knuckle_joint", "right_inner_finger_joint", "right_inner_knuckle_joint")


class GraphVLARobotiq85(Robotiq85Gripper):
    def __init__(self, idn=0):
        GripperModel.__init__(self, str(ASSETS / "gripper.xml"), idn=idn)

    @property
    def init_qpos(self):
        return np.zeros(6)


class GraphVLAUR5e(UR5e):
    def __init__(self, idn=0):
        ManipulatorModel.__init__(self, str(ASSETS / "arm.xml"), idn=idn)

    @property
    def init_qpos(self):
        qpos = super().init_qpos.copy()
        # Align actor-frame yaw with Panda demonstrations before control starts.
        qpos[-1] += np.pi
        return qpos

    @property
    def default_gripper(self):
        return "GraphVLARobotiq85"

    @property
    def base_xpos_offset(self):
        offsets = super().base_xpos_offset
        offsets.update(kitchen_table=offsets["table"],
                       study_table=lambda length: (-0.25 - length / 2, 0, 0),
                       coffee_table=lambda length: (-0.16 - length / 2, 0, 0.41),
                       living_room_table=lambda length: (-0.16 - length / 2, 0, 0.42))
        return offsets


class MountedGraphVLAUR5e(GraphVLAUR5e):
    pass


class OnTheGroundGraphVLAUR5e(GraphVLAUR5e):
    @property
    def default_mount(self):
        return None


GRIPPER_MAPPING["GraphVLARobotiq85"] = GraphVLARobotiq85
for robot_type in (GraphVLAUR5e, MountedGraphVLAUR5e, OnTheGroundGraphVLAUR5e):
    ROBOT_CLASS_MAPPING[robot_type.__name__] = SingleArm


class UR5eEnv(OffScreenRenderEnv):
    """Reuse Panda scene init states by named scene joints, never by robot indices."""
    ROBOT_NAME = "GraphVLAUR5e"
    JOINT_NAMES = JOINT_NAMES
    PAD_CONTACTS = (("left_fingerpad_collision", -1), ("right_fingerpad_collision", -1))

    def __init__(self, **kwargs):
        super().__init__(robots=[self.ROBOT_NAME], **kwargs)
        source_kwargs = dict(kwargs, use_camera_obs=False, has_offscreen_renderer=False,
                             camera_depths=False)
        source = OffScreenRenderEnv(**source_kwargs)
        try:
            self._source_shape = (source.sim.model.nq, source.sim.model.nv)
            self._scene_indices = []
            source_names = {n for n in source.sim.model.joint_names if not n.startswith(("robot", "gripper"))}
            target_names = {n for n in self.sim.model.joint_names if not n.startswith(("robot", "gripper"))}
            if source_names != target_names:
                raise ValueError("Source and target scene joints differ")
            for kind in ("qpos", "qvel"):
                src, dst = [], []
                for name in sorted(source_names):
                    for model, indices in ((source.sim.model, src), (self.sim.model, dst)):
                        addr = getattr(model, f"get_joint_{kind}_addr")(name)
                        indices.extend(range(*addr) if isinstance(addr, tuple) else [addr])
                if len(src) != len(dst):
                    raise ValueError("Source and target scene joint dimensions differ")
                self._scene_indices.append((np.asarray(src, dtype=int), np.asarray(dst, dtype=int)))
        finally:
            source.close()

    def set_init_state(self, init_state):
        source = np.asarray(init_state, dtype=np.float64)
        nq, nv = self._source_shape
        if source.shape != (1 + nq + nv,):
            raise ValueError(f"Expected Panda scene state of length {1 + nq + nv}, got {source.shape}")
        target = self.sim.get_state()
        for values, incoming, (src, dst) in zip(
                (target.qpos, target.qvel), (source[1:1+nq], source[1+nq:]), self._scene_indices):
            values[dst] = incoming[src]
        target.time = source[0]
        result = super().set_init_state(target.flatten())
        self.robots[0].controller.reset_goal()
        return result


def observation_state(env):
    """Canonical midpoint pose and native gripper qpos, from the live simulator."""
    sim, robot = env.sim, env.robots[0]
    # mj_step can leave derived body/geom poses one integration step behind qpos.
    sim.forward()
    prefix = robot.gripper.naming_prefix
    tips = []
    for pad, sign in env.PAD_CONTACTS:
        name = prefix + pad
        gid = sim.model.geom_name2id(name)
        tips.append(sim.data.get_geom_xpos(name) + sim.data.get_geom_xmat(name) @ np.array(
            [0.0, sign * sim.model.geom_size[gid, 1], 0.0]))
    rotation = sim.data.get_body_xmat(prefix + "actor_frame")
    joints = [float(sim.data.get_joint_qpos(prefix + name)) for name in env.JOINT_NAMES]
    return np.concatenate([np.mean(tips, axis=0), R.from_matrix(rotation).as_rotvec(), joints])


def controller_action(action, env):
    """Convert an absolute midpoint pose to the native OSC grip_site pose."""
    action = np.asarray(action, dtype=np.float64).copy()
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Absolute action must have seven finite values")
    state = observation_state(env)
    current_rotation = R.from_rotvec(state[3:6]).as_matrix()
    target_rotation = R.from_rotvec(action[3:6]).as_matrix()
    sim, robot = env.sim, env.robots[0]
    site = robot.eef_site_id
    site_offset = current_rotation.T @ (sim.data.site_xpos[site] - state[:3])
    site_rotation = current_rotation.T @ sim.data.site_xmat[site].reshape(3, 3)
    action[:3] += target_rotation @ site_offset
    action[3:6] = R.from_matrix(target_rotation @ site_rotation).as_rotvec()
    return action.astype(np.float32)
