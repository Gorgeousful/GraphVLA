"""Local Sawyer registration; reuse named scene transfer and midpoint control."""
from pathlib import Path
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers.rethink_gripper import RethinkGripper
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.models.robots.manipulators.sawyer_robot import Sawyer
from robosuite.robots import ROBOT_CLASS_MAPPING, SingleArm
from examples.libero.embodiment.ur5e.integration import UR5eEnv

ASSETS = Path(__file__).resolve().parent


class GraphVLARethink(RethinkGripper):
    def __init__(self, idn=0):
        GripperModel.__init__(self, str(ASSETS / "gripper.xml"), idn=idn)


class GraphVLASawyer(Sawyer):
    def __init__(self, idn=0):
        ManipulatorModel.__init__(self, str(ASSETS / "arm.xml"), idn=idn)

    @property
    def default_gripper(self):
        return "GraphVLARethink"

    @property
    def base_xpos_offset(self):
        offsets = super().base_xpos_offset
        offsets.update(kitchen_table=offsets["table"],
                       study_table=lambda length: (-0.25 - length / 2, 0, 0),
                       coffee_table=lambda length: (-0.16 - length / 2, 0, 0.41),
                       living_room_table=lambda length: (-0.16 - length / 2, 0, 0.42))
        return offsets


class MountedGraphVLASawyer(GraphVLASawyer):
    pass


class OnTheGroundGraphVLASawyer(GraphVLASawyer):
    @property
    def default_mount(self):
        return None


GRIPPER_MAPPING["GraphVLARethink"] = GraphVLARethink
for robot_type in (GraphVLASawyer, MountedGraphVLASawyer, OnTheGroundGraphVLASawyer):
    ROBOT_CLASS_MAPPING[robot_type.__name__] = SingleArm


class SawyerEnv(UR5eEnv):
    ROBOT_NAME = "GraphVLASawyer"
    JOINT_NAMES = ("l_finger_joint", "r_finger_joint")
    PAD_CONTACTS = (("l_fingerpad_g0", -1), ("r_fingerpad_g0", 1))
