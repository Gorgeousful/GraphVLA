from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import json
import os


class NodeRole(str, Enum):
    ACTOR = "actor"
    PATIENT = "patient"
    TARGET = "target"


class ActionType(tuple, Enum):
    # Canonical primitive or macro actions.
    # Value format: (name, arity, definition).
    # Ternary relation: actor, patient, explicit target.
    PUT = ("put", 3, "pick and place the patient to the target.")
    MOVE = ("sweep", 3, "sweep or push the patient along a surface to the target.")
    POUR = ("pour", 3, "pour the contents of the patient/container into the target.")
    ALIGN = ("align", 3, "align the patient with the target or reference object.")
    INSERT = ("insert", 3, "insert the patient into the target.")
    PLACE = ("place", 3, "place the already-held patient onto or into the target.")

    # Binary relation: actor, patient.
    LIFT = ("lift", 2, "raise the patient upward from its initial support.")
    REMOVE = ("remove", 2, "take the patient away from its current support.")
    PUSH = ("push", 2, "push the patient in a direction without an explicit target.")
    SLIDE = ("slide", 2, "slide the patient along its constrained track.")
    ROTATE = ("rotate", 2, "rotate the patient around an axis.")
    PRESS = ("press", 2, "press the patient.")
    SHAKE = ("shake", 2, "shake the patient.")


@dataclass
class Node:
    id: int
    name: str
    need_object: bool
    role: NodeRole
    point: Optional[NDArray] = None
    canon_pcd: Optional[NDArray] = None
    pos: Optional[NDArray] = None
    rot6d: Optional[NDArray] = None
    gripper: Optional[NDArray] = None  # 0 close 1 open

    def __post_init__(self) -> None:
        # common
        self.role = NodeRole(self.role)

        if self.point is not None and self.point.shape[-1] != 2:
            raise ValueError(f"point should have shape (N,2), got {self.point.shape}")
        if self.canon_pcd is not None and self.canon_pcd.shape[-2:] != (512, 3):
            raise ValueError(f"canon_pcd should have shape (512, 3), got {self.canon_pcd.shape}")
        if self.pos is not None and self.pos.shape[-1] != 3:
            raise ValueError(f"pos should have shape (H, 3), got {self.pos.shape}")
        if self.rot6d is not None and self.rot6d.shape[-1] != 6:
            raise ValueError(f"rot6d should have shape (H, 6), got {self.rot6d.shape}")
        if self.gripper is not None and self.gripper.shape[-1] != 1:
            raise ValueError(f"gripper should have shape (H, 1), got {self.gripper.shape}")

        # specific
        if self.role == NodeRole.ACTOR:
            self.need_object = False
        if self.role == NodeRole.PATIENT:
            self.need_object = True
        if self.role == NodeRole.ACTOR and self.canon_pcd is not None:
            self.canon_pcd = np.zeros_like(self.canon_pcd)
        if self.role != NodeRole.ACTOR and self.gripper is not None:
            self.gripper = np.zeros_like(self.gripper)


@dataclass
class SubtaskStructure:
    subtask: str
    action_type: ActionType
    action_degree: Optional[str] = None
    node_list: Optional[list[Node]] = None

    def __post_init__(self) -> None:
        # common
        self.action_type = ActionType(self.action_type)


@dataclass
class TaskStructure:
    task: str
    subtask_list: list[SubtaskStructure]

    def __post_init__(self) -> None:
        pass



def taskstructure_to_json(taskstructure, json_path=None) -> dict:
    data = {
        "task": taskstructure.task,
        "subtasks": [],
    }

    for subtask in taskstructure.subtask_list:
        nodes = []
        for node in (subtask.node_list or []):
            nodes.append({
                "id": int(node.id),
                "name": node.name,
                "need_object": bool(node.need_object),
                "role": node.role.value,
                "point": None if node.point is None else np.asarray(node.point).tolist(),
                "canon_pcd": None if node.canon_pcd is None else np.asarray(node.canon_pcd).tolist(),
                "pos": None if node.pos is None else np.asarray(node.pos).tolist(),
                "rot6d": None if node.rot6d is None else np.asarray(node.rot6d).tolist(),
                "gripper": None if node.gripper is None else np.asarray(node.gripper).tolist(),
            })

        data["subtasks"].append({
            "subtask": subtask.subtask,
            "action_type": subtask.action_type.value[0],
            "action_degree": subtask.action_degree,
            "nodes": nodes,
        })

    if json_path is not None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")

    return data
    

def json_to_taskstructure(json_data) -> TaskStructure:
    if isinstance(json_data, (str, bytes, os.PathLike)):
        with open(json_data, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json_data

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict or JSON path, got {type(data).__name__}")

    raw_subtasks = data.get("subtasks")
    if not isinstance(raw_subtasks, list):
        raise ValueError("Task JSON must contain a list field named subtasks")

    subtask_list = []
    for raw_subtask in raw_subtasks:
        if not isinstance(raw_subtask, dict):
            raise ValueError(
                f"Each subtask must be a dict, got {type(raw_subtask).__name__}"
            )

        node_list = []
        for fallback_id, raw_node in enumerate(raw_subtask.get("nodes", [])):
            if not isinstance(raw_node, dict):
                raise ValueError(
                    f"Each node must be a dict, got {type(raw_node).__name__}"
                )

            node_list.append(Node(
                id=int(raw_node.get("id", fallback_id)),
                name=str(raw_node["name"]),
                need_object=bool(raw_node.get("need_object", True)),
                role=NodeRole(raw_node["role"]),
                point=None if raw_node.get("point") is None else np.asarray(raw_node["point"], dtype=np.float32),
                canon_pcd=None if raw_node.get("canon_pcd") is None else np.asarray(raw_node["canon_pcd"], dtype=np.float32),
                pos=None if raw_node.get("pos") is None else np.asarray(raw_node["pos"], dtype=np.float32),
                rot6d=None if raw_node.get("rot6d") is None else np.asarray(raw_node["rot6d"], dtype=np.float32),
                gripper=None if raw_node.get("gripper") is None else np.asarray(raw_node["gripper"], dtype=np.float32),
            ))

        raw_action_type = raw_subtask["action_type"]
        action_type = next(
            (item for item in ActionType if item.value[0] == raw_action_type),
            None,
        )
        if action_type is None:
            raise ValueError(f"Invalid action_type: {raw_action_type}")

        subtask_list.append(SubtaskStructure(
            subtask=str(raw_subtask["subtask"]),
            action_type=action_type,
            action_degree=raw_subtask.get("action_degree"),
            node_list=node_list,
        ))

    return TaskStructure(
        task=str(data.get("task", "")),
        subtask_list=subtask_list,
    )


def json_to_substructure(json_data) -> SubtaskStructure:
    if isinstance(json_data, (str, bytes, os.PathLike)):
        with open(json_data, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json_data

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict or JSON path, got {type(data).__name__}")

    node_list = []
    for fallback_id, raw_node in enumerate(data.get("nodes", [])):
        if not isinstance(raw_node, dict):
            raise ValueError(f"Each node must be a dict, got {type(raw_node).__name__}")

        node_list.append(Node(
            id=int(raw_node.get("id", fallback_id)),
            name=str(raw_node["name"]),
            need_object=bool(raw_node.get("need_object", True)),
            role=NodeRole(raw_node["role"]),
            point=None if raw_node.get("point") is None else np.asarray(raw_node["point"], dtype=np.float32),
            canon_pcd=None if raw_node.get("canon_pcd") is None else np.asarray(raw_node["canon_pcd"], dtype=np.float32),
            pos=None if raw_node.get("pos") is None else np.asarray(raw_node["pos"], dtype=np.float32),
            rot6d=None if raw_node.get("rot6d") is None else np.asarray(raw_node["rot6d"], dtype=np.float32),
            gripper=None if raw_node.get("gripper") is None else np.asarray(raw_node["gripper"], dtype=np.float32),
        ))

    raw_action_type = data["action_type"]
    action_type = next(
        (item for item in ActionType if item.value[0] == raw_action_type),
        None,
    )
    if action_type is None:
        raise ValueError(f"Invalid action_type: {raw_action_type}")

    return SubtaskStructure(
        subtask=str(data["subtask"]),
        action_type=action_type,
        action_degree=data.get("action_degree"),
        node_list=node_list,
    )


def json_to_jsonl(json_path_list, jsonl_path):
    if isinstance(json_path_list, (str, bytes, os.PathLike)):
        json_path_list = [json_path_list]

    jsonl_dir = os.path.dirname(os.fspath(jsonl_path))
    if jsonl_dir:
        os.makedirs(jsonl_dir, exist_ok=True)

    data_list = []
    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for json_path in json_path_list:
            with open(json_path, "r", encoding="utf-8") as in_f:
                data = json.load(in_f)
            data_list.append(data)
            json.dump(data, out_f, ensure_ascii=False, separators=(",", ":"))
            out_f.write("\n")
    return data_list
