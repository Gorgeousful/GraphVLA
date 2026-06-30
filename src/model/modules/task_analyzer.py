import sys
sys.path.insert(0, "/data0/luokang/research/GraphVLA/src")
import base64
import io
import re
from textwrap import dedent
from pathlib import Path
from typing import Any, Union
import numpy as np
from openai import OpenAI
from PIL import Image
import json
from src.common.schema import ActionType, Node, NodeRole, SubtaskStructure, TaskStructure
from src.common.schema import taskstructure_to_json, json_to_taskstructure
from rich.console import Console
cs = Console()

# sjtu base_url: https://models.sjtu.edu.cn/api/v1
# siliconflow base_url: https://api.siliconflow.cn/v1
# siliconflow model example: Pro/zai-org/GLM-5.1 Qwen/Qwen3.5-397B-A17B

class Client:
    def __init__(self, 
                api_key: str, 
                base_url: str = "https://models.sjtu.edu.cn/api/v1", 
                model: str = "glm"): 

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300.0,
            max_retries=2,
        )
        self.model = model
    
    def image_to_base64(self, image_input: Union[str, np.ndarray, Image.Image]) -> str:
        """
        将图片转换为Base64格式
        
        Args:
            image_input: 图片文件路径、numpy数组或PIL Image对象
            
        Returns:
            Base64编码的图片数据URI
        """
        if isinstance(image_input, Image.Image):
            # PIL Image对象
            image = image_input.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_data = buffer.getvalue()
            mime_type = "image/jpeg"
        elif isinstance(image_input, str):
            # 文件路径
            image = Image.open(image_input).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_data = buffer.getvalue()
            mime_type = "image/jpeg"
        elif isinstance(image_input, np.ndarray):
            # numpy数组
            image = Image.fromarray(image_input)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_data = buffer.getvalue()
            mime_type = "image/jpeg"
        else:
            raise ValueError("image_input must be a file path (str), numpy array, or PIL Image")
        
        base64_data = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"
     
    def chat(self, 
             text: str, 
             image_input: Union[str, np.ndarray, list[Union[str, np.ndarray]], None]= None, 
             system_prompt: str = "You are a helpful assistant.",
             temperature: float = 0.0, top_p: float = 1.0) -> str:

        content = []
        if image_input is not None:
            # 统一处理为列表
            if isinstance(image_input, (str, np.ndarray)):
                image_inputs = [image_input]
            else:
                image_inputs = image_input
            for img in image_inputs:
                image_base64 = self.image_to_base64(img)
                content.append({"type": "image_url", "image_url": {"url": image_base64}})

        content.append({"type": "text", "text": text})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({
            "role": "user",
            "content": content,
        })
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            cs.print(f"[red]API call failed: {type(e).__name__}: {e}[/red]")
            raise
        
        return completion.choices[0].message.content


class TaskAnalyzer:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://models.sjtu.edu.cn/api/v1",
        model: str = "qwen3vl",
    ):
        self.client = Client(api_key, base_url, model)

        self.binary_action_types = [
            {"name": action_type.value[0], "definition": action_type.value[2]}
            for action_type in ActionType
            if action_type.value[1] == 2
        ]
        self.ternary_action_types = [
            {"name": action_type.value[0], "definition": action_type.value[2]}
            for action_type in ActionType
            if action_type.value[1] == 3
        ]
        self.node_roles = [role.value for role in NodeRole]

    #: Update
    def analyze(self, task_desc, special_situations=None, cache_dir=None, update=False):
        if cache_dir is None:
            return self.analyze_task(task_desc, special_situations)

        task_cache_dir = Path(cache_dir) / task_desc.replace(" ", "_")
        taskstructure_json = task_cache_dir / "taskstructure.json"

        if taskstructure_json.exists() and not update:
            return json_to_taskstructure(taskstructure_json)

        taskstructure = self.analyze_task(task_desc, special_situations)
        task_cache_dir.mkdir(parents=True, exist_ok=True)
        taskstructure_to_json(taskstructure, taskstructure_json)
        return taskstructure

    def analyze_task(
        self,
        task_desc: str,
        special_situations: str | None = None,
    ) -> TaskStructure:
        split_data = self.split_task(task_desc)

        analyzed_subtasks = [
            self.analyze_one_subtask(
                subtask_text=item["subtask"],
                special_situations=special_situations,
            )
            for item in split_data["subtasks"]
        ]

        return self._to_task_structure(
            {
                "task": task_desc,
                "subtasks": analyzed_subtasks,
            }
        )

    #: Common helpers
    def _json_system_prompt(self) -> str:
        return dedent("""\
            You are a robotics task decomposition expert.
            Return ONLY a valid JSON object, with no markdown, code fences,
            comments, or explanatory text.
        """)

    def _request_json(
        self,
        prompt: str,
        validator=None,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        last_error = None
        last_response = ""

        for attempt in range(1, max_attempts + 1):
            retry_note = ""
            if attempt > 1:
                retry_note = dedent(f"""\

                    The previous response could not be parsed or did not satisfy the required format.
                    Error: {last_error}
                    Return ONLY one valid JSON object.
                """)

            last_response = self.client.chat(
                text=prompt + retry_note,
                system_prompt=self._json_system_prompt(),
                temperature=0.0,
                top_p=1.0,
            )

            try:
                data = self._parse_json(last_response)
                return validator(data) if validator else data
            except (TypeError, ValueError, KeyError) as exc:
                last_error = exc

        preview = str(last_response).strip().replace("\n", " ")[:200]
        raise ValueError(
            f"Cannot get valid JSON after {max_attempts} attempts. "
            f"Last error: {last_error}. Preview: {preview}"
        )

    def _parse_json(self, response: str | dict | list) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if isinstance(response, list):
            raise ValueError("Expected a JSON object, got list")
        if not isinstance(response, str):
            raise TypeError(f"Expected str, dict, or list, got {type(response).__name__}")

        text = (
            response.strip()
            .lstrip("\ufeff")
            .replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
        )

        candidates = [text]
        candidates.extend(
            block.strip()
            for block in re.findall(
                r"```(?:json|JSON)?\s*(.*?)```",
                text,
                flags=re.DOTALL,
            )
        )

        json_candidates = list(candidates)
        for candidate in candidates:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                json_candidates.append(candidate[start:end + 1])

        for candidate in json_candidates:
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate.strip())
            if not cleaned:
                continue

            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                continue

            if isinstance(data, dict):
                return data

            raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

        preview = text.replace("\n", " ")[:200]
        raise ValueError(f"Cannot parse response as JSON object. Preview: {preview}")

    def _to_task_structure(self, data: dict[str, Any]) -> TaskStructure:
        raw_subtasks = data.get("subtasks")
        if not isinstance(raw_subtasks, list):
            raise ValueError("Task JSON must contain a list field named 'subtasks'")

        subtask_list = []

        for raw_subtask in raw_subtasks:
            if not isinstance(raw_subtask, dict):
                raise ValueError(
                    f"Each subtask must be a dict, got {type(raw_subtask).__name__}"
                )

            raw_nodes = raw_subtask.get("nodes")
            if not isinstance(raw_nodes, list):
                raise ValueError("Each subtask must contain a list field named 'nodes'")

            node_list = []

            for fallback_id, raw_node in enumerate(raw_nodes):
                if not isinstance(raw_node, dict):
                    raise ValueError(
                        f"Each node must be a dict, got {type(raw_node).__name__}"
                    )

                role = NodeRole(raw_node["role"])

                need_object = raw_node.get("need_object", role == NodeRole.PATIENT)
                if isinstance(need_object, str):
                    value = need_object.strip().lower()
                    if value not in ("true", "false"):
                        raise ValueError(f"Invalid need_object value: {need_object}")
                    need_object = value == "true"
                else:
                    need_object = bool(need_object)

                node_list.append(
                    Node(
                        id=int(raw_node.get("id", fallback_id)),
                        name=str(raw_node["name"]),
                        need_object=need_object,
                        role=role,
                        canon_pcd=None,
                        pos=None,
                        rot6d=None,
                        gripper=None,
                    )
                )

            raw_action_type = raw_subtask["action_type"]
            action_type = next(
                (
                    action_type
                    for action_type in ActionType
                    if action_type.value[0] == raw_action_type
                ),
                None,
            )
            if action_type is None:
                raise ValueError(f"Invalid action_type: {raw_action_type}")

            subtask_list.append(
                SubtaskStructure(
                    subtask=str(raw_subtask["subtask"]),
                    action_type=action_type,
                    action_degree=raw_subtask.get("action_degree"),
                    node_list=node_list,
                )
            )

        return TaskStructure(
            task=str(data.get("task", "")),
            subtask_list=subtask_list,
        )

    #: Stage 1: task -> subtasks
    def split_task(self, task_desc: str) -> dict[str, Any]:
        prompt = self._build_split_prompt(task_desc)
        return self._request_json(
            prompt=prompt,
            validator=lambda data: self._validate_split_data(data, task_desc),
        )

    def _build_split_prompt(self, task_desc: str) -> str:
        task_desc_json = json.dumps(task_desc, ensure_ascii=False)

        return dedent(f"""\
            - Split the instruction by explicitly stated action phrases only.
            - This is a surface-level language split, not a planning step.
            - DO NOT add implicit prerequisite actions that are not explicitly stated in the instruction.
            - Replace pronouns (e.g., "it", "them", "this") with the actual objects they refer to in each subtask.
            - When a single action phrase involves multiple objects, split it into separate subtasks — one per object.
            - Treat coordinated adjectives before a singular noun as describing one object, not multiple objects. 

            Task:
            {task_desc_json}

            JSON format:
            {{
                "task": {task_desc_json},
                "subtasks": [
                    {{
                        "id": 0,
                        "subtask": "<subtask text>"
                    }}
                ]
            }}

            Example:
            {{
                "task": "put the white mug between the box and the cup to the left of the plate and open the drawer",
                "subtasks": [
                    {{
                        "id": 0,
                        "subtask": "put the white mug between the box and the cup to the left of the plate"
                    }},
                    {{
                        "id": 1,
                        "subtask": "open the drawer"
                    }}
                ]
            }}
        """)

    def _validate_split_data(
        self,
        data: dict[str, Any],
        task_desc: str,
    ) -> dict[str, Any]:
        raw_subtasks = data.get("subtasks")
        if not isinstance(raw_subtasks, list):
            raise ValueError("Split JSON must contain a list field named 'subtasks'")

        subtasks = []
        for fallback_id, item in enumerate(raw_subtasks):
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("subtask")
            else:
                raise ValueError(f"Invalid split subtask type: {type(item).__name__}")

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Each split subtask must contain non-empty text")

            subtasks.append(
                {
                    "id": fallback_id,
                    "subtask": text.strip(),
                }
            )

        if not subtasks:
            raise ValueError("At least one subtask is required")

        return {
            "task": str(data.get("task", task_desc)),
            "subtasks": subtasks,
        }

    #: Stage 2: one isolated subtask -> structured graph
    def analyze_one_subtask(
        self,
        subtask_text: str,
        special_situations: str | None = None,
    ) -> dict[str, Any]:
        prompt = self._build_one_subtask_prompt(
            subtask_text=subtask_text,
            special_situations=special_situations,
        )

        return self._request_json(
            prompt=prompt,
            validator=lambda data: self._validate_one_subtask_analysis(
                data=data,
                subtask_text=subtask_text,
            ),
        )

    def _build_one_subtask_prompt(
        self,
        subtask_text: str,
        special_situations: str | None = None,
    ) -> str:
        subtask_json = json.dumps(subtask_text, ensure_ascii=False)
        binary_action_types = json.dumps(self.binary_action_types, ensure_ascii=False)
        ternary_action_types = json.dumps(self.ternary_action_types, ensure_ascii=False)
        special_situations = special_situations or ""

        return dedent(f"""\
            Analyze the robotic manipulation subtask and output its structured graph.

            Subtask:
            {subtask_json}

            Allowed action_types:
            - binary action_type (actor, patient):
            {binary_action_types}
            - ternary action_type (actor, patient, target):
            {ternary_action_types}

            Allowed roles:
            - "actor": the robot gripper.
            - "patient": the manipulated object or affordance, including qualifiers that identify which instance is being manipulated, such as 'A near B', 'A from B', 'A on B'
            - "target": the target object.

            Notes:
            - You must first determine the action type and decide whether it is binary or ternary. If it is binary, you should not include a target node.
            - Put scalar constraints on the patient action into action_degree. e.g. 'on', 'in', 'right', 'left', 'front', 'behind', 'outward', 'inward'.
            - For nodes with role patient or target, need_object defaults to true, unless explicitly specified in special situations.

            Special situations for node name:
            {special_situations}

            JSON format:
            {{
                "subtask": <subtask description>,
                "action_type": "<one allowed action_type>",
                "action_degree": "<scalar constraint or null>",
                "nodes": [
                    {{"id": 0, "name": "robotic gripper", "role": "actor", "need_object": false}},
                    {{"id": 1, "name": "<patient description>", "role": "patient", "need_object": true}},
                    {{"id": 2, "name": "<target description if needed>", "role": "target", "need_object": true}}
                ]
            }}

            Example:
            {{
                "subtask": "open the drawer",
                "action_type": "slide",
                "action_degree": "outward",
                "nodes": [
                    {{"id": 0, "name": "robotic gripper", "role": "actor", "need_object": false}},
                    {{"id": 1, "name": "the drawer handle", "role": "patient", "need_object": true}}
                ]
            }}
        """)

    def _validate_one_subtask_analysis(
        self,
        data: dict[str, Any],
        subtask_text: str,
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError(f"Subtask analysis must be a dict, got {type(data).__name__}")

        raw_action_type = data.get("action_type")
        valid_action_types = {action_type.value[0] for action_type in ActionType}
        if raw_action_type not in valid_action_types:
            raise ValueError(f"Invalid action_type: {raw_action_type}")

        raw_nodes = data.get("nodes")
        if not isinstance(raw_nodes, list):
            raise ValueError("Subtask analysis must contain a list field named 'nodes'")

        if len(raw_nodes) < 2:
            raise ValueError("Subtask analysis must contain at least actor and patient nodes")

        valid_roles = {role.value for role in NodeRole}
        for node in raw_nodes:
            if not isinstance(node, dict):
                raise ValueError(f"Each node must be a dict, got {type(node).__name__}")
            if node.get("role") not in valid_roles:
                raise ValueError(f"Invalid node role: {node.get('role')}")
            if not isinstance(node.get("name"), str) or not node["name"].strip():
                raise ValueError("Each node must contain a non-empty name")

        data["subtask"] = subtask_text
        data.setdefault("action_degree", None)
        return data


       
if __name__ == "__main__":
    analyzer = TaskAnalyzer(api_key= "sk-UZpG2yYwDE5itw7s57eIJA", 
                base_url= "https://models.sjtu.edu.cn/api/v1", 
                model= "glm")

    mode = "libero"

    if mode == "common":
        task_descs = [
            "put the white mug on the plate and put the chocolate pudding to the right of the plate",
            "push the red box in the microwave and put the blue box on the top of the microwave",
            "put the white mug between cup and yelow box to the left of the plate and close the drawer"
        ]
        special_situations = ""

    elif mode == "libero":
        task_descs = [
            "put the white mug on the left plate and put the yellow and white mug on the right plate",
            "put the white mug on the plate and put the chocolate pudding to the right of the plate",
            "put the yellow and white mug in the microwave and close it",
            "turn on the stove and put the moka pot on it",
            "put both the alphabet soup and the cream cheese box in the basket",
            "put both the alphabet soup and the tomato sauce in the basket",
            "put both moka pots on the stove",
            "put both the cream cheese box and the butter in the basket",
            "put the black bowl in the bottom drawer of the cabinet and close it",
            "pick up the book and place it in the back compartment of the caddy",
        ]
        special_situations = """
        - 'turn on/off the stove' -> 'rotate the stove knob right/left'
        """

    elif mode == "calvin":
        special_situations = """
        - 'turn on/off the lightbulb" is done by sliding the light switch up/down.
        - 'turn on/off the led' is done by pressing the black button.
        - 'slide the door' is done by sliding the door handle.
        - 'open/close the drawer' is done by sliding the drawer handle.
        """
        task_descs = [
            "rotate the red block right",
            "rotate the red block left",
            "rotate the blue block right",
            "rotate the blue block left",
            "rotate the pink block right",
            "rotate the pink block left",
            "push the red block right",
            "push the red block left",
            "push the blue block right",
            "push the blue block left",
            "push the pink block right",
            "push the pink block left",
            "slide the door left",
            "slide the door right",
            "open the drawer",
            "close the drawer",
            "turn on the lightbulb",
            "turn off the lightbulb",
            "lift the red block from the table",
            "lift the blue block from the table",
            "lift the pink block from the table",
            "lift the red block from the sliding cabinet",
            "lift the blue block from the sliding cabinet",
            "lift the pink block from the sliding cabinet",
            "lift the red block from the drawer",
            "lift the blue block from the drawer",
            "lift the pink block from the drawer",
            "place the object in the sliding cabinet",
            "place the object in the drawer",
            "sweep the object into the drawer",
            "stack the blocks",
            "remove the top block from the stack",
            "turn on the led",
            "turn off the led",
        ]


    for task_desc in task_descs:
        response = analyzer.analyze(task_desc, 
        special_situations, 
        cache_dir="/data0/luokang/research/GraphVLA/examples/libero/cache",
        update=False)
        cs.print(response, markup=False)
    
