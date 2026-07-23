import os
import re
import cv2
import copy
import json
import torch
import importlib.util
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union
import subprocess
from io import BytesIO
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.common.schema import ActionType, Node, NodeRole, SubtaskStructure, TaskStructure
from rich.console import Console
cs = Console()


class NodeLocatorRobo:
    """
    A unified class for performing inference using RoboBrain 2.5 models.
    """ 
    def __init__(self, model_id="/data0/luokang/dataset/luokang/ckpts/RoboBrain2.5-8B-NV", device_map="auto"):
        """
        Initialize the model and processor.
        
        Args:
            model_id (str): Path or Hugging Face model identifier
            device_map (str): Device mapping strategy ("auto", "cuda:0", etc.)
        """
        cs.print("Loading Checkpoint ...")
        self.model_id = model_id
        self.device_map = device_map
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            dtype="auto", 
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _input_device(self):
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def resize(self, images, scale=1.0):
        """
        Resize images in-memory. Accepts file paths, PIL Images, or numpy arrays.
        Always returns a list of PIL Images.

        Args:
            images (list): List of file paths (str), PIL Images, or numpy arrays.
            scale (float): Scale factor applied to both width and height.

        Returns:
            list[PIL.Image.Image]: Resized (or original) PIL Images.
        """
        result = []
        for img in images:
            if isinstance(img, str):
                pil_img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            if scale != 1.0:
                new_w, new_h = int(pil_img.width * scale), int(pil_img.height * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

            result.append(pil_img)
        return result

    def inference(self, text, image, task="pointing",
                  plot=False, plot_output_dir=None, image_name=None,
                  do_sample=False, temperature=0.7, resize_scale=1.0):
        """
        Perform inference with text and images input.

        Args:
            text (str): The input text prompt.
            image: Image(s) as file path(s), PIL Image(s), or numpy array(s).
                   A single item or a list.
            task (str): The task type: "general", "pointing", "trajectory", "grounding".
            plot (bool): Whether to plot results on image.
            plot_output_dir (str): Directory to save annotated images. Defaults to
                                   image directory (for paths) or __tmp__/node_locator_plots.
            image_name (str): Base name for the output plot file (e.g. "video_001_first.png").
                              Defaults to "{task}_annotated.png".
            do_sample (bool): Whether to use sampling during generation.
            temperature (float): Temperature for sampling.
            resize_scale (float): Scale factor to resize images. 1.0 = no resize.
        """
        # Normalize to list
        if not isinstance(image, list):
            image = [image]

        assert task in ["general", "pointing", "trajectory", "grounding"], \
            f"Invalid task type: {task}. Supported tasks are 'general', 'pointing', 'trajectory', 'grounding'."
        assert task == "general" or (task in ["pointing", "trajectory", "grounding"] and len(image) == 1), \
            "Pointing, grounding, and trajectory tasks require exactly one image."

        # Convert everything to PIL Images (resize in-memory, no disk I/O)
        image = self.resize(image, scale=resize_scale)

        if task == "pointing":
            # cs.print("Pointing task detected. Adding pointing prompt.")
            text = f"{text}. Please provide its 2D coordinates. Your answer should be formatted as a tuple, i.e. [(x, y)], where the tuple contains the x and y coordinates of a point satisfying the conditions above."
        elif task == "trajectory":
            # cs.print("Trajectory task detected. Adding trajectory prompt.")
            text = f"Please predict 3D end-effector-centric waypoints to complete the task successfully. The task is \"{text}\". Your answer should be formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains the x and y coordinates and the depth of the point."
        elif task == "grounding":
            # cs.print("Grounding task detected. Adding grounding prompt.")
            text = f"Please provide the bounding box coordinate of the region this sentence describes: {text}."

        # cs.print(f"\n{'='*20} INPUT {'='*20}\n{text}\n{'='*47}\n")

        # PIL Images are passed directly to process_vision_info (no file paths needed)
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in image],
                    {"type": "text", "text": f"{text}"},
                ],
            },
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._input_device())

        # Inference
        # cs.print("Running inference ...")
        generated_ids = self.model.generate(**inputs, max_new_tokens=768, do_sample=do_sample, temperature=temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answer_text = output_text[0] if output_text else ""

        # Extract coordinates based on task (always, not just when plotting)
        extraced_points, extraced_boxes, extraced_trajectories = None, None, None
        if task in ["pointing", "trajectory", "grounding"]:
            if task == "trajectory":
                trajectory_pattern = r'(\d+),\s*(\d+),\s*([+-]?\d+\.\d+)'
                trajectory_matches = re.findall(trajectory_pattern, answer_text)
                extraced_trajectories = [[(int(x), int(y), float(d)) for x, y, d in trajectory_matches]]
                # cs.print(f"Extracted trajectory points: {extraced_trajectories}")
            elif task == "pointing":
                point_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                point_matches = re.findall(point_pattern, answer_text)
                extraced_points = [(int(x), int(y)) for x, y in point_matches]
                # cs.print(f"Extracted points: {extraced_points}")
            elif task == "grounding":
                box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
                box_matches = re.findall(box_pattern, answer_text)
                extraced_boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in box_matches]
                # cs.print(f"Extracted bounding boxes: {extraced_boxes}")

        # Plotting functionality
        if plot and task in ["pointing", "trajectory", "grounding"]:
            cs.print("Plotting enabled. Drawing results on the image ...")

            if plot_output_dir is None:
                plot_output_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "__tmp__", "node_locator_plots",
                )
            os.makedirs(plot_output_dir, exist_ok=True)

            # Build output filename from image_name or fallback to task name
            if image_name:
                base, ext = os.path.splitext(image_name)
                plot_filename = f"{base}_{task}_annotated{ext}"
            else:
                plot_filename = f"{task}_annotated.png"

            self.draw_on_image(
                image[0],
                points=extraced_points,
                boxes=extraced_boxes,
                trajectories=extraced_trajectories,
                output_path=os.path.join(plot_output_dir, plot_filename),
            )

        # Build return dict: always include answer, plus task-specific coordinates
        result = {"answer": answer_text}
        if task == "pointing":
            result["points"] = extraced_points
        elif task == "grounding":
            result["boxes"] = extraced_boxes
        elif task == "trajectory":
            result["trajectories"] = extraced_trajectories

        return result

    def draw_on_image(self, image_input, points=None, boxes=None, trajectories=None, output_path=None):
        """
        Draw points, bounding boxes, and trajectories on an image.

        Parameters:
            image_input: PIL Image, numpy array, or file path (str)
            points: List of points in format [(x, y), ...] where x,y are relative (0~1000)
            boxes: List of boxes in format [[x1, y1, x2, y2], ...] where coords are relative (0~1000)
            trajectories: List of trajectories in format [[(x, y), (x, y), ...], ...]
                        or [[(x, y, d), ...], ...] where x,y are relative (0~1000)
            output_path: Path to save the output image. Required for PIL/numpy inputs.
        """
        try:
            # Read the image into a cv2 numpy array
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
                if image is None:
                    raise FileNotFoundError(f"Unable to read image: {image_input}")
            elif isinstance(image_input, Image.Image):
                image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                image = image_input.copy()
            else:
                raise TypeError(f"Unsupported image_input type: {type(image_input)}")

            h, w = image.shape[:2]

            def rel_to_abs(x_rel, y_rel):
                """Convert relative (0~1000) to absolute pixel coords, clamped to image bounds."""
                x = int(round((x_rel / 1000.0) * w))
                y = int(round((y_rel / 1000.0) * h))
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                return x, y

            # Draw points
            if points:
                for point in points:
                    x_rel, y_rel = point
                    x, y = rel_to_abs(x_rel, y_rel)
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red solid circle

            # Draw bounding boxes
            if boxes:
                for box in boxes:
                    x1r, y1r, x2r, y2r = box
                    x1, y1 = rel_to_abs(x1r, y1r)
                    x2, y2 = rel_to_abs(x2r, y2r)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # Draw trajectories
            if trajectories:
                for trajectory in trajectories:
                    if not trajectory or len(trajectory) < 2:
                        continue

                    # Convert all trajectory points to absolute pixels
                    abs_pts = []
                    for p in trajectory:
                        # support (x,y) or (x,y,d)
                        x_rel, y_rel = p[0], p[1]
                        abs_pts.append(rel_to_abs(x_rel, y_rel))

                    # Connect trajectory points with lines
                    for i in range(1, len(abs_pts)):
                        cv2.line(image, abs_pts[i - 1], abs_pts[i], (0, 0, 255), 2)  # Blue line

                    # Draw a larger point at the trajectory end
                    start_x, start_y = abs_pts[0]
                    cv2.circle(image, (start_x, start_y), 7, (0, 255, 0), -1)  # Red start point

                    # Draw a larger point at the trajectory end
                    end_x, end_y = abs_pts[-1]
                    cv2.circle(image, (end_x, end_y), 7, (255, 0, 0), -1)  # Blue end point

            # Determine output path
            if not output_path:
                if isinstance(image_input, str):
                    name, ext = os.path.splitext(image_input)
                    output_path = f"{name}_annotated{ext}"
                else:
                    raise ValueError("output_path is required when image_input is not a file path")

            # Save the result
            cv2.imwrite(output_path, image)
            cs.print(f"Annotated image saved to: {output_path}")
            return output_path

        except Exception as e:
            cs.print(f"Error processing image: {e}")
            return None


class NodeLocatorRynn:
    """
    A lightweight RynnBrain affordance point locator.
    """
    def __init__(self, model_id="/data0/luokang/dataset/luokang/ckpts/RynnBrain-8B", device_map="auto"):
        """
        Initialize the RynnBrain model and processor.

        Args:
            model_id (str): Path or Hugging Face model identifier.
            device_map (str): Device mapping strategy ("auto", "cuda:0", etc.).
        """
        cs.print("Loading RynnBrain Checkpoint ...")
        self.model_id = model_id
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map=device_map,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def resize(self, images, scale=1.0):
        """
        Resize images in-memory. Accepts file paths, PIL Images, or numpy arrays.
        Always returns a list of PIL Images.
        """
        result = []
        for img in images:
            if isinstance(img, str):
                pil_img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            if scale != 1.0:
                new_w, new_h = int(pil_img.width * scale), int(pil_img.height * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

            result.append(pil_img)
        return result

    def inference(self, text, image, task="affordance",
                  plot=False, plot_output_dir=None, image_name=None,
                  do_sample=False, temperature=0.7, resize_scale=1.0):
        """
        Perform affordance point localization with RynnBrain.

        Args:
            text (str): The node/object description to localize an affordance point for.
            image: Image as a file path, PIL Image, or numpy array.
            task (str): Must be "affordance".
            plot (bool): Whether to draw the predicted affordance point on the image.
            plot_output_dir (str): Directory to save annotated images.
            image_name (str): Base name for the output plot file.
            do_sample (bool): Whether to use sampling during generation.
            temperature (float): Temperature for sampling.
            resize_scale (float): Scale factor to resize images. 1.0 = no resize.
        """
        assert task == "affordance", "Only supports task='affordance'."

        if not isinstance(image, list):
            image = [image]
        assert len(image) == 1, "Affordance localization requires exactly one image."

        image = self.resize(image, scale=resize_scale)
        prompt = (
            f"Identify one affordance point for the area that {text}.\n"
            "Provide one normalized pixel coordinate in the format "
            "<affordance> (x, y) </affordance> with both coordinate values "
            "normalized to the standardized pixel coordinate system spanning 0 to 1000."
        )

        cs.print(f"\n{'='*20} INPUT {'='*20}\n{prompt}\n{'='*47}\n")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image[0]},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        cs.print("Running RynnBrain affordance inference ...")
        generation_kwargs = {
            "max_new_tokens": 128,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer_text = output_text[0] if output_text else ""

        content_match = re.search(r"<affordance>(.*?)</affordance>", answer_text, flags=re.DOTALL)
        content = content_match.group(1) if content_match else answer_text
        point_matches = re.findall(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)", content)
        points = [(int(float(x)), int(float(y))) for x, y in point_matches[:1]]
        cs.print(f"Extracted affordance points: {points}")

        if plot:
            if plot_output_dir is None:
                plot_output_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "__tmp__", "node_locator_plots",
                )
            os.makedirs(plot_output_dir, exist_ok=True)

            if image_name:
                base, ext = os.path.splitext(image_name)
                plot_filename = f"{base}_{task}_annotated{ext or '.png'}"
            else:
                plot_filename = f"{task}_annotated.png"

            self.draw_on_image(
                image[0],
                points=points,
                output_path=os.path.join(plot_output_dir, plot_filename),
            )

        return {"answer": answer_text, "points": points}

    def draw_on_image(self, image_input, points=None, output_path=None):
        """
        Draw affordance points on an image.

        Parameters:
            image_input: PIL Image, numpy array, or file path (str).
            points: List of points in format [(x, y), ...] where x,y are relative (0~1000).
            output_path: Path to save the output image. Required for PIL/numpy inputs.
        """
        try:
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
                if image is None:
                    raise FileNotFoundError(f"Unable to read image: {image_input}")
            elif isinstance(image_input, Image.Image):
                image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                image = image_input.copy()
            else:
                raise TypeError(f"Unsupported image_input type: {type(image_input)}")

            h, w = image.shape[:2]

            def rel_to_abs(x_rel, y_rel):
                x = int(round((x_rel / 1000.0) * w))
                y = int(round((y_rel / 1000.0) * h))
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                return x, y

            if points:
                for point in points:
                    x_rel, y_rel = point
                    x, y = rel_to_abs(x_rel, y_rel)
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            if not output_path:
                if isinstance(image_input, str):
                    name, ext = os.path.splitext(image_input)
                    output_path = f"{name}_annotated{ext}"
                else:
                    raise ValueError("output_path is required when image_input is not a file path")

            cv2.imwrite(output_path, image)
            cs.print(f"Annotated image saved to: {output_path}")
            return output_path

        except Exception as e:
            cs.print(f"Error processing image: {e}")
            return None


class NodeLocatorLA:
    """LocateAnything grounding exposed through the NodeLocatorRobo interface."""

    def __init__(
        self,
        model_id="/data0/luokang/dataset/luokang/ckpts/LocateAnything-3B",
        device_map="auto",
    ):
        from locateanything_worker import LocateAnythingWorker

        device = "cuda" if device_map == "auto" else str(device_map)
        cs.print("Loading LocateAnything Checkpoint ...")
        self.model_id = model_id
        self.device_map = device_map
        self.worker = LocateAnythingWorker(model_id, device=device)

    def inference(
        self,
        text,
        image,
        task="pointing",
        plot=False,
        plot_output_dir=None,
        image_name=None,
        do_sample=False,
        temperature=0.7,
        resize_scale=1.0,
    ):
        """Ground ``text`` and return bbox centers as normalized 0--1000 points."""
        del do_sample, temperature
        if task not in ("pointing", "grounding"):
            raise ValueError("NodeLocatorLA only supports task=pointing or task=grounding.")
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError("Pointing and grounding require exactly one image.")
            image = image[0]
        image = NodeLocatorRobo.resize(self, [image], scale=resize_scale)[0]

        result = self.worker.ground_multi(image, text)
        answer = result.get("answer", "")
        pixel_boxes = self.worker.parse_boxes(answer, image.width, image.height)
        boxes = [
            [
                box["x1"] / image.width * 1000.0,
                box["y1"] / image.height * 1000.0,
                box["x2"] / image.width * 1000.0,
                box["y2"] / image.height * 1000.0,
            ]
            for box in pixel_boxes
        ]
        points = [
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            for x1, y1, x2, y2 in boxes
        ]

        output = {"answer": answer}
        if task == "pointing":
            output["points"] = points
        else:
            output["boxes"] = boxes

        if plot:
            if plot_output_dir is None:
                plot_output_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "__tmp__", "node_locator_plots",
                )
            os.makedirs(plot_output_dir, exist_ok=True)
            base, ext = os.path.splitext(image_name or task)
            plot_filename = f"{base}_{task}_annotated{ext or '.png'}"
            NodeLocatorRobo.draw_on_image(
                self,
                image,
                points=points if task == "pointing" else None,
                boxes=boxes if task == "grounding" else None,
                output_path=os.path.join(plot_output_dir, plot_filename),
            )

        return output


def OnlineTest():
    #: task
    tasks = [
        TaskStructure(
            task='rotate the red block right',
            subtask_list=[
                SubtaskStructure(
                    subtask='rotate the red block right',
                    action_type=ActionType.ROTATE,
                    action_degree='right',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the red block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='rotate the red block left',
            subtask_list=[
                SubtaskStructure(
                    subtask='rotate the red block left',
                    action_type=ActionType.ROTATE,
                    action_degree='left',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the red block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='rotate the blue block right',
            subtask_list=[
                SubtaskStructure(
                    subtask='rotate the blue block right',
                    action_type=ActionType.ROTATE,
                    action_degree='right',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the blue block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='rotate the blue block left',
            subtask_list=[
                SubtaskStructure(
                    subtask='rotate the blue block left',
                    action_type=ActionType.ROTATE,
                    action_degree='left',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the blue block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='rotate the pink block right',
            subtask_list=[
                SubtaskStructure(
                    subtask='rotate the pink block right',
                    action_type=ActionType.ROTATE,
                    action_degree='right',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the pink block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='rotate the pink block left',
            subtask_list=[
                SubtaskStructure(
                    subtask='rotate the pink block left',
                    action_type=ActionType.ROTATE,
                    action_degree='left',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='pink block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),

        TaskStructure(
            task='push the red block right',
            subtask_list=[
                SubtaskStructure(
                    subtask='push the red block right',
                    action_type=ActionType.PUSH,
                    action_degree='right',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the red block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='push the red block left',
            subtask_list=[
                SubtaskStructure(
                    subtask='push the red block left',
                    action_type=ActionType.PUSH,
                    action_degree='left',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the red block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='push the blue block right',
            subtask_list=[
                SubtaskStructure(
                    subtask='push the blue block right',
                    action_type=ActionType.PUSH,
                    action_degree='right',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the blue block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='push the blue block left',
            subtask_list=[
                SubtaskStructure(
                    subtask='push the blue block left',
                    action_type=ActionType.PUSH,
                    action_degree='left',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the blue block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='push the pink block right',
            subtask_list=[
                SubtaskStructure(
                    subtask='push the pink block right',
                    action_type=ActionType.PUSH,
                    action_degree='right',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the pink block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='push the pink block left',
            subtask_list=[
                SubtaskStructure(
                    subtask='push the pink block left',
                    action_type=ActionType.PUSH,
                    action_degree='left',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the pink block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),

        TaskStructure(
            task='slide the door left',
            subtask_list=[
                SubtaskStructure(
                    subtask='slide the door left',
                    action_type=ActionType.SLIDE,
                    action_degree='left',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the door handle', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='slide the door right',
            subtask_list=[
                SubtaskStructure(
                    subtask='slide the door right',
                    action_type=ActionType.SLIDE,
                    action_degree='right',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the door handle', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='open the drawer',
            subtask_list=[
                SubtaskStructure(
                    subtask='open the drawer',
                    action_type=ActionType.SLIDE,
                    action_degree='outward',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the drawer handle', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='close the drawer',
            subtask_list=[
                SubtaskStructure(
                    subtask='close the drawer',
                    action_type=ActionType.SLIDE,
                    action_degree='inward',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the drawer handle', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),

        TaskStructure(
            task='lift the red block from the table',
            subtask_list=[
                SubtaskStructure(
                    subtask='lift the red block from the table',
                    action_type=ActionType.LIFT,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the red block on the table', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='lift the blue block from the table',
            subtask_list=[
                SubtaskStructure(
                    subtask='lift the blue block from the table',
                    action_type=ActionType.LIFT,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the blue block', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='lift the pink block from the table',
            subtask_list=[
                SubtaskStructure(
                    subtask='lift the pink block from the table',
                    action_type=ActionType.LIFT,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the pink block from the table', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='lift the red block from the sliding cabinet',
            subtask_list=[
                SubtaskStructure(
                    subtask='lift the red block from the sliding cabinet',
                    action_type=ActionType.LIFT,
                    action_degree='upward',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the red block from the sliding cabinet', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='lift the blue block from the sliding cabinet',
            subtask_list=[
                SubtaskStructure(
                    subtask='lift the blue block from the sliding cabinet',
                    action_type=ActionType.LIFT,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the blue block from the sliding cabinet', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='lift the pink block from the sliding cabinet',
            subtask_list=[
                SubtaskStructure(
                    subtask='lift the pink block from the sliding cabinet',
                    action_type=ActionType.LIFT,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(
                            id=1,
                            name='the pink block from the sliding cabinet',
                            role=NodeRole.PATIENT,
                        )
                    ]
                )
            ]
        ),
        TaskStructure(
            task='lift the red block from the drawer',
            subtask_list=[
                SubtaskStructure(
                    subtask='lift the red block from the drawer',
                    action_type=ActionType.LIFT,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(
                            id=1,
                            name='the red block from the drawer',
                            role=NodeRole.PATIENT,
                        )
                    ]
                )
            ]
        ),
        TaskStructure(
            task='lift the blue block from the drawer',
            subtask_list=[
                SubtaskStructure(
                    subtask='lift the blue block from the drawer',
                    action_type=ActionType.LIFT,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(
                            id=1,
                            name='the blue block from the drawer',
                            role=NodeRole.PATIENT,
                        )
                    ]
                )
            ]
        ),
        TaskStructure(
            task='lift the pink block from the drawer',
            subtask_list=[
                SubtaskStructure(
                    subtask='lift the pink block from the drawer',
                    action_type=ActionType.LIFT,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(
                            id=1,
                            name='the pink block from the drawer',
                            role=NodeRole.PATIENT,
                        )
                    ]
                )
            ]
        ),

        TaskStructure(
            task='place the block in the sliding cabinet',
            subtask_list=[
                SubtaskStructure(
                    subtask='place the block in the sliding cabinet',
                    action_type=ActionType.PLACE,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the block', role=NodeRole.PATIENT),
                        Node(id=2, name='in the sliding cabinet', role=NodeRole.TARGET)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='place the block in the drawer',
            subtask_list=[
                SubtaskStructure(
                    subtask='place the block in the drawer',
                    action_type=ActionType.PLACE,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the block', role=NodeRole.PATIENT),
                        Node(id=2, name='in the drawer', role=NodeRole.TARGET)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='sweep the block into the drawer',
            subtask_list=[
                SubtaskStructure(
                    subtask='sweep the block into the drawer',
                    action_type=ActionType.MOVE,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the block', role=NodeRole.PATIENT),
                        Node(id=2, name='into the drawer', role=NodeRole.TARGET)
                    ]
                )
            ]
        ),

        TaskStructure(
            task='put the block on top of another block',
            subtask_list=[
                SubtaskStructure(
                    subtask='put the block on top of another block',
                    action_type=ActionType.PUT,
                    action_degree='on',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the block', role=NodeRole.PATIENT),
                        Node(id=2, name='another block', role=NodeRole.TARGET)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='remove the top block from the stack',
            subtask_list=[
                SubtaskStructure(
                    subtask='remove the top block from the stack',
                    action_type=ActionType.REMOVE,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the top block from the stack', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),

        TaskStructure(
            task='turn on the lightbulb',
            subtask_list=[
                SubtaskStructure(
                    subtask='turn on the lightbulb',
                    action_type=ActionType.SLIDE,
                    action_degree='upward',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the light switch', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='turn off the lightbulb',
            subtask_list=[
                SubtaskStructure(
                    subtask='turn off the lightbulb',
                    action_type=ActionType.SLIDE,
                    action_degree='down',
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the light switch', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='turn on the led',
            subtask_list=[
                SubtaskStructure(
                    subtask='turn on the led',
                    action_type=ActionType.PRESS,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the black button', role=NodeRole.PATIENT)
                    ]
                )
            ]
        ),
        TaskStructure(
            task='turn off the led',
            subtask_list=[
                SubtaskStructure(
                    subtask='turn off the led',
                    action_type=ActionType.PRESS,
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', role=NodeRole.ACTOR),
                        Node(id=1, name='the black button', role=NodeRole.PATIENT)
                    ]
                )
            ]
        )
    ]

    #: data
    tasks_demo_dir = (
        "/data0/luokang/dataset/luokang/lerobot/calvin-abc-lerobot-depth_v21/"
        "meta/tasks_demo/"
    )
    all_video_files = sorted(
        f for f in os.listdir(tasks_demo_dir) if f.endswith(".mp4")
    )

    def extract_first_frame(video_path):
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", video_path,
                "-frames:v", "1",
                "-f", "image2pipe",
                "-vcodec", "png",
                "-",
            ],
            capture_output=True, check=True,
        )
        return Image.open(BytesIO(result.stdout)).convert("RGB")

    task_to_frame = {}
    for task, vf in zip(tasks, all_video_files):
        video_path = os.path.join(tasks_demo_dir, vf)
        task_to_frame[task.task] = extract_first_frame(video_path)
        cs.print(f"[green]✓[/green] '{task.task}' -> {vf}")

    #: infer
    node_locator = NodeLocatorRobo(model_id="/data0/luokang/dataset/luokang/ckpts/RoboBrain2.5-8B-NV")
    # node_locator = NodeLocatorRynn(model_id="/data0/luokang/dataset/luokang/ckpts/RynnBrain-4B")
    for task, vf in zip(tasks, all_video_files):
        pil_image = task_to_frame.get(task.task)
        if pil_image is None:
            cs.print(f"[yellow]Skipping task '{task.task}': no demo video available[/yellow]")
            continue
        video_basename = os.path.splitext(vf)[0]
        for i, subtask in enumerate(task.subtask_list):
            for j, node in enumerate(subtask.node_list):
                if node.role == NodeRole.ACTOR:
                    continue
                prompt = node.name
                cs.print(f"\nLocating node {node.id}: {prompt}")
                result = node_locator.inference(
                    text=prompt,
                    image=pil_image,
                    plot=True,
                    image_name=f"{video_basename}_first_node{node.id}_{prompt}.png",
                    plot_output_dir="/data0/luokang/research/GraphVLA/__tmp__/node_locator_plots",
                    resize_scale=2.0,
                )
        cs.print(task)


if __name__ == "__main__":
    OnlineTest()