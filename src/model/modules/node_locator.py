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
from common.schema import ActionType, Node, NodeRole, SubtaskStructure, TaskStructure
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
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            dtype="auto", 
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
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
            cs.print("Pointing task detected. Adding pointing prompt.")
            text = f"{text}. Please provide its 2D coordinates. Your answer should be formatted as a tuple, i.e. [(x, y)], where the tuple contains the x and y coordinates of a point satisfying the conditions above."
        elif task == "trajectory":
            cs.print("Trajectory task detected. Adding trajectory prompt.")
            text = f"Please predict 3D end-effector-centric waypoints to complete the task successfully. The task is \"{text}\". Your answer should be formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains the x and y coordinates and the depth of the point."
        elif task == "grounding":
            cs.print("Grounding task detected. Adding grounding prompt.")
            text = f"Please provide the bounding box coordinate of the region this sentence describes: {text}."

        cs.print(f"\n{'='*20} INPUT {'='*20}\n{text}\n{'='*47}\n")

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
        inputs = inputs.to("cuda")

        # Inference
        cs.print("Running inference ...")
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
                cs.print(f"Extracted trajectory points: {extraced_trajectories}")
            elif task == "pointing":
                point_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                point_matches = re.findall(point_pattern, answer_text)
                extraced_points = [(int(x), int(y)) for x, y in point_matches]
                cs.print(f"Extracted points: {extraced_points}")
            elif task == "grounding":
                box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
                box_matches = re.findall(box_pattern, answer_text)
                extraced_boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in box_matches]
                cs.print(f"Extracted bounding boxes: {extraced_boxes}")

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


class NodeLocatorCalvin:
    def __init__(
        self,
        dataset_root="/data0/luokang/dataset/luokang/lerobot/calvin-abc-lerobot-depth_v21",
    ):
        self.tasks_mapping = {
            # rotate 2
            "rotate the red block right": ["grasp the red block, then rotate it right",
                                    "grasp the red block, then turn it right",
                                    "grasp the red block and rotate it right",
                                    "grasp the red block and turn it right",
                                    "take the red block and rotate it right",
                                    "take the red block and turn it right",
                                    "rotate right the red block",
                                    "rotate the red block 90 degrees to the right",
                                    "rotate the red block to the right",
                                    "rotate the red block towards the right",
                                    "turn the red block right"],
            "rotate the red block left": ["grasp the red block, then rotate it left",
                                    "grasp the red block, then turn it left",
                                    "grasp the red block and rotate it left",
                                    "grasp the red block and turn it left",
                                    "take the red block and rotate it left",
                                    "take the red block and turn it left",
                                    "rotate the red block 90 degrees to the left",
                                    "rotate left the red block",
                                    "rotate the red block to the left",
                                    "rotate the red block towards the left",
                                    "turn the red block left"],
            "rotate the blue block right": ["grasp the blue block, then rotate it right",
                                    "grasp the blue block, then turn it right",
                                    "grasp the blue block and rotate it right",
                                    "grasp the blue block and turn it right",
                                    "take the blue block and rotate it right",
                                    "take the blue block and turn it right",
                                    "rotate the blue block 90 degrees to the right",
                                    "rotate right the blue block",
                                    "rotate the blue block to the right",
                                    "rotate the blue block towards the right",
                                    "turn the blue block right"],
            "rotate the blue block left": ["grasp the blue block, then rotate it left",
                                    "grasp the blue block, then turn it left",
                                    "grasp the blue block and rotate it left",
                                    "grasp the blue block and turn it left",
                                    "take the blue block and rotate it left",
                                    "take the blue block and turn it left",
                                    "rotate the blue block 90 degrees to the left",
                                    "rotate left the blue block",
                                    "rotate the blue block to the left",
                                    "rotate the blue block towards the left",
                                    "turn the blue block left"],
            "rotate the pink block right": ["grasp the pink block, then rotate it right",
                                    "grasp the pink block, then turn it right",
                                    "grasp the pink block and rotate it right",
                                    "grasp the pink block and turn it right",
                                    "take the pink block and rotate it right",
                                    "take the pink block and turn it right",
                                    "rotate the pink block 90 degrees to the right",
                                    "rotate right the pink block",
                                    "rotate the pink block to the right",
                                    "rotate the pink block towards the right",
                                    "turn the pink block right"],
            "rotate the pink block left": ["grasp the pink block, then rotate it left",
                                    "grasp the pink block, then turn it left",
                                    "grasp the pink block and rotate it left",
                                    "grasp the pink block and turn it left",
                                    "take the pink block and rotate it left",
                                    "take the pink block and turn it left",
                                    "rotate the pink block 90 degrees to the left",
                                    "rotate left the pink block",
                                    "rotate the pink block to the left",
                                    "rotate the pink block towards the left",
                                    "turn the pink block left"],

            # push 2
            "push the red block right": ["push the red block towards the right",
                                "push right the red block",
                                "push the red block to the right",
                                "go push the red block to the right",
                                "slide the red block towards the right",
                                "slide right the red block",
                                "slide the red block to the right",
                                "sweep the red block to the right",
                                "go slide the red block to the right"],
            "push the red block left": ["push the red block towards the left",
                                "push left the red block",
                                "push the red block to the left",
                                "go push the red block to the left",
                                "slide the red block towards the left",
                                "slide left the red block",
                                "slide the red block to the left",
                                "sweep the red block to the left",
                                "go slide the red block to the left"],
            "push the blue block right": ["push the blue block towards the right",
                                "push right the blue block",
                                "push the blue block to the right",
                                "go push the blue block to the right",
                                "slide the blue block towards the right",
                                "slide right the blue block",
                                "slide the blue block to the right",
                                "sweep the blue block to the right",
                                "go slide the blue block to the right"],
            "push the blue block left": ["push the blue block towards the left",
                                "push left the blue block",
                                "push the blue block to the left",
                                "go push the blue block to the left",
                                "slide the blue block towards the left",
                                "slide left the blue block",
                                "slide the blue block to the left",
                                "sweep the blue block to the left",
                                "go slide the blue block to the left"],
            "push the pink block right": ["push the pink block towards the right",
                                "push right the pink block",
                                "push the pink block to the right",
                                "go push the pink block to the right",
                                "slide the pink block towards the right",
                                "slide right the pink block",
                                "slide the pink block to the right",
                                "sweep the pink block to the right",
                                "go slide the pink block to the right"],
            "push the pink block left": ["push the pink block towards the left",
                                "push left the pink block",
                                "push the pink block to the left",
                                "go push the pink block to the left",
                                "slide the pink block towards the left",
                                "slide left the pink block",
                                "slide the pink block to the left",
                                "sweep the pink block to the left",
                                "go slide the pink block to the left"],

            # slide 2
            "slide the door left": [ "grasp the door handle, then slide the door to the left",
                                "grasp the door handle, then move the door to the left",
                                "grasp the door handle and slide the door to the left",
                                "grasp the door handle and move the door to the left",
                                "move the door all the way to the left",
                                "slide the door all the way to the left",
                                "move the door to the left",
                                "slide the door to the left",
                                "push the door to the left",
                                "move the door to the left side",
                                "slide the door to the left side",
                                "push the door to the left side",
                                "slide the door to the left, then let it go",
                                "move the door all the way to the left and let go",
                                "move the sliding door to the left",
                                "push the sliding door to the left"],
            "slide the door right": [ "grasp the door handle, then slide the door to the right",
                                "grasp the door handle, then move the door to the right",
                                "grasp the door handle and slide the door to the right",
                                "grasp the door handle and move the door to the right",
                                "move the door all the way to the right",
                                "slide the door all the way to the right",
                                "move the door to the right",
                                "slide the door to the right",
                                "push the door to the right",
                                "move the door to the right side",
                                "slide the door to the right side",
                                "push the door to the right side",
                                "slide the door to the right, then let it go",
                                "move the door all the way to the right and let go",
                                "move the sliding door to the right",
                                "push the sliding door to the right"],
            "open the drawer": [ "grasp the drawer handle, then open it",
                        "grasp the drawer handle and open it",
                        "grasp the handle of the drawer, then open it",
                        "grasp the handle of the drawer and open it",
                        "open the drawer",
                        "go open the drawer",
                        "pull the handle of the drawer",
                        "pull the drawer",
                        "open the cabinet drawer"],
            "close the drawer": [ "grasp the drawer handle, then close it",
                        "grasp the drawer handle and close it",
                        "grasp the handle of the drawer, then close it",
                        "grasp the handle of the drawer and close it",
                        "close the drawer",
                        "go close the drawer",
                        "push the handle of the drawer",
                        "push the drawer",
                        "close the cabinet drawer"],

            # lift 2
            "lift the red block from the table": ["lift the red block from the table",
                                "pick up the red block on the table",
                                "pick up the red block from the table",
                                "lift the red block",
                                "pick up the red block",
                                "lift the red block up",
                                "grasp the red block on the table and lift it up",
                                "grasp the red block and lift it up",
                                "grasp the red block on the table, then lift it up",
                                "grasp the red block, then lift it up"],
            "lift the blue block from the table": ["lift the blue block from the table",
                                "pick up the blue block on the table",
                                "pick up the blue block from the table",
                                "lift the blue block",
                                "pick up the blue block",
                                "lift the blue block up",
                                "grasp the blue block on the table and lift it up",
                                "grasp the blue block and lift it up",
                                "grasp the blue block on the table, then lift it up",
                                "grasp the blue block, then lift it up"],
            "lift the pink block from the table": ["lift the pink block from the table",
                                "pick up the pink block on the table",
                                "pick up the pink block from the table",
                                "lift the pink block",
                                "pick up the pink block",
                                "lift the pink block up",
                                "grasp the pink block on the table and lift it up",
                                "grasp the pink block and lift it up",
                                "grasp the pink block on the table, then lift it up",
                                "grasp the pink block, then lift it up"],

            "lift the red block from the sliding cabinet": [ "pick up the red block from the shelf",
                                    "pick up the red block from the sliding cabinet",
                                    "pick up the red block in the sliding cabinet",
                                    "grasp the red block lying on the shelf",
                                    "grasp the red block lying in the cabinet",
                                    "grasp the red block lying in the sliding cabinet",
                                    "grasp the red block lying in the slider",
                                    "lift the red block lying on the shelf",
                                    "lift the red block lying in the cabinet",
                                    "lift the red block lying in the sliding cabinet",
                                    "lift the red block lying in the slider",
                                    "in the slider pick up the red block",
                                    "in the cabinet pick up the red block",
                                    "in the slider grasp the red block",
                                    "in the cabinet grasp the red block",
                                    "in the sliding cabinet grasp the red block",
                                    "lift the red block on the shelf"],
            "lift the blue block from the sliding cabinet": [ "pick up the blue block from the shelf",
                                    "pick up the blue block from the sliding cabinet",
                                    "pick up the blue block in the sliding cabinet",
                                    "grasp the blue block lying on the shelf",
                                    "grasp the blue block lying in the cabinet",
                                    "grasp the blue block lying in the sliding cabinet",
                                    "grasp the blue block lying in the slider",
                                    "lift the blue block lying on the shelf",
                                    "lift the blue block lying in the cabinet",
                                    "lift the blue block lying in the sliding cabinet",
                                    "lift the blue block lying in the slider",
                                    "in the slider pick up the blue block",
                                    "in the cabinet pick up the blue block",
                                    "in the slider grasp the blue block",
                                    "in the cabinet grasp the blue block",
                                    "in the sliding cabinet grasp the blue block",
                                    "lift the blue block on the shelf"],
            "lift the pink block from the sliding cabinet": ["pick up the pink block from the shelf",
                                    "pick up the pink block from the sliding cabinet",
                                    "pick up the pink block in the sliding cabinet",
                                    "grasp the pink block lying on the shelf",
                                    "grasp the pink block lying in the cabinet",
                                    "grasp the pink block lying in the sliding cabinet",
                                    "grasp the pink block lying in the slider",
                                    "lift the pink block lying on the shelf",
                                    "lift the pink block lying in the cabinet",
                                    "lift the pink block lying in the sliding cabinet",
                                    "lift the pink block lying in the slider",
                                    "in the slider pick up the pink block",
                                    "in the cabinet pick up the pink block",
                                    "in the slider grasp the pink block",
                                    "in the cabinet grasp the pink block",
                                    "in the sliding cabinet grasp the pink block",
                                    "lift the pink block on the shelf"],

            "lift the red block from the drawer": ["grasp the red block from the drawer",
                                    "grasp the red block lying in the drawer",
                                    "grasp the red block in the drawer",
                                    "pick up the red block lying in the drawer",
                                    "pick up the red block from the drawer",
                                    "pick up the red block in the drawer",
                                    "go towards the red block in the drawer and pick it up",
                                    "go towards the red block in the drawer and grasp it",
                                    "go towards the red block in the drawer and lift it",
                                    "lift the red block in the drawer",
                                    "lift the red block lying in the drawer"],
            "lift the blue block from the drawer": ["grasp the blue block from the drawer",
                                    "grasp the blue block lying in the drawer",
                                    "grasp the blue block in the drawer",
                                    "pick up the blue block lying in the drawer",
                                    "pick up the blue block from the drawer",
                                    "pick up the blue block in the drawer",
                                    "go towards the blue block in the drawer and pick it up",
                                    "go towards the blue block in the drawer and grasp it",
                                    "go towards the blue block in the drawer and lift it",
                                    "lift the blue block in the drawer",
                                    "lift the blue block lying in the drawer"],
            "lift the pink block from the drawer": ["grasp the pink block from the drawer",
                                    "grasp the pink block lying in the drawer",
                                    "grasp the pink block in the drawer",
                                    "pick up the pink block lying in the drawer",
                                    "pick up the pink block from the drawer",
                                    "pick up the pink block in the drawer",
                                    "go towards the pink block in the drawer and pick it up",
                                    "go towards the pink block in the drawer and grasp it",
                                    "go towards the pink block in the drawer and lift it",
                                    "lift the pink block in the drawer",
                                    "lift the pink block lying in the drawer"],

            # place 3
            "place the block in the sliding cabinet": [ "place in slider",
                            "put it in the slider",
                            "place the block in the sliding cabinet",
                            "place the object in the sliding cabinet",
                            "place the grasped object in the sliding cabinet",
                            "put the block in the sliding cabinet",
                            "put the object in the sliding cabinet",
                            "put the grasped object in the sliding cabinet",
                            "place the block in the cabinet",
                            "place the object in the cabinet",
                            "place the grasped object in the cabinet",
                            "put the block in the cabinet",
                            "put the object in the cabinet",
                            "put the grasped object in the cabinet",
                            "place the block in the slider",
                            "place the object in the slider",
                            "place the grasped object in the slider",
                            "put the block in the slider",
                            "put the object in the slider",
                            "put the grasped object in the slider"],
            "place the block in the drawer": [ "place the block in the drawer",
                            "place the object in the drawer",
                            "place the grasped object in the drawer",
                            "put the block in the drawer",
                            "put the object in the drawer",
                            "put the grasped object in the drawer",
                            "store the block in the drawer",
                            "store the object in the drawer",
                            "store the grasped object in the drawer",
                            "move to the drawer and place the object",
                            "go towards the drawer and place the object",
                            "move to the drawer, then place the object",
                            "move to the drawer and store the object",
                            "go towards the drawer and store the object",
                            "move to the drawer, then store the object"],

            # sweep 3
            "sweep the block into the drawer": ["push the object into the drawer",
                            "push the block into the drawer",
                            "slide the object into the drawer",
                            "slide the block into the drawer",
                            "sweep the object into the drawer",
                            "sweep the block into the drawer",
                            "push the object that it falls into the drawer"],

            # put 3
            "put the block on top of another block": ["stack blocks on top of each other",
                        "stack the blocks",
                        "stack the object on top of another object",
                        "place the block on top of another block",
                        "place the grasped block on top of another block",
                        "put the grasped block on top of a block",
                        "put the block on top of another block",
                        "stack the block on top of another block"],

            # remove 2
            "remove the top block from the stack": ["collapse the stacked blocks",
                            "take off the stacked block",
                            "unstack the blocks",
                            "go to the tower of blocks and take off the top one",
                            "remove a block from the stack",
                            "take off the block that is on top of the other one",
                            "remove the top block"],

            # slide 2
            "turn on the lightbulb": ["turn on the light bulb",
                                "turn on the yellow light",
                                "turn on the yellow lamp",
                                "move up the switch",
                                "push the switch upwards",
                                "slide up the switch",
                                "move the light switch to turn on the light bulb",
                                "toggle the light switch to turn on the light bulb",
                                "move the light switch to turn on the yellow light",
                                "toggle the light switch to turn on the yellow light"],
            "turn off the lightbulb": ["turn off the light bulb",
                                "turn off the yellow light",
                                "turn off the yellow lamp",
                                "move down the switch",
                                "push the switch downwards",
                                "slide down the switch",
                                "move the light switch to turn off the light bulb",
                                "toggle the light switch to turn off the light bulb",
                                "move the light switch to turn off the yellow light",
                                "toggle the light switch to turn off the yellow light"],

            # press 2
            "turn on the led": ["turn on the led light",
                        "turn on the led",
                        "turn on the led lamp",
                        "turn on the green light",
                        "turn on the green lamp",
                        "push down the button to turn on the led light",
                        "push down the button to turn on the led",
                        "push down the button to turn on the green light",
                        "push the button to turn on the led light",
                        "push the button to turn on the led",
                        "push the button to turn on the green light",
                        "toggle the button to turn on the led light",
                        "toggle the button to turn on the led",
                        "toggle the button to turn on the green light"],
            "turn off the led": ["turn off the led light",
                        "turn off the led",
                        "turn off the led lamp",
                        "turn off the green light",
                        "turn off the green lamp",
                        "push down the button to turn off the led light",
                        "push down the button to turn off the led",
                        "push down the button to turn off the green light",
                        "push the button to turn off the led light",
                        "push the button to turn off the led",
                        "push the button to turn off the green light",
                        "toggle the button to turn off the led light",
                        "toggle the button to turn off the led",
                        "toggle the button to turn off the green light"],
        }
        self.tasks = [
            TaskStructure(
                task='rotate the red block right',
                subtask_list=[
                    SubtaskStructure(
                        subtask='rotate the red block right',
                        action_type=ActionType.ROTATE,
                        action_degree='right',
                        node_list=[
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the red block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the red block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the blue block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the blue block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the pink block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='pink block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the red block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the red block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the blue block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the blue block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the pink block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the pink block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the door handle', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the door handle', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the drawer handle', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the drawer handle', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the red block on the table', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the blue block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the pink block from the table', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the red block from the sliding cabinet', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the blue block from the sliding cabinet', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(
                                id=1,
                                name='the pink block from the sliding cabinet',
                                need_object=True,
                                role=NodeRole.PATIENT,
                                canon_pcd=None,
                                pos=None,
                                rot6d=None,
                                gripper=None
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(
                                id=1,
                                name='the red block from the drawer',
                                need_object=True,
                                role=NodeRole.PATIENT,
                                canon_pcd=None,
                                pos=None,
                                rot6d=None,
                                gripper=None
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(
                                id=1,
                                name='the blue block from the drawer',
                                need_object=True,
                                role=NodeRole.PATIENT,
                                canon_pcd=None,
                                pos=None,
                                rot6d=None,
                                gripper=None
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(
                                id=1,
                                name='the pink block from the drawer',
                                need_object=True,
                                role=NodeRole.PATIENT,
                                canon_pcd=None,
                                pos=None,
                                rot6d=None,
                                gripper=None
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=2, name='in the sliding cabinet', need_object=True, role=NodeRole.TARGET, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=2, name='in the drawer', need_object=True, role=NodeRole.TARGET, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=2, name='into the drawer', need_object=True, role=NodeRole.TARGET, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        action_degree=None,
                        node_list=[
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=2, name='on top of another block', need_object=False, role=NodeRole.TARGET, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the top block from the stack', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the light switch', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the light switch', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the black button', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                            Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                            Node(id=1, name='the black button', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
                        ]
                    )
                ]
            )
        ]


class NodeLocatorLibero:
    def __init__(
        self,
        dataset_root="/data0/luokang/dataset/luokang/lerobot/libero/libero_all_no_noops_1.0.0_lerobot_10hz"
    ):
        pass


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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the red block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the red block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the blue block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the blue block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the pink block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='pink block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the red block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the red block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the blue block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the blue block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the pink block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the pink block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the door handle', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the door handle', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the drawer handle', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the drawer handle', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the red block on the table', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the blue block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the pink block from the table', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the red block from the sliding cabinet', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the blue block from the sliding cabinet', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(
                            id=1,
                            name='the pink block from the sliding cabinet',
                            need_object=True,
                            role=NodeRole.PATIENT,
                            canon_pcd=None,
                            pos=None,
                            rot6d=None,
                            gripper=None
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(
                            id=1,
                            name='the red block from the drawer',
                            need_object=True,
                            role=NodeRole.PATIENT,
                            canon_pcd=None,
                            pos=None,
                            rot6d=None,
                            gripper=None
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(
                            id=1,
                            name='the blue block from the drawer',
                            need_object=True,
                            role=NodeRole.PATIENT,
                            canon_pcd=None,
                            pos=None,
                            rot6d=None,
                            gripper=None
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(
                            id=1,
                            name='the pink block from the drawer',
                            need_object=True,
                            role=NodeRole.PATIENT,
                            canon_pcd=None,
                            pos=None,
                            rot6d=None,
                            gripper=None
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=2, name='in the sliding cabinet', need_object=True, role=NodeRole.TARGET, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=2, name='in the drawer', need_object=True, role=NodeRole.TARGET, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=2, name='into the drawer', need_object=True, role=NodeRole.TARGET, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                    action_degree=None,
                    node_list=[
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the block', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=2, name='on top of another block', need_object=False, role=NodeRole.TARGET, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the top block from the stack', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the light switch', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the light switch', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the black button', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                        Node(id=0, name='robotic gripper', need_object=False, role=NodeRole.ACTOR, canon_pcd=None, pos=None, rot6d=None, gripper=None),
                        Node(id=1, name='the black button', need_object=True, role=NodeRole.PATIENT, canon_pcd=None, pos=None, rot6d=None, gripper=None)
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
                task.subtask_list[i].node_list[j].point = np.array(result['points'])
        cs.print(task)


def OfflineTest():
    pass


if __name__ == "__main__":
    OnlineTest()