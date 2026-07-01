#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
import numpy as np

from src.models.video_depth import onlineVideoDepthAnything
from src.utils.loading_utils import load_video_as_numpy, save_video_mp4, save_predictions_tiff, save_side_by_side, colorize_pred

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def process_single_video(video_path, args, model):
    print(f"Processing: {video_path}")

    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    

    # Load video
    original = load_video_as_numpy(video_path)

    # Inference
    pred, (h,w)= model.infer_video_depth(
        original,
        device=args.device,
        preprocess_device=args.preprocess_device,
        input_size=args.input_size,
        print_process_res=args.print_process_res,
        fp32=args.fp32,
        output_raw=args.save_raw,
        return_process_res=True,
    )

    out_pred_path = os.path.join(args.output_dir, f"{vid_name}_{h}x{w}_pred.mp4")
    out_tiff_path = os.path.join(args.output_dir, f"{vid_name}_{h}x{w}_raw.tiff")
    out_side_path = os.path.join(args.output_dir, f"{vid_name}_{h}x{w}_sidebyside.mp4")

    pred_depth = pred[0]   # [T,H,W]
    pred_color = colorize_pred(pred_depth)  # färbig für MP4 & side-by-side

    # Save prediction video
    save_video_mp4(pred_depth, out_pred_path)

    # Save raw depth
    if args.save_raw:
        save_predictions_tiff(pred_depth, out_tiff_path)

    # Save side-by-side
    save_side_by_side(original, pred_color, out_side_path)

    print(f"Done: {video_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, help="Pfad zu EINEM Video")
    parser.add_argument('--input_dir', type=str, help="Ordner mit Videos")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--save_raw', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--preprocess_device', type=str, default='cpu')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--print_process_res', action='store_true')
    parser.add_argument('--lazy_forward', action='store_true')
    parser.add_argument('--config', type=str, default='./configs/oVDA_c16.yaml')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    model = onlineVideoDepthAnything(**config['net'])
    ckpt = torch.load(config['pretrained_path'], map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()

    # Lazy forward mode
    if args.lazy_forward:
        model.lazy_forward(
            image_list=args.input_video,
            output_dir=args.output_dir,
            device=args.device,
            preprocess_device=args.preprocess_device,
            input_size=args.input_size,
            fp32=args.fp32,
            print_process_res=args.print_process_res,
            output_raw=args.save_raw,
            offset=1.,
            save_rgb=True,
        )
        return

    # Standard inference
    # Folder mode
    if args.input_dir:
        video_exts = (".mp4", ".avi", ".mov", ".mkv")
        all_videos = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith(video_exts)
        ]

        for v in all_videos:
            process_single_video(v, args, model)

    # Single-video mode
    elif args.input_video:
        process_single_video(args.input_video, args, model)

    else:
        raise ValueError("Bitte entweder --input_video oder --input_dir angeben.")


if __name__ == '__main__':
    main()