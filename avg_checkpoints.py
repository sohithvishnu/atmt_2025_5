#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
average_checkpoints.py
--------------------------------
Averages model checkpoints.
- EXCLUDES 'last' and 'best' checkpoints to prevent double-counting.
- Sorts by the EPOCH number found in the filename.
- PRESERVES 'args' so translate.py can load the configuration.

Usage:
python average_checkpoints.py --checkpoint-dir cz-en/checkpoints/ --num-last 3
"""

import os
import re
import torch
import argparse
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser(description="Average the weights of multiple model checkpoints.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Path to the directory containing checkpoints.")
    parser.add_argument("--inputs", type=str, nargs="+", default=None,
                        help="List of checkpoint files to average (overrides --num-last).")
    parser.add_argument("--num-last", type=int, default=3,
                        help="Number of last checkpoints to average.")
    parser.add_argument("--output", type=str, default="checkpoint_avg.pt",
                        help="Output file to save the averaged checkpoint.")
    return parser.parse_args()

def extract_epoch(filename):
    """
    Extracts the epoch number from filenames like 'checkpoint0_7.039.pt'.
    """
    s = re.findall(r"\d+", filename)
    if s:
        return int(s[0])
    return -1

def sorted_checkpoints(checkpoint_dir):
    """
    Returns a list of checkpoint files sorted by epoch number.
    Ignored files: '*avg*', '*last*', '*best*'.
    """
    files = []
    for f in os.listdir(checkpoint_dir):
        if not f.endswith(".pt"):
            continue
            
        # EXCLUDE 'avg', 'last', and 'best' to avoid duplicates
        if "avg" in f or "last" in f or "best" in f:
            continue
            
        files.append(f)
    
    sorted_files = sorted(files, key=extract_epoch)
    full_paths = [os.path.join(checkpoint_dir, f) for f in sorted_files]
    return full_paths

def average_checkpoints(ckpt_files):
    """Load and average multiple checkpoint state_dicts."""
    assert len(ckpt_files) > 0, "No checkpoints to average."

    print(f"📦 Averaging {len(ckpt_files)} checkpoints:")
    for f in ckpt_files:
        print(f"   - {f}")

    avg_state_dict = None
    saved_args = None  # To store the configuration args
    
    for i, ckpt_path in enumerate(ckpt_files):
        state = torch.load(ckpt_path, map_location="cpu")

        # Capture args from the first checkpoint we encounter (assuming they are all the same)
        if saved_args is None and "args" in state:
            saved_args = state["args"]

        if "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state

        if avg_state_dict is None:
            avg_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if torch.is_floating_point(v):
                    avg_state_dict[k] = v.clone().float()
                else:
                    avg_state_dict[k] = v.clone()
        else:
            for k, v in state_dict.items():
                if k in avg_state_dict:
                    if torch.is_floating_point(v):
                        avg_state_dict[k] += v.float()
                    else:
                        avg_state_dict[k] = v

    # Average the floats
    for k in avg_state_dict.keys():
        if torch.is_floating_point(avg_state_dict[k]):
            avg_state_dict[k] /= len(ckpt_files)

    print("✅ Averaging complete.")
    return avg_state_dict, saved_args

def save_averaged_checkpoint(avg_state_dict, saved_args, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save both model weights and the args
    save_obj = {
        "model": avg_state_dict,
        "args": saved_args
    }
    
    torch.save(save_obj, output_path)
    print(f"💾 Saved averaged checkpoint (with args) to: {output_path}")

def main():
    args = get_args()

    if args.inputs:
        ckpt_files = args.inputs
    elif args.checkpoint_dir:
        ckpt_files = sorted_checkpoints(args.checkpoint_dir)
        if len(ckpt_files) > args.num_last:
            ckpt_files = ckpt_files[-args.num_last:]
    else:
        raise ValueError("Please provide either --checkpoint-dir or --inputs.")

    if len(ckpt_files) == 0:
        raise ValueError("No checkpoints found to average. Check your directory or filters.")

    avg_state_dict, saved_args = average_checkpoints(ckpt_files)
    save_averaged_checkpoint(avg_state_dict, saved_args, args.output)

if __name__ == "__main__":
    main()
