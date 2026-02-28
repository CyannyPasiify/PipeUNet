#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from pathlib import Path
from monai.transforms import LoadImage, SaveImage, SpatialCropd, Compose

def parse_args():
    parser = argparse.ArgumentParser(
        description='Crop image and mask around a specified center location with given size',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -v input.nii.gz -m mask.nii.gz -l 100 120 50 -s 64 64 32 -ov cropped_image.nii.gz -om cropped_mask.nii.gz
        """)
    
    parser.add_argument('-v', '--volume', type=str, required=True, help='Path to the input image file')
    parser.add_argument('-m', '--mask', type=str, required=True, help='Path to the input mask file')
    parser.add_argument('-l', '--location', type=int, nargs=3, required=True, help='Crop center coordinates (cx, cy, cz) in index space')
    parser.add_argument('-s', '--size', type=int, nargs=3, required=True, help='Crop dimensions (sx, sy, sz)')
    parser.add_argument('-ov', '--output_volume', type=str, required=True, help='Output path for the cropped image (float32)')
    parser.add_argument('-om', '--output_mask', type=str, required=True, help='Output path for the cropped mask (uint8)')
    
    return parser.parse_args()

def crop_image_mask(volume_path, mask_path, location, size, output_volume_path, output_mask_path):
    # Load the image and mask with MONAI
    load_image = LoadImage(image_only=False, ensure_channel_first=True)
    volume_data, volume_meta = load_image(str(volume_path))
    mask_data, _ = load_image(str(mask_path))
    
    # Create a dictionary with both image and mask
    data = {
        "image": volume_data,
        "mask": mask_data
    }
    
    # Define the crop transform using SpatialCropd
    crop_transform = Compose([
        SpatialCropd(
            keys=["image", "mask"],
            roi_center=location,  # roi_center should be (x, y, z) in index space
            roi_size=size         # roi_size should be (sx, sy, sz)
        )
    ])
    
    # Apply the crop transform
    cropped_data = crop_transform(data)
    
    # Extract cropped image and mask
    cropped_volume = cropped_data["image"]
    cropped_mask = cropped_data["mask"]
    
    # Create output directories if they don't exist
    output_volume_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save cropped image (float32)
    save_image = SaveImage(
        output_dir=str(output_volume_path.parent),
        output_format="nii.gz",
        separate_folder=False,
        print_log=True,
        output_dtype=np.float32
    )
    save_image(cropped_volume, meta_data=volume_meta, filename=str(output_volume_path).replace(".nii.gz", ""))
    
    # Save cropped mask (uint8)
    save_mask = SaveImage(
        output_dir=str(output_mask_path.parent),
        output_format="nii.gz",
        separate_folder=False,
        print_log=True,
        output_dtype=np.uint8
    )
    save_mask(cropped_mask, meta_data=volume_meta, filename=str(output_mask_path).replace(".nii.gz", ""))

def main():
    args = parse_args()
    
    print(f"Input volume: {args.volume}")
    print(f"Input mask: {args.mask}")
    print(f"Crop center: {args.location}")
    print(f"Crop size: {args.size}")
    print(f"Output volume: {args.output_volume}")
    print(f"Output mask: {args.output_mask}")
    
    # Convert paths to Path objects
    volume_path = Path(args.volume)
    mask_path = Path(args.mask)
    output_volume_path = Path(args.output_volume)
    output_mask_path = Path(args.output_mask)
    
    try:
        crop_image_mask(
            volume_path,
            mask_path,
            tuple(args.location),
            tuple(args.size),
            output_volume_path,
            output_mask_path
        )
        print("\nCropping completed successfully!")
    except Exception as e:
        print(f"\nError during cropping: {e}")

if __name__ == '__main__':
    main()