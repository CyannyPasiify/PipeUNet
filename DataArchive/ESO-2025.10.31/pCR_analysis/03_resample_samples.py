#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
接收-r/--root_dir指定根目录，根目录下包含{site}/{site}_{pid}样本目录。-s/--respacing指定重采样间距（单位：mm），默认[1.0, 1.0, 1.0]。-t/--thresh指定图像值阈值范围[min, max]，用于过滤蒙版前景体素，默认[-140.0, 1000.0]。
  对每个样本，读取其{site}_{pid}_pre_volume.nii.gz、{site}_{pid}_pre_mask.nii.gz、{site}_{pid}_post_volume.nii.gz和{site}_{pid}_post_mask.nii.gz文件，将图像和蒙版重采样到指定间距，并应用阈值过滤。重采样结果保存在样本目录下，文件名为{site}_{pid}_(pre|post)_std_resampled_volume.nii.gz和{site}_{pid}_(pre|post)_std_resampled_mask.nii.gz。
  使用MONAI库的Spacingd进行重采样，LoadImage和SaveImage读取和保存图像文件，使用pathlib处理路径。使用tqdm显示进度。
  对全部变量和函数参数添加类型注解。
  除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Resampling and Thresholding Script for ESO-2025.10.31 Dataset

This script resamples images and masks to standardized spacing and applies threshold filtering
 to remove foreground voxels outside the specified intensity range.

Parameters:
    -r, --root_dir: Root directory containing {site}/{site}_{pid} sample directories
    -s, --respacing: List of resampling spacing values in mm for x,y,z directions (default: [1.0, 1.0, 1.0])
    -t, --thresh: Image value threshold range [min, max] for filtering mask foreground voxels (default: [-140.0, 1000.0])

Usage Examples:
    python 03_resample_samples.py -r /path/to/root
    python 03_resample_samples.py --root_dir /path/to/root --respacing 1.0 1.0 1.0 --thresh -140.0 1000.0
"""

import argparse
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Pattern, Tuple
from tqdm import tqdm
from monai.transforms import (
    LoadImage, 
    SaveImage, 
    Spacingd, 
    Compose
)
from monai.data import MetaTensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Resample images and masks to standardized spacing with threshold filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/root
  %(prog)s --root_dir /path/to/root --respacing 1.0 1.0 1.0 --thresh -140.0 1000.0
        """
    )
    
    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing {site}/{site}_{pid} sample directories'
    )
    
    parser.add_argument(
        '-s', '--respacing',
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help='List of resampling spacing values in mm for x,y,z directions (default: [1.0, 1.0, 1.0])'
    )
    
    parser.add_argument(
        '-t', '--thresh',
        type=float,
        nargs=2,
        default=[-140.0, 1000.0],
        help='Image value threshold range [min, max] for filtering mask foreground voxels (default: [-140.0, 1000.0])'
    )
    
    return parser.parse_args()


def find_sample_dirs(root_dir: str) -> List[Path]:
    root_path: Path = Path(root_dir)
    sample_dirs: List[Path] = []
    
    for site_dir in root_path.iterdir():
        if site_dir.is_dir():
            for sample_dir in site_dir.iterdir():
                if sample_dir.is_dir() and re.match(r'^[^_]+_[^_]+$', sample_dir.name):
                    sample_dirs.append(sample_dir)
    
    return sorted(sample_dirs)


def resample_image_mask(image_path: Path, mask_path: Path, respacing: Tuple[float, float, float], thresh: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    load_image = LoadImage(image_only=False, ensure_channel_first=True)
    image_data, image_meta = load_image(str(image_path))
    mask_data, mask_meta = load_image(str(mask_path))
    
    data = {
        "image": image_data,
        "mask": mask_data
    }
    
    resample_transform = Compose([
        Spacingd(
            keys=["image", "mask"],
            pixdim=respacing,
            mode=["bilinear", "nearest"],
            align_corners=[True, None],
            padding_mode=["border", "constant"]
        )
    ])
    
    resampled_data = resample_transform(data)
    
    resampled_image: MetaTensor = resampled_data["image"]
    resampled_mask: MetaTensor = resampled_data["mask"]
    
    resampled_image_np = resampled_image[0].numpy()
    resampled_mask_np = resampled_mask[0].numpy()
    
    foreground_mask = resampled_mask_np > 0
    outside_threshold = (resampled_image_np < thresh[0]) | (resampled_image_np > thresh[1])
    resampled_mask_np[foreground_mask & outside_threshold] = 0
    
    resampled_meta = resampled_image.meta
    
    return resampled_image_np, resampled_mask_np, resampled_meta


def process_sample_dir(sample_dir: Path, respacing: Tuple[float, float, float], thresh: Tuple[float, float]) -> int:
    files_created: int = 0
    
    pre_volume_pattern: Pattern = re.compile(r'^([^_]+_[^_]+)_pre_volume\.nii\.gz$')
    post_volume_pattern: Pattern = re.compile(r'^([^_]+_[^_]+)_post_volume\.nii\.gz$')
    pre_mask_pattern: Pattern = re.compile(r'^([^_]+_[^_]+)_pre_mask\.nii\.gz$')
    post_mask_pattern: Pattern = re.compile(r'^([^_]+_[^_]+)_post_mask\.nii\.gz$')
    
    pre_volume: Optional[Path] = None
    post_volume: Optional[Path] = None
    pre_mask: Optional[Path] = None
    post_mask: Optional[Path] = None
    base_name: str = sample_dir.name
    
    for file_path in sample_dir.iterdir():
        if file_path.is_file():
            if pre_volume is None:
                pre_match = pre_volume_pattern.match(file_path.name)
                if pre_match:
                    pre_volume = file_path
            if post_volume is None:
                post_match = post_volume_pattern.match(file_path.name)
                if post_match:
                    post_volume = file_path
            if pre_mask is None:
                pre_match = pre_mask_pattern.match(file_path.name)
                if pre_match:
                    pre_mask = file_path
            if post_mask is None:
                post_match = post_mask_pattern.match(file_path.name)
                if post_match:
                    post_mask = file_path
    
    if pre_volume and pre_mask:
        try:
            print(f"Resampling pre volume and mask for {sample_dir.name}...")
            
            resampled_image, resampled_mask, meta_data = resample_image_mask(pre_volume, pre_mask, respacing, thresh)
            
            image_output_filestem: Path = sample_dir / f"{base_name}_pre_std_resampled_volume"
            save_image = SaveImage(
                output_dir=str(sample_dir),
                output_format="nii.gz",
                separate_folder=False,
                print_log=True
            )
            save_image(resampled_image, meta_data=meta_data, filename=str(image_output_filestem))
            files_created += 1
            
            mask_output_filestem: Path = sample_dir / f"{base_name}_pre_std_resampled_mask"
            save_image = SaveImage(
                output_dir=str(sample_dir),
                output_format="nii.gz",
                separate_folder=False,
                print_log=True,
                output_dtype=np.uint8
            )
            save_image(resampled_mask, meta_data=meta_data, filename=str(mask_output_filestem))
            files_created += 1
            
        except Exception as e:
            print(f"Error processing pre volume/mask for {sample_dir.name}: {e}")
    
    if post_volume and post_mask:
        try:
            print(f"Resampling post volume and mask for {sample_dir.name}...")
            
            resampled_image, resampled_mask, meta_data = resample_image_mask(post_volume, post_mask, respacing, thresh)
            
            image_output_filestem: Path = sample_dir / f"{base_name}_post_std_resampled_volume"
            save_image = SaveImage(
                output_dir=str(sample_dir),
                output_format="nii.gz",
                separate_folder=False,
                print_log=True
            )
            save_image(resampled_image, meta_data=meta_data, filename=str(image_output_filestem))
            files_created += 1
            
            mask_output_filestem: Path = sample_dir / f"{base_name}_post_std_resampled_mask"
            save_image = SaveImage(
                output_dir=str(sample_dir),
                output_format="nii.gz",
                separate_folder=False,
                print_log=True,
                output_dtype=np.uint8
            )
            save_image(resampled_mask, meta_data=meta_data, filename=str(mask_output_filestem))
            files_created += 1
            
        except Exception as e:
            print(f"Error processing post volume/mask for {sample_dir.name}: {e}")
    
    return files_created


def main() -> None:
    args: argparse.Namespace = parse_args()
    
    print(f"Finding sample directories in: {args.root_dir}")
    sample_dirs: List[Path] = find_sample_dirs(args.root_dir)
    
    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Using resampling spacing: {args.respacing} mm")
    print(f"Using image threshold range: {args.thresh}")
    
    total_files_created: int = 0
    
    with tqdm(sample_dirs, desc="Processing samples", unit="sample") as pbar:
        for sample_dir in pbar:
            pbar.set_description(f"Processing sample: {sample_dir.name}")
            files_created: int = process_sample_dir(sample_dir, tuple(args.respacing), tuple(args.thresh))
            total_files_created += files_created
    
    print(f"Resampling and thresholding completed!")
    print(f"Total files created: {total_files_created}")


if __name__ == '__main__':
    main()