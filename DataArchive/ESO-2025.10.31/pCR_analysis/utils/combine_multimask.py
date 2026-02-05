#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
接收-r/--root_dir指定根目录，根目录下包含{site}/{site}_{pid}样本目录。样本目录中包含形如{site}_{pid}_(pre|post)(_{mask_type})?_mask.nii.gz的若干蒙版文件，其中(_{mask_type})?部分表示_{mask_type}内容是可选的。 -m/--mask_type用于指定选中的蒙版文件的extra_type列表，默认值为['', 'peritumor_3.0mm','peritumor_5.0mm','peritumor_7.0mm']。
  -p/--phase用于指定阶段列表，默认为['pre','post']。
 遍历所有phase指定阶段，对每个阶段按顺序获取mask_type指定的mask文件{site}_{pid}_{phase}(_{mask_type})?_mask.nii.gz，如果mask_type=''，则对应于{site}_{pid}_{phase}_mask.nii.gz，为这些蒙版文件中的前景区域分配由-i/--reindex列表指定的label_index，默认为[1,2,3,4]，然后合并为一个多值蒙版，对于前景重叠的情况，优先取用mask_type列表中索引靠后的蒙版对应的label_index。-f/--infix指定输出蒙版的中缀，默认为'multi'。将合并的蒙版输出到原样本目录，命名为{site}_{pid}_{phase}(_{infix})?_mask.nii.gz，如果infix=''，则应输出为{site}_{pid}_{phase}_mask.nii.gz。
 使用MONAI库LoadImage和SaveImage读取和保存图像文件，使用pathlib处理路径。使用tqdm显示进度。
 对全部变量和函数参数添加类型注解。
  除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Multi-mask Combination Script for ESO-2025.10.31 Dataset

This script combines multiple mask files into a single multi-value mask by assigning different label indices
to each mask type. It handles overlaps by prioritizing masks that appear later in the mask_type list.

Parameters:
    -r, --root_dir: Root directory containing {site}/{site}_{pid} sample directories
    -m, --mask_type: List of extra mask types to process (default: ['', 'peritumor_3.0mm', 'peritumor_5.0mm', 'peritumor_7.0mm'])
    -p, --phase: List of phases to process (default: ['pre', 'post'])
    -i, --reindex: List of label indices to assign to each mask type (default: [1, 2, 3, 4])
    -f, --infix: Infix for output mask filename (default: 'multi')
    --skip_existing: Skip processing if output mask file already exists

Usage Examples:
    python combine_multimask.py -r /path/to/root
    python combine_multimask.py --root_dir /path/to/root --mask_type '' 'peritumor_3.0mm' --phase 'pre' --reindex 1 2 --infix 'combined'
    python combine_multimask.py -r /path/to/root --skip_existing
"""

import argparse
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Pattern, Tuple
from tqdm import tqdm
from monai.transforms import LoadImage, SaveImage
from monai.data import MetaTensor


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Combine multiple masks into a single multi-value mask',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/root
  %(prog)s --root_dir /path/to/root --mask_type '' 'peritumor_3.0mm' --phase 'pre' --reindex 1 2 --infix 'combined'
  %(prog)s -r /path/to/root --skip_existing
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing {site}/{site}_{pid} sample directories'
    )

    parser.add_argument(
        '-m', '--mask_type',
        type=str,
        nargs='+',
        default=['', 'peritumor_3.0mm', 'peritumor_5.0mm', 'peritumor_7.0mm'],
        help='List of extra mask types to process (default: [\'\', \'peritumor_3.0mm\', \'peritumor_5.0mm\', \'peritumor_7.0mm\'])'
    )

    parser.add_argument(
        '-p', '--phase',
        type=str,
        nargs='+',
        default=['pre', 'post'],
        help='List of phases to process (default: [\'pre\', \'post\'])'
    )

    parser.add_argument(
        '-i', '--reindex',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4],
        help='List of label indices to assign to each mask type (default: [1, 2, 3, 4])'
    )

    parser.add_argument(
        '-f', '--infix',
        type=str,
        default='multi',
        help='Infix for output mask filename (default: \'multi\')'
    )

    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip processing if output mask file already exists'
    )

    return parser.parse_args()


def find_sample_dirs(root_dir: str) -> List[Path]:
    """
    Find all sample directories in the specified root directory.
    
    Args:
        root_dir: Root directory containing {site}/{site}_{pid} sample directories
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path: Path = Path(root_dir)
    sample_dirs: List[Path] = []

    # Iterate through site directories
    for site_dir in root_path.iterdir():
        if site_dir.is_dir():
            # Iterate through sample directories in site directory
            for sample_dir in site_dir.iterdir():
                if sample_dir.is_dir() and re.match(r'^[^_]+_[^_]+$', sample_dir.name):
                    sample_dirs.append(sample_dir)

    return sorted(sample_dirs)


def combine_masks(sample_dir: Path, phases: List[str], mask_types: List[str], label_indices: List[int],
                  infix: str, skip_existing: bool = False) -> int:
    """
    Process a single sample directory to combine multiple masks into a multi-value mask.

    Args:
        sample_dir: Path to the sample directory
        phases: List of phases to process
        mask_types: List of mask types to process
        label_indices: List of label indices to assign to each mask type
        infix: Infix for output mask filename
        skip_existing: Skip processing if output mask file already exists

    Returns:
        int: Number of combined masks created
    """
    masks_created: int = 0
    load_image = LoadImage(image_only=False)

    base_name: str = sample_dir.name

    for phase in phases:
        try:
            # Construct output filename
            if infix:
                output_filename: str = f"{base_name}_{phase}_{infix}_mask.nii.gz"
            else:
                output_filename: str = f"{base_name}_{phase}_mask.nii.gz"

            output_path: Path = sample_dir / output_filename

            # Check if output file already exists
            if skip_existing and output_path.exists():
                print(f"Skipping {phase} mask for {sample_dir.name}: Output file already exists")
                continue

            # Initialize combined mask
            combined_mask: Optional[np.ndarray] = None
            meta_data: Optional[Dict] = None

            # Process each mask type in order (former types have higher priority)
            for mask_type, label_idx in (zip(reversed(mask_types), reversed(label_indices))):
                # Construct mask filename
                if mask_type:
                    mask_filename: str = f"{base_name}_{phase}_{mask_type}_mask.nii.gz"
                else:
                    mask_filename: str = f"{base_name}_{phase}_mask.nii.gz"

                mask_path: Path = sample_dir / mask_filename

                # Check if mask file exists
                if not mask_path.exists():
                    print(f"Warning: Mask file not found: {mask_path}")
                    continue

                # Load mask
                mask_data: MetaTensor
                mask_meta: Dict
                mask_data, mask_meta = load_image(str(mask_path))

                # Initialize combined mask if not already done
                if combined_mask is None:
                    combined_mask = np.zeros_like(mask_data, dtype=np.uint8)
                    meta_data = mask_meta

                # Assign label index to foreground voxels
                foreground: np.ndarray = (mask_data.numpy() > 0)
                combined_mask[foreground] = label_idx

            # Save combined mask if it was initialized
            if combined_mask is not None and meta_data is not None:
                # Save using MONAI
                save_image = SaveImage(
                    output_dir=str(sample_dir),
                    output_format="nii.gz",
                    separate_folder=False,
                    print_log=True,
                    output_dtype=np.uint8
                )

                save_image(combined_mask, meta_data=meta_data, filename=str(output_path).replace('.nii.gz', ''))
                masks_created += 1

        except Exception as e:
            print(f"Error processing sample {sample_dir.name} for phase {phase}: {e}")

    return masks_created


def main() -> None:
    """
    Main function to orchestrate the multi-mask combination process.
    """
    args: argparse.Namespace = parse_args()

    print(f"Finding sample directories in: {args.root_dir}")
    sample_dirs: List[Path] = find_sample_dirs(args.root_dir)

    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Using mask types: {args.mask_type}")
    print(f"Using phases: {args.phase}")
    print(f"Using label indices: {args.reindex}")
    print(f"Using output infix: '{args.infix}'")

    # Validate that mask_type and reindex have the same length
    if len(args.mask_type) != len(args.reindex):
        print(f"Error: Length of mask_type ({len(args.mask_type)}) must match length of reindex ({len(args.reindex)})")
        return

    total_masks_created: int = 0

    with tqdm(sample_dirs, desc="Processing samples", unit="sample") as pbar:
        for sample_dir in pbar:
            pbar.set_description(f"Processing sample: {sample_dir.name}")
            masks_created: int = combine_masks(
                sample_dir,
                args.phase,
                args.mask_type,
                args.reindex,
                args.infix,
                args.skip_existing
            )
            total_masks_created += masks_created

    print(f"Multi-mask combination completed!")
    print(f"Total combined masks created: {total_masks_created}")


if __name__ == '__main__':
    main()
