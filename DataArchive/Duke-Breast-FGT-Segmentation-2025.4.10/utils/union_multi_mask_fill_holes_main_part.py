# -*- coding: utf-8 -*-
"""
接收-r/--root_dir指定数据集根目录，此根目录下有[train,val,test]子集目录，子集目录下有形如Breast_MRI_{sid:03d}的样本目录，样本pid=Breast_MRI_{sid:03d}，样本目录下包含形如{pid}_mask.nii.gz的多值蒙版文件以及{pid}_mask_mass.nii.gz乳房整体蒙版文件，读取{pid}_mask.nii.gz文件，将其中的前景部分的值全部设置为1，然后执行fill_holes，如果补洞后结果与{pid}_mask_mass.nii.gz不完全一致，则报告不一致性，当指定了--overwrite选项时，将补洞后的蒙版输出覆盖{pid}_mask_mass.nii.gz；当指定了--overwrite选项时，还将原{pid}_mask.nii.gz多值蒙版中label_index>1的区域按照原label_index重新填入补洞后蒙版中并覆盖{pid}_mask.nii.gz文件。 
 使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。 使用tqdm显示进度，并在进度条前的desc中展示当前正在处理样本的pid。对所有变量和函数参数进行类型注解。 
   除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Union Multi-Label Mask and Fill Holes Script for Duke-Breast-FGT-Segmentation Dataset

This script processes multi-label mask files by setting foreground pixels to 1, filling holes,
and optionally overwriting the mass mask and multi-label mask with processed versions.

Parameters:
    -r, --root_dir: Root directory containing [train, val, test] subset directories
    --overwrite: Overwrite existing mask files with processed versions
    -ht, --hole_thresh: Threshold for hole filling (default: 40 voxels, only fill holes smaller than this threshold)

Usage Examples:
    python union_multi_mask_fill_holes_small_removal.py -r /path/to/root
    python union_multi_mask_fill_holes_small_removal.py --root_dir /path/to/root --overwrite
    python union_multi_mask_fill_holes_small_removal.py -r /path/to/root -ht 40
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from monai.transforms import LoadImage, SaveImage
from scipy import ndimage
from typing import List, Dict, Tuple, Optional, Union


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir and overwrite
    """
    parser = argparse.ArgumentParser(
        description='Union multi-label mask and fill holes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/root
  %(prog)s --root_dir /path/to/root --overwrite
  %(prog)s -r /path/to/root -ht 40
  %(prog)s --root_dir /path/to/root --hole_thresh 40 --overwrite
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing [train, val, test] subset directories'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing mask files with processed versions'
    )

    parser.add_argument(
        '-ht', '--hole_thresh',
        type=int,
        default=40,
        help='Threshold for hole filling (default: 40 voxels, only fill holes smaller than this threshold)'
    )

    return parser.parse_args()


def find_sample_dirs(root_dir: Union[str, Path]) -> List[Path]:
    """
    Find all sample directories in subset directories.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing subset directories
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path: Path = Path(root_dir)
    sample_dirs: List[Path] = []
    
    # Subset directories: train, val, test
    subset_dirs = ['train', 'val', 'test']
    
    for subset in subset_dirs:
        subset_path = root_path / subset
        if subset_path.exists() and subset_path.is_dir():
            for sample_dir in subset_path.iterdir():
                if sample_dir.is_dir() and sample_dir.name.startswith('Breast_MRI_'):
                    sample_dirs.append(sample_dir)
    
    return sorted(sample_dirs)


def fill_holes(mask_data: np.ndarray, spacing: tuple, hole_thresh: int) -> np.ndarray:
    """
    Fill holes in binary mask by applying 2D binary_fill_holes to each slice
    along the axis with the largest spacing, only filling holes smaller than hole_thresh.
    
    Args:
        mask_data (np.ndarray): Binary mask data array
        spacing (tuple): Spacing of the image in each dimension
        hole_thresh (int): Maximum voxel count for holes to be filled
        
    Returns:
        np.ndarray: Mask data with small holes filled
    """
    # Find axis with largest spacing
    max_spacing = np.max(spacing)
    max_spacing_axis = np.where(spacing == max_spacing)[0][-1]  # if multiple max axes, take the last one as it is most likely to be Z-axis
    
    # Create output array
    filled_mask = mask_data.copy()
    
    # Iterate over each slice along the max spacing axis
    for i in range(filled_mask.shape[max_spacing_axis]):
        # Extract 2D slice
        if max_spacing_axis == 0:
            slice_data = filled_mask[i, :, :]
        elif max_spacing_axis == 1:
            slice_data = filled_mask[:, i, :]
        else:  # max_spacing_axis == 2
            slice_data = filled_mask[:, :, i]
        
        # Find all holes (background regions surrounded by foreground)
        # First invert the mask to make holes foreground
        inverted_slice = 1 - slice_data
        
        # Label connected regions in the inverted mask (these are the holes)
        labeled_holes, num_holes = ndimage.label(inverted_slice)
        
        # Create a mask for holes to fill
        holes_to_fill = np.zeros_like(slice_data)
        
        # For each hole, check if it's smaller than the threshold
        for hole_label in range(1, num_holes + 1):
            hole_mask = labeled_holes == hole_label
            hole_size = np.sum(hole_mask)
            
            if hole_size < hole_thresh:
                holes_to_fill[hole_mask] = 1
        
        # Fill only the small holes
        filled_slice = slice_data | holes_to_fill
        
        # Put filled slice back
        if max_spacing_axis == 0:
            filled_mask[i, :, :] = filled_slice
        elif max_spacing_axis == 1:
            filled_mask[:, i, :] = filled_slice
        else:  # max_spacing_axis == 2
            filled_mask[:, :, i] = filled_slice
    
    return filled_mask.astype(np.uint8)


def reserve_largest_connected_region(mask_data: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected region in the binary mask.
    
    Args:
        mask_data (np.ndarray): Binary mask data array
        
    Returns:
        np.ndarray: Mask data with only the largest connected region kept
    """
    # Label connected regions
    labeled_mask, num_labels = ndimage.label(mask_data)
    
    # If no regions, return empty mask
    if num_labels == 0:
        return np.zeros_like(mask_data, dtype=np.uint8)
    
    # Count voxels in each region
    region_sizes = np.bincount(labeled_mask.ravel())
    
    # Find the largest region (excluding background)
    max_size = np.max(region_sizes[1:])
    if max_size == 0:
        return np.zeros_like(mask_data, dtype=np.uint8)
    
    largest_label = np.argmax(region_sizes[1:]) + 1  # +1 to account for background (label 0)
    
    # Create mask for largest region
    filtered_mask = (labeled_mask == largest_label).astype(np.uint8)
    
    return filtered_mask.astype(np.uint8)


def process_sample_dir(sample_dir: Path, overwrite: bool, hole_thresh: int) -> Tuple[int, int, int]:
    """
    Process a single sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        overwrite (bool): Whether to overwrite existing mask files
        hole_thresh (int): Threshold for hole filling (only fill holes smaller than this threshold)
        
    Returns:
        Tuple[int, int, int]: (processed_count, inconsistency_count, overwritten_count)
    """
    pid: str = sample_dir.name
    
    # Find mask files
    mask_file: Optional[Path] = None
    mask_mass_file: Optional[Path] = None
    
    for file_path in sample_dir.iterdir():
        if file_path.is_file():
            if file_path.name == f'{pid}_mask.nii.gz':
                mask_file = file_path
            elif file_path.name == f'{pid}_mask_mass.nii.gz':
                mask_mass_file = file_path
    
    if not mask_file or not mask_mass_file:
        print(f"Warning: Missing required files for {pid}")
        return 0, 0, 0
    
    processed_count: int = 0
    inconsistency_count: int = 0
    overwritten_count: int = 0
    
    try:
        # Load multi-label mask
        loader: LoadImage = LoadImage(image_only=False, dtype=None)
        mask_data, mask_meta = loader(str(mask_file))
        
        # Load mass mask
        mass_data, mass_meta = loader(str(mask_mass_file))
        
        # Create binary mask by setting all foreground pixels to 1
        binary_mask: np.ndarray = (mask_data > 0).astype(np.uint8)
        
        # Get spacing from metadata
        spacing = mask_meta.get('spacing', (1.0, 1.0, 1.0))
        
        # Fill holes in binary mask
        filled_mask: np.ndarray = fill_holes(binary_mask, spacing, hole_thresh)
        
        # Reserve the largest connected region in the filled mask
        filtered_mask: np.ndarray = reserve_largest_connected_region(filled_mask)
        
        # Check consistency with mass mask
        is_consistent = np.array_equal(filtered_mask, mass_data)
        
        if not is_consistent:
            inconsistency_count += 1
            print(f"Warning: Inconsistency detected for {pid}")
            print(f"  Processed mask differs from mass mask")
            
            # Calculate differences
            diff_pixels = np.sum(filtered_mask != mass_data)
            print(f"  Different pixels: {diff_pixels}")
            
            # Get first 3 different pixel coordinates
            diff_coords = np.argwhere(filtered_mask != mass_data)
            if len(diff_coords) > 0:
                print("  First 3 different pixel coordinates:")
                for i, coord in enumerate(diff_coords[:3]):
                    print(f"    {i+1}. {tuple(coord.tolist())}")
        
        # Overwrite files if requested
        if not is_consistent and overwrite:
            # Save filtered mask to mass mask file
            saver: SaveImage = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.uint8)
            saver(filtered_mask, meta_data=mass_meta, filename=str(sample_dir / f'{pid}_mask_mass'))
            overwritten_count += 1
            
            # Restore multi-label information to filtered mask
            # Find regions with label_index > 1 in original mask
            for label_index in np.unique(mask_data):
                if label_index > 1:
                    # Create mask for this label
                    label_mask = (mask_data == label_index).astype(np.uint8)
                    
                    # Add this label to filtered mask
                    filtered_mask[np.logical_and(filtered_mask == 1, label_mask > 0)] = label_index
            
            # Save multi-label filtered mask
            saver(filtered_mask, meta_data=mask_meta, filename=str(sample_dir / f'{pid}_mask'))
            overwritten_count += 1
        
        processed_count += 1
        
    except Exception as e:
        print(f"Error processing {pid}: {str(e)}")
    
    return processed_count, inconsistency_count, overwritten_count


def process_root_dir(root_dir: Union[str, Path], overwrite: bool, hole_thresh: int) -> None:
    """
    Process all sample directories in the root directory.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing subset directories
        overwrite (bool): Whether to overwrite existing mask files
        hole_thresh (int): Threshold for hole filling (only fill holes smaller than this threshold)
    """
    root_path: Path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    sample_dirs: List[Path] = find_sample_dirs(root_path)

    if not sample_dirs:
        print(f"Warning: No sample directories found in {root_dir}")
        return

    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Overwrite mode: {'Enabled' if overwrite else 'Disabled'}")
    print(f"Hole filling threshold: {hole_thresh} voxels")
    print()

    total_processed: int = 0
    total_inconsistencies: int = 0
    total_overwritten: int = 0

    with tqdm(total=len(sample_dirs), desc='Processing samples') as pbar:
        for sample_dir in sample_dirs:
            pbar.set_description(f'Processing {sample_dir.name}')
            
            processed, inconsistencies, overwritten = process_sample_dir(sample_dir, overwrite, hole_thresh)
            
            total_processed += processed
            total_inconsistencies += inconsistencies
            total_overwritten += overwritten
            
            pbar.update(1)

    print(f"\nProcessing completed!")
    print(f"Total samples processed: {total_processed}")
    print(f"Total inconsistencies found: {total_inconsistencies}")
    print(f"Total files overwritten: {total_overwritten}")


def main() -> None:
    """
    Main function to orchestrate the mask union and hole filling process.
    """
    args: argparse.Namespace = parse_args()

    print(f"Processing data from: {args.root_dir}")
    print(f"Overwrite mode: {'Enabled' if args.overwrite else 'Disabled'}")
    print(f"Hole filling threshold: {args.hole_thresh} voxels")

    process_root_dir(args.root_dir, args.overwrite, args.hole_thresh)

    print("Mask union and hole filling completed successfully!")


if __name__ == '__main__':
    main()
