#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
接收-r/--root_dir指定根目录，根目录下包含{site}/{site}_{pid}样本目录。-p/--peri_expand_range指定瘤周扩张距离（单位：mm）列表，默认[3.0,5.0,7.0]。 
 对每个样本，获取其{site}_{pid}_pre_mask.nii.gz和{site}_{pid}_post_mask.nii.gz的治疗前和治疗后蒙版，根据-p指定的每个扩张距离，封装一个ellipsoid函数用于构造结构元，将蒙版的前景部分边界向外扩张指定距离，最后减去原有蒙版前景区域构成peri_expand_range(mm)瘤周区域蒙版，将蒙版保存在样本目录下，文件名为{site}_{pid}_(pre|post)_peritumor_{peri_expand_range}mm_mask.nii.gz。 
 使用MONAI库LoadImage和SaveImage读取和保存图像文件，使用pathlib处理路径。使用tqdm显示进度。 
 对全部变量和函数参数添加类型注解。 
  除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Peritumor Region Extraction Script for ESO-2025.10.31 Dataset

This script extracts peritumor regions from pre and post-treatment masks by expanding the foreground boundaries
by specified distances and subtracting the original foreground region. It also filters regions by intensity values
and removes small connected regions.

Parameters:
    -r, --root_dir: Root directory containing {site}/{site}_{pid} sample directories
    -p, --peri_expand_range: List of peritumor expansion distances in millimeters (default: [3.0, 5.0, 7.0])
    -m, --mask_type: List of mask types to process (default: [''])
    -v, --volume_type: List of volume types to process (default: [''])
    -vr, --valid_value_range: Valid intensity value range [min, max] (default: [-140, 3000])
    -st, --small_thresh: Threshold for removing small connected regions (default: 10 voxels)
    --skip_existing: Skip processing if peritumor files already exist

Usage Examples:
    python 02_extract_peritumor.py -r /path/to/root
    python 02_extract_peritumor.py --root_dir /path/to/root --peri_expand_range 3.0 5.0 7.0
    python 02_extract_peritumor.py -r /path/to/root --valid_value_range -140 3000 --small_thresh 10
    python 02_extract_peritumor.py -r /path/to/root --skip_existing
    python 02_extract_peritumor.py -r /path/to/root --mask_type '' 'std_resampled' --volume_type '' 'std_resampled'
"""

import argparse
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Pattern
from tqdm import tqdm
from scipy.ndimage import binary_dilation, binary_fill_holes, label
from monai.transforms import LoadImage, SaveImage
from monai.data import MetaTensor


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Extract peritumor regions by expanding mask boundaries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/root
  %(prog)s --root_dir /path/to/root --peri_expand_range 3.0 5.0 7.0 --mask_type '' 'std_resampled'
  %(prog)s -r /path/to/root --valid_value_range -140 3000 --small_thresh 10
  %(prog)s -r /path/to/root --skip_existing
        """)
    
    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing {site}/{site}_{pid} sample directories'
    )
    
    parser.add_argument(
        '-p', '--peri_expand_range',
        type=float,
        nargs='+',
        default=[3.0, 5.0, 7.0],
        help='List of peritumor expansion distances in millimeters (default: [3.0, 5.0, 7.0])'
    )
    
    parser.add_argument(
        '-m', '--mask_type',
        type=str,
        nargs='+',
        default=[''],
        help='List of mask types to process (default: [\'\'])'
    )

    parser.add_argument(
        '-v', '--volume_type',
        type=str,
        nargs='+',
        default=[''],
        help='List of volume types to process (default: [\'\'])'
    )

    parser.add_argument(
        '-vr', '--valid_value_range',
        type=float,
        nargs=2,
        default=[-140, 3000],
        help='Valid intensity value range [min, max] (default: [-140, 3000])'
    )
    
    parser.add_argument(
        '-st', '--small_thresh',
        type=int,
        default=10,
        help='Threshold for removing small connected regions (default: 10 voxels)'
    )

    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip processing if peritumor files already exist'
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


def ellipsoid(rx: int, ry: int, rz: int) -> np.ndarray:
    """
    Create a 3D ellipsoidal structuring element.
    
    Args:
        rx: Radius in x direction (voxels)
        ry: Radius in y direction (voxels)
        rz: Radius in z direction (voxels)
        
    Returns:
        np.ndarray: 3D numpy array with 1s inside the ellipsoid and 0s outside
    """
    # Create a grid of coordinates
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    x, y, z = np.mgrid[-rx:rx+1, -ry:ry+1, -rz:rz+1]
    
    # Calculate ellipsoid equation
    ellipsoid_mask: np.ndarray = (x**2 / rx**2 + y**2 / ry**2 + z**2 / rz**2) <= 1
    
    return ellipsoid_mask.astype(np.uint8)


def fill_holes(mask_data: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary mask by processing each Z-slice individually.
    
    Args:
        mask_data: Binary mask data array
        
    Returns:
        np.ndarray: Mask data with holes filled in each Z-slice
    """
    # Create a copy of the input mask
    filled_mask = mask_data.copy()
    
    # Process each Z-slice individually
    for z in range(mask_data.shape[2]):
        # Extract the 2D slice
        slice_data = mask_data[:, :, z]
        # Fill holes in the slice
        filled_slice = binary_fill_holes(slice_data)
        # Put the filled slice back
        filled_mask[:, :, z] = filled_slice
    
    return filled_mask.astype(np.uint8)


def filter_by_intensity(image_data: np.ndarray, mask: np.ndarray, valid_range: Tuple[float, float]) -> np.ndarray:
    """
    Filter mask by intensity values in the image.
    
    Args:
        image_data: Original image data array
        mask: Binary mask array
        valid_range: Valid intensity value range [min, max]
        
    Returns:
        np.ndarray: Filtered mask with only valid intensity regions
    """
    filtered_mask = mask.copy()
    
    # Create mask for valid intensity values
    valid_intensity = (image_data >= valid_range[0]) & (image_data <= valid_range[1])
    
    # Set invalid intensity regions to 0
    filtered_mask[~valid_intensity] = 0
    
    return filtered_mask.astype(np.uint8)


def remove_small_regions(mask: np.ndarray, threshold: int) -> np.ndarray:
    """
    Remove small connected regions from a binary mask.
    
    Args:
        mask: Binary mask array
        threshold: Minimum number of voxels for a region to be kept
        
    Returns:
        np.ndarray: Mask with small regions removed
    """
    
    # Label connected regions
    labeled_mask, num_regions = label(mask)
    
    # Remove small regions
    result = np.zeros_like(mask, dtype=np.uint8)
    
    for i in range(1, num_regions + 1):
        region = (labeled_mask == i)
        if region.sum() >= threshold:
            result[region] = 1
    
    return result


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    
    Args:
        mask: Binary mask array
        
    Returns:
        np.ndarray: Mask with only the largest connected component
    """
    # Label connected regions
    labeled_mask, num_regions = label(mask)
    
    if num_regions == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    
    # Find the largest connected component
    largest_size = 0
    largest_component = 0
    
    for i in range(1, num_regions + 1):
        region_size = np.sum(labeled_mask == i)
        if region_size > largest_size:
            largest_size = region_size
            largest_component = i
    
    # Create mask with only the largest component
    result = np.zeros_like(mask, dtype=np.uint8)
    result[labeled_mask == largest_component] = 1
    
    return result


def extract_peritumor_region(mask_path: Path, volume_path: Optional[Path], expand_distance: float, valid_range: Tuple[float, float], small_thresh: int) -> np.ndarray:
    """
    Extract peritumor region by expanding the foreground boundary and subtracting the original foreground.
    
    Args:
        mask_path: Path to the original mask NIfTI file
        volume_path: Path to the corresponding volume NIfTI file
        expand_distance: Expansion distance in millimeters
        valid_range: Valid intensity value range [min, max]
        small_thresh: Threshold for removing small connected regions
        
    Returns:
        np.ndarray: 3D numpy array containing the peritumor region
    """
    # Load the mask using MONAI
    load_image = LoadImage(image_only=False)
    mask_data: np.ndarray
    meta_data: Dict
    mask_data, meta_data = load_image(str(mask_path))
    
    # Get voxel spacing from metadata
    voxel_spacing: Tuple[float, float, float] = tuple(meta_data['pixdim'][1:4])
    
    # Create binary mask (foreground = 1, background = 0)
    binary_mask: np.ndarray = (mask_data > 0).astype(np.uint8)
    
    # Calculate number of voxels to expand in each dimension
    rx: int
    ry: int
    rz: int
    rx, ry, rz = [int(np.ceil(expand_distance / spacing)) for spacing in voxel_spacing]
    
    # Get ROI bounding box
    coords = np.argwhere(binary_mask)
    if coords.size == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Expand bounding box by (rx, ry, rz) without exceeding image bounds
    img_shape = binary_mask.shape
    start = np.maximum(0, min_coords - [rx, ry, rz])
    end = np.minimum(img_shape, max_coords + [rx, ry, rz] + 1)  # +1 because Python slicing is exclusive
    
    # Extract ROI with padding
    roi = binary_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    
    # Fill holes in ROI
    filled_roi = fill_holes(roi)
    
    # Create a new mask with filled ROI
    filled_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    filled_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = filled_roi
    
    # Create ellipsoidal structuring element
    structuring_element: np.ndarray = ellipsoid(rx, ry, rz)
    
    # Dilate the filled mask
    dilated_mask: np.ndarray = binary_dilation(filled_mask, structure=structuring_element)
    
    # Convert to uint8
    dilated_mask: np.ndarray = dilated_mask.astype(np.uint8)
    
    # Load the volume file to get intensity values and perform threshold suppression
    if volume_path and volume_path.exists():
        image_data: np.ndarray
        image_data, _ = load_image(str(volume_path))
        
        # Filter by intensity values (threshold suppression)
        dilated_mask = filter_by_intensity(image_data, dilated_mask, valid_range)
    
    # Keep only the largest connected component
    dilated_mask = keep_largest_connected_component(dilated_mask)
    
    # Subtract the original filled mask to get peritumor region
    peritumor_mask: np.ndarray = (dilated_mask.astype(bool)
                                  & np.logical_not(filled_mask.astype(bool))).astype(np.uint8)
    
    # Convert to uint8
    peritumor_mask: np.ndarray = peritumor_mask.astype(np.uint8)
    
    # Remove small connected regions
    peritumor_mask = remove_small_regions(peritumor_mask, small_thresh)
    
    return peritumor_mask


def process_sample_dir(sample_dir: Path, peri_expand_range: List[float], mask_type: str, volume_type: str, valid_range: Tuple[float, float], small_thresh: int, skip_existing: bool = False) -> int:
    """
    Process a single sample directory to extract peritumor regions for all specified expansion distances and mask types.
    
    Args:
        sample_dir: Path to the sample directory
        peri_expand_range: List of peritumor expansion distances in millimeters
        mask_type: Mask type to process
        volume_type: Volume type to process
        valid_range: Valid intensity value range [min, max]
        small_thresh: Threshold for removing small connected regions
        skip_existing: Skip processing if peritumor files already exist
        
    Returns:
        int: Number of peritumor masks created
    """
    masks_created: int = 0

    # Initialize MONAI components
    load_image = LoadImage(image_only=False)

    # Generate mask patterns based on mask_type
    base_name: str = sample_dir.name
    
    if mask_type:
        pre_mask_pattern: Pattern = re.compile(rf'^([^_]+_[^_]+)_pre_{mask_type}_mask\.nii\.gz$')
        post_mask_pattern: Pattern = re.compile(rf'^([^_]+_[^_]+)_post_{mask_type}_mask\.nii\.gz$')
    else:
        pre_mask_pattern: Pattern = re.compile(r'^([^_]+_[^_]+)_pre_mask\.nii\.gz$')
        post_mask_pattern: Pattern = re.compile(r'^([^_]+_[^_]+)_post_mask\.nii\.gz$')
    
    # Generate volume patterns based on volume_type
    if volume_type:
        pre_volume_pattern: Pattern = re.compile(rf'^([^_]+_[^_]+)_pre_{volume_type}_volume\.nii\.gz$')
        post_volume_pattern: Pattern = re.compile(rf'^([^_]+_[^_]+)_post_{volume_type}_volume\.nii\.gz$')
    else:
        pre_volume_pattern: Pattern = re.compile(r'^([^_]+_[^_]+)_pre_volume\.nii\.gz$')
        post_volume_pattern: Pattern = re.compile(r'^([^_]+_[^_]+)_post_volume\.nii\.gz$')

    pre_mask: Optional[Path] = None
    post_mask: Optional[Path] = None
    pre_volume: Optional[Path] = None
    post_volume: Optional[Path] = None

    for file_path in sample_dir.iterdir():
        if file_path.is_file():
            # Check for mask files
            pre_mask_match = pre_mask_pattern.match(file_path.name)
            post_mask_match = post_mask_pattern.match(file_path.name)
            
            # Check for volume files
            pre_volume_match = pre_volume_pattern.match(file_path.name)
            post_volume_match = post_volume_pattern.match(file_path.name)
            
            if pre_mask_match:
                pre_mask = file_path
            elif post_mask_match:
                post_mask = file_path
            elif pre_volume_match:
                pre_volume = file_path
            elif post_volume_match:
                post_volume = file_path
    
    # Process pre mask if found
    if pre_mask:
        for distance in peri_expand_range:
            # Check if output file already exists
            if mask_type:
                output_file: Path = sample_dir / f"{base_name}_pre_peritumor_{distance}mm_{mask_type}_mask.nii.gz"
            else:
                output_file: Path = sample_dir / f"{base_name}_pre_peritumor_{distance}mm_mask.nii.gz"
            
            if skip_existing and output_file.exists():
                print(f"Skipping pre peritumor file for {sample_dir.name} with distance {distance}mm: File already exists")
                continue
                
            try:
                # Load original data and metadata for saving
                meta_data: Dict
                _, meta_data = load_image(str(pre_mask))
                
                # Extract peritumor region
                peritumor_mask: np.ndarray = extract_peritumor_region(pre_mask, pre_volume, distance, valid_range, small_thresh)
                
                # Create output filestem with mask_type if specified
                if mask_type:
                    output_filestem: Path = sample_dir / f"{base_name}_pre_peritumor_{distance}mm_{mask_type}_mask"
                else:
                    output_filestem: Path = sample_dir / f"{base_name}_pre_peritumor_{distance}mm_mask"
                
                # Create SaveImage instance with custom filename
                save_image = SaveImage(
                    output_dir=str(sample_dir),
                    output_format="nii.gz",
                    separate_folder=False,
                    print_log=True,
                    output_dtype=np.uint8
                )
                
                # Save using MONAI with custom filename
                save_image(peritumor_mask, meta_data=meta_data, filename=output_filestem)
                masks_created += 1
            except Exception as e:
                print(f"Error processing pre mask for {sample_dir.name} with distance {distance}mm: {e}")
    
    # Process post mask if found
    if post_mask:
        for distance in peri_expand_range:
            # Check if output file already exists
            if mask_type:
                output_file: Path = sample_dir / f"{base_name}_post_peritumor_{distance}mm_{mask_type}_mask.nii.gz"
            else:
                output_file: Path = sample_dir / f"{base_name}_post_peritumor_{distance}mm_mask.nii.gz"
            
            if skip_existing and output_file.exists():
                print(f"Skipping post peritumor file for {sample_dir.name} with distance {distance}mm: File already exists")
                continue
                
            try:
                # Load original data and metadata for saving
                meta_data: Dict
                _, meta_data = load_image(str(post_mask))
                
                # Extract peritumor region
                peritumor_mask: np.ndarray = extract_peritumor_region(post_mask, post_volume, distance, valid_range, small_thresh)
                
                # Create output filestem with mask_type if specified
                if mask_type:
                    output_filestem: Path = sample_dir / f"{base_name}_post_peritumor_{distance}mm_{mask_type}_mask"
                else:
                    output_filestem: Path = sample_dir / f"{base_name}_post_peritumor_{distance}mm_mask"
                
                # Create SaveImage instance with custom filename
                save_image = SaveImage(
                    output_dir=str(sample_dir),
                    output_format="nii.gz",
                    separate_folder=False,
                    print_log=True,
                    output_dtype=np.uint8
                )
                
                # Save using MONAI with custom filename
                save_image(peritumor_mask, meta_data=meta_data, filename=output_filestem)
                masks_created += 1
            except Exception as e:
                print(f"Error processing post mask for {sample_dir.name} with distance {distance}mm: {e}")
    
    return masks_created


def main() -> None:
    """
    Main function to orchestrate the peritumor region extraction process.
    """
    args: argparse.Namespace = parse_args()
    
    # Validate that mask_type and volume_type have the same length
    if len(args.mask_type) != len(args.volume_type):
        print(f"Error: Number of mask types ({len(args.mask_type)}) must match number of volume types ({len(args.volume_type)})")
        return
    
    print(f"Finding sample directories in: {args.root_dir}")
    sample_dirs: List[Path] = find_sample_dirs(args.root_dir)
    
    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Using expansion distances: {args.peri_expand_range} mm")
    print(f"Using mask types: {args.mask_type}")
    print(f"Using volume types: {args.volume_type}")
    print(f"Valid intensity range: {args.valid_value_range}")
    print(f"Small region threshold: {args.small_thresh} voxels")
    print(f"Skip existing files: {args.skip_existing}")
    
    total_masks_created: int = 0
    
    # Convert valid_value_range to tuple
    valid_range: Tuple[float, float] = tuple(args.valid_value_range)
    
    with tqdm(sample_dirs, desc="Processing samples", unit="sample") as pbar:
        for sample_dir in pbar:
            pbar.set_description(f"Processing sample: {sample_dir.name}")
            for mask_type, volume_type in zip(args.mask_type, args.volume_type):
                masks_created: int = process_sample_dir(
                    sample_dir, 
                    args.peri_expand_range, 
                    mask_type, 
                    volume_type,
                    valid_range, 
                    args.small_thresh, 
                    args.skip_existing
                )
                total_masks_created += masks_created
    
    print(f"Peritumor region extraction completed!")
    print(f"Total peritumor masks created: {total_masks_created}")


if __name__ == '__main__':
    main()
