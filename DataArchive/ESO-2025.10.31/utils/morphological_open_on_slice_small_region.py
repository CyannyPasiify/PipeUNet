# -*- coding: utf-8 -*-
'''参照VerSe数据集的形态学闭运算脚本生成。该脚本对除了表示背景的二值蒙版以外的全部前景二值蒙版执行开运算，移除前景部分的小连通区域。在冲突处理时，对于那些不包含在任何处理后前景二值蒙版中的点，将其设置为背景。 
 -r/--root_dir指定根目录，根目录下包含若干层级的子目录，最低层为site中心目录，中心目录下包含phase阶段目录，即治疗前pre和治疗后post目录，阶段目录中包含{site}_{pid}_{phase}样本目录，每个中包含多个形如{site}_{pid}_{phase}_mask_{label_index}_{label_name}.nii.gz的蒙版文件。 
  对全部变量和函数参数添加类型注解。 
  除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。'''

"""
Morphological Open Operation Script for ESO-2025.10.31 Dataset

This script applies morphological opening operations to foreground binary masks slice by slice,
removing small connected regions from foreground areas. The processed masks are saved to the specified output directory
while preserving the original relative path structure.

The directory structure is as follows:
Root Directory
└── Site Directory
    ├── pre (pre-treatment)
    │   └── {site}_{pid}_pre (sample directories)
    │       └── {site}_{pid}_pre_mask_{label_index}_{label_name}.nii.gz (binary masks)
    └── post (post-treatment)
        └── {site}_{pid}_post (sample directories)
            └── {site}_{pid}_post_mask_{label_index}_{label_name}.nii.gz (binary masks)

Parameters:
    -r, --root_dir: Root directory containing sample directories
    -o, --output_dir: Output root directory for processed masks
    -k, --kernel_size: Two integers (kx, ky) for kernel size
    -f, --filter_halfedge: Integer for filter half-edge (default: 3, actual kernel size is 2*filter_halfedge+1)
    -t, --region_area_thresh: Area threshold in mm² for small regions (default: 10.0)
    --skip_existing: Skip processing if output multi-label mask already exists. If multi-label mask exists but some binary masks are missing, export missing binary masks from existing multi-label mask

Usage Examples:
    python morphological_open_on_slice_small_region.py -r /path/to/input -o /path/to/output -k 3 3
    python morphological_open_on_slice_small_region.py --root_dir /path/to/input --output_dir /path/to/output --kernel_size 3 3 --filter_halfedge 5 --region_area_thresh 10.0
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.ndimage import binary_opening, label
from monai.transforms import LoadImage, SaveImage
from typing import List, Dict, Tuple, Optional, Union


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir, output_dir, kernel_size, filter_halfedge, and region_area_thresh
    """
    parser = argparse.ArgumentParser(
        description='Apply morphological open operation to foreground binary masks slice by slice',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/input -o /path/to/output -k 3 3
  %(prog)s --root_dir /path/to/input --output_dir /path/to/output --kernel_size 3 3 --filter_halfedge 5 --region_area_thresh 10.0
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing sample directories'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Output root directory for processed masks'
    )

    parser.add_argument(
        '-k', '--kernel_size',
        type=int,
        nargs=2,
        required=True,
        help='Two integers (kx, ky) for kernel size'
    )

    parser.add_argument(
        '-f', '--filter_halfedge',
        type=int,
        default=3,
        help='Integer for filter half-edge (default: 3, actual kernel size is 2*filter_halfedge+1)'
    )

    parser.add_argument(
        '-t', '--region_area_thresh',
        type=float,
        default=10.0,
        help='Area threshold in mm² for small regions (default: 10.0)'
    )

    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip processing if output multi-label mask already exists. If multi-label mask exists but some binary masks are missing, export missing binary masks from existing multi-label mask'
    )

    return parser.parse_args()


def find_small_regions_mask(binary_mask_slice: np.ndarray, spacing: Tuple[float, float], area_thresh: float) -> np.ndarray:
    """
    Find small connected regions in a binary mask slice.
    
    Args:
        binary_mask_slice (np.ndarray): Binary mask slice (2D array)
        spacing (Tuple[float, float]): Tuple of (dx, dy) for pixel spacing in mm
        area_thresh (float): Area threshold in mm² for small regions
        
    Returns:
        np.ndarray: Boolean mask indicating voxels in small regions
    """
    labeled_array, num_features = label(binary_mask_slice, structure=np.ones((3, 3), dtype=int))

    small_region_mask = np.zeros_like(binary_mask_slice, dtype=bool)

    for region_label in range(1, num_features + 1):
        region_mask = (labeled_array == region_label)
        region_voxels = np.sum(region_mask)

        dx, dy = spacing
        area = region_voxels * dx * dy

        if area < area_thresh:
            small_region_mask = small_region_mask | region_mask

    return small_region_mask


def apply_morphological_operations_slice(binary_mask_slice: np.ndarray, kernel_size: Tuple[int, int], spacing: Tuple[float, float], area_thresh: float) -> np.ndarray:
    """
    Apply morphological opening operation to small foreground connected regions in a single slice.
    Only apply opening operation to foreground regions to remove small connected components.
    
    Args:
        binary_mask_slice (np.ndarray): Binary mask slice (2D array)
        kernel_size (Tuple[int, int]): Tuple of (kx, ky) for kernel size
        spacing (Tuple[float, float]): Tuple of (dx, dy) for pixel spacing in mm
        area_thresh (float): Area threshold in mm² for small regions
        
    Returns:
        np.ndarray: Processed binary mask slice
    """
    kx, ky = kernel_size
    structure = np.ones((kx, ky), dtype=bool)

    original_slice = binary_mask_slice.copy()

    # Only apply opening operation to remove small foreground regions
    opened_slice = binary_opening(original_slice, structure=structure)

    # Find small foreground regions
    small_foreground_mask = find_small_regions_mask(original_slice, spacing, area_thresh)

    processed_slice = original_slice.copy()

    if np.any(small_foreground_mask):
        # Apply opening result only to small foreground regions
        foreground_small_regions = small_foreground_mask & (original_slice == 1)
        np.putmask(processed_slice, foreground_small_regions, opened_slice)

    return processed_slice


def process_binary_mask_slice_by_slice(binary_mask_data: np.ndarray, kernel_size: Tuple[int, int], spacing: Tuple[float, float], area_thresh: float) -> np.ndarray:
    """
    Process binary mask slice by slice with morphological opening operations on small foreground regions only.
    
    Args:
        binary_mask_data (np.ndarray): Binary mask data (3D array)
        kernel_size (Tuple[int, int]): Tuple of (kx, ky) for kernel size
        spacing (Tuple[float, float]): Tuple of (dx, dy) for pixel spacing in mm
        area_thresh (float): Area threshold in mm² for small regions
        
    Returns:
        np.ndarray: Processed binary mask data
    """
    processed_data = np.zeros_like(binary_mask_data)

    for z in range(binary_mask_data.shape[2]):
        slice_data = binary_mask_data[:, :, z]
        processed_slice = apply_morphological_operations_slice(slice_data, kernel_size, spacing, area_thresh)
        processed_data[:, :, z] = processed_slice

    return processed_data


def resolve_conflicts(binary_masks_dict: Dict[int, np.ndarray], filter_halfedge: int) -> Optional[np.ndarray]:
    """
    Resolve conflicts where multiple labels claim the same voxel using vectorized operations.
    For voxels with no foreground labels after processing, set them to background (0).
    
    Args:
        binary_masks_dict (Dict[int, np.ndarray]): Dictionary mapping label values to binary mask arrays
        filter_halfedge (int): Half-edge of filter region for conflict resolution (actual kernel size is 2*filter_halfedge+1)
        
    Returns:
        Optional[np.ndarray]: Multi-label mask with resolved conflicts, or None if no masks were provided
    """
    if not binary_masks_dict:
        return None

    sorted_labels: List[int] = sorted(binary_masks_dict.keys())
    binary_masks: List[np.ndarray] = [binary_masks_dict[label] for label in sorted_labels]

    stacked_masks: np.ndarray = np.stack(binary_masks, axis=0)
    label_counts: np.ndarray = np.sum(stacked_masks, axis=0)

    no_conflict_mask: np.ndarray = (label_counts == 1)
    conflict_mask: np.ndarray = (label_counts > 1)

    # Initialize with background (0)
    combined_mask: np.ndarray = np.zeros(stacked_masks.shape[1:], dtype=np.uint8)

    # Assign non-conflict voxels
    for i, label in enumerate(sorted_labels):
        label_mask = stacked_masks[i] & no_conflict_mask
        np.putmask(combined_mask, label_mask, label)

    # Resolve conflicts
    conflict_count = np.sum(conflict_mask)
    if conflict_count > 0:
        print(f"Found {conflict_count} conflict voxels, resolving...")

        conflict_coords = np.argwhere(conflict_mask)
        for x, y, z in conflict_coords:
            x_min = max(0, x - filter_halfedge)
            x_max = min(stacked_masks.shape[1], x + filter_halfedge + 1)
            y_min = max(0, y - filter_halfedge)
            y_max = min(stacked_masks.shape[2], y + filter_halfedge + 1)

            best_label = sorted_labels[0]
            max_count = -1

            for i, label in enumerate(sorted_labels):
                if stacked_masks[i, x, y, z] == 0:
                    continue

                region = stacked_masks[i, x_min:x_max, y_min:y_max, z]
                count = np.sum(region > 0)

                if count > max_count:
                    max_count = count
                    best_label = label

            combined_mask[x, y, z] = best_label

    # Voxels with no foreground labels remain background (0)
    return combined_mask


def find_sample_dirs(root_dir: Union[str, Path]) -> List[Path]:
    """
    Find sample directories with the structure: Root → Site → Phase → {site}_{pid}_{phase}.
    
    Args:
        root_dir (Union[str, Path]): Root directory to search
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path: Path = Path(root_dir)
    sample_dirs: List[Path] = []

    # Iterate through all directories and find sample directories containing mask files
    for sample_dir in root_path.rglob('*_*_*'):
        if sample_dir.is_dir():
            # Check if the directory contains binary mask files
            mask_files = list(sample_dir.glob('*_mask_*.nii.gz'))
            if mask_files:
                # Validate the directory name format: {site}_{pid}_{phase}
                if re.match(r'^[^_]+_[^_]+_(pre|post)$', sample_dir.name):
                    sample_dirs.append(sample_dir)

    return sorted(sample_dirs)


def process_sample_dir(sample_dir: Path, root_path: Path, output_root_path: Path, kernel_size: Tuple[int, int], filter_halfedge: int, area_thresh: float, skip_existing: bool) -> Tuple[int, int, int]:
    """
    Process a single sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        root_path (Path): Root directory path for relative path calculation
        output_root_path (Path): Output root directory path
        kernel_size (Tuple[int, int]): Tuple of (kx, ky) for kernel size
        filter_halfedge (int): Half-edge of filter region for conflict resolution
        area_thresh (float): Area threshold in mm² for small regions
        skip_existing (bool): Skip processing if output multi-label mask already exists
        
    Returns:
        Tuple[int, int, int]: (binary_masks_processed, multi_mask_saved, error_count)
    """
    loader: LoadImage = LoadImage(image_only=False, dtype=None)
    saver: SaveImage = SaveImage(output_postfix='', output_dtype=np.uint8)

    sample_name: str = sample_dir.name

    # Find all binary mask files in the sample directory
    binary_mask_files: List[Path] = sorted(sample_dir.glob('*_mask_*.nii.gz'))

    if not binary_mask_files:
        return 0, 0, 0

    binary_masks_processed: int = 0
    multi_mask_saved: int = 0
    error_count: int = 0

    # Calculate output multi-label mask path early
    # Find the base filename for the multi-label mask
    first_mask = binary_mask_files[0]
    base_name_match = re.match(r'([^_]+_[^_]+_[^_]+)_mask_.*\.nii\.gz', first_mask.name)
    if base_name_match:
        base_name: str = base_name_match.group(1)
        
        # Calculate output path while preserving relative structure
        rel_path: Path = sample_dir.relative_to(root_path)
        output_dir: Path = output_root_path / rel_path
        output_multi_mask_path: Path = output_dir / f'{base_name}_mask.nii.gz'

        # Check if skip_existing is True and output multi-label mask exists
        if skip_existing and output_multi_mask_path.exists():
            print(f"Existing multi-label mask found for {sample_name}, checking binary masks...")
            
            # Check if all binary masks exist
            all_binary_masks_exist: bool = True
            missing_binary_masks: List[Path] = []
            
            for binary_mask_file in binary_mask_files:
                output_binary_path: Path = output_root_path / binary_mask_file.relative_to(root_path)
                if not output_binary_path.exists():
                    all_binary_masks_exist = False
                    missing_binary_masks.append(binary_mask_file)
            
            if all_binary_masks_exist:
                print(f"All binary masks exist for {sample_name}, skipping...")
                return 0, 0, 0
            else:
                print(f"Some binary masks missing for {sample_name}, regenerating from existing multi-label mask...")
                
                try:
                    # Load existing multi-label mask
                    combined_mask_data, mask_meta = loader(str(output_multi_mask_path))
                    combined_mask = combined_mask_data.numpy()
                    
                    # Export missing binary masks
                    for binary_mask_file in missing_binary_masks:
                        mask_match = re.match(r'.*_mask_(\d+)_([^_]+(?:_[^_]+)*)\.nii\.gz', binary_mask_file.name)
                        if mask_match:
                            label_index: int = int(mask_match.group(1))
                            label_name: str = mask_match.group(2)
                            
                            # Create binary mask from multi-label mask
                            binary_mask: np.ndarray = (combined_mask == label_index).astype(np.uint8)
                            
                            # Generate output filename
                            output_binary_mask: Path = output_root_path / binary_mask_file.relative_to(root_path)
                            
                            # Ensure output directory exists
                            output_binary_mask.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Save binary mask
                            saver(binary_mask, meta_data=mask_meta, filename=str(output_binary_mask).replace('.nii.gz', ''))
                            binary_masks_processed += 1
                except Exception as e:
                    print(f"Error exporting missing binary masks for {sample_name}: {str(e)}")
                    error_count += 1
                
                return binary_masks_processed, 0, error_count
    else:
        # Fallback: use sample name as base name
        base_name: str = sample_name
        rel_path: Path = sample_dir.relative_to(root_path)
        output_dir: Path = output_root_path / rel_path

    binary_masks_dict: Dict[int, np.ndarray] = {}
    binary_masks_meta_dict: Dict[int, Dict] = {}

    try:
        # Process all binary masks and build dictionary
        for binary_mask_file in binary_mask_files:
            # Extract label index from filename
            mask_match = re.match(r'.*_mask_(\d+)_.*\.nii\.gz', binary_mask_file.name)
            if not mask_match:
                continue

            label_index: int = int(mask_match.group(1))

            try:
                # Load mask data and metadata
                binary_mask_data, binary_mask_meta = loader(str(binary_mask_file))

                # Get pixel spacing
                pixdim: Optional[List[float]] = binary_mask_meta.get('pixdim', None)
                if pixdim is not None:
                    spacing: Tuple[float, float] = (pixdim[1], pixdim[2])
                else:
                    spacing: Tuple[float, float] = (1.0, 1.0)

                # Apply morphological operations slice by slice
                processed_binary_mask: np.ndarray = process_binary_mask_slice_by_slice(
                    binary_mask_data.numpy(), kernel_size, spacing, area_thresh
                )

                binary_masks_dict[label_index] = processed_binary_mask
                binary_masks_meta_dict[label_index] = binary_mask_meta

            except Exception as e:
                print(f"Error processing {binary_mask_file.name}: {str(e)}")
                error_count += 1

        # Generate multi-label mask if we have processed masks
        if binary_masks_dict:
            combined_mask: Optional[np.ndarray] = resolve_conflicts(binary_masks_dict, filter_halfedge)

            if combined_mask is not None:
                mask_meta: Dict = next(iter(binary_masks_meta_dict.values()))

                # Calculate output path while preserving relative structure
                output_multi_mask: Path = output_dir / f'{base_name}_mask.nii.gz'

                # Create output directory if it doesn't exist
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save multi-label mask
                saver(combined_mask, meta_data=mask_meta, filename=str(output_multi_mask).replace('.nii.gz', ''))
                multi_mask_saved += 1

                # Export binary masks based on the processed multi-label mask
                for binary_mask_file in binary_mask_files:
                    mask_match = re.match(r'.*_mask_(\d+)_([^_]+(?:_[^_]+)*)\.nii\.gz', binary_mask_file.name)
                    if mask_match:
                        label_index: int = int(mask_match.group(1))
                        label_name: str = mask_match.group(2)

                        # Create binary mask from multi-label mask
                        binary_mask: np.ndarray = (combined_mask == label_index).astype(np.uint8)

                        # Generate output filename
                        output_binary_mask: Path = output_dir / f'{base_name}_mask_{label_index:02d}_{label_name}.nii.gz'

                        # Save binary mask
                        saver(binary_mask, meta_data=mask_meta,
                              filename=str(output_binary_mask).replace('.nii.gz', ''))
                        binary_masks_processed += 1

    except Exception as e:
        print(f"Error processing sample {sample_name}: {str(e)}")
        error_count += 1

    return binary_masks_processed, multi_mask_saved, error_count


def process_root_dir(root_dir: Union[str, Path], output_dir: Union[str, Path], kernel_size: Tuple[int, int], filter_halfedge: int, area_thresh: float, skip_existing: bool) -> None:
    """
    Process all sample directories in the root directory recursively.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing sample directories
        output_dir (Union[str, Path]): Output root directory for processed masks
        kernel_size (Tuple[int, int]): Tuple of (kx, ky) for kernel size
        filter_halfedge (int): Half-edge of filter region for conflict resolution
        area_thresh (float): Area threshold in mm² for small regions
        skip_existing (bool): Skip processing if output multi-label mask already exists
    """
    root_path: Path = Path(root_dir)
    output_root_path: Path = Path(output_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    # Find all sample directories
    sample_dirs: List[Path] = find_sample_dirs(root_path)

    if not sample_dirs:
        print(f"Warning: No sample directories found in {root_dir}")
        return

    total_binary_masks_processed: int = 0
    total_multi_masks_saved: int = 0
    total_errors: int = 0

    # Process each sample directory
    for sample_dir in tqdm(sample_dirs, desc='Processing samples'):
        binary_masks_processed, multi_mask_saved, errors = process_sample_dir(
            sample_dir, root_path, output_root_path, kernel_size, filter_halfedge, area_thresh, skip_existing
        )
        total_binary_masks_processed += binary_masks_processed
        total_multi_masks_saved += multi_mask_saved
        total_errors += errors

    # Print summary
    print(f"\nProcessing completed!")
    print(f"Total binary masks processed: {total_binary_masks_processed}")
    print(f"Total multi-label masks saved: {total_multi_masks_saved}")
    print(f"Total errors encountered: {total_errors}")


def main() -> None:
    """
    Main function to orchestrate the morphological operations process.
    """
    # Parse command line arguments
    args = parse_args()

    # Print configuration
    print(f"Processing data from: {args.root_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Filter half-edge: {args.filter_halfedge}")
    print(f"Region area threshold: {args.region_area_thresh} mm²")
    print(f"Skip existing: {args.skip_existing}")

    # Process the root directory
    process_root_dir(
        args.root_dir,
        args.output_dir,
        tuple(args.kernel_size),
        args.filter_halfedge,
        args.region_area_thresh,
        args.skip_existing
    )

    print("Morphological open operations completed successfully!")


if __name__ == '__main__':
    main()
