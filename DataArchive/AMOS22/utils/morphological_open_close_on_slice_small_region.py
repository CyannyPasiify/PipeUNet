# -*- coding: utf-8 -*-
"""
形态学开闭运算脚本。使用argparse。接收-r/--root_dir指定根目录，根目录下包含若干层级的子目录，
最低层为以amos_xxxx*命名的样本目录，每个样本目录中包含形如*_mask.nii.gz的蒙版文件，
*_mask_<2位标签值zz><标签名(其中可能带下划线)>.nii.gz的二值蒙版文件。对每个样本读取其每个二值蒙版，
逐slice处理蒙版的每层，使用-k/--kernel_size（输入2个整数kx,ky）指定的kx*ky结构元依次执行闭运算和开运算，
将处理完毕的蒙版保持原相对路径保存到-o/--output_dir指定的新根目录下。将所有二值蒙版的前景解析为对应的标签值
然后合成为1张多值蒙版，对于冲突的前景点（有多个标签值），在前景包含此冲突点的二值蒙版中检查该点所在slice
周围-f/--filter_halfedge区域的体素，选择前景支持体素点数量较多的那一个标签值作为最终值。
首先分别对原slice执行闭运算和开运算，然后对该slice的每个二值蒙版的前景和背景区域进行连通域分析，
找到前景和背景的小连通域，前景小连通域选取开运算结果（删除孤岛），背景小连通域选取闭运算结果（补洞），
其它区域保留原slice内容。对于处理后没有任何前景标签的体素点，检查在其所在slice中每个二值蒙版
filter_halfedge范围内的前景支持点数量，将此点标签设置为支持点数量最多的那个标签值。
"""

"""
Morphological Open/Close Operation Script for AMOS22 Dataset

This script applies morphological closing and opening operations to binary masks slice by slice,
then reconstructs the multi-label mask from processed binary masks, handling conflicts by
comparing foreground voxel counts in filter regions. First, separate closing and opening operations
are applied to the original slice. Then, connected component analysis is performed on foreground
and background regions of each binary mask slice to identify small connected regions.
Small foreground regions use opening results (remove islands), small background regions use closing
results (fill holes), and other regions retain original slice content. For voxels with no foreground
labels after processing, check foreground support point counts within filter_halfedge in each binary mask
in the same slice, and assign the label with the most support points.

Parameters:
    -r, --root_dir: Root directory containing sample directories at any depth
    -o, --output_dir: Output root directory for processed masks
    -k, --kernel_size: Two integers (kx, ky) for kernel size
    -f, --filter_halfedge: Integer for filter half-edge (default: 3, i.e., 7x7 kernel), for conflict solving and non-value filling
    -t, --region_area_thresh: Area threshold in mm² for small regions (default: 10.0)

Usage Examples:
    python morphological_open_close_on_slice_small_region.py -r /path/to/input -o /path/to/output -k 3 3
    python morphological_open_close_on_slice_small_region.py --root_dir /path/to/input --output_dir /path/to/output --kernel_size 3 3 --filter_halfedge 5 --region_area_thresh 10.0
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label
from monai.transforms import LoadImage, SaveImage
from typing import List, Dict, Tuple, Optional, Union


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir, output_dir, kernel_size, filter_halfedge, and region_area_thresh
    """
    parser = argparse.ArgumentParser(
        description='Apply morphological open/close operations to binary masks slice by slice',
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
        help='Root directory containing sample directories at any depth'
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
    Apply morphological closing and opening operations to small connected regions in a single slice.
    First apply separate closing and opening operations, then use connected component analysis
    to identify small regions. Small foreground regions use opening results (remove islands),
    small background regions use closing results (fill holes), and other regions retain original content.
    
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

    closed_slice = binary_closing(original_slice, structure=structure)
    opened_slice = binary_opening(original_slice, structure=structure)

    small_foreground_mask = find_small_regions_mask(original_slice, spacing, area_thresh)
    small_background_mask = find_small_regions_mask(1 - original_slice, spacing, area_thresh)

    processed_slice = original_slice.copy()

    if np.any(small_foreground_mask):
        foreground_small_regions = small_foreground_mask & (original_slice == 1)
        np.putmask(processed_slice, foreground_small_regions, opened_slice)

    if np.any(small_background_mask):
        background_small_regions = small_background_mask & (original_slice == 0)
        np.putmask(processed_slice, background_small_regions, closed_slice)

    return processed_slice


def process_binary_mask_slice_by_slice(binary_mask_data: np.ndarray, kernel_size: Tuple[int, int], spacing: Tuple[float, float], area_thresh: float) -> np.ndarray:
    """
    Process binary mask slice by slice with morphological operations on small regions only.
    
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
    For voxels with no foreground labels after processing, check foreground support point counts
    within filter_halfedge in each binary mask in the same slice, and assign the label with the most support points.
    
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
    no_label_mask: np.ndarray = (label_counts == 0)

    combined_mask: np.ndarray = np.zeros(stacked_masks.shape[1:], dtype=np.uint8)

    for i, label in enumerate(sorted_labels):
        label_mask = stacked_masks[i] & no_conflict_mask
        np.putmask(combined_mask, label_mask, label)

    conflict_count = np.sum(conflict_mask)

    if conflict_count > 0:
        print(f"Found {conflict_count} conflict voxels, resolving...")

    if not np.any(conflict_mask) and not np.any(no_label_mask):
        return combined_mask

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

    no_label_count = np.sum(no_label_mask)

    if no_label_count > 0:
        print(f"Found {no_label_count} voxels with no foreground labels, assigning based on support...")

    if not np.any(no_label_mask):
        return combined_mask

    no_label_coords = np.argwhere(no_label_mask)

    for x, y, z in no_label_coords:
        x_min = max(0, x - filter_halfedge)
        x_max = min(stacked_masks.shape[1], x + filter_halfedge + 1)
        y_min = max(0, y - filter_halfedge)
        y_max = min(stacked_masks.shape[2], y + filter_halfedge + 1)

        best_label = sorted_labels[0]
        max_count = -1

        for i, label in enumerate(sorted_labels):
            region = stacked_masks[i, x_min:x_max, y_min:y_max, z]
            count = np.sum(region > 0)

            if count > max_count:
                max_count = count
                best_label = label

        if max_count > 0:
            combined_mask[x, y, z] = best_label

    return combined_mask


def find_sample_dirs(root_dir: Union[str, Path]) -> List[Path]:
    """
    Recursively find all sample directories named amos_xxxx*.
    
    Args:
        root_dir (Union[str, Path]): Root directory to search
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path = Path(root_dir)
    sample_dirs = []

    for path in root_path.rglob('amos_*'):
        if path.is_dir():
            sample_dirs.append(path)

    return sorted(sample_dirs)


def process_sample_dir(sample_dir: Path, root_path: Path, output_root_path: Path, kernel_size: Tuple[int, int], filter_halfedge: int, area_thresh: float) -> Tuple[int, int, int]:
    """
    Process a single sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        root_path (Path): Root directory path for relative path calculation
        output_root_path (Path): Output root directory path
        kernel_size (Tuple[int, int]): Tuple of (kx, ky) for kernel size
        filter_halfedge (int): Half-edge of filter region for conflict resolution (actual kernel size is 2*filter_halfedge+1)
        area_thresh (float): Area threshold in mm² for small regions
        
    Returns:
        Tuple[int, int, int]: (binary_masks_processed, multi_mask_saved, error_count)
    """
    loader: LoadImage = LoadImage(image_only=False, dtype=None)
    saver: SaveImage = SaveImage(output_postfix='', output_dtype=np.uint8)

    sample_name: str = sample_dir.name

    match = re.match(r'amos_(\d{4})', sample_name)
    if not match:
        return 0, 0, 0

    sample_id: str = match.group(1)

    binary_mask_files: List[Path] = sorted(sample_dir.glob('*_mask_*.nii.gz'))

    if not binary_mask_files:
        return 0, 0, 0

    binary_masks_processed: int = 0
    multi_mask_saved: int = 0
    error_count: int = 0

    binary_masks_dict: Dict[int, np.ndarray] = {}
    binary_masks_meta_dict: Dict[int, Dict] = {}

    try:
        # First process all binary masks and build dictionary, but don't save them yet
        for binary_mask_file in binary_mask_files:
            binary_match = re.match(r'.+_mask_(\d{2})_(.+)\.nii\.gz', binary_mask_file.name)
            if not binary_match:
                continue

            label_value: int = int(binary_match.group(1))

            try:
                binary_mask_data, binary_mask_meta = loader(str(binary_mask_file))

                pixdim: Optional[List[float]] = binary_mask_meta.get('pixdim', None)
                if pixdim is not None:
                    spacing: Tuple[float, float] = (pixdim[1], pixdim[2])
                else:
                    spacing: Tuple[float, float] = (1.0, 1.0)

                processed_binary_mask: np.ndarray = process_binary_mask_slice_by_slice(binary_mask_data.numpy(), kernel_size,
                                                                           spacing, area_thresh)

                binary_masks_dict[label_value] = processed_binary_mask
                binary_masks_meta_dict[label_value] = binary_mask_meta

            except Exception as e:
                print(f"Error processing {binary_mask_file.name}: {str(e)}")
                error_count += 1

        if binary_masks_dict:
            combined_mask: Optional[np.ndarray] = resolve_conflicts(binary_masks_dict, filter_halfedge)

            if combined_mask is not None:
                mask_meta: Dict = next(iter(binary_masks_meta_dict.values()))
                # Find original mask filename pattern
                original_mask_files: List[Path] = list(sample_dir.glob('*_mask.nii.gz'))
                output_filestem: Path = (output_root_path / original_mask_files[0]
                                   .relative_to(root_path).parent / f'amos_{sample_id}_mask')
                
                # First save the combined multi-label mask
                output_filestem.parent.mkdir(parents=True, exist_ok=True)
                saver(combined_mask, meta_data=mask_meta, filename=output_filestem)
                multi_mask_saved += 1
                
                # Then export binary masks based on the processed multi-label mask
                for binary_mask_file in binary_mask_files:
                    binary_match = re.match(r'.+_mask_(\d{2})_(.+)\.nii\.gz', binary_mask_file.name)
                    if not binary_match:
                        continue
                    
                    label_value: int = int(binary_match.group(1))
                    
                    # Create binary mask from multi-label mask
                    binary_mask: np.ndarray = (combined_mask == label_value).astype(np.uint8)
                    
                    # Generate output filename
                    rel_path: Path = binary_mask_file.relative_to(root_path)
                    output_file_path: Path = output_root_path / rel_path
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save binary mask
                    saver(binary_mask, meta_data=mask_meta,
                          filename=str(output_file_path).replace('.nii.gz', ''))
                    binary_masks_processed += 1

    except Exception as e:
        print(f"Error processing sample {sample_name}: {str(e)}")
        error_count += 1

    return binary_masks_processed, multi_mask_saved, error_count


def process_root_dir(root_dir: Union[str, Path], output_dir: Union[str, Path], kernel_size: Tuple[int, int], filter_halfedge: int, area_thresh: float) -> None:
    """
    Process all sample directories in the root directory recursively.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing sample directories at any depth
        output_dir (Union[str, Path]): Output root directory for processed masks
        kernel_size (Tuple[int, int]): Tuple of (kx, ky) for kernel size
        filter_halfedge (int): Half-edge of filter region for conflict resolution (actual kernel size is 2*filter_halfedge+1)
        area_thresh (float): Area threshold in mm² for small regions
    """
    root_path: Path = Path(root_dir)
    output_root_path: Path = Path(output_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    sample_dirs: List[Path] = find_sample_dirs(root_path)

    if not sample_dirs:
        print(f"Warning: No sample directories (amos_xxxx*) found in {root_dir}")
        return

    total_binary_masks_processed: int = 0
    total_multi_masks_saved: int = 0
    total_errors: int = 0

    for sample_dir in tqdm(sample_dirs, desc='Processing samples'):
        binary_masks_processed, multi_mask_saved, errors = process_sample_dir(
            sample_dir, root_path, output_root_path, kernel_size, filter_halfedge, area_thresh
        )
        total_binary_masks_processed += binary_masks_processed
        total_multi_masks_saved += multi_mask_saved
        total_errors += errors

    print(f"\nProcessing completed!")
    print(f"Total binary masks processed: {total_binary_masks_processed}")
    print(f"Total multi-label masks saved: {total_multi_masks_saved}")
    print(f"Total errors encountered: {total_errors}")


def main() -> None:
    """
    Main function to orchestrate the morphological operations process.
    """
    args = parse_args()

    print(f"Processing data from: {args.root_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Filter half-edge: {args.filter_halfedge}")
    print(f"Region area threshold: {args.region_area_thresh} mm²")

    process_root_dir(args.root_dir, args.output_dir, tuple(args.kernel_size), args.filter_halfedge,
                     args.region_area_thresh)

    print("Morphological operations completed successfully!")


if __name__ == '__main__':
    main()
