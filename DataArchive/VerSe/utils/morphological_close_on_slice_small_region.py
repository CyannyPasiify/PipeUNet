# -*- coding: utf-8 -*-
"""
形态学闭运算脚本。使用argparse。接收-r/--root_dir指定根目录，根目录下包含若干层级的子目录，
第一级为VerSe*格式的子数据集目录，第二级为train/val/test子集目录，第三级为样本目录，
每个样本目录中包含形如*_mask.nii.gz的蒙版文件，
*_mask_<2位标签值zz><标签名(其中可能带下划线)>.nii.gz的二值蒙版文件。对每个样本读取其每个二值蒙版，
逐slice处理蒙版的每层，使用-k/--kernel_size（输入2个整数kx,ky）指定的kx*ky结构元执行闭运算，
将处理完毕的蒙版保持原相对路径保存到-o/--output_dir指定的新根目录下。将所有二值蒙版的前景解析为对应的标签值
然后合成为1张多值蒙版，对于冲突的前景点（有多个标签值），在前景包含此冲突点的二值蒙版中检查该点所在slice
周围-f/--filter_halfedge区域的体素，选择前景支持体素点数量较多的那一个标签值作为最终值。
对该slice的每个二值蒙版的背景区域进行连通域分析，找到背景的小连通域，背景小连通域选取闭运算结果（补洞），
其它区域保留原slice内容。对于处理后没有任何前景标签的体素点，检查在其所在slice中每个二值蒙版
filter_halfedge范围内的前景支持点数量，将此点标签设置为支持点数量最多的那个标签值。
"""

"""
Morphological Close Operation Script for VerSe Dataset

This script applies morphological closing operation to binary masks slice by slice to fill small holes,
and reconstructs the multi-label mask from processed binary masks, handling conflicts by comparing foreground
voxel counts in filter regions. Connected component analysis is performed on background regions of each 
binary mask slice to identify small connected regions. Small background regions use closing results (fill holes),
and other regions retain original content. For voxels with no foreground labels after processing, check 
foreground support point counts within filter_halfedge in each binary mask in the same slice, and assign 
the label with the most support points.

Parameters:
    -r, --root_dir: Root directory containing sample directories
    -o, --output_dir: Output root directory for processed masks
    -k, --kernel_size: Two integers (kx, ky) for kernel size
    -f, --filter_halfedge: Integer for filter half-edge (default: 3, i.e., 7x7 kernel), for conflict solving and non-value filling
    -t, --region_area_thresh: Area threshold in mm² for small regions (default: 10.0)
    --skip_existing: Skip processing if output multi-label mask already exists. If multi-label mask exists but some binary masks are missing, export missing binary masks from existing multi-label mask

Usage Examples:
    python morphological_close_on_slice_small_region.py -r /path/to/Verse -o /path/to/output -k 3 3
    python morphological_close_on_slice_small_region.py --root_dir /path/to/Verse --output_dir /path/to/output --kernel_size 3 3 --filter_halfedge 5 --region_area_thresh 10.0
"""

import argparse
import re
from typing import List
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.ndimage import binary_closing, label
from monai.transforms import LoadImage, SaveImage


def parse_args():
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir, output_dir, kernel_size, filter_halfedge, and region_area_thresh
    """
    parser = argparse.ArgumentParser(
        description='Apply morphological closing operation to binary masks slice by slice to fill small holes',
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
        help='Skip processing if output multi-label mask already exists'
    )

    return parser.parse_args()


def find_small_regions_mask(binary_mask_slice, spacing, area_thresh):
    """
    Find small connected regions in a binary mask slice.
    
    Args:
        binary_mask_slice (np.ndarray): Binary mask slice (2D array)
        spacing (tuple): Tuple of (dx, dy) for pixel spacing in mm
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


def apply_morphological_operations_slice(binary_mask_slice, kernel_size, spacing, area_thresh):
    """
    Apply morphological closing operation to small background connected regions in a single slice.
    Only fill small holes using closing operation for background regions.
    
    Args:
        binary_mask_slice (np.ndarray): Binary mask slice (2D array)
        kernel_size (tuple): Tuple of (kx, ky) for kernel size
        spacing (tuple): Tuple of (dx, dy) for pixel spacing in mm
        area_thresh (float): Area threshold in mm² for small regions
        
    Returns:
        np.ndarray: Processed binary mask slice
    """
    kx, ky = kernel_size
    structure = np.ones((kx, ky), dtype=bool)

    original_slice = binary_mask_slice.copy()

    # Only apply closing operation
    closed_slice = binary_closing(original_slice, structure=structure)

    # Only check for small background regions (holes)
    small_background_mask = find_small_regions_mask(1 - original_slice, spacing, area_thresh)

    processed_slice = original_slice.copy()

    # Fill small holes with closing results
    if np.any(small_background_mask):
        background_small_regions = small_background_mask & (1 - original_slice)
        np.putmask(processed_slice, background_small_regions, closed_slice)

    return processed_slice


def process_binary_mask_slice_by_slice(binary_mask_data, kernel_size, spacing, area_thresh):
    """
    Process binary mask slice by slice with morphological closing operations on small background regions only.
    
    Args:
        binary_mask_data (np.ndarray): Binary mask data (3D array)
        kernel_size (tuple): Tuple of (kx, ky) for kernel size
        spacing (tuple): Tuple of (dx, dy) for pixel spacing in mm
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


def resolve_conflicts(binary_masks_dict, filter_halfedge):
    """
    Resolve conflicts where multiple labels claim the same voxel using vectorized operations.
    For voxels with no foreground labels after processing, check foreground support point counts
    within filter_halfedge in each binary mask in the same slice, and assign the label with the most support points.
    
    Args:
        binary_masks_dict (dict): Dictionary mapping label values to binary mask arrays
        filter_halfedge (int): Half-edge of filter region for conflict resolution (actual kernel size is 2*filter_halfedge+1)
        
    Returns:
        np.ndarray: Multi-label mask with resolved conflicts
    """
    if not binary_masks_dict:
        return None

    sorted_labels = sorted(binary_masks_dict.keys())
    binary_masks = [binary_masks_dict[label] for label in sorted_labels]

    stacked_masks = np.stack(binary_masks, axis=0)

    label_counts = np.sum(stacked_masks, axis=0)

    no_conflict_mask = (label_counts == 1)
    conflict_mask = (label_counts > 1)
    no_label_mask = (label_counts == 0)

    combined_mask = np.zeros(stacked_masks.shape[1:], dtype=np.uint8)

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

        # Check which labels are present at this specific conflict point
        present_labels = []
        for i, label in enumerate(sorted_labels):
            if stacked_masks[i, x, y, z] == 1:
                present_labels.append((i, label))

        # Filter out label 00 (value 0) if other non-zero labels are present at this point
        has_non_zero_labels = any(label != 0 for _, label in present_labels)
        if has_non_zero_labels:
            present_labels = [item for item in present_labels if item[1] != 0]

        best_label = present_labels[0][1] if present_labels else 0
        max_count = -1

        for i, label in present_labels:
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


def find_sample_dirs(root_dir):
    """
    Find sample directories with specified structure: Root → VerSe* → train/val/test → Sample directory.
    
    Args:
        root_dir (Path): Root directory to search
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path: Path = Path(root_dir)
    sample_dirs: List[Path] = []

    # Iterate through first-level VerSe* subdataset directories
    for sub_dataset_dir in root_path.glob('VerSe*'):
        if not sub_dataset_dir.is_dir():
            continue

        # Iterate through second-level train/val/test subset directories
        for subset_dir in sub_dataset_dir.glob('*'):
            if not subset_dir.is_dir():
                continue

            # Iterate through third-level sample directories
            for sample_dir in subset_dir.glob('*'):
                if sample_dir.is_dir():
                    # Check if the directory contains mask files
                    mask_files = list(sample_dir.glob('*_mask_*.nii.gz'))
                    if mask_files:
                        sample_dirs.append(sample_dir)

    return sorted(sample_dirs)


def process_sample_dir(sample_dir, root_path, output_root_path, kernel_size, filter_halfedge, area_thresh, skip_existing):
    """
    Process a single sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        root_path (Path): Root directory path for relative path calculation
        output_root_path (Path): Output root directory path
        kernel_size (tuple): Tuple of (kx, ky) for kernel size
        filter_halfedge (int): Half-edge of filter region for conflict resolution (actual kernel size is 2*filter_halfedge+1)
        area_thresh (float): Area threshold in mm² for small regions
        skip_existing (bool): Skip processing if output multi-label mask already exists
        
    Returns:
        tuple: (binary_masks_processed, multi_mask_saved, error_count)
    """
    loader = LoadImage(image_only=False, dtype=None)
    saver = SaveImage(output_postfix='', output_dtype=np.uint8)

    sample_name = sample_dir.name

    binary_mask_files = sorted(sample_dir.glob('*_mask_*.nii.gz'))

    if not binary_mask_files:
        return 0, 0, 0

    binary_masks_processed = 0
    multi_mask_saved = 0
    error_count = 0

    try:
        # Get sample ID for output filename
        sample_id = sample_dir.name
        # Find original mask filename pattern
        original_mask_files = list(sample_dir.glob('*_mask.nii.gz'))
        if original_mask_files:
            original_mask_name = original_mask_files[0].name
            id_part = original_mask_name[:-len('_mask.nii.gz')]
            output_filestem = (output_root_path / original_mask_files[0]
                               .relative_to(root_path).parent / f'{id_part}_mask')
        else:
            # Fallback to sample name
            # Try to find the first binary mask file to get the path structure
            if binary_mask_files:
                first_mask_file = binary_mask_files[0]
                output_filestem = (output_root_path / first_mask_file
                                   .relative_to(root_path).parent / f'{sample_id}_mask')
            else:
                return 0, 0, 0

        output_multi_mask_path = Path(f'{output_filestem}.nii.gz')

        # Check if skip_existing is True and output multi-label mask exists
        if skip_existing and output_multi_mask_path.exists():
            print(f"Existing multi-label mask found for {sample_name}, checking binary masks...")
            
            # Check if all binary masks exist
            all_binary_masks_exist = True
            missing_binary_masks = []
            
            for binary_mask_file in binary_mask_files:
                rel_path = binary_mask_file.relative_to(root_path)
                output_binary_path = output_root_path / rel_path
                if not output_binary_path.exists():
                    all_binary_masks_exist = False
                    missing_binary_masks.append(binary_mask_file)
            
            if all_binary_masks_exist:
                print(f"All binary masks exist for {sample_name}, skipping...")
                return 0, 0, 0
            else:
                print(f"Some binary masks missing for {sample_name}, regenerating from existing multi-label mask...")
                # Load existing multi-label mask
                combined_mask, mask_meta = loader(str(output_multi_mask_path))
                combined_mask = combined_mask.numpy()
                
                # Export missing binary masks
                for binary_mask_file in missing_binary_masks:
                    # Extract label value and name from filename
                    binary_match = re.match(r'.+_mask_([0-9a-fA-F]{2})_(.+?)\.nii\.gz', binary_mask_file.name)
                    if not binary_match:
                        # Try alternative pattern for VerSe dataset
                        binary_match = re.match(r'.+_mask_(\d{2})_(.+?)\.nii\.gz', binary_mask_file.name)
                        if not binary_match:
                            continue

                    label_value = int(binary_match.group(1))

                    # Create binary mask from multi-label mask
                    binary_mask = (combined_mask == label_value).astype(np.uint8)

                    # Generate output filename
                    rel_path = binary_mask_file.relative_to(root_path)
                    output_file_path = output_root_path / rel_path
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save binary mask
                    saver(binary_mask, meta_data=mask_meta, filename=str(output_file_path).replace('.nii.gz', ''))
                    binary_masks_processed += 1
                
                return binary_masks_processed, 0, 0

        # If we reach here, either skip_existing is False or multi-label mask doesn't exist
        binary_masks_dict = {}
        binary_masks_meta_dict = {}

        # First process all binary masks and build dictionary, but don't save them yet
        for binary_mask_file in binary_mask_files:
            # Extract label value and name from filename
            binary_match = re.match(r'.+_mask_([0-9a-fA-F]{2})_(.+?)\.nii\.gz', binary_mask_file.name)
            if not binary_match:
                # Try alternative pattern for VerSe dataset
                binary_match = re.match(r'.+_mask_(\d{2})_(.+?)\.nii\.gz', binary_mask_file.name)
                if not binary_match:
                    continue

            label_value = int(binary_match.group(1))

            try:
                binary_mask_data, binary_mask_meta = loader(str(binary_mask_file))

                pixdim = binary_mask_meta.get('pixdim', None)
                if pixdim is not None:
                    spacing = (pixdim[1], pixdim[2])
                else:
                    spacing = (1.0, 1.0)

                # Only apply morphological operations to non-zero labels
                if label_value != 0:
                    processed_binary_mask = process_binary_mask_slice_by_slice(binary_mask_data.numpy(), kernel_size,
                                                                               spacing, area_thresh)
                else:
                    # Skip morphological operations for label 00, use original mask
                    processed_binary_mask = binary_mask_data.numpy().astype(np.uint8)

                binary_masks_dict[label_value] = processed_binary_mask
                binary_masks_meta_dict[label_value] = binary_mask_meta

            except Exception as e:
                print(f"Error processing {binary_mask_file.name}: {str(e)}")
                error_count += 1

        if binary_masks_dict:
            combined_mask = resolve_conflicts(binary_masks_dict, filter_halfedge)

            if combined_mask is not None:
                mask_meta = next(iter(binary_masks_meta_dict.values()))

                # First save the combined multi-label mask
                output_filestem.parent.mkdir(parents=True, exist_ok=True)
                saver(combined_mask, meta_data=mask_meta, filename=output_filestem)
                multi_mask_saved += 1

                # Then export binary masks based on the processed multi-label mask
                for binary_mask_file in binary_mask_files:
                    # Extract label value and name from filename
                    binary_match = re.match(r'.+_mask_([0-9a-fA-F]{2})_(.+?)\.nii\.gz', binary_mask_file.name)
                    if not binary_match:
                        binary_match = re.match(r'.+_mask_(\d{2})_(.+?)\.nii\.gz', binary_mask_file.name)
                        if not binary_match:
                            continue

                    label_value = int(binary_match.group(1))

                    # Create binary mask from multi-label mask
                    binary_mask = (combined_mask == label_value).astype(np.uint8)

                    # Generate output filename
                    rel_path = binary_mask_file.relative_to(root_path)
                    output_file_path = output_root_path / rel_path
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save binary mask
                    saver(binary_mask, meta_data=mask_meta, filename=str(output_file_path).replace('.nii.gz', ''))
                    binary_masks_processed += 1

    except Exception as e:
        print(f"Error processing sample {sample_name}: {str(e)}")
        error_count += 1

    return binary_masks_processed, multi_mask_saved, error_count


def process_root_dir(root_dir, output_dir, kernel_size, filter_halfedge, area_thresh, skip_existing):
    """
    Process all sample directories in the root directory recursively.
    
    Args:
        root_dir (str or Path): Root directory containing sample directories
        output_dir (str or Path): Output root directory for processed masks
        kernel_size (tuple): Tuple of (kx, ky) for kernel size
        filter_halfedge (int): Half-edge of filter region for conflict resolution (actual kernel size is 2*filter_halfedge+1)
        area_thresh (float): Area threshold in mm² for small regions
        skip_existing (bool): Skip processing if output multi-label mask already exists
    """
    root_path = Path(root_dir)
    output_root_path = Path(output_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    sample_dirs = find_sample_dirs(root_path)

    if not sample_dirs:
        print(f"Warning: No sample directories containing mask files found in {root_dir}")
        return

    total_binary_masks_processed = 0
    total_multi_masks_saved = 0
    total_errors = 0

    for sample_dir in tqdm(sample_dirs, desc='Processing samples'):
        binary_masks_processed, multi_mask_saved, errors = process_sample_dir(
            sample_dir, root_path, output_root_path, kernel_size, filter_halfedge, area_thresh, skip_existing
        )
        total_binary_masks_processed += binary_masks_processed
        total_multi_masks_saved += multi_mask_saved
        total_errors += errors

    print(f"\nProcessing completed!")
    print(f"Total binary masks processed: {total_binary_masks_processed}")
    print(f"Total multi-label masks saved: {total_multi_masks_saved}")
    print(f"Total errors encountered: {total_errors}")


def main():
    """
    Main function to orchestrate the morphological operations process.
    """
    args = parse_args()

    print(f"Processing data from: {args.root_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Filter half-edge: {args.filter_halfedge}")
    print(f"Region area threshold: {args.region_area_thresh} mm²")
    print(f"Skip existing: {args.skip_existing}")

    process_root_dir(args.root_dir, args.output_dir, tuple(args.kernel_size), args.filter_halfedge,
                     args.region_area_thresh, args.skip_existing)

    print("Morphological closing operations completed successfully!")


if __name__ == '__main__':
    main()
