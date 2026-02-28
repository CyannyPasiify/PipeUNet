# -*- coding: utf-8 -*-
"""
Binary Mask Union and Multi-label Mask Creation Script

This script processes binary masks from two root directories, takes union of corresponding binary masks,
resolves conflicts, creates multi-label masks, and exports both multi-label and binary masks.

Parameters:
    -ro, --root_origin: Root directory for origin masks (d:\Datasets\VerSe\grouped)
    -rd, --root_derived: Root directory for derived masks (d:\Datasets\VerSe\grouped_morph)
    -o, --output_dir: Output directory to save merged results

Usage Example:
    python bar.py -ro d:\Datasets\VerSe\grouped -rd d:\Datasets\VerSe\grouped_morph -o d:\Datasets\VerSe\merged_results
"""

import argparse
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from monai.transforms import LoadImage, SaveImage


def parse_args():
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_origin, root_derived, output_dir, and skip_existing
    """
    parser = argparse.ArgumentParser(
        description='Binary Mask Union and Multi-label Mask Creation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -ro d:\Datasets\VerSe\grouped -rd d:\Datasets\VerSe\grouped_morph -o d:\Datasets\VerSe\merged_results
  %(prog)s -ro d:\Datasets\VerSe\grouped -rd d:\Datasets\VerSe\grouped_morph -o d:\Datasets\VerSe\merged_results --skip_existing
        """
    )

    parser.add_argument(
        '-ro', '--root_origin',
        type=str,
        required=True,
        help='Root directory for origin masks'
    )

    parser.add_argument(
        '-rd', '--root_derived',
        type=str,
        required=True,
        help='Root directory for derived masks'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Output directory to save merged results'
    )

    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip processing if output multi-label mask already exists'
    )

    return parser.parse_args()


def find_sample_dirs(root_dir):
    """
    Find sample directories with specified structure: Root → VerSe* → train/val/test → Sample directory.
    
    Args:
        root_dir (Path): Root directory to search
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path = Path(root_dir)
    sample_dirs = []

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


def get_paired_sample_dirs(root_origin, root_derived):
    """
    Get paired sample directories from both root directories.
    
    Args:
        root_origin (Path): Root directory for origin masks
        root_derived (Path): Root directory for derived masks
        
    Returns:
        List[tuple]: List of tuples containing (origin_sample_dir, derived_sample_dir)
    """
    origin_samples = find_sample_dirs(root_origin)
    derived_samples = find_sample_dirs(root_derived)
    
    # Create mapping from relative path to sample directory for both roots
    origin_rel_map = {sample.relative_to(root_origin): sample for sample in origin_samples}
    derived_rel_map = {sample.relative_to(root_derived): sample for sample in derived_samples}
    
    # Find common relative paths
    common_rel_paths = set(origin_rel_map.keys()) & set(derived_rel_map.keys())
    
    # Create paired list
    paired_samples = []
    for rel_path in sorted(common_rel_paths):
        paired_samples.append((origin_rel_map[rel_path], derived_rel_map[rel_path]))
    
    return paired_samples


def process_sample_dir(origin_sample_dir, derived_sample_dir, root_origin, root_derived, output_dir, skip_existing):
    """
    Process a single sample directory pair.
    
    Args:
        origin_sample_dir (Path): Path to the origin sample directory
        derived_sample_dir (Path): Path to the derived sample directory
        root_origin (Path): Root directory for origin masks
        root_derived (Path): Root directory for derived masks
        output_dir (Path): Output directory to save merged results
        skip_existing (bool): Skip processing if output multi-label mask already exists
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        # Determine output paths while preserving directory structure first to check for skip_existing
        rel_path = origin_sample_dir.relative_to(root_origin)
        output_sample_dir = output_dir / rel_path
        output_sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the base filename for the mask
        origin_mask_files = list(origin_sample_dir.glob('*_mask.nii.gz'))
        derived_mask_files = list(derived_sample_dir.glob('*_mask.nii.gz'))
        
        if origin_mask_files:
            base_filename = origin_mask_files[0].name
        elif derived_mask_files:
            base_filename = derived_mask_files[0].name
        else:
            # Extract base name from binary mask filenames
            binary_files = list(origin_sample_dir.glob('*_mask_*.nii.gz')) + list(derived_sample_dir.glob('*_mask_*.nii.gz'))
            if binary_files:
                binary_match = re.match(r'(.+)_mask_\d{2}_.*\.nii\.gz', binary_files[0].name)
                if binary_match:
                    base_filename = f"{binary_match.group(1)}_mask.nii.gz"
                else:
                    base_filename = f"{origin_sample_dir.name}_mask.nii.gz"
            else:
                base_filename = f"{origin_sample_dir.name}_mask.nii.gz"
        
        # Save merged multi-label mask
        output_multi_mask_path = output_sample_dir / base_filename
        
        # Check if skip_existing is True and output multi-label mask already exists
        if skip_existing and output_multi_mask_path.exists():
            print(f"Existing multi-label mask found for {origin_sample_dir.name}, skipping...")
            return True, None
        
        # Load binary masks from both directories
        origin_masks = load_binary_masks(origin_sample_dir)
        derived_masks = load_binary_masks(derived_sample_dir)
        
        # Take union of corresponding binary masks (ignoring label 00)
        union_masks = {}  
        all_mask_data = {}  # Store (mask_array, meta_data, label_name) for each label
        
        # Get all unique label indices (excluding 00)
        all_labels = set(origin_masks.keys()) | set(derived_masks.keys())
        all_labels = {label for label in all_labels if label != 0}  # Ignore label 00
        
        # Process each label
        for label_index in all_labels:
            origin_mask_data = origin_masks.get(label_index, None)
            derived_mask_data = derived_masks.get(label_index, None)
            
            if origin_mask_data is not None and derived_mask_data is not None:
                # Both masks exist, take union
                origin_mask = origin_mask_data[0]
                derived_mask = derived_mask_data[0]
                union_mask = np.logical_or(origin_mask, derived_mask).astype(np.uint8)
                # Use origin metadata and label name
                mask_meta = origin_mask_data[1]
                label_name = origin_mask_data[2]
            elif origin_mask_data is not None:
                # Only origin mask exists
                union_mask = origin_mask_data[0]
                mask_meta = origin_mask_data[1]
                label_name = origin_mask_data[2]
            else:
                # Only derived mask exists
                union_mask = derived_mask_data[0]
                mask_meta = derived_mask_data[1]
                label_name = derived_mask_data[2]
            
            union_masks[label_index] = union_mask
            all_mask_data[label_index] = (union_mask, mask_meta, label_name)
        
        if not union_masks:
            return False, f"No valid binary masks found for {origin_sample_dir.name}"
        
        # Get metadata from one of the masks
        mask_meta = next(iter(all_mask_data.values()))[1] if all_mask_data else None
        
        # Create multi-label mask with conflict resolution
        multi_mask = create_multi_label_mask(union_masks)
        
        if multi_mask is None:
            return False, f"Failed to create multi-label mask for {origin_sample_dir.name}"
        save_multi_label_mask(multi_mask, mask_meta, output_multi_mask_path)
        
        # Export binary masks from the merged multi-label mask
        export_binary_masks(multi_mask, mask_meta, output_sample_dir, origin_masks, derived_masks)
        
        return True, None
        
    except Exception as e:
        return False, f"Error processing {origin_sample_dir.name}: {str(e)}"


def load_binary_masks(sample_dir):
    """
    Load binary masks from a sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        
    Returns:
        dict: Dictionary mapping label indices to (binary_mask_array, metadata)
    """
    loader = LoadImage(image_only=False, dtype=None)
    binary_masks = {}
    
    # Get all binary mask files
    binary_mask_files = sorted(sample_dir.glob('*_mask_*.nii.gz'))
    
    for binary_mask_file in binary_mask_files:
        # Extract label index and name from filename
        binary_match = re.match(r'.+_mask_([0-9a-fA-F]{2})_(.+?)\.nii\.gz', binary_mask_file.name)
        if not binary_match:
            # Try alternative pattern
            binary_match = re.match(r'.+_mask_(\d{2})_(.+?)\.nii\.gz', binary_mask_file.name)
            if not binary_match:
                continue
        
        label_index = int(binary_match.group(1))
        label_name = binary_match.group(2)
        
        try:
            binary_mask_data, binary_mask_meta = loader(str(binary_mask_file))
            binary_masks[label_index] = (binary_mask_data.numpy().astype(np.uint8), binary_mask_meta, label_name)
        except Exception as e:
            print(f"Error loading {binary_mask_file.name}: {str(e)}")
            continue
    
    return binary_masks


def create_multi_label_mask(binary_masks_dict):
    """
    Create a multi-label mask from binary masks, resolving conflicts.
    
    Args:
        binary_masks_dict (dict): Dictionary mapping label indices to binary mask arrays
        
    Returns:
        np.ndarray: Multi-label mask array
    """
    if not binary_masks_dict:
        return None
    
    # Get the first mask to determine shape
    first_mask = next(iter(binary_masks_dict.values()))
    multi_mask = np.zeros(first_mask.shape, dtype=np.uint8)
    
    # First, apply all non-conflicting regions
    label_count = np.zeros(first_mask.shape, dtype=int)
    
    for label in binary_masks_dict.keys():
        mask = binary_masks_dict[label]
        label_count[mask > 0] += 1
    
    # Apply non-conflicting regions
    for label in binary_masks_dict.keys():
        mask = binary_masks_dict[label]
        non_conflict_mask = (mask > 0) & (label_count == 1)
        multi_mask[non_conflict_mask] = label
    
    # Resolve conflicts
    conflict_mask = (label_count > 1)
    if np.any(conflict_mask):
        print(f"Found {np.sum(conflict_mask)} conflict voxels, resolving...")
        resolve_mask_conflicts(binary_masks_dict, multi_mask, conflict_mask)
    
    return multi_mask


def resolve_mask_conflicts(binary_masks_dict, multi_mask, conflict_mask):
    """
    Resolve conflicts in the multi-label mask by selecting the label with the most support in the neighborhood.
    
    Args:
        binary_masks_dict (dict): Dictionary mapping label indices to binary mask arrays
        multi_mask (np.ndarray): Multi-label mask array to modify
        conflict_mask (np.ndarray): Boolean mask indicating conflict voxels
    """
    conflict_coords = np.argwhere(conflict_mask)
    filter_halfedge = 3  # Use a 7x7x7 neighborhood
    
    for coord in conflict_coords:
        x, y, z = coord
        x_min = max(0, x - filter_halfedge)
        x_max = min(multi_mask.shape[0], x + filter_halfedge + 1)
        y_min = max(0, y - filter_halfedge)
        y_max = min(multi_mask.shape[1], y + filter_halfedge + 1)
        z_min = max(0, z - filter_halfedge)
        z_max = min(multi_mask.shape[2], z + filter_halfedge + 1)
        
        best_label = None
        max_support = -1
        
        for label in binary_masks_dict.keys():
            mask = binary_masks_dict[label]
            if mask[x, y, z] == 0:
                continue
            
            # Count support in the neighborhood
            neighborhood = mask[x_min:x_max, y_min:y_max, z_min:z_max]
            support = np.sum(neighborhood > 0)
            
            if support > max_support:
                max_support = support
                best_label = label
        
        if best_label is not None:
            multi_mask[x, y, z] = best_label


def save_multi_label_mask(multi_mask, mask_meta, output_path):
    """
    Save a multi-label mask to file.
    
    Args:
        multi_mask (np.ndarray): Multi-label mask array
        mask_meta (dict): Metadata for the mask
        output_path (Path): Output path for the mask
    """
    saver = SaveImage(output_postfix='', output_dtype=np.uint8)
    saver(multi_mask, meta_data=mask_meta, filename=str(output_path).replace('.nii.gz', ''))
    print(f"Saved merged multi-label mask to {output_path}")


def export_binary_masks(multi_mask, mask_meta, output_dir, origin_masks, derived_masks):
    """
    Export binary masks from a multi-label mask.
    
    Args:
        multi_mask (np.ndarray): Multi-label mask array
        mask_meta (dict): Metadata for the mask
        output_dir (Path): Output directory for binary masks
        origin_masks (dict): Origin binary masks (for label names)
        derived_masks (dict): Derived binary masks (for label names)
    """
    saver = SaveImage(output_postfix='', output_dtype=np.uint8)
    
    # Create a combined dictionary of label names
    label_names = {}
    for label, (_, _, name) in origin_masks.items():
        label_names[label] = name
    for label, (_, _, name) in derived_masks.items():
        if label not in label_names:
            label_names[label] = name
    
    # Get all unique labels in the multi-label mask
    unique_labels = list(range(0, 27))
    
    for label in unique_labels:
        # Create binary mask
        binary_mask = (multi_mask == label).astype(np.uint8)
        
        # Get label name
        label_name = label_names.get(label, f"label_{label:02d}")
        
        # Determine base filename
        multi_mask_files = list(output_dir.glob('*_mask.nii.gz'))
        if multi_mask_files:
            base_name = multi_mask_files[0].name.replace('_mask.nii.gz', '')
        else:
            # Extract base name from directory
            base_name = output_dir.name
        
        # Create output path
        output_filename = f"{base_name}_mask_{label:02d}_{label_name}.nii.gz"
        output_path = output_dir / output_filename
        
        # Save binary mask
        saver(binary_mask, meta_data=mask_meta, filename=str(output_path).replace('.nii.gz', ''))
        print(f"Exported binary mask for label {label:02d} to {output_path}")


def main():
    """
    Main function to orchestrate the binary mask union and multi-label mask creation process.
    """
    args = parse_args()
    
    # Convert paths to Path objects
    root_origin = Path(args.root_origin)
    root_derived = Path(args.root_derived)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get paired sample directories
    paired_samples = get_paired_sample_dirs(root_origin, root_derived)
    
    if not paired_samples:
        print("No paired sample directories found. Exiting.")
        return
    
    print(f"Found {len(paired_samples)} paired sample directories to process.")
    
    # Process each sample directory pair with progress bar
    success_count = 0
    error_count = 0
    
    for origin_sample, derived_sample in tqdm(paired_samples, desc="Processing samples"):
        success, error_msg = process_sample_dir(
            origin_sample, derived_sample, 
            root_origin, root_derived, 
            output_dir, args.skip_existing)
        
        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"Error processing {origin_sample.name}: {error_msg}")
    
    # Print summary statistics
    print("\nProcessing Summary:")
    print(f"Total paired samples: {len(paired_samples)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {error_count}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
