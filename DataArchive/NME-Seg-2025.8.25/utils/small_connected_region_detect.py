# -*- coding: utf-8 -*-
"""
Small Connected Region Detection Script for NME-Seg-2025.8.25 Dataset

This script detects small connected regions in binary mask files and reports their
center coordinates and volumes. Regions with physical volume below the threshold are reported.

Parameters:
    -r, --root_dir: Root directory containing nested subdirectories with sample directories ({seq}_{pid})
    -t, --region_volume_thresh: Volume threshold in mm³ for small regions (default: 100.0)
    -o, --output_manifest: Path to output Excel manifest file

Usage Examples:
    python small_connected_region_detect.py -r /path/to/root -o output.xlsx
    python small_connected_region_detect.py --root_dir /path/to/root --region_volume_thresh 100.0 --output_manifest output.xlsx
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from monai.transforms import LoadImage
from scipy import ndimage


def parse_args():
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir, region_volume_thresh, and output_manifest
    """
    parser = argparse.ArgumentParser(
        description='Detect small connected regions in binary mask files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/root -o output.xlsx
  %(prog)s --root_dir /path/to/root --region_volume_thresh 100.0 --output_manifest output.xlsx
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing nested subdirectories with sample directories ({seq}_{pid})'
    )

    parser.add_argument(
        '-t', '--region_volume_thresh',
        type=float,
        default=100.0,
        help='Volume threshold in mm³ for small regions (default: 100.0)'
    )

    parser.add_argument(
        '-o', '--output_manifest',
        type=str,
        required=True,
        help='Path to output Excel manifest file'
    )

    return parser.parse_args()


def detect_small_regions(mask_data, spacing, threshold):
    """
    Detect small connected regions in binary mask.
    
    Args:
        mask_data (np.ndarray): Binary mask data array
        spacing (tuple): Spacing values (dx, dy, dz) in mm
        threshold (float): Volume threshold for small regions
        
    Returns:
        list: List of small regions with format [(x, y, z, vol), ...]
    """
    labeled_array, num_features = ndimage.label(mask_data, structure=np.ones((3, 3, 3)))

    small_regions = []

    for label in range(1, num_features + 1):
        region_mask = (labeled_array == label)
        region_voxels = np.sum(region_mask)

        dx, dy, dz = spacing
        volume = region_voxels * dx * dy * dz

        if volume < threshold:
            coords = np.argwhere(region_mask)
            center = coords.mean(axis=0)
            small_regions.append((center[0], center[1], center[2], volume))

    return small_regions


def process_sample_dir(sample_dir, threshold, loader):
    """
    Process all binary mask files in a sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        threshold (float): Volume threshold for small regions
        loader (LoadImage): MONAI LoadImage instance
        
    Returns:
        dict: Dictionary containing ID and small regions for each mask
    """
    sample_name = sample_dir.name
    match = re.match(r'([^_]+)_([^_]+)', sample_name)
    if not match:
        return None

    seq, pid = match.group(1), match.group(2)
    record = {'ID': sample_name}

    # Find all binary mask files in the sample directory
    mask_files = sorted(sample_dir.glob('*_mask_*.nii.gz')) + sorted(sample_dir.glob('*_mask_*.nii'))

    for mask_file in mask_files:
        binary_match = re.match(rf'{seq}_{pid}_mask_(.+)\.nii\.gz', mask_file.name)
        if binary_match:
            label_name = binary_match.group(1)
        else:
            continue

        try:
            mask_data, mask_meta = loader(str(mask_file))
            spacing = [mask_meta['pixdim'][1], mask_meta['pixdim'][2], mask_meta['pixdim'][3]]

            small_regions = detect_small_regions(mask_data, spacing, threshold)

            if small_regions:
                region_str = '; '.join([f"({x:.0f},{y:.0f},{z:.0f},{v:.2f})" for x, y, z, v in small_regions])
                record[f'mask_{label_name}'] = region_str
            else:
                record[f'mask_{label_name}'] = ''

        except Exception as e:
            print(f"Error processing {mask_file.name}: {str(e)}")
            record[f'mask_{label_name}'] = ''

    return record


def find_sample_dirs(root_dir):
    """
    Recursively find all sample directories named {seq}_{pid}.
    
    Args:
        root_dir (Path): Root directory to search
        
    Returns:
        list: List of Path objects pointing to sample directories
    """
    root_path = Path(root_dir)
    sample_dirs = []

    # Regular expression to match {seq}_{pid} format
    sample_pattern = re.compile(r'[^_]+_[^_]+')

    for path in root_path.rglob('*'):
        if path.is_dir() and sample_pattern.fullmatch(path.name):
            # Check if this is a leaf directory (no subdirectories matching the pattern)
            is_leaf = True
            for subpath in path.iterdir():
                if subpath.is_dir() and sample_pattern.fullmatch(subpath.name):
                    is_leaf = False
                    break
            if is_leaf:
                sample_dirs.append(path)

    return sorted(sample_dirs)


def process_root_dir(root_dir, threshold, output_manifest):
    """
    Process all sample directories in the root directory recursively.
    
    Args:
        root_dir (str or Path): Root directory containing nested subdirectories
        threshold (float): Volume threshold for small regions
        output_manifest (str): Path to output Excel manifest file
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    sample_dirs = find_sample_dirs(root_dir)

    if not sample_dirs:
        print(f"Warning: No sample directories found in {root_dir}")
        return

    print(f"Found {len(sample_dirs)} sample directories")

    loader = LoadImage(image_only=False, dtype=None)
    all_records = []

    for sample_dir in tqdm(sample_dirs, desc='Processing samples'):
        record = process_sample_dir(sample_dir, threshold, loader)
        if record:
            all_records.append(record)

    if all_records:
        df = pd.DataFrame(all_records)

        # Sort by ID
        df = df.sort_values(by='ID', ascending=True)

        output_path = Path(output_manifest)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_excel(output_path, sheet_name=f'Small Connected Regions {threshold}', index=False)

        print(f"\nProcessing completed!")
        print(f"Total samples processed: {len(all_records)}")
        print(f"Output manifest saved to: {output_manifest}")

        mask_cols = [col for col in df.columns if col.startswith('mask_')]
        for col in mask_cols:
            non_empty = df[col].apply(lambda x: x != '' if x is not None else False).sum()
            if non_empty > 0:
                print(f"  {col}: {non_empty} samples with small regions")
    else:
        print("No records found to save.")


def main():
    """
    Main function to orchestrate the small connected region detection process.
    """
    args = parse_args()

    print(f"Processing data from: {args.root_dir}")
    print(f"Volume threshold: {args.region_volume_thresh} mm³")
    print(f"Output manifest: {args.output_manifest}")

    process_root_dir(args.root_dir, args.region_volume_thresh, args.output_manifest)

    print("Small connected region detection completed successfully!")


if __name__ == '__main__':
    main()