# -*- coding: utf-8 -*-
"""
Small Connected Region Detection Script for ACDC Dataset

This script detects small connected regions in binary mask files and reports their
center coordinates and volumes. Regions with physical volume below the threshold are reported.

Parameters:
    -r, --root_dir: Root directory containing subset subdirectories (e.g., train, test)
    -t, --region_volume_thresh: Volume threshold in mm³ for small regions (default: 300.0)
    -o, --output_manifest: Path to output Excel manifest file

Usage Examples:
    python small_connected_region_detect.py -r /path/to/grouped -o output.xlsx
    python small_connected_region_detect.py --root_dir /path/to/grouped --region_volume_thresh 300.0 --output_manifest output.xlsx
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
  %(prog)s -r /path/to/grouped -o output.xlsx
  %(prog)s --root_dir /path/to/grouped --region_volume_thresh 300.0 --output_manifest output.xlsx
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing subset subdirectories (e.g., train, test)'
    )

    parser.add_argument(
        '-t', '--region_volume_thresh',
        type=float,
        default=300.0,
        help='Volume threshold in mm³ for small regions (default: 300.0)'
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


def process_pair_dir(pair_dir, threshold, loader):
    """
    Process all binary mask files in a pair directory.
    
    Args:
        pair_dir (Path): Path to the pair directory
        threshold (float): Volume threshold for small regions
        loader (LoadImage): MONAI LoadImage instance
        
    Returns:
        dict: Dictionary containing ID and small regions for each mask
    """
    pair_name = pair_dir.name
    match = re.match(r'patient(\d{3})_frame(\d{2})', pair_name)
    if not match:
        return None

    patient_id = match.group(1)
    frame_id = match.group(2)
    combined_id = f'patient{patient_id}_frame{frame_id}'
    record = {'ID': combined_id}

    mask_files = sorted(pair_dir.glob('*_mask_*.nii.gz'))

    for mask_file in mask_files:
        binary_match = re.match(rf'{combined_id}_mask_(.+)\.nii\.gz', mask_file.name)
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


def process_root_dir(root_dir, threshold, output_manifest):
    """
    Process all subset directories in the root directory.
    
    Args:
        root_dir (str or Path): Root directory containing subset subdirectories
        threshold (float): Volume threshold for small regions
        output_manifest (str): Path to output Excel manifest file
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    subset_dirs = [d for d in root_path.iterdir() if d.is_dir()]

    if not subset_dirs:
        print(f"Warning: No subset directories found in {root_dir}")
        return

    print(f"Found {len(subset_dirs)} subset directories")
    print(f"Volume threshold: {threshold} mm³\n")

    loader = LoadImage(image_only=False, dtype=None)
    all_records = []

    for subset_dir in tqdm(subset_dirs, desc='Processing subsets'):
        pair_dirs = sorted(subset_dir.glob('patient*_frame*'))

        for pair_dir in tqdm(pair_dirs, desc=f'  Processing {subset_dir.name}', leave=False):
            record = process_pair_dir(pair_dir, threshold, loader)
            all_records.append(record)

    if all_records:
        df = pd.DataFrame(all_records)
        df = df.sort_values(by='ID', ascending=True)

        output_path = Path(output_manifest)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_excel(output_path, sheet_name=f'Small Connected Regions {threshold}', index=False)

        print(f"\nProcessing completed!")
        print(f"Total samples processed: {len(all_records)}")
        print(f"Output manifest saved to: {output_manifest}")

        small_region_cols = [col for col in df.columns if 'small_region' in col]
        for col in small_region_cols:
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