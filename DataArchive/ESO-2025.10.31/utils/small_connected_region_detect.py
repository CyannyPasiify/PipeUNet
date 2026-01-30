# -*- coding: utf-8 -*-
"""
Small Connected Region Detection Script for ESO-2025 Dataset

This script detects small connected regions in binary mask files and reports their
center coordinates and volumes. Regions with physical volume below the threshold are reported.

Parameters:
    -r, --root_dir: Root directory containing nested subdirectories with sample directories ({site}_{pid}_{phase})
    -t, --region_volume_thresh: Volume threshold in mm³ for small regions (default: 300.0)
    -o, --output_manifest: Path to output Excel manifest file

Usage Examples:
    python small_connected_region_detect.py -r /path/to/root -o output.xlsx
    python small_connected_region_detect.py --root_dir /path/to/root --region_volume_thresh 300.0 --output_manifest output.xlsx
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from monai.transforms import LoadImage
from scipy import ndimage
from typing import List, Dict, Tuple, Optional, Union, Any


def parse_args() -> argparse.Namespace:
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
        help='Root directory containing nested subdirectories with sample directories ({site}_{pid}_{phase})'
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


def detect_small_regions(mask_data: np.ndarray, spacing: Tuple[float, float, float], threshold: float) -> List[Tuple[float, float, float, float]]:
    """
    Detect small connected regions in binary mask.
    
    Args:
        mask_data (np.ndarray): Binary mask data array
        spacing (Tuple[float, float, float]): Spacing values (dx, dy, dz) in mm
        threshold (float): Volume threshold for small regions
        
    Returns:
        List[Tuple[float, float, float, float]]: List of small regions with format [(x, y, z, vol), ...]
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


def process_sample_dir(sample_dir: Path, threshold: float, loader: LoadImage) -> Optional[Dict[str, Any]]:
    """
    Process all binary mask files in a sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        threshold (float): Volume threshold for small regions
        loader (LoadImage): MONAI LoadImage instance
        
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing ID and small regions for each mask, or None if invalid sample directory
    """
    sample_name: str = sample_dir.name
    match = re.match(r'([^_]+)_([^_]+)_(pre|post)', sample_name)
    if not match:
        return None

    site_id: str = match.group(1)
    pid: str = match.group(2)
    phase: str = match.group(3)
    sample_id: str = f'{site_id}_{pid}_{phase}'
    record: Dict[str, Any] = {'ID': sample_id}

    mask_files = sorted(sample_dir.glob(f'*_mask_*.nii.gz'))

    for mask_file in mask_files:
        # Extract label name from filename
        binary_match = re.match(r'.+_mask_(.+)\.nii\.gz', mask_file.name)
        if binary_match:
            label_name = binary_match.group(1)
        else:
            continue

        try:
            mask_data, mask_meta = loader(str(mask_file))
            spacing: Tuple[float, float, float] = (mask_meta['pixdim'][1], mask_meta['pixdim'][2], mask_meta['pixdim'][3])

            small_regions: List[Tuple[float, float, float, float]] = detect_small_regions(mask_data, spacing, threshold)

            if small_regions:
                region_str: str = '; '.join([f"({x:.0f},{y:.0f},{z:.0f},{v:.2f})" for x, y, z, v in small_regions])
                record[f'mask_{label_name}'] = region_str
            else:
                record[f'mask_{label_name}'] = ''

        except Exception as e:
            print(f"Error processing {mask_file.name}: {str(e)}")
            record[f'mask_{label_name}'] = ''

    return record


def find_sample_dirs(root_dir: Union[str, Path]) -> List[Path]:
    """
    Recursively find all sample directories named {site}_{pid}_{phase}.
    
    Args:
        root_dir (Union[str, Path]): Root directory to search
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path: Path = Path(root_dir)
    sample_dirs: List[Path] = []

    for path in root_path.rglob('*_*_*'):
        if path.is_dir() and re.match(r'[A-Z]+_[0-9]+_(pre|post)', path.name):
            sample_dirs.append(path)

    return sorted(sample_dirs)


def process_root_dir(root_dir: Union[str, Path], threshold: float, output_manifest: Union[str, Path]) -> None:
    """
    Process all sample directories in the root directory recursively.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing nested subdirectories
        threshold (float): Volume threshold for small regions
        output_manifest (Union[str, Path]): Path to output Excel manifest file
    """
    root_path: Path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    sample_dirs: List[Path] = find_sample_dirs(root_dir)

    if not sample_dirs:
        print(f"Warning: No sample directories found in {root_dir}")
        return

    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Volume threshold: {threshold} mm³\n")

    loader: LoadImage = LoadImage(image_only=False, dtype=None)
    all_records: List[Dict[str, Any]] = []

    for sample_dir in tqdm(sample_dirs, desc='Processing samples'):
        record: Optional[Dict[str, Any]] = process_sample_dir(sample_dir, threshold, loader)
        if record:
            all_records.append(record)

    if all_records:
        df: pd.DataFrame = pd.DataFrame(all_records)

        ids = df['ID'].str.extract(r'([^_]+)_([^_]+)_(pre|post)')
        df['site'] = ids[0]
        df['pid'] = ids[1]
        df['phase'] = ids[2]
        df = df.sort_values(by=['site', 'pid', 'phase'], ascending=True)
        df = df.drop(columns=['site', 'pid', 'phase'])

        output_path: Path = Path(output_manifest)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_excel(output_path, sheet_name=f'Small Connected Regions {threshold}', index=False)

        print(f"\nProcessing completed!")
        print(f"Total samples processed: {len(all_records)}")
        print(f"Output manifest saved to: {output_manifest}")

        mask_cols: List[str] = [col for col in df.columns if col.startswith('mask_')]
        for col in mask_cols:
            non_empty: int = df[col].apply(lambda x: x != '' if x is not None else False).sum()
            if non_empty > 0:
                print(f"  {col}: {non_empty} samples with small regions")
    else:
        print("No records found to save.")


def main() -> None:
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