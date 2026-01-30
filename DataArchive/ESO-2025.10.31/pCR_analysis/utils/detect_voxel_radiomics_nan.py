#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测影像组学Voxel特征图中的NaN值。使用argparse。接收-r/--root_dir指定根目录，根目录下包含若干层级的子目录，第一层是site中心目录，中心目录下包含包含{site}_{pid}样本目录，样本目录中包含形如{site}_{pid}_(pre|post)_{radiomics_type}_voxel_radiomics的治疗前|后影像组学特征目录，voxel_radiomics目录下包含若干{site}_{pid}_(pre|post)_{radiomics_name}.nii.gz的影像组学Voxel特征图。-rt/--radiomics_type用于指定选用的影像组学来源列表，默认为['std_resampled_volume_std_resampled_mask'] 
  -p/--phase用于指定阶段列表，默认为['pre','post']。 
  遍历每个样本目录，在样本目录中遍历phase指定阶段的影像组学特征目录，对voxel_radiomics目录下的全部影像组学特征图文件按radiomics_name进行排序；然后顺序读取影像组学特征图：如果其中存在NaN值则在控制台报告NaN值的坐标。可选参数-o/--report_file（默认None）可以指定Excel报告表格输出路径，如果不为None，则输出一个表格，包含以下列： 
 site，pid，phase，path存在NaN的影像组学特征路径（相对于根目录-r），NaN Locations记录NaN值坐标的列表list
"""

"""
Detect NaN Values in Voxel-based Radiomics Features

This script detects NaN values in voxel-based radiomics feature maps and reports
coordinates of NaN values. It can also generate an Excel report with NaN locations.

Parameters:
    -r, --root_dir: Root directory containing site directories with {site}_{pid} sample directories
    -rt, --radiomics_type: List of radiomics types to process (default: ['std_resampled_volume_std_resampled_mask'])
    -p, --phase: List of phases to process (default: ['pre', 'post'])
    -o, --report_file: Optional Excel report output path (default: None)

Usage Examples:
    python detect_voxel_radiomics_nan.py -r /path/to/root
    python detect_voxel_radiomics_nan.py --root_dir /path/to/root --radiomics_type std_resampled_volume_std_resampled_mask custom --phase pre
    python detect_voxel_radiomics_nan.py -r /path/to/root -o nan_report.xlsx
"""

import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from monai.transforms import LoadImage
from typing import List, Dict, Tuple, Optional


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Detect NaN values in voxel-based radiomics features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/root
  %(prog)s --root_dir /path/to/root --radiomics_type std_resampled_volume_std_resampled_mask custom --phase pre
  %(prog)s -r /path/to/root -o nan_report.xlsx
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing site directories with {site}_{pid} sample directories'
    )

    parser.add_argument(
        '-rt', '--radiomics_type',
        type=str,
        nargs='+',
        default=['std_resampled_volume_std_resampled_mask'],
        help='List of radiomics types to process (default: [\'std_resampled_volume_std_resampled_mask\'])'
    )

    parser.add_argument(
        '-p', '--phase',
        type=str,
        nargs='+',
        default=['pre', 'post'],
        help='List of phases to process (default: [\'pre\', \'post\'])'
    )

    parser.add_argument(
        '-o', '--report_file',
        type=str,
        default=None,
        help='Optional Excel report output path (default: None)'
    )

    return parser.parse_args()


def find_sample_dirs(root_dir: str) -> List[Path]:
    """
    Find all sample directories in the specified root directory.
    
    Args:
        root_dir: Root directory containing site directories with {site}_{pid} sample directories
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path: Path = Path(root_dir)
    sample_dirs: List[Path] = []

    for site_dir in root_path.iterdir():
        if site_dir.is_dir():
            for sample_dir in site_dir.iterdir():
                if sample_dir.is_dir() and re.match(r'^[^_]+_[^_]+$', sample_dir.name):
                    sample_dirs.append(sample_dir)

    return sorted(sample_dirs)


def parse_site_pid(sample_name: str) -> Tuple[str, str]:
    """
    Parse site and pid from sample directory name.
    
    Args:
        sample_name: Sample directory name in format {site}_{pid}
        
    Returns:
        Tuple[str, str]: (site, pid)
    """
    parts = sample_name.split('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return '', sample_name


def find_voxel_radiomics_dirs(sample_dir: Path, phase: str, radiomics_type: str) -> List[Path]:
    """
    Find voxel radiomics directories for a specific phase and radiomics type.
    
    Args:
        sample_dir: Sample directory path
        phase: Phase (pre/post)
        radiomics_type: Radiomics type
        
    Returns:
        List[Path]: List of voxel radiomics directories
    """
    base_name: str = sample_dir.name
    voxel_radiomics_dirs: List[Path] = []
    
    radiomics_dir_pattern: re.Pattern = re.compile(rf'^{base_name}_{phase}_{radiomics_type}_voxel_radiomics$')

    for item in sample_dir.iterdir():
        if item.is_dir() and radiomics_dir_pattern.match(item.name):
            voxel_radiomics_dirs.append(item)

    return voxel_radiomics_dirs


def load_radiomics_features(radiomics_dir: Path, base_name: str, phase: str) -> List[Tuple[str, Path, np.ndarray]]:
    """
    Load radiomics feature maps from a directory, sorted by radiomics name.
    
    Args:
        radiomics_dir: Directory containing radiomics feature maps
        base_name: Sample base name
        phase: Phase (pre/post)
        
    Returns:
        List[Tuple[str, Path, np.ndarray]]: List of tuples containing (radiomics_name, file_path, feature_data)
    """
    radiomics_files: List[Path] = list(radiomics_dir.glob(f"{base_name}_{phase}_*.nii.gz"))

    radiomics_info = []
    for file_path in radiomics_files:
        match = re.match(rf'^{base_name}_{phase}_(.+)\.nii\.gz$', file_path.name)
        if match:
            radiomics_name = match.group(1)
            radiomics_info.append((radiomics_name, file_path))

    # Sort by radiomics name
    radiomics_info.sort(key=lambda x: x[0])
    
    # Load feature maps
    loaded_features = []
    loader = LoadImage(image_only=True, dtype=None)
    
    for radiomics_name, file_path in radiomics_info:
        try:
            feature_data = loader(str(file_path))
            loaded_features.append((radiomics_name, file_path, feature_data))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return loaded_features


def detect_nan_values(feature_data: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Detect NaN values in a feature map and return their coordinates.
    
    Args:
        feature_data: Feature map data
        
    Returns:
        List[Tuple[int, int, int]]: List of (x, y, z) coordinates of NaN values
    """
    nan_coords = np.argwhere(np.isnan(feature_data))
    return [tuple(coord.tolist()) for coord in nan_coords]


def process_sample(sample_dir: Path, phases: List[str], radiomics_types: List[str], root_dir: Path) -> List[Dict]:
    """
    Process a single sample directory for NaN detection.
    
    Args:
        sample_dir: Sample directory path
        phases: List of phases to process
        radiomics_types: List of radiomics types to process
        root_dir: Root directory path
        
    Returns:
        List[Dict]: List of NaN detection results
    """
    results = []
    base_name = sample_dir.name
    site, pid = parse_site_pid(base_name)
    
    for phase in phases:
        for radiomics_type in radiomics_types:
            # Find voxel radiomics directories
            radiomics_dirs = find_voxel_radiomics_dirs(sample_dir, phase, radiomics_type)
            
            for radiomics_dir in radiomics_dirs:
                # Load radiomics features
                features = load_radiomics_features(radiomics_dir, base_name, phase)
                
                for radiomics_name, file_path, feature_data in features:
                    # Detect NaN values
                    nan_coords = detect_nan_values(feature_data)
                    
                    if nan_coords:
                        # Calculate relative path
                        rel_path = file_path.relative_to(root_dir)
                        
                        # Report NaN values
                        print(f"NaN values found in {file_path}:")
                        print(f"  Number of NaN values: {len(nan_coords)}")
                        print(f"  First 10 NaN coordinates: {nan_coords[:10]}")
                        if len(nan_coords) > 10:
                            print(f"  ... and {len(nan_coords) - 10} more")
                        print()
                        
                        # Add to results
                        results.append({
                            'site': site,
                            'pid': pid,
                            'phase': phase,
                            'path': str(rel_path),
                            'NaN Locations': nan_coords
                        })
    
    return results


def generate_report(results: List[Dict], report_file: str) -> None:
    """
    Generate Excel report from NaN detection results.
    
    Args:
        results: List of NaN detection results
        report_file: Excel report output path
    """
    if not results:
        print("No NaN values found. Report not generated.")
        return
    
    df = pd.DataFrame(results)
    
    # Create directory if it doesn't exist
    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(report_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='NaN Detection', index=False)
    
    print(f"Excel report generated: {report_file}")


def main() -> None:
    """
    Main function to orchestrate NaN detection workflow.
    """
    # Parse command line arguments
    args: argparse.Namespace = parse_args()
    
    # Print configuration
    print(f"Root directory: {args.root_dir}")
    print(f"Radiomics types: {args.radiomics_type}")
    print(f"Phases: {args.phase}")
    print(f"Report file: {args.report_file}")
    print()
    
    # Find all sample directories
    root_path = Path(args.root_dir)
    sample_dirs = find_sample_dirs(str(root_path))
    print(f"Found {len(sample_dirs)} sample directories")
    print()
    
    # Process each sample
    all_results = []
    with tqdm(total=len(sample_dirs), desc="Detecting NaN values", unit="sample") as pbar:
        for sample_dir in sample_dirs:
            results = process_sample(sample_dir, args.phase, args.radiomics_type, root_path)
            all_results.extend(results)
            pbar.update(1)
    
    # Generate report if requested
    if args.report_file:
        generate_report(all_results, args.report_file)
    
    # Summary
    print("=" * 80)
    print("NaN Detection Summary")
    print("=" * 80)
    print(f"Total samples processed: {len(sample_dirs)}")
    print(f"Files with NaN values: {len(all_results)}")
    if all_results:
        print("NaN values were found in some files. Check the output above for details.")
    else:
        print("No NaN values found in any files!")
    print("=" * 80)


if __name__ == '__main__':
    main()
