# -*- coding: utf-8 -*-
"""
参照AMOS22/05_gen_dataset_manifest.py生成VerSe数据集清单脚本。使用argparse。接收-r/--root_dir指定根目录，根目录下包含若干层级的子目录，
第一级为VerSe*格式的子数据集目录，第二级为train/val/test子集目录，第三级为样本目录，每个样本目录中包含形如*_volume.nii.gz的图像文件，*_mask.nii.gz的蒙版文件，
*_mask_<具体释义(其中可能带下划线)>.nii.gz的二值蒙版文件，以及*_info.yaml的元信息文件。接收-o/--output_manifest_file
指定输出Excel清单文件路径，清单文件中依次记录以下信息：
  ID：记录文件所属样本的ID。
  archive：记录文件所属的子数据集，从元信息文件中的archive字段获取。
  subset：记录样本所属的子集（train/val/test）。
  subject：从元信息文件中的subject-id字段获取。
  split：从元信息文件中的split字段获取，如果此字段不存在，则留空白。
  info：记录元信息文件的路径，相对于数据集根，使用/路径连接符。
  volume：记录图像文件的路径，相对于数据集根，使用/路径连接符。
  mask：记录蒙版文件的路径，相对于数据集根，使用/路径连接符。
  若干 mask_<具体释义>：记录<具体释义>二值蒙版文件的路径，相对于数据集根，使用/路径连接符。
  szx：记录图像的x规格。
  szy：记录图像的y规格。
  szz：记录图像的z规格。
  spx：记录图像的x间距。
  spy：记录图像的y间距。
  spz：记录图像的z间距。
  orientation_from：记录图像的L/R、A/P、S/I朝向，记录开始端侧，例如LPS、RAI等，如果其方向不标准（transfomr空间变换矩阵的前3×3不是对角阵），则记录为Oblique(closest to *)，*表示最接近的L/R、A/P、S/I朝向。
  orientation_to：记录图像的L/R、A/P、S/I朝向，记录结束端侧，如果其方向不标准（transfomr空间变换矩阵的前3×3不是对角阵），则记录为Oblique(closest to *)，*表示最接近的L/R、A/P、S/I朝向。
  vol_dtype：图像文件的数据类型。
  mask_dtype：蒙版文件的数据类型。
  transform：记录图像携带的4×4空间变换矩阵。
扫描期间检查每个样本图像与全部蒙版的空间变换矩阵，规格，间距，原点，朝向的一致性，以及全部蒙版文件数据类型的一致性，如果存在不一致则将信息输出到控制台
使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。
"""

"""
Dataset Manifest Generation Script for VerSe Dataset

This script generates a comprehensive manifest of the dataset, including volume and mask file paths,
metadata information from YAML files, and consistency checks between images and masks.

Parameters:
    -r, --root_dir: Root directory containing sample directories at any depth
    -o, --output_manifest_file: Output Excel manifest file path
    -s, --sheet_name: Sheet name in the Excel file (default: Manifest)
    -v, --verbose: Add debug column with processing exception information

Usage Examples:
    python 05_gen_dataset_manifest.py -r /path/to/grouped -o /path/to/dataset_manifest.xlsx
    python 05_gen_dataset_manifest.py --root_dir /path/to/grouped --output_manifest_file /path/to/dataset_manifest.xlsx -s Manifest
    python 05_gen_dataset_manifest.py -r /path/to/grouped -o /path/to/dataset_manifest.xlsx --verbose
"""

import re
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional, Any
from tqdm import tqdm
import numpy as np
import pandas as pd
from monai.transforms import LoadImage


# Command line argument parsing function
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir and output_manifest_file
    """
    parser = argparse.ArgumentParser(
        description='Generate dataset manifest with metadata and consistency checks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/grouped -o /path/to/dataset_manifest.xlsx
  %(prog)s --root_dir /path/to/grouped --output_manifest_file /path/to/dataset_manifest.xlsx
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing sample directories at any depth'
    )

    parser.add_argument(
        '-o', '--output_manifest_file',
        type=str,
        required=True,
        help='Output Excel manifest file path'
    )

    parser.add_argument(
        '-s', '--sheet_name',
        type=str,
        default='Manifest',
        help='Sheet name in the Excel file (default: Manifest)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Add debug column with processing exception information'
    )
    
    parser.add_argument(
        '-i', '--inclusion',
        type=str,
        nargs='+',
        default=['VerSe19', 'VerSe20'],
        help='List of sub-datasets to include (default: [\'VerSe19\', \'VerSe20\'])'
    )

    return parser.parse_args()


# Check if matrix is diagonal
def is_diagonal_matrix(matrix: np.ndarray, tol: float = 0.0) -> bool:
    """
    Check if a matrix is diagonal (all off-diagonal elements are close to zero).

    Args:
        matrix (np.ndarray): Input matrix to check
        tol (float): Tolerance for considering an element as zero

    Returns:
        bool: True if matrix is diagonal, False otherwise
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return np.allclose(matrix[mask], 0, atol=tol)


# Get orientation string from affine matrix
def get_orientation_string(affine: np.ndarray) -> Tuple[str, str]:
    """
    Extract orientation string from affine matrix.

    Args:
        affine (np.ndarray): 4x4 affine transformation matrix

    Returns:
        Tuple[str, str]: (orientation_from, orientation_to) strings, or ('Oblique(closest to X)', 'Oblique(closest to X)') if non-standard
    """
    rotation = affine[:3, :3]

    def get_axis_label(vec):
        max_idx = np.argmax(np.abs(vec))
        val = vec[max_idx]

        if max_idx == 0:
            return 'R' if val > 0 else 'L'
        elif max_idx == 1:
            return 'A' if val > 0 else 'P'
        else:
            return 'S' if val > 0 else 'I'

    if not is_diagonal_matrix(rotation):
        orientation_from = ''.join([get_axis_label(-rotation[:, i]) for i in range(3)])
        orientation_to = ''.join([get_axis_label(rotation[:, i]) for i in range(3)])
        return f'Oblique(closest to {orientation_from})', f'Oblique(closest to {orientation_to})'

    orientation_from = ''.join([get_axis_label(-rotation[:, i]) for i in range(3)])
    orientation_to = ''.join([get_axis_label(rotation[:, i]) for i in range(3)])

    return orientation_from, orientation_to


# Check metadata consistency
def check_metadata_consistency(volume_meta: Dict[str, Any], mask_meta_dict: Dict[str, Dict[str, Any]]) -> Tuple[
    bool, List[str]]:
    """
    Check consistency between volume and all masks.
    
    Args:
        volume_meta (Dict[str, Any]): Volume metadata dictionary
        mask_meta_dict (Dict[str, Dict[str, Any]]): Dictionary of mask metadata dictionaries
        
    Returns:
        Tuple[bool, List[str]]: (is_consistent, issues) where issues is a list of inconsistency messages
    """
    issues = []

    volume_shape = volume_meta['spatial_shape']
    volume_affine = volume_meta['affine']
    volume_origin = volume_affine[:3, 3]
    volume_spacing = np.array([volume_meta['pixdim'][1], volume_meta['pixdim'][2], volume_meta['pixdim'][3]])

    volume_rotation = volume_affine[:3, :3]
    volume_is_oblique = not np.allclose(np.abs(volume_rotation), np.diag(np.diag(np.abs(volume_rotation))), atol=0.0)

    for mask_name, mask_meta in mask_meta_dict.items():
        mask_shape = mask_meta['spatial_shape']
        mask_affine = mask_meta['affine']
        mask_origin = mask_affine[:3, 3]
        mask_spacing = np.array([mask_meta['pixdim'][1], mask_meta['pixdim'][2], mask_meta['pixdim'][3]])

        mask_rotation = mask_affine[:3, :3]
        mask_is_oblique = not np.allclose(np.abs(mask_rotation), np.diag(np.diag(np.abs(mask_rotation))), atol=0.0)

        if not (volume_shape == mask_shape).all():
            issues.append(f"Shape mismatch with {mask_name}: volume={volume_shape}, mask={mask_shape}")

        if not np.allclose(volume_affine, mask_affine, atol=0.0):
            issues.append(f"Affine mismatch with {mask_name}")

        if not np.allclose(volume_origin, mask_origin, atol=0.0):
            issues.append(f"Origin mismatch with {mask_name}: volume={volume_origin}, mask={mask_origin}")

        if not np.allclose(volume_spacing, mask_spacing, atol=0.0):
            issues.append(f"Spacing mismatch with {mask_name}: volume={volume_spacing}, mask={mask_spacing}")

        if volume_is_oblique != mask_is_oblique:
            issues.append(f"Orientation obliqueness mismatch with {mask_name}")

    return len(issues) == 0, issues


# Check mask data type consistency
def check_mask_dtype_consistency(mask_dtype_dict: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Check consistency of data types across all masks.
    
    Args:
        mask_dtype_dict (Dict[str, str]): Dictionary of mask data types
        
    Returns:
        Tuple[bool, List[str]]: (is_consistent, issues) where issues is a list of inconsistency messages
    """
    issues = []

    if not mask_dtype_dict:
        return True, issues

    dtypes = list(mask_dtype_dict.values())
    unique_dtypes = set(dtypes)

    if len(unique_dtypes) > 1:
        issues.append(f"Data type inconsistency across masks: {unique_dtypes}")

    return len(issues) == 0, issues


# Process sample directory
def process_sample_dir(sample_dir: Path, root_path: Path, loader: LoadImage, verbose: bool = False, inclusion: List[str] = None) -> Optional[Dict[str, Any]]:
    """
    Process a single sample directory and extract metadata.
    
    Args:
        sample_dir (Path): Path to the sample directory
        root_path (Path): Root directory path for relative path calculation
        loader (LoadImage): MONAI LoadImage instance
        verbose (bool): Whether to collect debug information
        inclusion (List[str]): List of sub-datasets to include
        
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing metadata for this sample, or None if error or sample should be excluded
    """
    sample_name: str = sample_dir.name

    # Find info file
    info_files: List[Path] = list(sample_dir.glob('*_info.yaml'))
    if not info_files:
        if verbose:
            print(f"No info file found in {sample_dir}")
        return None

    info_file = info_files[0]

    # Read metadata
    try:
        with open(info_file, 'r', encoding='utf-8') as f:
            metadata: Dict[str, str] = yaml.safe_load(f)
    except Exception as e:
        if verbose:
            print(f"Error reading info file {info_file}: {e}")
        return None

    # Construct sample ID
    subject: str = metadata.get('subject', '')
    split: str = metadata.get('split', '')

    if split:
        # Extract the last characters of split [5:]
        try:
            split_suffix = split[5:]
            sample_id = f"{subject}-{split_suffix}"
        except IndexError:
            sample_id = subject
    else:
        sample_id = subject

    # Find image, mask, and binary mask files
    volume_files: List[Path] = list(sample_dir.glob('*_volume.nii.gz'))
    mask_files: List[Path] = list(sample_dir.glob('*_mask.nii.gz'))
    binary_mask_files: List[Path] = sorted(sample_dir.glob('*_mask_*.nii.gz'))

    debug_info: List[str] = []

    if not volume_files:
        debug_info.append(f"Missing volume file")

    if not mask_files:
        debug_info.append(f"Missing mask file")

    volume_file: Optional[Path] = volume_files[0] if volume_files else None
    mask_file: Optional[Path] = mask_files[0] if mask_files else None

    # Calculate relative paths
    info_rel_path: str = str(info_file.relative_to(root_path).as_posix())
    volume_rel_path: str = str(volume_file.relative_to(root_path).as_posix()) if volume_file else ''
    mask_rel_path: str = str(mask_file.relative_to(root_path).as_posix()) if mask_file else ''

    volume_data: Optional[np.ndarray] = None
    volume_meta: Optional[Dict[str, Any]] = None
    mask_data: Optional[np.ndarray] = None
    mask_meta: Optional[Dict[str, Any]] = None

    # Load image file and extract metadata
    try:
        if volume_file:
            volume_data, volume_meta = loader(str(volume_file))
    except Exception as e:
        debug_info.append(f"Error loading volume file: {e}")

    # Load mask file and extract metadata
    try:
        if mask_file:
            mask_data, mask_meta = loader(str(mask_file))
    except Exception as e:
        debug_info.append(f"Error loading mask file: {e}")

    # Process mask metadata and binary masks
    mask_meta_dict: Dict[str, Dict[str, Any]] = {}
    mask_dtype_dict: Dict[str, str] = {}
    binary_mask_paths: Dict[str, str] = {}
    binary_mask_existence: Dict[str, int] = {}

    if mask_meta:
        mask_meta_dict['mask'] = mask_meta
        mask_dtype_dict['mask'] = str(mask_data.dtype) if mask_data is not None else ''

    # Process binary mask files
    for binary_mask_file in binary_mask_files:
        binary_match = re.match(r'.+_mask_(.+)\.nii\.gz', binary_mask_file.name)
        if binary_match:
            label_name = binary_match.group(1)
            try:
                binary_mask_data, binary_mask_meta = loader(str(binary_mask_file))
                mask_meta_dict[f'mask_{label_name}'] = binary_mask_meta
                mask_dtype_dict[f'mask_{label_name}'] = str(binary_mask_data.dtype)
                binary_mask_paths[f'mask_{label_name}'] = str(binary_mask_file.relative_to(root_path).as_posix())

                # Check if binary mask has foreground pixels
                has_foreground = int(np.any(binary_mask_data > 0))
                binary_mask_existence[f'mask_{label_name}_existence'] = has_foreground
            except Exception as e:
                debug_info.append(f"Error loading binary mask {label_name}: {e}")

    # Perform consistency checks
    consistency_issues = []
    dtype_issues = []

    if volume_meta and mask_meta_dict:
        is_consistent, consistency_issues = check_metadata_consistency(volume_meta, mask_meta_dict)

    if mask_dtype_dict:
        dtype_consistent, dtype_issues = check_mask_dtype_consistency(mask_dtype_dict)

    if consistency_issues or dtype_issues:
        print(f"\nConsistency issues in {sample_name}:")
        for issue in consistency_issues + dtype_issues:
            print(f"  - {issue}")

    # Extract image dimensions and spacing
    volume_shape: Union[np.ndarray, List[str]] = volume_meta['spatial_shape'] if volume_meta else ['', '', '']
    volume_spacing: List[Union[float, str]] = []
    orientation_from: str = ''
    orientation_to: str = ''
    transform_string: str = ''
    volume_dtype: str = ''

    if volume_meta:
        volume_pixdim = volume_meta.get('pixdim', None)
        if volume_pixdim is not None:
            volume_spacing = [volume_pixdim[1], volume_pixdim[2], volume_pixdim[3]]
        else:
            volume_spacing = ['', '', '']

        orientation_from, orientation_to = get_orientation_string(volume_meta['affine'])
        transform_string = format_transform_for_excel(volume_meta['affine'])
        volume_dtype = str(volume_data.numpy().dtype) if volume_data is not None else ''

    # Extract mask data type
    mask_dtype: str = str(mask_data.numpy().dtype) if mask_data is not None else ''

    # Generate valid_labels string from binary_mask_existence
    valid_labels_list: List[str] = []
    for label_name, existence in binary_mask_existence.items():
        if existence == 1:
            mask_key = label_name.replace('_existence', '')
            valid_labels_list.append(mask_key.replace('mask_', ''))

    valid_labels: str = ','.join(sorted(valid_labels_list)) if valid_labels_list else ''

    # Get sample's inclusion list
    sample_inclusion = metadata.get('inclusion', [])
    
    # Set default inclusion if None
    if inclusion is None:
        inclusion = ['VerSe19', 'VerSe20']
    
    # Filter samples: skip if no overlap between sample_inclusion and inclusion
    if not any(elem in sample_inclusion for elem in inclusion):
        if verbose:
            print(f"Skipping sample {sample_id}: no inclusion match with {inclusion}")
        return None
    
    # Generate in_xxx columns for each inclusion element
    inclusion_columns = {}
    for elem in inclusion:
        # Create column name: in_lowercase(elem)
        col_name = f"in_{elem.lower()}"
        # Set value to 1 if elem is in sample_inclusion, else 0
        inclusion_columns[col_name] = 1 if elem in sample_inclusion else 0
    
    # Build result dictionary
    record = {
        'ID': sample_id,
        'archive': metadata.get('archive', ''),
        'subset': metadata.get('subset', ''),
        'subject': metadata.get('subject', ''),
        'split': metadata.get('split', '')
    }
    
    # Add inclusion columns
    record.update(inclusion_columns)
    
    # Add remaining columns
    record.update({
        'CT_image_series': metadata.get('CT_image_series', ''),
        'sex': metadata.get('sex', ''),
        'age': metadata.get('age', ''),
        'valid_labels': valid_labels,
        'info': info_rel_path,
        'volume': volume_rel_path,
        'mask': mask_rel_path
    })

    # Add binary mask paths and existence information
    record.update(binary_mask_paths)
    record.update(binary_mask_existence)

    record.update({
        'szx': volume_shape[0] if len(volume_shape) > 0 else '',
        'szy': volume_shape[1] if len(volume_shape) > 1 else '',
        'szz': volume_shape[2] if len(volume_shape) > 2 else '',
        'spx': volume_spacing[0],
        'spy': volume_spacing[1],
        'spz': volume_spacing[2],
        'orientation_from': orientation_from,
        'orientation_to': orientation_to,
        'vol_dtype': volume_dtype,
        'mask_dtype': mask_dtype,
        'transform': transform_string
    })

    # Add debug information
    if verbose and debug_info:
        record['debug'] = '; '.join(debug_info)

    return record


# Find all sample directories
def find_sample_dirs(root_path: Path) -> List[Path]:
    """
    Find all sample directories in the specified root path.
    
    Args:
        root_path (Path): Root directory path
    
    Returns:
        List[Path]: List of Path objects for all sample directories
    """
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
                    sample_dirs.append(sample_dir)

    return sorted(sample_dirs)


# Format transformation matrix as string
def format_transform_for_excel(transform: Optional[np.ndarray]) -> str:
    """
    Format a 4x4 transformation matrix as a string for Excel.
    
    Args:
        transform (Optional[np.ndarray]): 4x4 transformation matrix
    
    Returns:
        str: Formatted string representation of the matrix
    """
    if transform is None:
        return ""

    # Convert matrix to string with 6 decimal places
    return "\n".join(["\t".join([f"{x:.6f}" for x in row]) for row in transform])


# Main function
def main() -> None:
    """
    Main function to orchestrate the dataset manifest generation process.
    """
    # Parse command line arguments
    args = parse_args()
    root_path: Path = Path(args.root_dir)
    output_path: Path = Path(args.output_manifest_file)
    sheet_name: str = args.sheet_name
    verbose: bool = args.verbose

    # Create MONAI LoadImage instance
    loader: LoadImage = LoadImage(image_only=False, dtype=None)
    inclusion_list: List[str] = args.inclusion

    # Find all sample directories
    print(f"Finding sample directories in {root_path}...")
    sample_dirs = find_sample_dirs(root_path)
    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Including samples from: {', '.join(inclusion_list)}")

    # Process sample directories and collect results
    print("Processing samples...")
    records: List[Dict[str, Any]] = []
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        record = process_sample_dir(sample_dir, root_path, loader, verbose, inclusion_list)
        if record is not None:
            records.append(record)

    if not records:
        print("No valid records found. Exiting.")
        return

    # Create DataFrame and save to Excel file
    print(f"Creating Excel manifest with {len(records)} records...")
    df = pd.DataFrame(records)

    # Save to Excel file
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Manifest successfully created at {output_path}")
    except Exception as e:
        print(f"Error creating Excel file: {e}")
        return


if __name__ == "__main__":
    main()
