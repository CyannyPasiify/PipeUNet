# -*- coding: utf-8 -*-
"""
使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。 使用tqdm显示进度，并在进度条前的desc中展示当前正在处理样本的pid。 
   除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Dataset Manifest Generation Script for Duke-Breast-FGT-Segmentation Dataset

This script generates a comprehensive manifest of the Duke-Breast-FGT-Segmentation dataset, including volume and mask file paths,
metadata information from YAML files, and consistency checks between images and masks.

Parameters:
    -r, --root_dir: Root directory containing nested sample directories at any depth
    -o, --output_manifest_file: Output Excel manifest file path
    -s, --sheet_name: Sheet name in the Excel file (default: Manifest)
    -v, --verbose: Add debug column with processing exception information

Usage Examples:
    python 04_gen_dataset_manifest.py -r /path/to/root -o /path/to/dataset_manifest.xlsx
    python 04_gen_dataset_manifest.py --root_dir /path/to/root --output_manifest_file /path/to/dataset_manifest.xlsx -s Manifest
    python 04_gen_dataset_manifest.py -r /path/to/root -o /path/to/dataset_manifest.xlsx --verbose
"""

import re
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from monai.transforms import LoadImage
from typing import List, Dict, Tuple, Optional, Union, Any


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
  %(prog)s -r /path/to/root -o /path/to/dataset_manifest.xlsx
  %(prog)s --root_dir /path/to/root --output_manifest_file /path/to/dataset_manifest.xlsx
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing nested sample directories at any depth'
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

    return parser.parse_args()


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


def get_orientation_string(affine: np.ndarray) -> Tuple[str, str]:
    """
    Extract orientation string from affine matrix.

    Args:
        affine (np.ndarray): 4x4 affine transformation matrix

    Returns:
        tuple: (orientation_from, orientation_to) strings, or ('Oblique(closest to X)', 'Oblique(closest to X)') if non-standard
    """
    rotation = affine[:3, :3]

    def get_axis_label(vec: np.ndarray) -> str:
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


def format_transform_for_excel(transform: np.ndarray) -> str:
    """
    Format a 4x4 transform matrix for Excel display.

    Args:
        transform (np.ndarray): 4x4 transformation matrix

    Returns:
        str: Formatted string representation of the matrix
    """
    return '; '.join([', '.join([f'{val:.6f}' for val in row]) for row in transform])


def check_metadata_consistency(volume_meta: dict, mask_meta_dict: Dict[str, dict]) -> Tuple[bool, List[str]]:
    """
    Check consistency between volume and all masks.
    
    Args:
        volume_meta (dict): Volume metadata dictionary
        mask_meta_dict (dict): Dictionary of mask metadata dictionaries
        
    Returns:
        tuple: (is_consistent, issues) where issues is a list of inconsistency messages
    """
    issues: List[str] = []

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


def check_mask_dtype_consistency(mask_dtype_dict: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Check consistency of data types across all masks.
    
    Args:
        mask_dtype_dict (dict): Dictionary of mask data types
        
    Returns:
        tuple: (is_consistent, issues) where issues is a list of inconsistency messages
    """
    issues: List[str] = []

    if not mask_dtype_dict:
        return True, issues

    dtypes = list(mask_dtype_dict.values())
    unique_dtypes = set(dtypes)

    if len(unique_dtypes) > 1:
        issues.append(f"Data type inconsistency across masks: {unique_dtypes}")

    return len(issues) == 0, issues


def find_sample_dirs(root_dir: Union[str, Path]) -> List[Path]:
    """
    Recursively find all sample directories named Breast_MRI_{sid:03d}.
    
    Args:
        root_dir (Union[str, Path]): Root directory to search
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path = Path(root_dir)
    sample_dirs: List[Path] = []

    # For Duke-Breast dataset, we look for Breast_MRI_* directories
    for path in root_path.rglob('Breast_MRI_*'):
        if path.is_dir() and re.match(r'Breast_MRI_\d+', path.name):
            sample_dirs.append(path)
    
    return sorted(sample_dirs)


def process_sample_dir(sample_dir: Path, root_path: Path, loader: LoadImage, verbose: bool = False) -> Optional[
    Dict[str, Any]]:
    """
    Process a single sample directory and extract metadata.
    
    Args:
        sample_dir (Path): Path to the sample directory
        root_path (Path): Root directory path for relative path calculation
        loader (LoadImage): MONAI LoadImage instance
        verbose (bool): Whether to collect debug information
        
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing metadata for this sample, or None if error
    """
    sample_name: str = sample_dir.name

    # For Duke-Breast dataset, sample name is like Breast_MRI_001
    match = re.match(r'Breast_MRI_(\d+)', sample_name)
    if not match:
        return None

    pid: str = sample_name
    sample_id: str = pid

    # Find files in the sample directory
    info_files: List[Path] = list(sample_dir.glob(f'{sample_id}_info.yaml'))
    volume_files: List[Path] = list(sample_dir.glob(f'{sample_id}_volume.nii.gz'))
    mask_files: List[Path] = list(sample_dir.glob(f'{sample_id}_mask.nii.gz'))
    mask_mass_files: List[Path] = list(sample_dir.glob(f'{sample_id}_mask_mass.nii.gz'))

    debug_info: List[str] = []

    if not volume_files:
        debug_info.append(f"Missing volume file")

    if not mask_files:
        debug_info.append(f"Missing mask file")

    if not mask_mass_files:
        debug_info.append(f"Missing mask_mass file")

    volume_file: Optional[Path] = volume_files[0] if volume_files else None
    mask_file: Optional[Path] = mask_files[0] if mask_files else None
    mask_mass_file: Optional[Path] = mask_mass_files[0] if mask_mass_files else None

    info_rel_path: str = ''
    metadata: Dict[str, Any] = {}

    # Load metadata from YAML file
    if info_files:
        info_file: Path = info_files[0]
        info_rel_path = str(info_file.relative_to(root_path).as_posix())

        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
        except Exception as e:
            debug_info.append(f"Error loading YAML file: {str(e)}")

    volume_data = None
    volume_meta = None
    mask_data = None
    mask_meta = None
    mask_mass_data = None
    mask_mass_meta = None

    volume_shape = ['', '', '']
    volume_spacing = ['', '', '']
    orientation_from: str = ''
    orientation_to: str = ''
    volume_rel_path: Path = Path()
    mask_rel_path: Path = Path()
    mask_mass_rel_path: Path = Path()

    try:
        # Load volume data if available
        if volume_file:
            volume_data, volume_meta = loader(str(volume_file))
            volume_shape = volume_meta['spatial_shape']
            volume_affine = volume_meta['affine']
            volume_pixdim = volume_meta.get('pixdim', None)

            if volume_pixdim is not None:
                volume_spacing = [volume_pixdim[1], volume_pixdim[2], volume_pixdim[3]]
            else:
                volume_spacing = ['', '', '']

            orientation_from, orientation_to = get_orientation_string(volume_affine)
            volume_rel_path = volume_file.relative_to(root_path)

        # Load mask data if available
        if mask_file:
            mask_data, mask_meta = loader(str(mask_file))
            mask_rel_path = mask_file.relative_to(root_path)

        # Load mask_mass data if available
        if mask_mass_file:
            mask_mass_data, mask_mass_meta = loader(str(mask_mass_file))
            mask_mass_rel_path = mask_mass_file.relative_to(root_path)

        # Process binary masks
        mask_meta_dict: Dict[str, Any] = {}
        mask_dtype_dict: Dict[str, str] = {}
        binary_mask_paths: Dict[str, str] = {}
        binary_mask_existence: Dict[str, int] = {}

        if mask_meta:
            mask_meta_dict['mask'] = mask_meta
            mask_dtype_dict['mask'] = str(mask_data.dtype)

        if mask_mass_meta:
            mask_meta_dict['mask_mass'] = mask_mass_meta
            mask_dtype_dict['mask_mass'] = str(mask_mass_data.dtype)

        binary_mask_files: List[Path] = sorted(sample_dir.glob(f'{sample_id}_mask_*.nii.gz'))

        for binary_mask_file in binary_mask_files:
            binary_match = re.match(r'.+_mask_(\d+_([\w_]+))\.nii\.gz', binary_mask_file.name)
            if binary_match:
                label_name = binary_match.group(1)
                binary_mask_data, binary_mask_meta = loader(str(binary_mask_file))

                mask_meta_dict[f'mask_{label_name}'] = binary_mask_meta
                mask_dtype_dict[f'mask_{label_name}'] = str(binary_mask_data.dtype)

                rel_path = binary_mask_file.relative_to(root_path)
                binary_mask_paths[f'mask_{label_name}'] = str(rel_path.as_posix())

                # Check if binary mask has foreground pixels
                has_foreground: int = int(np.any(binary_mask_data > 0))
                binary_mask_existence[f'mask_{label_name}_existence'] = has_foreground

        # Perform consistency checks if both volume and mask data are available
        if volume_meta and mask_meta_dict:
            is_consistent, consistency_issues = check_metadata_consistency(volume_meta, mask_meta_dict)

            dtype_consistent, dtype_issues = check_mask_dtype_consistency(mask_dtype_dict)

            if consistency_issues or dtype_issues:
                debug_info.extend(consistency_issues)
                debug_info.extend(dtype_issues)
                print(f"\nConsistency issues in {sample_name}:")
                for issue in consistency_issues + dtype_issues:
                    print(f"  - {issue}")

        # Generate valid_labels list
        valid_labels_list: List[str] = []
        for label_name, existence in binary_mask_existence.items():
            if existence == 1:
                mask_key = label_name.replace('_existence', '')
                valid_labels_list.append(mask_key.replace('mask_', ''))

        valid_labels: str = ','.join(sorted(valid_labels_list)) if valid_labels_list else ''

        # Create record dictionary
        record: Dict[str, Any] = {
            'ID': sample_id,
            'valid_labels': valid_labels,
            'info': info_rel_path,
            'volume': str(volume_rel_path.as_posix()) if volume_rel_path else '',
            'mask': str(mask_rel_path.as_posix()) if mask_rel_path else '',
            'mask_mass': str(mask_mass_rel_path.as_posix()) if mask_mass_rel_path else ''
        }

        # Add binary mask existence flags
        sorted_existence_keys = sorted(binary_mask_existence.keys())
        for key in sorted_existence_keys:
            record[key] = binary_mask_existence[key]

        # Add image and mask metadata
        record.update({
            'szx': volume_shape[0] if len(volume_shape) > 0 else '',
            'szy': volume_shape[1] if len(volume_shape) > 1 else '',
            'szz': volume_shape[2] if len(volume_shape) > 2 else '',
            'spx': volume_spacing[0],
            'spy': volume_spacing[1],
            'spz': volume_spacing[2],
            'orientation_from': orientation_from,
            'orientation_to': orientation_to,
            'vol_dtype': str(volume_data.numpy().dtype) if volume_data is not None else '',
            'mask_dtype': str(mask_data.numpy().dtype) if mask_data is not None else '',
            'transform': format_transform_for_excel(volume_meta['affine']) if volume_meta is not None else ''
        })

        # Add binary mask paths
        record.update(binary_mask_paths)

        # Add metadata from YAML file
        # Ignore 'pid'
        yaml_fields = ['subset']

        for field in yaml_fields:
            record[field] = metadata.get(field, '')

        # Add debug information if verbose mode is enabled
        if verbose:
            record['debug'] = '; '.join(debug_info) if debug_info else ''

        return record

    except Exception as e:
        error_msg: str = f"Error processing {sample_dir}: {str(e)}"
        debug_info.append(error_msg)
        print(error_msg)

        valid_labels_list: List[str] = []
        for label_name, existence in binary_mask_existence.items():
            if existence == 1:
                mask_key = label_name.replace('_existence', '')
                valid_labels_list.append(mask_key.replace('mask_', ''))

        valid_labels: str = ','.join(sorted(valid_labels_list)) if valid_labels_list else ''

        # Create minimal record with error info if verbose
        if verbose:
            result: Dict[str, Any] = {
                'ID': sample_id,
                'valid_labels': valid_labels,
                'info': info_rel_path,
                'volume': str(volume_rel_path.as_posix()) if volume_rel_path else '',
                'mask': str(mask_rel_path.as_posix()) if mask_rel_path else '',
                'debug': '; '.join(debug_info) if debug_info else ''
            }
            return result

        return None


def main() -> None:
    """
    Main function to orchestrate the dataset manifest generation process.
    """
    args: argparse.Namespace = parse_args()

    root_path: Path = Path(args.root_dir)
    output_file: Path = Path(args.output_manifest_file)
    sheet_name: str = args.sheet_name
    verbose: bool = args.verbose

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_path}")
        return

    print(f"Processing data from: {root_path}")
    print(f"Output manifest file: {output_file}")

    # Create LoadImage instance
    loader: LoadImage = LoadImage(image_only=False, dtype=None)

    # Find all sample directories
    sample_dirs: List[Path] = find_sample_dirs(root_path)

    if not sample_dirs:
        print(f"Warning: No sample directories found in {root_path}")
        return

    print(f"Found {len(sample_dirs)} sample directories")

    # Process all sample directories
    records: List[Dict[str, Any]] = []

    with tqdm(total=len(sample_dirs), desc='Processing samples') as pbar:
        for sample_dir in sample_dirs:
            pbar.set_description(f'Processing {sample_dir.name}')
            record = process_sample_dir(sample_dir, root_path, loader, verbose)
            if record:
                records.append(record)
            pbar.update(1)

    if not records:
        print(f"Warning: No valid records generated from {len(sample_dirs)} sample directories")
        return

    print(f"Generated {len(records)} valid records")

    # Create DataFrame from records
    df = pd.DataFrame(records)

    # Reorder columns to match the required sequence
    required_columns = [
        'ID', 'subset', 'valid_labels', 'info', 'volume', 'mask'
    ]

    # Create columns for binary masks and their existence flags
    mask_columns = [col for col in df.columns
                    if col.startswith('mask_') and not col.endswith(('_existence', '_dtype'))]
    existence_columns = [col for col in df.columns if col.endswith('_existence')]

    # Image and mask properties
    image_properties = [
        'szx', 'szy', 'szz', 'spx', 'spy', 'spz', 'orientation_from', 'orientation_to',
        'vol_dtype', 'mask_dtype', 'transform'
    ]

    # Combine all columns in the correct order
    all_columns = (required_columns + mask_columns + existence_columns + image_properties)

    # Add debug column if verbose
    if verbose and 'debug' in df.columns:
        all_columns.append('debug')

    # Filter and reorder columns
    df = df[all_columns]

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write DataFrame to Excel
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Dataset manifest saved to: {output_file}")
    except Exception as e:
        print(f"Error writing Excel file: {str(e)}")
        return

    print(f"Dataset manifest generation completed successfully!")


if __name__ == '__main__':
    main()