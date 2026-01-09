# -*- coding: utf-8 -*-
"""
参照03_extract_binary_masks.py生成数据集清单脚本。使用argparse。接收-r/--root_dir指定根目录，根目录下包含多个子集子目录，
子集子目录下包含以patientxxx_frameyy命名的二级子目录，每个二级子目录中包含形如patientxxx_frameyy_volume.nii.gz的图像文件，
patientxxx_frameyy_mask.nii.gz的蒙版文件，以及patientxxx_frameyy_mask_<具体释义>.nii.gz的二值蒙版文件，接收-o/--output_manifest_file
指定输出Excel清单文件路径，清单文件中依次记录以下信息：
  patient：记录文件所属病人的3位数字编号xxx。
  frame：记录文件所属帧号的2位数字编号yy。
  volume：记录图像文件的路径，相对于数据集根，使用/路径连接符。
  mask：记录蒙版文件的路径，相对于数据集根，使用/路径连接符。
  若干 mask_<具体释义>：记录<具体释义>二值蒙版文件的路径，相对于数据集根，使用/路径连接符。
  szx：记录图像的x规格。
  szy：记录图像的y规格。
  szz：记录图像的z规格。
  spx：记录图像的x间距。
  spy：记录图像的y间距。
  spz：记录图像的z间距。
  orientation_from：记录图像的L/R、A/P、S/I朝向，记录开始端侧，例如LPS、RAI等，如果其方向不标准（transfomr空间变换矩阵的前3×3不是对角阵），则记录为Oblique。
  orientation_to：记录图像的L/R、A/P、S/I朝向，记录结束端侧，如果其方向不标准（transfomr空间变换矩阵的前3×3不是对角阵），则记录为Oblique。
  vol_dtype：图像文件的数据类型。
  mask_dtype：蒙版文件的数据类型。
  transform：记录图像携带的4×4空间变换矩阵。
扫描期间检查每个样本图像与全部蒙版的空间变换矩阵，规格，间距，原点，朝向的一致性，以及全部蒙版文件数据类型的一致性，如果存在不一致则将信息输出到控制台
使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。
"""

"""
Dataset Manifest Generation Script for ACDC Dataset

This script generates a comprehensive manifest of the dataset, including volume and mask file paths,
metadata information, and consistency checks between images and masks.

Parameters:
    -r, --root_dir: Root directory containing subset subdirectories (e.g., train, test)
    -o, --output_manifest_file: Output Excel manifest file path
    -s, --sheet_name: Sheet name in the Excel file (default: Manifest)
    -v, --verbose: Add debug column with processing exception information

Usage Examples:
    python 04_gen_dataset_manifest.py -r /path/to/grouped -o /path/to/dataset_manifest.xlsx
    python 04_gen_dataset_manifest.py --root_dir /path/to/grouped --output_manifest_file /path/to/dataset_manifest.xlsx -s Manifest
    python 04_gen_dataset_manifest.py -r /path/to/grouped -o /path/to/dataset_manifest.xlsx --verbose
"""

import re
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from monai.transforms import LoadImage


def parse_args():
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
        help='Root directory containing subset subdirectories (e.g., train, test)'
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


def is_diagonal_matrix(matrix, tol=0.0):
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


def get_orientation_string(affine):
    """
    Extract orientation string from affine matrix.

    Args:
        affine (np.ndarray): 4x4 affine transformation matrix

    Returns:
        tuple: (orientation_from, orientation_to) strings, or ('Oblique', 'Oblique') if non-standard
    """
    rotation = affine[:3, :3]

    if not is_diagonal_matrix(rotation):
        return 'Oblique', 'Oblique'

    def get_axis_label(vec):
        max_idx = np.argmax(np.abs(vec))
        val = vec[max_idx]

        if max_idx == 0:
            return 'R' if val > 0 else 'L'
        elif max_idx == 1:
            return 'A' if val > 0 else 'P'
        else:
            return 'S' if val > 0 else 'I'

    orientation_from = ''.join([get_axis_label(-rotation[:, i]) for i in range(3)])
    orientation_to = ''.join([get_axis_label(rotation[:, i]) for i in range(3)])

    return orientation_from, orientation_to


def check_metadata_consistency(volume_meta, mask_meta_dict):
    """
    Check consistency between volume and all masks.
    
    Args:
        volume_meta (dict): Volume metadata dictionary
        mask_meta_dict (dict): Dictionary of mask metadata dictionaries
        
    Returns:
        tuple: (is_consistent, issues) where issues is a list of inconsistency messages
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


def check_mask_dtype_consistency(mask_dtype_dict):
    """
    Check consistency of data types across all masks.
    
    Args:
        mask_dtype_dict (dict): Dictionary of mask data types
        
    Returns:
        tuple: (is_consistent, issues) where issues is a list of inconsistency messages
    """
    issues = []

    if not mask_dtype_dict:
        return True, issues

    dtypes = list(mask_dtype_dict.values())
    unique_dtypes = set(dtypes)

    if len(unique_dtypes) > 1:
        issues.append(f"Data type inconsistency across masks: {unique_dtypes}")

    return len(issues) == 0, issues


def parse_info_yaml(info_yaml_path):
    """
    Parse info.yaml file to extract patient information.
    
    Args:
        info_yaml_path (str or Path): Path to the info.yaml file
        
    Returns:
        dict: Dictionary containing group, height, phase, tot_frame, weight information
    """
    info = {
        'group': None,
        'height': None,
        'phase': None,
        'tot_frame': None,
        'weight': None
    }
    
    try:
        with open(info_yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            
            if yaml_data:
                if 'group' in yaml_data:
                    info['group'] = yaml_data['group']
                if 'height' in yaml_data:
                    info['height'] = yaml_data['height']
                if 'phase' in yaml_data:
                    info['phase'] = yaml_data['phase']
                if 'tot_frame' in yaml_data:
                    info['tot_frame'] = yaml_data['tot_frame']
                if 'weight' in yaml_data:
                    info['weight'] = yaml_data['weight']
    except Exception as e:
        pass
    
    return info


def process_pair_dir(pair_dir, root_path, loader, verbose=False):
    """
    Process a single pair directory and extract metadata.
    
    Args:
        pair_dir (Path): Path to the pair directory
        root_path (Path): Root directory path for relative path calculation
        loader (LoadImage): MONAI LoadImage instance
        verbose (bool): Whether to collect debug information
        
    Returns:
        dict: Dictionary containing metadata for this pair, or None if error
    """
    pair_name = pair_dir.name

    match = re.match(r'patient(\d{3})_frame(\d{2})', pair_name)
    if not match:
        return None

    patient_id = match.group(1)
    frame_id = match.group(2)
    combined_id = f'patient{patient_id}_frame{frame_id}'

    info_file = pair_dir / f'{combined_id}_info.yaml'
    volume_file = pair_dir / f'{combined_id}_volume.nii.gz'
    mask_file = pair_dir / f'{combined_id}_mask.nii.gz'

    debug_info = []

    if not info_file.exists():
        debug_info.append(f"Missing info file: {info_file.name}")
        if verbose:
            return None
        return None

    if not volume_file.exists():
        debug_info.append(f"Missing volume file: {volume_file.name}")
        if verbose:
            return None
        return None

    if not mask_file.exists():
        debug_info.append(f"Missing mask file: {mask_file.name}")
        if verbose:
            return None
        return None

    try:
        volume_data, volume_meta = loader(str(volume_file))
        mask_data, mask_meta = loader(str(mask_file))

        volume_shape = volume_meta['spatial_shape']
        volume_affine = volume_meta['affine']
        volume_pixdim = volume_meta.get('pixdim', None)

        if volume_pixdim is not None:
            volume_spacing = [volume_pixdim[1], volume_pixdim[2], volume_pixdim[3]]
        else:
            volume_spacing = [1.0, 1.0, 1.0]

        orientation_from, orientation_to = get_orientation_string(volume_affine)

        info_data = parse_info_yaml(info_file)

        mask_meta_dict = {'mask': mask_meta}
        mask_dtype_dict = {'mask': str(mask_data.dtype)}
        binary_mask_paths = {}

        binary_mask_files = sorted(pair_dir.glob(f'{combined_id}_mask_*.nii.gz'))

        for binary_mask_file in binary_mask_files:
            binary_match = re.match(rf'{combined_id}_mask_(.+)\.nii\.gz', binary_mask_file.name)
            if binary_match:
                label_name = binary_match.group(1)
                binary_mask_data, binary_mask_meta = loader(str(binary_mask_file))

                mask_meta_dict[f'mask_{label_name}'] = binary_mask_meta
                mask_dtype_dict[f'mask_{label_name}'] = str(binary_mask_data.dtype)

                rel_path = binary_mask_file.relative_to(root_path)
                binary_mask_paths[f'mask_{label_name}'] = str(rel_path).replace('\\', '/')

        is_consistent, consistency_issues = check_metadata_consistency(volume_meta, mask_meta_dict)

        dtype_consistent, dtype_issues = check_mask_dtype_consistency(mask_dtype_dict)

        if consistency_issues or dtype_issues:
            debug_info.extend(consistency_issues)
            debug_info.extend(dtype_issues)
            print(f"\nConsistency issues in {pair_name}:")
            for issue in consistency_issues + dtype_issues:
                print(f"  - {issue}")

        info_rel_path = info_file.relative_to(root_path)
        volume_rel_path = volume_file.relative_to(root_path)
        mask_rel_path = mask_file.relative_to(root_path)

        record = {
            'ID': f'patient{patient_id}_frame{frame_id}',
            'patient': patient_id,
            'frame': frame_id,
            'group': info_data['group'] if info_data['group'] is not None else '',
            'phase': info_data['phase'] if info_data['phase'] is not None else '',
            'tot_frame': info_data['tot_frame'] if info_data['tot_frame'] is not None else '',
            'height': info_data['height'] if info_data['height'] is not None else '',
            'weight': info_data['weight'] if info_data['weight'] is not None else '',
            'info': str(info_rel_path.as_posix()),
            'volume': str(volume_rel_path.as_posix()),
            'mask': str(mask_rel_path.as_posix()),
            'szx': volume_shape[0] if len(volume_shape) > 0 else '',
            'szy': volume_shape[1] if len(volume_shape) > 1 else '',
            'szz': volume_shape[2] if len(volume_shape) > 2 else '',
            'spx': volume_spacing[0],
            'spy': volume_spacing[1],
            'spz': volume_spacing[2],
            'orientation_from': orientation_from,
            'orientation_to': orientation_to,
            'vol_dtype': str(volume_data.numpy().dtype),
            'mask_dtype': str(mask_data.numpy().dtype),
            'transform': format_transform_for_excel(volume_affine.numpy())
        }

        record.update(binary_mask_paths)

        if verbose:
            record['debug'] = '; '.join(debug_info) if debug_info else ''

        return record

    except Exception as e:
        error_msg = f"Error processing {pair_dir}: {str(e)}"
        debug_info.append(error_msg)
        print(error_msg)
        if verbose:
            return {
                'ID': f'patient{patient_id}_frame{frame_id}',
                'patient': patient_id,
                'frame': frame_id,
                'group': '',
                'phase': '',
                'tot_frame': '',
                'height': '',
                'weight': '',
                'info': '',
                'volume': '',
                'mask': '',
                'szx': '',
                'szy': '',
                'szz': '',
                'spx': '',
                'spy': '',
                'spz': '',
                'orientation_from': '',
                'orientation_to': '',
                'vol_dtype': '',
                'mask_dtype': '',
                'transform': '',
                'debug': error_msg
            }
        return None


def format_transform_for_excel(transform):
    """
    Format 4x4 transformation matrix for Excel output.
    
    Args:
        transform (np.ndarray or str): 4x4 affine transformation matrix or empty string
        
    Returns:
        str: Formatted string representation of the matrix
    """
    formatter = {'float_kind': lambda x: f'{x:>14.8f}'}
    matrix_str = np.array2string(transform, formatter=formatter, separator='')
    return matrix_str


def process_root_dir(root_dir, output_manifest_file, sheet_name='Manifest', verbose=False):
    """
    Process all subset directories and generate manifest.
    
    Args:
        root_dir (str or Path): Root directory containing subset subdirectories
        output_manifest_file (str or Path): Output Excel manifest file path
        sheet_name (str): Sheet name in the Excel file
        verbose (bool): Whether to add debug column with exception information
    """
    root_path = Path(root_dir)
    output_path = Path(output_manifest_file)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    subset_dirs = [d for d in root_path.iterdir() if d.is_dir()]

    if not subset_dirs:
        print(f"Warning: No subset directories found in {root_dir}")
        return

    loader = LoadImage(image_only=False, dtype=None)

    all_records = []
    all_binary_mask_keys = set()

    print(f"Found {len(subset_dirs)} subset directories")
    print(f"Generating manifest...\n")

    for subset_dir in tqdm(subset_dirs, desc='Processing subsets'):
        pair_dirs = sorted(subset_dir.glob('patient*_frame*'))

        for pair_dir in tqdm(pair_dirs, desc=f'  {subset_dir.name}', leave=False):
            record = process_pair_dir(pair_dir, root_path, loader, verbose=verbose)

            if record is not None:
                binary_mask_keys = [k for k in record.keys() if
                                    k.startswith('mask_') and k not in ['mask', 'mask_dtype']]
                all_binary_mask_keys.update(binary_mask_keys)
                all_records.append(record)

    if not all_records:
        print("No valid records found")
        return

    df = pd.DataFrame(all_records)

    df['patient'] = df['patient'].astype(int)
    df['frame'] = df['frame'].astype(int)
    df = df.sort_values(by=['patient', 'frame'], ascending=[True, True])
    df['patient'] = df['patient'].astype(str).str.zfill(3)
    df['frame'] = df['frame'].astype(str).str.zfill(2)

    column_order = ['ID', 'patient', 'frame', 'group', 'phase', 'tot_frame', 'height', 'weight', 'info', 'volume', 'mask']
    column_order.extend(sorted(all_binary_mask_keys))
    column_order.extend(['szx', 'szy', 'szz', 'spx', 'spy', 'spz', 'orientation_from', 'orientation_to',
                         'vol_dtype', 'mask_dtype', 'transform'])

    if verbose:
        column_order.append('debug')

    df = df[column_order]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"\nManifest generation completed!")
    print(f"Total records: {len(all_records)}")
    print(f"Binary mask types found: {sorted(all_binary_mask_keys)}")
    print(f"Output file: {output_path}")
    print(f"Sheet name: {sheet_name}")


def main():
    """
    Main function to orchestrate the manifest generation process.
    """
    args = parse_args()

    print(f"Processing data from: {args.root_dir}")
    print(f"Output manifest file: {args.output_manifest_file}")
    print(f"Sheet name: {args.sheet_name}")
    if args.verbose:
        print("Verbose mode: Enabled (debug column will be added)")

    process_root_dir(args.root_dir, args.output_manifest_file, args.sheet_name, args.verbose)

    print("Dataset manifest generation completed successfully!")


if __name__ == '__main__':
    main()
