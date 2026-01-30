# -*- coding: utf-8 -*-
"""
 -r/--root_dir指定根目录，根目录下包含若干层级的子目录，其结构为{output_dir}/{site}/{collection}/{subset}/{seq}_{pid}，{seq}_{pid}为样本目录，其中包含形如{seq}_{pid}_info.yaml的元信息文件，{seq}_{pid}_volume.nii.gz的图像文件，{seq}_{pid}_mask.nii.gz的蒙版文件，{seq}_{pid}_mask_{label_index}_{label_name}.nii.gz的二值蒙版文件。 
 接收-o/--output_manifest_file指定输出Excel清单文件路径，清单文件中依次记录以下信息： 
    ID：记录文件所属样本的ID（{seq}_{pid}）。 
    site：记录样本所属的中心，从yaml中获取。 
    collection：记录文件所属的选集，从yaml中获取。 
    subset：记录文件所属的子集，从yaml中获取。 
    seq：记录文件的所属样本的前置序号（str），从yaml中获取。 
    pid：记录文件所属样本的编号（str），从yaml中获取。 
    valid_labels：记录不为空的二值蒙版对应的标签列表，每个标签按照{label_index}_{label_name}格式记录，逗号','分隔。 
    info：记录元信息文件的路径，相对于数据集根，使用POSIX路径连接符。 
    volume：记录图像文件的路径，相对于数据集根，使用POSIX路径连接符。 
    mask：记录蒙版文件的路径，相对于数据集根，使用POSIX路径连接符。 
    若干 mask_{label_index}_{label_name}：记录{label_index}_{label_name}二值蒙版文件的路径，相对于数据集根，使用POSIX路径连接符。 
    若干 mask_{label_index}_{label_name}_existence：如果{label_index}_{label_name}二值蒙版没有前景，则记录为0，否则记录为1。 
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
 扫描期间检查每个样本图像与全部蒙版的空间变换矩阵，规格，间距，原点，朝向的一致性，以及全部蒙版文件数据类型的一致性，如果存在不一致则将信息输出到控制台。 
   使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。使用tqdm在进度条前方显示正在处理的样本ID。 
   为所有变量和函数参数添加类型注解。 
   除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Dataset Manifest Generation Script for NME-Seg-2025.8.25 Dataset

This script generates a comprehensive manifest of the NME-Seg-2025.8.25 dataset, including volume and mask file paths,
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
    Recursively find all sample directories matching {seq}_{pid} pattern.
    
    Args:
        root_dir (Union[str, Path]): Root directory to search
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path = Path(root_dir)
    sample_dirs: List[Path] = []

    # Recursively search for sample directories
    for path in root_path.rglob('*_*'):
        if path.is_dir() and re.match(r'^[^_]+_[^_]+$', path.name):
            # Check if this directory contains an info file
            info_files = list(path.glob(f'{path.name}_info.yaml'))
            if info_files:
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

    # Get seq and pid from the directory name
    match = re.match(r'([^_]+)_([^_]+)', sample_name)
    if not match:
        return None

    seq: str = match.group(1)
    pid: str = match.group(2)
    sample_id: str = f'{seq}_{pid}'

    # Find files in the sample directory
    info_files: List[Path] = list(sample_dir.glob(f'{sample_id}_info.yaml'))
    volume_files: List[Path] = list(sample_dir.glob(f'{sample_id}_volume.nii.gz'))
    mask_files: List[Path] = list(sample_dir.glob(f'{sample_id}_mask.nii.gz'))

    debug_info: List[str] = []

    if not info_files:
        debug_info.append(f"Missing info file")

    if not volume_files:
        debug_info.append(f"Missing volume file")

    if not mask_files:
        debug_info.append(f"Missing mask file")

    info_file: Optional[Path] = info_files[0] if info_files else None
    volume_file: Optional[Path] = volume_files[0] if volume_files else None
    mask_file: Optional[Path] = mask_files[0] if mask_files else None

    # Load metadata from YAML file
    yaml_metadata: Dict[str, str] = {}
    if info_file:
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                yaml_metadata = yaml.safe_load(f)
        except Exception as e:
            debug_info.append(f"Error loading YAML file: {str(e)}")

    # Get metadata from YAML
    site: str = yaml_metadata.get('site', '')
    collection: str = yaml_metadata.get('collection', '')
    subset: str = yaml_metadata.get('subset', '')
    yaml_seq: str = yaml_metadata.get('seq', seq)
    yaml_pid: str = yaml_metadata.get('pid', pid)

    volume_data = None
    volume_meta = None
    mask_data = None
    mask_meta = None

    volume_shape = ['', '', '']
    volume_spacing = ['', '', '']
    orientation_from: str = ''
    orientation_to: str = ''
    volume_rel_path: str = ''
    mask_rel_path: str = ''
    info_rel_path: str = ''

    try:
        # Calculate relative paths
        if info_file:
            info_rel_path = str(info_file.relative_to(root_path).as_posix())

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
            volume_rel_path = str(volume_file.relative_to(root_path).as_posix())

        # Load mask data if available
        if mask_file:
            mask_data, mask_meta = loader(str(mask_file))
            mask_rel_path = str(mask_file.relative_to(root_path).as_posix())

        # Process binary masks
        mask_meta_dict: Dict[str, Any] = {}
        mask_dtype_dict: Dict[str, str] = {}
        binary_mask_paths: Dict[str, str] = {}
        binary_mask_existence: Dict[str, int] = {}
        valid_labels: List[str] = []

        if mask_meta:
            mask_meta_dict['mask'] = mask_meta
            mask_dtype_dict['mask'] = str(mask_data.dtype)

        binary_mask_files: List[Path] = sorted(sample_dir.glob(f'{sample_id}_mask_*.nii.gz'))

        for binary_mask_file in binary_mask_files:
            binary_match = re.match(r'.+_mask_([0-9]+_.*)\.nii\.gz', binary_mask_file.name)
            if binary_match:
                label_name = binary_match.group(1)
                binary_mask_data, binary_mask_meta = loader(str(binary_mask_file))

                mask_meta_dict[f'mask_{label_name}'] = binary_mask_meta
                mask_dtype_dict[f'mask_{label_name}'] = str(binary_mask_data.dtype)

                rel_path = str(binary_mask_file.relative_to(root_path).as_posix())
                binary_mask_paths[f'mask_{label_name}'] = rel_path

                # Check if binary mask has foreground pixels
                has_foreground: int = int(np.any(binary_mask_data.numpy() > 0))
                binary_mask_existence[f'mask_{label_name}_existence'] = has_foreground

                if has_foreground:
                    valid_labels.append(label_name)

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

        # Create the metadata dictionary
        metadata_dict: Dict[str, Any] = {
            'ID': sample_id,
            'site': site,
            'collection': collection,
            'subset': subset,
            'seq': yaml_seq,
            'pid': yaml_pid,
            'valid_labels': ','.join(valid_labels),
            'info': info_rel_path,
            'volume': volume_rel_path,
            'mask': mask_rel_path,
            'szx': volume_shape[0] if isinstance(volume_shape, (list, tuple, np.ndarray)) else '',
            'szy': volume_shape[1] if isinstance(volume_shape, (list, tuple, np.ndarray)) else '',
            'szz': volume_shape[2] if isinstance(volume_shape, (list, tuple, np.ndarray)) else '',
            'spx': volume_spacing[0] if isinstance(volume_spacing, (list, tuple, np.ndarray)) else '',
            'spy': volume_spacing[1] if isinstance(volume_spacing, (list, tuple, np.ndarray)) else '',
            'spz': volume_spacing[2] if isinstance(volume_spacing, (list, tuple, np.ndarray)) else '',
            'orientation_from': orientation_from,
            'orientation_to': orientation_to,
            'vol_dtype': str(volume_data.numpy().dtype) if volume_data is not None else '',
            'mask_dtype': str(mask_data.numpy().dtype) if mask_data is not None else '',
            'transform': format_transform_for_excel(volume_meta['affine']) if volume_meta is not None else ''
        }

        # Add binary mask paths and existence flags
        metadata_dict.update(binary_mask_paths)
        metadata_dict.update(binary_mask_existence)

        # Add debug info if verbose
        if verbose and debug_info:
            metadata_dict['debug'] = '; '.join(debug_info)

        return metadata_dict

    except Exception as e:
        if verbose:
            print(f"Error processing {sample_name}: {str(e)}")
        return None


def generate_manifest(root_dir: Union[str, Path], output_file: Union[str, Path], sheet_name: str = 'Manifest',
                      verbose: bool = False) -> None:
    """
    Generate dataset manifest Excel file.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing sample directories
        output_file (Union[str, Path]): Output Excel file path
        sheet_name (str): Sheet name in the Excel file
        verbose (bool): Whether to include debug information
    """
    root_path = Path(root_dir)
    output_path = Path(output_file)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all sample directories
    sample_dirs: List[Path] = find_sample_dirs(root_path)

    if not sample_dirs:
        print(f"No sample directories found in {root_dir}")
        return

    print(f"Found {len(sample_dirs)} sample directories")

    # Initialize MONAI loader
    loader = LoadImage(image_only=False, dtype=None)

    # Process each sample directory
    sample_metadata_list: List[Dict[str, Any]] = []

    with tqdm(sample_dirs, desc='Processing samples') as pbar:
        for sample_dir in pbar:
            # Update progress bar description with current sample ID
            sample_id = sample_dir.name
            pbar.set_description(f'Processing sample: {sample_id}')

            metadata = process_sample_dir(sample_dir, root_path, loader, verbose)
            if metadata:
                sample_metadata_list.append(metadata)

    if not sample_metadata_list:
        print(f"No valid samples found in {root_dir}")
        return

    # Create DataFrame from metadata list
    df = pd.DataFrame(sample_metadata_list)

    # Define the required columns in order
    basic_columns = ['ID', 'site', 'collection', 'subset', 'seq', 'pid',
                        'valid_labels', 'info', 'volume', 'mask', ]

    metric_columns = ['szx', 'szy', 'szz',
                      'spx', 'spy', 'spz',
                      'orientation_from', 'orientation_to',
                      'vol_dtype', 'mask_dtype', 'transform']

    # Get all binary mask columns
    binary_mask_columns = [col for col in df.columns if
                           col != 'mask_dtype' and col.startswith('mask_') and not col.endswith('_existence')]
    binary_mask_existence_columns = [col for col in df.columns if col.endswith('_existence')]

    # Combine all columns in the correct order
    all_columns = basic_columns + binary_mask_columns + binary_mask_existence_columns + metric_columns

    # Add any missing required columns
    for col in basic_columns + metric_columns:
        if col not in df.columns:
            df[col] = ''

    # Reorder columns
    df = df[all_columns]
    df.sort_values(by=['ID'], inplace=True)

    # Save to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nManifest generated successfully: {output_path}")
    print(f"Total samples processed: {len(df)}")


def main() -> None:
    """
    Main function to orchestrate the manifest generation process.
    """
    args: argparse.Namespace = parse_args()

    print(f"Processing data from: {args.root_dir}")
    print(f"Generating manifest at: {args.output_manifest_file}")
    print(f"Sheet name: {args.sheet_name}")

    generate_manifest(args.root_dir, args.output_manifest_file, args.sheet_name, args.verbose)

    print("Manifest generation completed successfully!")


if __name__ == '__main__':
    main()
