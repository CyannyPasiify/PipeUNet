# -*- coding: utf-8 -*-
"""
接收-r/--root_dir指定数据存档根目录，此根目录下有图像Images和蒙版Masks目录，Images子目录下有形如Breast_MRI_{sid:03d}的图像样本目录，样本pid=Breast_MRI_{sid:03d}，图像样本目录下包含若干图像文件；Masks子目录下包含形如Breast_MRI_{sid:03d}的样本目录，样本目录下包含若干蒙版文件，形如Segmentation_{pid}_{seg_type}.nii.gz 
接收-o/--output_manifest_file指定输出Excel清单文件路径，清单文件中依次记录以下信息： 
  file_path：记录文件的路径，从数据集根目录开始。 
  pid：记录文件所属样本的编号（str）。 
  type：images图像记录为v3d，masks蒙版记录为m3d。 
  split：样本所属的子集。
  primary：记录此图像文件是否为首选，1-首选，2-非首选或者是蒙版
  szx：记录图像的x规格。 
  szy：记录图像的y规格。 
  szz：记录图像的z规格。 
  spx：记录图像的x间距。 
  spy：记录图像的y间距。 
  spz：记录图像的z间距。 
  orientation_from：记录图像的L/R、A/P、S/I朝向，记录开始端侧，例如LPS、RAI等，如果其方向不标准（transfomr空间变换矩阵的前3×3不是对角阵），则记录为Oblique。 
  orientation_to：记录图像的L/R、A/P、S/I朝向，记录结束端侧，如果其方向不标准（transfomr空间变换矩阵的前3×3不是对角阵），则记录为Oblique。 
  dtype：记录图像文件的数据类型。 
  transform：记录图像携带的4×4空间变换矩阵。 
  diff_trans_f_norm：只对蒙版文件进行记录，其它留空。记录{pid}_*.nii.gz蒙版文件与其对应的首选图像文件的transform空间变换矩阵的差矩阵的F范数。 
使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。 使用tqdm显示进度，并展示当前正在处理样本的pid。 
接收可选参数-s/--sheet_name（默认Manifest），如果指定则将工作表重命名为指定名称。 
除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Data Archive Manifest Generator for Duke-Breast-FGT-Segmentation-2025.4.10 Dataset

This script generates a comprehensive manifest Excel file for the Duke-Breast-FGT-Segmentation dataset,
extracting metadata from NIfTI image files including dimensions, spacings, orientations, and transformation matrices.

Key Features:
    - Scans both Images and Masks directories recursively
    - Processes NIfTI files in Breast_MRI_{sid:03d} sample directories
    - Extracts detailed metadata including dimensions, spacings, and orientations
    - Calculates transformation matrix differences between masks and corresponding images
    - Supports dataset split information from JSON files (train/val/test)
    - Displays progress with tqdm, showing current sample IDs
    - Generates well-formatted Excel files with clear column organization

Parameters:
    -r, --root_dir: Root directory of the dataset containing Images and Masks directories
    -o, --output_manifest_file: Output path for the generated Excel manifest file
    -s, --sheet_name: Optional sheet name for the Excel worksheet (default: Manifest)
    -d, --dataset_split_json: Optional JSON file containing dataset splits (train/val/test) with sample IDs

The manifest file will include the following columns:
    file_path: Path from dataset root directory
    pid: Patient ID (Breast_MRI_{sid:03d})
    type: File type (v3d for images, m3d for masks)
    split: Dataset split (train/val/test) if provided in split JSON
    primary: 1 if this is the primary image for the sample, 0 otherwise
    szx, szy, szz: Image dimensions
    spx, spy, spz: Pixel spacings
    orientation_from, orientation_to: Image orientation
    dtype: Data type
    transform: 4x4 affine transformation matrix
    diff_trans_f_norm: Frobenius norm of transform difference between mask and corresponding primary image

Usage Examples:
    # Basic usage
    python 01_gen_archive_manifest.py -r /path/to/Duke-Breast-FGT-Segmentation-2025.4.10 -o /path/to/archive_manifest.xlsx
    
    # With custom sheet name
    python 01_gen_archive_manifest.py --root_dir /path/to/Duke-Breast-FGT-Segmentation-2025.4.10 --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name Duke_Manifest
    
    # With dataset split information
    python 01_gen_archive_manifest.py -r /path/to/Duke-Breast-FGT-Segmentation-2025.4.10 -o /path/to/archive_manifest.xlsx -d /path/to/dataset_split.json
"""


import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from monai.transforms import LoadImage
from typing import Dict, List, Tuple, Any, Optional, Union


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir, output_manifest_file, sheet_name, and dataset_split_json
    """

    parser = argparse.ArgumentParser(
        description='Generate data archive manifest for Duke-Breast-FGT-Segmentation-2025.4.10 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/Duke-Breast-FGT-Segmentation-2025.4.10 -o /path/to/archive_manifest.xlsx
  %(prog)s --root_dir /path/to/Duke-Breast-FGT-Segmentation-2025.4.10 --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name Duke_Manifest
  %(prog)s -r /path/to/Duke-Breast-FGT-Segmentation-2025.4.10 -o /path/to/archive_manifest.xlsx -d /path/to/dataset_split.json
        """
    )
    
    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing Images and Masks directories'
    )
    
    parser.add_argument(
        '-o', '--output_manifest_file',
        type=str,
        required=True,
        help='Output path for the generated Excel manifest file'
    )
    
    parser.add_argument(
        '-s', '--sheet_name',
        type=str,
        default='Manifest',
        help='Optional sheet name for the Excel worksheet (default: Manifest)'
    )
    
    parser.add_argument(
        '-d', '--dataset_split_json',
        type=str,
        default=None,
        help='Optional JSON file containing dataset splits (train/val/test) with sample IDs'
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
        Tuple[str, str]: (orientation_from, orientation_to) strings, or ('Oblique(closest to X)', 'Oblique(closest to X)') if non-standard
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


def get_image_metadata(nii_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from a NIfTI image file using MONAI LoadImage.
    
    Args:
        nii_path (Union[str, Path]): Path to the NIfTI file
        
    Returns:
        Dict[str, Any]: Dictionary containing image metadata including dimensions, spacings, orientation, and transform
    """
    loader = LoadImage(image_only=False, dtype=None)
    img_data, img_meta = loader(str(nii_path))
    
    affine = img_meta['affine'].numpy()
    shape = img_meta['spatial_shape'].tolist()
    
    pixdim = img_meta.get('pixdim', None)
    if pixdim is not None:
        zooms = [pixdim[i] for i in range(len(pixdim))]
    else:
        zooms = [1.0] * len(shape)
    
    orientation_from, orientation_to = get_orientation_string(affine)
    
    metadata = {
        'szx': shape[0] if len(shape) > 0 else '',
        'szy': shape[1] if len(shape) > 1 else '',
        'szz': shape[2] if len(shape) > 2 else '',
        'spx': zooms[1] if len(zooms) > 1 else '',
        'spy': zooms[2] if len(zooms) > 2 else '',
        'spz': zooms[3] if len(zooms) > 3 else '',
        'orientation_from': orientation_from,
        'orientation_to': orientation_to,
        'dtype': str(img_data.numpy().dtype),
        'transform': affine
    }
    
    return metadata


def load_dataset_split(split_json_path: Optional[Union[str, Path]]) -> Dict[str, Dict[str, str]]:
    """
    Load dataset split information from JSON file and create sample ID to split mapping.
    
    Args:
        split_json_path (Optional[Union[str, Path]]): Path to dataset split JSON file
        
    Returns:
        Dict[str, Dict[str, str]]: Mapping from sample ID to dict with 'split' and 'primary_image'
    """
    sample_info = {}
    
    if split_json_path:
        split_json_path = Path(split_json_path)
        if split_json_path.exists():
            try:
                with open(split_json_path, 'r', encoding='utf-8') as f:
                    split_data = json.load(f)
                
                for split_name in ['train', 'val', 'test']:
                    if split_name in split_data:
                        for item in split_data[split_name]:
                            if 'sample' in item:
                                sample_id = item['sample']
                                sample_info[sample_id] = {
                                    'subset': split_name,
                                    'primary_image': item.get('primary_image', '')
                                }
                
                print(f"Loaded dataset splits from: {split_json_path}")
                print(f"Total samples in splits: {len(sample_info)}")
            except Exception as e:
                print(f"Error loading dataset split JSON: {e}")
        else:
            print(f"Dataset split JSON file not found: {split_json_path}")
    
    return sample_info


def calculate_transform_f_norm_diff(affine1: np.ndarray, affine2: np.ndarray) -> float:
    """
    Calculate the Frobenius norm of the difference between two transformation matrices.
    
    Args:
        affine1 (np.ndarray): First 4x4 affine matrix
        affine2 (np.ndarray): Second 4x4 affine matrix
        
    Returns:
        float: Frobenius norm of the difference matrix
    """
    diff = affine1 - affine2
    f_norm = np.linalg.norm(diff, 'fro')
    return f_norm


def parse_mask_filename(filename: str) -> Tuple[str, str]:
    """
    Parse pid and seg_type from mask filename.
    
    Args:
        filename (str): Mask filename in format Segmentation_{pid}_{seg_type}.nii.gz
        
    Returns:
        Tuple[str, str]: (pid, seg_type)
    """
    match = re.match(r'Segmentation_([^_]+(?:_[^_]+)*)_([^_]+)\.nii\.gz', filename)
    if match:
        return match.group(1), match.group(2)
    return '', ''


def scan_dataset(root_dir: Union[str, Path], sample_info: Dict[str, Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Scan the dataset directory structure and collect all relevant image files.
    
    Args:
        root_dir (Union[str, Path]): Root directory path containing Images and Masks directories
        sample_info (Dict[str, Dict[str, str]]): Mapping from sample ID to dict with 'subset' and 'primary_image'
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing file information
    """
    root_path = Path(root_dir)
    file_info_list: List[Dict[str, Any]] = []
    
    if sample_info is None:
        sample_info = {}
    
    image_transforms: Dict[str, np.ndarray] = {}
    primary_images: Dict[str, str] = {}
    
    # Process Images directory
    images_path = root_path / 'Images'
    if images_path.exists():
        sample_dirs = sorted(images_path.glob('Breast_MRI_*'))
        with tqdm(sample_dirs, desc='Processing Images') as pbar:
            for sample_dir in pbar:
                pid = sample_dir.name
                pbar.set_description(f'Processing Images - {pid}')
                
                # Process all NIfTI files in the sample directory
                nii_files = sorted(sample_dir.glob('*.nii.gz'))
                for nii_file in nii_files:
                    # Extract metadata
                    metadata = get_image_metadata(str(nii_file))
                    
                    # Save image transform for future mask comparison
                    key = pid
                    image_transforms[key] = metadata['transform']
                    
                    # Get subset and primary image information for this sample
                    sample_data = sample_info.get(pid, {})
                    subset = sample_data.get('subset', '')
                    primary_image = sample_data.get('primary_image', '')
                    
                    # Check if this is the primary image
                    rel_path_str = str(nii_file.relative_to(root_path).as_posix())
                    is_primary = 1 if rel_path_str == primary_image else 0
                    
                    # Save image transform - always use primary image for mask comparison if available
                    if is_primary or pid not in primary_images:
                        primary_images[pid] = metadata['transform']
                    
                    # Create file info dictionary
                    file_info = {
                        'file_path': rel_path_str,
                        'pid': pid,
                        'type': 'v3d',
                        'subset': subset,
                        'primary': is_primary,
                        'szx': metadata['szx'],
                        'szy': metadata['szy'],
                        'szz': metadata['szz'],
                        'spx': metadata['spx'],
                        'spy': metadata['spy'],
                        'spz': metadata['spz'],
                        'orientation_from': metadata['orientation_from'],
                        'orientation_to': metadata['orientation_to'],
                        'dtype': metadata['dtype'],
                        'transform': metadata['transform'],
                        'diff_trans_f_norm': ''
                    }
                    file_info_list.append(file_info)
    
    # Process Masks directory
    masks_path = root_path / 'Masks'
    if masks_path.exists():
        sample_dirs = sorted(masks_path.glob('Breast_MRI_*'))
        with tqdm(sample_dirs, desc='Processing Masks') as pbar:
            for sample_dir in pbar:
                pid = sample_dir.name
                pbar.set_description(f'Processing Masks - {pid}')
                
                # Process all NIfTI files in the sample directory
                nii_files = sorted(sample_dir.glob('*.nii.gz'))
                for nii_file in nii_files:
                    # Extract metadata
                    metadata = get_image_metadata(str(nii_file))
                    
                    # Calculate transform difference using primary image if available
                    diff_f_norm = ''
                    if pid in primary_images:
                        diff_f_norm = calculate_transform_f_norm_diff(
                            metadata['transform'],
                            primary_images[pid]
                        )
                    
                    # Get subset information for this sample
                    sample_data = sample_info.get(pid, {})
                    subset = sample_data.get('subset', '')
                    
                    # Create file info dictionary
                    rel_path_str = str(nii_file.relative_to(root_path).as_posix())
                    file_info = {
                        'file_path': rel_path_str,
                        'pid': pid,
                        'type': 'm3d',
                        'subset': subset,
                        'primary': 0,  # Masks are never primary images
                        'szx': metadata['szx'],
                        'szy': metadata['szy'],
                        'szz': metadata['szz'],
                        'spx': metadata['spx'],
                        'spy': metadata['spy'],
                        'spz': metadata['spz'],
                        'orientation_from': metadata['orientation_from'],
                        'orientation_to': metadata['orientation_to'],
                        'dtype': metadata['dtype'],
                        'transform': metadata['transform'],
                        'diff_trans_f_norm': diff_f_norm
                    }
                    file_info_list.append(file_info)
    
    return file_info_list


def format_transform_for_excel(transform: Union[np.ndarray, str]) -> str:
    """
    Format 4x4 transformation matrix for Excel output.
    
    Args:
        transform (Union[np.ndarray, str]): 4x4 affine transformation matrix or empty string
        
    Returns:
        str: Formatted string representation of the matrix
    """
    if isinstance(transform, str):
        return transform
    formatter = {'float_kind': lambda x: f'{x:>14.8f}'}
    matrix_str = np.array2string(transform, formatter=formatter, separator='')
    return matrix_str


def generate_manifest_excel(file_info_list: List[Dict[str, Any]], 
                           output_path: Union[str, Path], 
                           sheet_name: str = 'Manifest') -> None:
    """
    Generate Excel manifest file from collected file information.
    
    Args:
        file_info_list (List[Dict[str, Any]]): List of dictionaries containing file metadata
        output_path (Union[str, Path]): Output path for the Excel file
        sheet_name (str): Name for the Excel worksheet
    """
    # Create DataFrame
    df = pd.DataFrame(file_info_list)
    
    # Format transform matrix for Excel
    if 'transform' in df.columns:
        df['transform'] = df['transform'].apply(format_transform_for_excel)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Manifest file generated successfully: {output_path}")
    print(f"Total files processed: {len(file_info_list)}")


def main() -> None:
    """
    Main function to orchestrate the manifest generation process.
    """
    args: argparse.Namespace = parse_args()
    
    # Load dataset split information if provided
    sample_info = load_dataset_split(args.dataset_split_json)
    
    print(f"Scanning dataset directory: {args.root_dir}")
    file_info_list = scan_dataset(args.root_dir, sample_info)
    
    print(f"Generating manifest Excel file: {args.output_manifest_file}")
    generate_manifest_excel(file_info_list, args.output_manifest_file, args.sheet_name)
    
    print("Manifest generation completed successfully!")


if __name__ == '__main__':
    main()
