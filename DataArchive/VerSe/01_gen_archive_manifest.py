# -*- coding: utf-8 -*-
"""
生成一个用于生成数据存档检查清单的工具脚本。使用argparse。接收-r/--root_dir指定数据集根目录，此根目录下包含VerSe*的若干子数据集目录，子数据集目录下有形如dataset-*起头的子集目录，即dataset-01training/dataset-02validation/dataset-03test（训练集/验证集/测试集），每个子集目录下包含2个资源目录，即rawdata图像目录和derivatives标注目录，资源目录下包含若干以样本编号<seq>为名称的样本目录，图像目录下的样本目录下包含唯一的.nii.gz图像文件（如有多个报告路径和异常），标注目录下的样本目录下包含唯一的.nii.gz蒙版文件（如有多个报告路径和异常），读取这些文件并记录相关信息。接收-o/--output_manifest_file指定输出Excel清单文件路径，清单文件中依次记录以下信息： 
  file_path：记录文件的路径，从数据集根目录开始。 
   subset：记录文件所属的数据集子集，dataset-01training对应于train、dataset-02validation对应于val、dataset-03test对应于test。 
   seq：记录文件所属样本的编号（样本目录的名称）。 
   type：图像记录为v3d，蒙版记录为m3d。 
   szx：记录图像的x规格。 
   szy：记录图像的y规格。 
   szz：记录图像的z规格。 
   spx：记录图像的x间距。 
   spy：记录图像的y间距。 
   spz：记录图像的z间距。 
   orientation_from：记录图像的L/R、A/P、S/I朝向，记录开始端侧，例如LPS、RAI等，如果其方向不标准（transfomr空间变换矩阵的前3×3不是对角阵），则记录为Oblique(closest to *)。 
   orientation_to：记录图像的L/R、A/P、S/I朝向，记录结束端侧，如果其方向不标准（transfomr空间变换矩阵的前3×3不是对角阵），则记录为Oblique(closest to *)。 
   dtype：记录图像文件的数据类型。 
   transform：记录图像携带的4×4空间变换矩阵。 
   diff_trans_f_norm：只对蒙版文件进行记录，其它留空。记录每个样本的蒙版文件与其对应图像文件的transform空间变换矩阵的差矩阵的F范数。 
  使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。 
  接收可选参数-s/--sheet_name，如果指定则将工作表重命名为指定名称。
"""

"""
Data Archive Manifest Generator for VerSe Dataset

This script generates a comprehensive manifest Excel file for the VerSe dataset, extracting metadata from NIfTI image files including dimensions,
spacings, orientations, and transformation matrices.

Parameters:
    -r, --root_dir: Root directory of the dataset containing VerSe* subdirectories with dataset-01training, dataset-02validation, dataset-03test subdirectories
    -o, --output_manifest_file: Output path for the generated Excel manifest file
    -s, --sheet_name: Optional sheet name for the Excel worksheet

Usage Examples:
    python 01_gen_archive_manifest.py -r /path/to/VerSe -o /path/to/archive_manifest.xlsx
    python 01_gen_archive_manifest.py --root_dir /path/to/VerSe --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name VerSe_Manifest
"""

import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from monai.transforms import LoadImage


def parse_args():
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir, output_manifest_file, and sheet_name
    """
    parser = argparse.ArgumentParser(
        description='Generate data archive manifest for VerSe dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/VerSe -o /path/to/archive_manifest.xlsx
  %(prog)s --root_dir /path/to/VerSe --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name VerSe_Manifest
        """
    )
    
    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing VerSe* subdirectories with dataset-01training, dataset-02validation, dataset-03test subdirectories'
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
        tuple: (orientation_from, orientation_to) strings, or ('Oblique(closest to X)', 'Oblique(closest to X)') if non-standard
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


def get_image_metadata(nii_path):
    """
    Extract metadata from a NIfTI image file using MONAI LoadImage.
    
    Args:
        nii_path (str or Path): Path to the NIfTI file
        
    Returns:
        dict: Dictionary containing image metadata including dimensions, spacings, orientation, and transform
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


def calculate_transform_f_norm_diff(affine1, affine2):
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


def parse_subset_from_dirname(dirname):
    """
    Parse subset name (train/val/test) from directory name.
    
    Args:
        dirname (str): Directory name (e.g., dataset-01training, dataset-02validation, dataset-03test)
        
    Returns:
        str: Subset name ('train', 'val', or 'test')
    """
    if dirname == 'dataset-01training':
        return 'train'
    elif dirname == 'dataset-02validation':
        return 'val'
    elif dirname == 'dataset-03test':
        return 'test'
    else:
        return ''


def parse_type_from_dirname(dirname):
    """
    Parse file type (image/mask) from directory name.
    
    Args:
        dirname (str): Directory name (e.g., rawdata, derivatives)
        
    Returns:
        str: File type ('v3d' for rawdata, 'm3d' for derivatives)
    """
    if dirname == 'rawdata':
        return 'v3d'
    elif dirname == 'derivatives':
        return 'm3d'
    else:
        return ''


def scan_dataset(root_dir):
    """
    Scan the VerSe dataset directory structure and collect all relevant image files.
    
    Args:
        root_dir (str or Path): Root directory path containing VerSe* subdirectories with dataset-01training, dataset-02validation, dataset-03test subdirectories
        
    Returns:
        list: List of dictionaries containing file information
    """
    root_path = Path(root_dir)
    file_info_list = []
    
    verse_subdirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith('VerSe')])
    
    if not verse_subdirs:
        print(f"Warning: No VerSe* subdirectories found in {root_dir}")
        return file_info_list
    
    print(f"Found {len(verse_subdirs)} VerSe subdirectories: {[d.name for d in verse_subdirs]}")
    
    for verse_subdir in tqdm(verse_subdirs, desc='Processing VerSe subdirectories'):
        dataset_subdirs = sorted([d for d in verse_subdir.iterdir() if d.is_dir() and d.name.startswith('dataset-')])
        
        for dataset_subdir in tqdm(dataset_subdirs, desc=f'  {verse_subdir.name}', leave=False):
            subset = parse_subset_from_dirname(dataset_subdir.name)
            
            if not subset:
                continue
            
            resource_dirs = sorted([d for d in dataset_subdir.iterdir() if d.is_dir()])
            
            for resource_dir in resource_dirs:
                file_type = parse_type_from_dirname(resource_dir.name)
                
                if not file_type:
                    continue
                
                sample_dirs = sorted([d for d in resource_dir.iterdir() if d.is_dir()])
                
                for sample_dir in sample_dirs:
                    seq_id = sample_dir.name
                    
                    nii_files = list(sample_dir.glob('*.nii.gz'))
                    
                    if len(nii_files) == 0:
                        print(f"Warning: No .nii.gz file found in {sample_dir}")
                        continue
                    
                    if len(nii_files) > 1:
                        print(f"Warning: Multiple .nii.gz files found in {sample_dir}, using first one: {[f.name for f in nii_files]}")
                    
                    nii_file = nii_files[0]
                    
                    metadata = get_image_metadata(str(nii_file))
                    
                    diff_f_norm = ''
                    if file_type == 'm3d':
                        image_transform = get_image_transform_for_sample(root_path, seq_id, subset)
                        if image_transform is not None:
                            diff_f_norm = calculate_transform_f_norm_diff(
                                metadata['transform'],
                                image_transform
                            )
                    
                    rel_path = nii_file.relative_to(root_path)
                    file_info = {
                        'file_path': str(rel_path),
                        'subset': subset,
                        'seq': seq_id,
                        'type': file_type,
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


def get_image_transform_for_sample(root_path, seq_id, subset):
    """
    Get the transformation matrix for the image file of a given sample.
    
    Args:
        root_path (Path): Root directory path
        seq_id (str): Sample ID (directory name)
        subset (str): Subset name (train/val/test)
        
    Returns:
        np.ndarray or None: 4x4 affine transformation matrix if found, None otherwise
    """
    subset_dir_map = {
        'train': 'dataset-01training',
        'val': 'dataset-02validation',
        'test': 'dataset-03test'
    }
    
    subset_dir_name = subset_dir_map.get(subset)
    if not subset_dir_name:
        return None
    
    verse_subdirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith('VerSe')])
    
    for verse_subdir in verse_subdirs:
        dataset_subdir = verse_subdir / subset_dir_name
        if not dataset_subdir.exists():
            continue
        
        rawdata_dir = dataset_subdir / 'rawdata'
        if not rawdata_dir.exists():
            continue
        
        sample_dir = rawdata_dir / seq_id
        if not sample_dir.exists():
            continue
        
        nii_files = list(sample_dir.glob('*.nii.gz'))
        if len(nii_files) == 0:
            continue
        
        nii_file = nii_files[0]
        metadata = get_image_metadata(str(nii_file))
        return metadata['transform']
    
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


def generate_manifest_excel(file_info_list, output_path, sheet_name='Manifest'):
    """
    Generate Excel manifest file from collected file information.
    
    Args:
        file_info_list (list): List of dictionaries containing file metadata
        output_path (str or Path): Output path for the Excel file
        sheet_name (str): Name for the Excel worksheet
    """
    df_data = []
    
    for file_info in tqdm(file_info_list, desc='Generating Excel rows'):
        row = {
            'file_path': file_info['file_path'],
            'subset': file_info['subset'],
            'seq': file_info['seq'],
            'type': file_info['type'],
            'szx': file_info['szx'],
            'szy': file_info['szy'],
            'szz': file_info['szz'],
            'spx': file_info['spx'],
            'spy': file_info['spy'],
            'spz': file_info['spz'],
            'orientation_from': file_info['orientation_from'],
            'orientation_to': file_info['orientation_to'],
            'dtype': file_info['dtype'],
            'transform': format_transform_for_excel(file_info['transform']),
            'diff_trans_f_norm': file_info['diff_trans_f_norm']
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df = df.sort_values(by=['subset', 'seq'], ascending=[True, True])
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    """
    Main function to orchestrate the manifest generation process.
    """
    args = parse_args()
    
    print(f"Scanning VerSe dataset from: {args.root_dir}")
    file_info_list = scan_dataset(args.root_dir)
    print(f"Found {len(file_info_list)} image files")
    
    print(f"Generating manifest Excel file: {args.output_manifest_file}")
    generate_manifest_excel(file_info_list, args.output_manifest_file, args.sheet_name)
    
    print("Manifest generation completed successfully!")


if __name__ == '__main__':
    main()
