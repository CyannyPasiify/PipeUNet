# -*- coding: utf-8 -*-
"""
生成一个用于生成数据存档检查清单的工具脚本。使用argparse。接收-r/--root_dir指定数据集根目录，此根目录下有training和testing两个子目录，每个子目录下有命名格式为patientxxx（xxx为3位数字编号，高位不足补0）的二级子目录，二级子目录下包含一个名为patientxxx_4d.nii.gz的4D图像（包含时间维），若干个名为patientxxx_frameyy.nii.gz（yy为2位数帧编号，高位不足补0）的单帧3D图像以及与之配套的名为patientxxx_frameyy_gt.nii.gz的蒙版文件，忽略其它后缀的文件。接收-o/--output_manifest_file指定输出Excel清单文件路径，清单文件中依次记录以下信息： 
 file_path：记录文件的路径，从数据集根目录开始。
 subset：记录文件所属的数据集子集，training或testing。
 patient：记录文件所属病人的3位数字编号xxx。
 frame：记录文件所属帧号的2位数字编号yy。
 type：帧图像记录为v3d，蒙版图像记录为m3d。
 szx：记录图像的x规格。
 szy：记录图像的y规格。
 szz：记录图像的z规格。
 spx：记录图像的x间距。
 spy：记录图像的y间距。
 spz：记录图像的z间距。
 orientation_from：记录图像的L/R、A/P、S/I朝向，记录开始端侧，例如LPS、RAI等，如果其方向不标准（transfomr空间变换矩阵的前3×3不是对角阵），则记录为Oblique。
 orientation_to：记录图像的L/R、A/P、S/I朝向，记录结束端侧，如果其方向不标准（transfomr空间变换矩阵的前3×3不是对角阵），则记录为Oblique。
 transform：记录图像携带的4×4空间变换矩阵。
 diff_trans_f_norm：只对蒙版文件进行记录，其它留空。记录patientxxx_frameyy_gt.nii.gz蒙版文件与其对应的patientxxx_frameyy.nii.gz图像文件的transform空间变换矩阵的差矩阵的F范数。
 使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。
 接收可选参数-s/--sheet_name，如果指定则将工作表重命名为指定名称。
 除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Data Archive Manifest Generator for ACDC Dataset

This script generates a comprehensive manifest Excel file for the ACDC (Automated Cardiac 
Diagnosis Challenge) dataset, extracting metadata from NIfTI image files including dimensions,
spacings, orientations, and transformation matrices.

Parameters:
    -r, --root_dir: Root directory of the dataset containing 'training' and 'testing' subdirectories
    -o, --output_manifest_file: Output path for the generated Excel manifest file
    -s, --sheet_name: Optional sheet name for the Excel worksheet

Usage Examples:
    python 01_gen_manifest.py -r /path/to/dataset -o manifest.xlsx
    python 01_gen_manifest.py --root_dir ./ACDC --output_manifest_file ./output/manifest.xlsx --sheet_name ACDC_Manifest
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
        description='Generate data archive manifest for ACDC dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/dataset -o manifest.xlsx
  %(prog)s --root_dir ./ACDC --output_manifest_file ./output/manifest.xlsx --sheet_name ACDC_Manifest
        """
    )
    
    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing training and testing subdirectories'
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
    
    orientation_from = ''.join([get_axis_label(rotation[:, i]) for i in range(3)])
    orientation_to = ''.join([get_axis_label(-rotation[:, i]) for i in range(3)])
    
    return orientation_from, orientation_to


def get_image_metadata(nii_path):
    """
    Extract metadata from a NIfTI image file using MONAI LoadImage.
    
    Args:
        nii_path (str or Path): Path to the NIfTI file
        
    Returns:
        dict: Dictionary containing image metadata including dimensions, spacings, orientation, and transform
    """
    loader = LoadImage(image_only=False)
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
        'szt': shape[3] if len(shape) > 3 else '',
        'spx': zooms[1] if len(zooms) > 1 else '',
        'spy': zooms[2] if len(zooms) > 2 else '',
        'spz': zooms[3] if len(zooms) > 3 else '',
        'spt': zooms[4] if len(zooms) > 4 else '',
        'orientation_from': orientation_from,
        'orientation_to': orientation_to,
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


def scan_dataset(root_dir):
    """
    Scan the dataset directory structure and collect all relevant image files.
    
    Args:
        root_dir (str or Path): Root directory path containing training and testing subdirectories
        
    Returns:
        list: List of dictionaries containing file information
    """
    root_path = Path(root_dir)
    file_info_list = []
    
    subdirs = ['training', 'testing']
    
    for subdir in tqdm(subdirs, desc='Processing subsets'):
        subdir_path = root_path / subdir
        if not subdir_path.exists():
            continue
        
        patient_dirs = sorted(subdir_path.glob('patient*'))
        
        for patient_dir in tqdm(patient_dirs, desc=f'  {subdir} patients', leave=False):
            patient_match = re.match(r'patient(\d{3})', patient_dir.name)
            if not patient_match:
                continue
            
            patient_id = patient_match.group(1)
            
            nii_files = list(patient_dir.glob('*.nii.gz'))
            
            frame_transforms = {}
            
            for nii_file in tqdm(nii_files, desc=f'    patient{patient_id}', leave=False):
                filename = nii_file.name
                
                frame_match = re.match(rf'patient{patient_id}_frame(\d{{2}})\.nii\.gz', filename)
                gt_match = re.match(rf'patient{patient_id}_frame(\d{{2}})_gt\.nii\.gz', filename)
                
                if frame_match:
                    frame_id = frame_match.group(1)
                    metadata = get_image_metadata(str(nii_file))
                    frame_transforms[frame_id] = metadata['transform']
                    
                    rel_path = nii_file.relative_to(root_path)
                    file_info = {
                        'file_path': str(rel_path),
                        'subset': subdir,
                        'patient': patient_id,
                        'frame': frame_id,
                        'type': 'v3d',
                        'szx': metadata['szx'],
                        'szy': metadata['szy'],
                        'szz': metadata['szz'],
                        'spx': metadata['spx'],
                        'spy': metadata['spy'],
                        'spz': metadata['spz'],
                        'orientation_from': metadata['orientation_from'],
                        'orientation_to': metadata['orientation_to'],
                        'transform': metadata['transform'],
                        'diff_trans_f_norm': ''
                    }
                    file_info_list.append(file_info)
                
                elif gt_match:
                    frame_id = gt_match.group(1)
                    metadata = get_image_metadata(str(nii_file))
                    
                    diff_f_norm = ''
                    if frame_id in frame_transforms:
                        diff_f_norm = calculate_transform_f_norm_diff(
                            metadata['transform'],
                            frame_transforms[frame_id]
                        )
                    
                    rel_path = nii_file.relative_to(root_path)
                    file_info = {
                        'file_path': str(rel_path),
                        'subset': subdir,
                        'patient': patient_id,
                        'frame': frame_id,
                        'type': 'm3d',
                        'szx': metadata['szx'],
                        'szy': metadata['szy'],
                        'szz': metadata['szz'],
                        'spx': metadata['spx'],
                        'spy': metadata['spy'],
                        'spz': metadata['spz'],
                        'orientation_from': metadata['orientation_from'],
                        'orientation_to': metadata['orientation_to'],
                        'transform': metadata['transform'],
                        'diff_trans_f_norm': diff_f_norm
                    }
                    file_info_list.append(file_info)
    
    return file_info_list


def format_transform_for_excel(transform):
    """
    Format 4x4 transformation matrix for Excel output.
    
    Args:
        transform (np.ndarray or str): 4x4 affine transformation matrix or empty string
        
    Returns:
        str: Formatted string representation of the matrix
    """
    matrix_str = np.array2string(transform, precision=6, suppress_small=True)
    return matrix_str.replace('\n', '; ')


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
            'patient': file_info['patient'],
            'frame': file_info['frame'],
            'type': file_info['type'],
            'szx': file_info['szx'],
            'szy': file_info['szy'],
            'szz': file_info['szz'],
            'spx': file_info['spx'],
            'spy': file_info['spy'],
            'spz': file_info['spz'],
            'orientation_from': file_info['orientation_from'],
            'orientation_to': file_info['orientation_to'],
            'transform': format_transform_for_excel(file_info['transform']),
            'diff_trans_f_norm': file_info['diff_trans_f_norm']
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    """
    Main function to orchestrate the manifest generation process.
    """
    args = parse_args()
    
    print(f"Scanning dataset from: {args.root_dir}")
    file_info_list = scan_dataset(args.root_dir)
    print(f"Found {len(file_info_list)} image files")
    
    print(f"Generating manifest Excel file: {args.output_manifest_file}")
    generate_manifest_excel(file_info_list, args.output_manifest_file, args.sheet_name)
    
    print("Manifest generation completed successfully!")


if __name__ == '__main__':
    main()