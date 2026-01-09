# -*- coding: utf-8 -*-
"""
生成一个用于生成数据存档检查清单的工具脚本。使用argparse。接收-r/--root_dir指定数据集根目录，此根目录下有以images/labels（图像/蒙版）起头，Tr/Ts/Va（train/test/val）结尾的若干子目录（即imagesTr，imagesTs，imagesVa，labelsTr，labelsTs，labelsVa中的一部分或全部），每个子目录下有命名格式为amos_xxxx.nii.gz（xxxx为4位数字编号，高位不足补0）的资源文件，其意义由上级子目录决定是图像或是蒙版。接收-o/--output_manifest_file指定输出Excel清单文件路径，清单文件中依次记录以下信息： 
 file_path：记录文件的路径，从数据集根目录开始。 
  subset：记录文件所属的数据集子集，train、val或test。 
  seq：记录文件所属样本的4位数字编号xxxx。 
  type：images图像记录为v3d，labels蒙版记录为m3d。 
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
  diff_trans_f_norm：只对蒙版文件进行记录，其它留空。记录amos_xxxx.nii.gz蒙版文件与其对应的amos_xxxx.nii.gz图像文件的transform空间变换矩阵的差矩阵的F范数。
 使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。
 接收可选参数-s/--sheet_name，如果指定则将工作表重命名为指定名称。
 除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Data Archive Manifest Generator for AMOS22 Dataset

This script generates a comprehensive manifest Excel file for the AMOS22 dataset, extracting metadata from NIfTI image files including dimensions,
spacings, orientations, and transformation matrices.

Parameters:
    -r, --root_dir: Root directory of the dataset containing imagesTr, imagesTs, imagesVa, labelsTr, labelsTs, labelsVa subdirectories
    -o, --output_manifest_file: Output path for the generated Excel manifest file
    -s, --sheet_name: Optional sheet name for the Excel worksheet
    -m, --labeled_data_meta: Optional CSV file path containing labeled data metadata with amos_id column

Usage Examples:
    python 01_gen_archive_manifest.py -r /path/to/AMOS22 -o /path/to/archive_manifest.xlsx
    python 01_gen_archive_manifest.py --root_dir /path/to/AMOS22 --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name Manifest
    python 01_gen_archive_manifest.py -r /path/to/AMOS22 -o /path/to/archive_manifest.xlsx -m /path/to/labeled_data.csv
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
        description='Generate data archive manifest for AMOS22 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/AMOS22 -o /path/to/archive_manifest.xlsx
  %(prog)s --root_dir /path/to/AMOS22 --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name AMOS22_Manifest
        """
    )
    
    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing imagesTr, imagesTs, imagesVa, labelsTr, labelsTs, labelsVa subdirectories'
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
        '-m', '--labeled_data_meta',
        type=str,
        default=None,
        help='Optional CSV file path containing labeled data metadata with amos_id column'
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
        dirname (str): Directory name (e.g., imagesTr, labelsTs, imagesVa)
        
    Returns:
        str: Subset name ('train', 'val', or 'test')
    """
    if dirname.endswith('Tr'):
        return 'train'
    elif dirname.endswith('Va'):
        return 'val'
    elif dirname.endswith('Ts'):
        return 'test'
    else:
        return ''


def parse_type_from_dirname(dirname):
    """
    Parse file type (image/mask) from directory name.
    
    Args:
        dirname (str): Directory name (e.g., imagesTr, labelsTs, imagesVa)
        
    Returns:
        str: File type ('v3d' for images, 'm3d' for labels)
    """
    if dirname.startswith('images'):
        return 'v3d'
    elif dirname.startswith('labels'):
        return 'm3d'
    else:
        return ''


def scan_dataset(root_dir):
    """
    Scan the dataset directory structure and collect all relevant image files.
    
    Args:
        root_dir (str or Path): Root directory path containing imagesTr, imagesTs, imagesVa, labelsTr, labelsTs, labelsVa subdirectories
        
    Returns:
        list: List of dictionaries containing file information
    """
    root_path = Path(root_dir)
    file_info_list = []
    
    image_transforms = {}
    
    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    for subdir in tqdm(subdirs, desc='Processing subdirectories'):
        dirname = subdir.name
        subset = parse_subset_from_dirname(dirname)
        file_type = parse_type_from_dirname(dirname)
        
        if not subset or not file_type:
            continue
        
        nii_files = sorted(subdir.glob('amos_*.nii.gz'))
        
        for nii_file in tqdm(nii_files, desc=f'  {dirname}', leave=False):
            filename = nii_file.name
            seq_match = re.match(r'amos_(\d{4})\.nii\.gz', filename)
            
            if not seq_match:
                continue
            
            seq_id = seq_match.group(1)
            metadata = get_image_metadata(str(nii_file))
            
            diff_f_norm = ''
            if file_type == 'm3d' and seq_id in image_transforms:
                diff_f_norm = calculate_transform_f_norm_diff(
                    metadata['transform'],
                    image_transforms[seq_id]
                )
            
            if file_type == 'v3d':
                image_transforms[seq_id] = metadata['transform']
            
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


def load_labeled_data_meta(csv_path):
    """
    Load labeled data metadata from CSV file and create DataFrame.
    
    Args:
        csv_path (str or Path): Path to the CSV file containing labeled data metadata
        
    Returns:
        pd.DataFrame: DataFrame with renamed columns and seq column (4-digit zero-padded amos_id)
    """
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_file)
    
    if 'amos_id' not in df.columns:
        raise ValueError(f"'amos_id' column not found in CSV file: {csv_path}")
    
    column_mapping = {
        "amos_id": 'seq',
        "Patient's Birth Date": 'birth_date',
        "Patient's Sex": 'sex',
        "Patient's Age": 'age',
        "Manufacturer's Model Name": 'manufacturer_model_name',
        "Manufacturer": 'manufacturer',
        "Acquisition Date": 'acquisition_date',
        "Site": 'site'
    }
    
    df = df.rename(columns=column_mapping)
    
    df['seq'] = pd.to_numeric(df['seq'], errors='coerce')
    df = df.dropna(subset=['seq'])
    df['seq'] = df['seq'].astype(int)
    df['seq'] = df['seq'].apply(lambda x: f'{x:04d}')
    
    df = df.set_index('seq')
    
    return df


def generate_manifest_excel(file_info_list, output_path, sheet_name='Manifest', labeled_data_meta=None):
    """
    Generate Excel manifest file from collected file information.
    
    Args:
        file_info_list (list): List of dictionaries containing file metadata
        output_path (str or Path): Output path for the Excel file
        sheet_name (str): Name for the Excel worksheet
        labeled_data_meta (pd.DataFrame, optional): DataFrame with labeled data metadata indexed by seq
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
    
    if labeled_data_meta is not None:
        df = df.join(labeled_data_meta, on='seq', how='left')
    
    df = df.sort_values(by='seq', ascending=True)
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    """
    Main function to orchestrate the manifest generation process.
    """
    args = parse_args()
    
    labeled_data_meta = None
    if args.labeled_data_meta is not None:
        print(f"Loading labeled data metadata from: {args.labeled_data_meta}")
        labeled_data_meta = load_labeled_data_meta(args.labeled_data_meta)
        print(f"Loaded metadata for {len(labeled_data_meta)} samples")
    
    print(f"Scanning dataset from: {args.root_dir}")
    file_info_list = scan_dataset(args.root_dir)
    print(f"Found {len(file_info_list)} image files")
    
    print(f"Generating manifest Excel file: {args.output_manifest_file}")
    generate_manifest_excel(file_info_list, args.output_manifest_file, args.sheet_name, labeled_data_meta)
    
    print("Manifest generation completed successfully!")


if __name__ == '__main__':
    main()