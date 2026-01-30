# -*- coding: utf-8 -*-
"""
接收-r/--root_dir指定数据集根目录，此根目录下有多个子集目录，形如NME-Seg-2025.8.25-{collection}，collection可为 
train-val：包含训练验证集，来自01中心（site=Tongji）。 
test-inner-01：包含内部测试集，来自01中心（site=Tongji）。 
test-outer-{cid}：包含外部测试集，cid可为['02', '03', '04']，分别来自site=Yunnan,Zhongnan,HKU-SZH。 
每个选集目录下可包含形如(images|labels)(Tr|Vd|Ts)的(图像|蒙版)(训练集|验证集|测试集)的子集subset目录，这些目录可能存在其中的一个或多个： 
train-val：包含(images|labels)(Tr|Vd)。 
test-inner-01：包含(images|labels)(Ts)。 
test-outer-{cid}：包含(images|labels)(Ts)。 
(images)(Tr|Vd|Ts)目录下包含按照NME-Seg_{seq}.{pid}.nii.gz命名的图像文件，其中ID={seq}.{pid}为每个样本的唯一标识符。(labels)(Tr|Vd|Ts)目录下包含按照NME-Seg_{seq}.{pid}.nii.gz命名的蒙版文件。接收-o/--output_manifest_file指定输出Excel清单文件路径，清单文件中依次记录以下信息： 
   file_path：记录文件的路径，从数据集根目录开始。 
    site：记录文件所属的中心。 
    collection：记录文件所属的选集。 
    subset：记录文件所属的子集。 
    seq：记录文件的所属样本的前置序号（str）。 
    pid：记录文件所属样本的编号（str）。 
    type：图像记录为v3d，蒙版记录为m3d。 
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
    diff_trans_f_norm：只对蒙版文件进行记录，其它留空。记录{site_id}_{pid}.nii.gz蒙版文件与其对应的同名图像文件的transform空间变换矩阵的差矩阵的F范数。 
   使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。使用tqdm在进度条前方显示正在处理的样本ID。 
   接收可选参数-s/--sheet_name，如果指定则将工作表重命名为指定名称。 
   为所有变量和函数参数添加类型注解。 
   除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Data Archive Manifest Generator for NME-Seg-2025.8.25 Dataset

This script generates a comprehensive manifest Excel file for the NME-Seg-2025.8.25 dataset,
extracting metadata from NIfTI image files including dimensions, spacings, orientations, and transformation matrices.

Parameters:
    -r, --root_dir: Root directory of the dataset containing collection directories
    -o, --output_manifest_file: Output path for the generated Excel manifest file
    -s, --sheet_name: Optional sheet name for the Excel worksheet

The manifest file will include the following columns:
    file_path: Path from dataset root directory
    site: Site abbreviation (Tongji, Yunnan, Zhongnan, HKU-SZH)
    collection: Collection name (train-val, test-inner-01, test-outer-02, etc.)
    subset: Subset name (imagesTr, imagesVd, imagesTs, labelsTr, labelsVd, labelsTs)
    seq: Sequence number of the sample
    pid: Patient ID
    type: File type (v3d for images, m3d for masks)
    szx, szy, szz: Image dimensions
    spx, spy, spz: Pixel spacings
    orientation_from, orientation_to: Image orientation
    dtype: Data type
    transform: 4x4 affine transformation matrix
    diff_trans_f_norm: Frobenius norm of transform difference between mask and image

Usage Examples:
    python 01_gen_archive_manifest.py -r /path/to/NME-Seg-2025.8.25 -o /path/to/archive_manifest.xlsx
    python 01_gen_archive_manifest.py --root_dir /path/to/NME-Seg-2025.8.25 --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name Manifest
"""

import re
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
        argparse.Namespace: Parsed arguments containing root_dir, output_manifest_file, and sheet_name
    """
    parser = argparse.ArgumentParser(
        description='Generate data archive manifest for NME-Seg-2025.8.25 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/NME-Seg-2025.8.25 -o /path/to/archive_manifest.xlsx
  %(prog)s --root_dir /path/to/NME-Seg-2025.8.25 --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name NME_Manifest
        """
    )
    
    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing subset directories'
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


def get_site_from_subset(subset_dirname: str) -> str:
    """
    Get site name from subset directory name.
    
    Args:
        subset_dirname (str): Subset directory name
        
    Returns:
        str: Site name
    """
    if 'train-val' in subset_dirname or 'test-inner-01' in subset_dirname:
        return 'Tongji'
    elif 'test-outer-02' in subset_dirname:
        return 'Yunnan'
    elif 'test-outer-03' in subset_dirname:
        return 'Zhongnan'
    elif 'test-outer-04' in subset_dirname:
        return 'HKU-SZH'
    else:
        return 'Unknown'


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse seq and pid from filename.
    
    Args:
        filename (str): Filename in format NME-Seg_{seq}.{pid}.nii.gz
        
    Returns:
        Tuple[str, str]: (seq, pid)
    """
    match = re.match(r'NME-Seg_([^.]+)\.([^.]+)\.nii\.gz', filename)
    if match:
        return match.group(1), match.group(2)
    return '', ''


def scan_dataset(root_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Scan the dataset directory structure and collect all relevant image files.
    
    Args:
        root_dir (Union[str, Path]): Root directory path containing collection directories
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing file information
    """
    root_path = Path(root_dir)
    file_info_list: List[Dict[str, Any]] = []
    
    image_transforms: Dict[str, np.ndarray] = {}
    
    # Scan collection directories
    collection_dirs = sorted(root_path.iterdir())
    for collection_dir in tqdm(collection_dirs, desc='Processing collections'):
        if not collection_dir.is_dir() or not collection_dir.name.startswith('NME-Seg-2025.8.25-'):
            continue
        
        # Extract collection name
        collection = collection_dir.name.split('NME-Seg-2025.8.25-')[1]
        site = get_site_from_subset(collection_dir.name)
        
        # Scan content directories (imagesTr, imagesVd, imagesTs, labelsTr, labelsVd, labelsTs)
        content_dirs = []
        for content_dir in collection_dir.iterdir():
            if content_dir.is_dir():
                dirname = content_dir.name
                if dirname.startswith('images'):
                    content_dirs.append((content_dir, 'v3d'))
                elif dirname.startswith('labels'):
                    content_dirs.append((content_dir, 'm3d'))
        
        for content_path, file_type in tqdm(content_dirs, desc=f'  {collection_dir.name} - Content', leave=False):
            # Process all NIfTI files
            nii_files = sorted(content_path.glob('*.nii.gz'))
            for nii_file in tqdm(nii_files, desc=f'    {content_path.name}', leave=False):
                filename = nii_file.name
                seq, pid = parse_filename(filename)
                
                if not seq or not pid:
                    continue
                
                # Extract subset name from content directory
                subset = content_path.name
                
                # Extract metadata
                metadata = get_image_metadata(str(nii_file))
                
                # Calculate transform difference if it's a mask
                diff_f_norm = ''
                if file_type == 'm3d':
                    # Get corresponding image file path
                    # Determine the corresponding images directory
                    images_dir_name = content_path.name.replace('labels', 'images')
                    image_file_path = content_path.parent / images_dir_name / filename
                    if image_file_path.exists():
                        # Get image transform
                        image_metadata = get_image_metadata(str(image_file_path))
                        diff_f_norm = calculate_transform_f_norm_diff(
                            metadata['transform'],
                            image_metadata['transform']
                        )
                
                # Save image transform for future mask comparison
                if file_type == 'v3d':
                    key = f'{seq}.{pid}'
                    image_transforms[key] = metadata['transform']
                
                # Create file info dictionary
                rel_path = nii_file.relative_to(root_path)
                file_info = {
                    'file_path': str(rel_path.as_posix()),
                    'site': site,
                    'collection': collection,
                    'subset': subset,
                    'seq': seq,
                    'pid': pid,
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
    df_data: List[Dict[str, Any]] = []
    
    for file_info in tqdm(file_info_list, desc='Generating Excel rows'):
        # Create a copy of the file info dictionary
        row_data = file_info.copy()
        
        # Format transform matrix for Excel
        row_data['transform'] = format_transform_for_excel(row_data['transform'])
        
        df_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Reorder columns to match the required order
    columns_order = [
        'file_path', 'site', 'collection', 'subset', 'seq', 'pid', 'type',
        'szx', 'szy', 'szz',
        'spx', 'spy', 'spz',
        'orientation_from', 'orientation_to',
        'dtype', 'transform', 'diff_trans_f_norm'
    ]
    
    # Ensure all required columns are present
    for col in columns_order:
        if col not in df.columns:
            df[col] = ''
    
    # Reorder columns
    df = df[columns_order]
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets[sheet_name]
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Manifest file generated successfully: {output_path}")
    print(f"Total files processed: {len(df)}")


def main() -> None:
    """
    Main function to orchestrate the manifest generation process.
    """
    args: argparse.Namespace = parse_args()
    
    print(f"Scanning dataset in: {args.root_dir}")
    file_info_list: List[Dict[str, Any]] = scan_dataset(args.root_dir)
    
    print(f"Processing {len(file_info_list)} files")
    generate_manifest_excel(file_info_list, args.output_manifest_file, args.sheet_name)


if __name__ == '__main__':
    main()
