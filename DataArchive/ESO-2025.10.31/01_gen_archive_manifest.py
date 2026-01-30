# -*- coding: utf-8 -*-
"""
接收-r/--root_dir指定数据集根目录，此根目录下有多中心site目录，每个中心子目录下有pre-therapy治疗前和post-therapy治疗后时期phase目录，时期目录下包含images图像目录和masks蒙版目录，图像目录下包含{site_id}_{pid}.nii.gz命名格式的图像资源文件，蒙版目录下包含{site_id}_{pid}.nii.gz命名格式的同名蒙版资源文件。同时读取根目录下形如data_meta_{site}.xlsx的元信息文件，此文件中记录了PID及对应样本的元信息属性。接收-o/--output_manifest_file指定输出Excel清单文件路径，清单文件中依次记录以下信息： 
  file_path：记录文件的路径，从数据集根目录开始。 
   site：记录文件所属的中心，[tongji,shiyan-taihe,xiangyang]目录下分别记录为[TJ,SYTH,XY]。 
   phase：记录文件所属的时期，pre-therapy记录为pre，post-therapy记录为post。 
   pid：记录文件所属样本的编号（str）。 
   type：images图像记录为v3d，masks蒙版记录为m3d。 
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
   附加列：从对应site的元信息文件中匹配PID获取的元信息属性（不包含PID）。 
  使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。 
  接收可选参数-s/--sheet_name，如果指定则将工作表重命名为指定名称。 
  除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Data Archive Manifest Generator for ESO-2025 Dataset

This script generates a comprehensive manifest Excel file for the ESO-2025 dataset, extracting metadata from NIfTI image files including dimensions,
spacings, orientations, and transformation matrices.

Parameters:
    -r, --root_dir: Root directory of the dataset containing site directories (tongji, shiyan-taihe, xiangyang)
    -o, --output_manifest_file: Output path for the generated Excel manifest file
    -s, --sheet_name: Optional sheet name for the Excel worksheet

The manifest file will include the following columns:
    file_path: Path from dataset root directory
    site: Site abbreviation (TJ, SYTH, XY)
    phase: Phase (pre for pre-therapy, post for post-therapy)
    pid: Patient ID
    type: File type (v3d for images, m3d for masks)
    szx, szy, szz: Image dimensions
    spx, spy, spz: Pixel spacings
    orientation_from, orientation_to: Image orientation
    dtype: Data type
    transform: 4x4 affine transformation matrix
    diff_trans_f_norm: Frobenius norm of transform difference between mask and image
    Additional columns: Metadata attributes from data_meta_{site}.xlsx files (matched by PID)

Usage Examples:
    python 01_gen_archive_manifest.py -r /path/to/ESO-2025.10.31 -o /path/to/archive_manifest.xlsx
    python 01_gen_archive_manifest.py --root_dir /path/to/ESO-2025.10.31 --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name Manifest
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
        description='Generate data archive manifest for ESO-2025 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/ESO-2025.10.31 -o /path/to/archive_manifest.xlsx
  %(prog)s --root_dir /path/to/ESO-2025.10.31 --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name ESO_Manifest
        """
    )
    
    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing site directories (tongji, shiyan-taihe, xiangyang)'
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


def get_site_abbreviation(site_dirname: str) -> str:
    """
    Get site abbreviation from site directory name.
    
    Args:
        site_dirname (str): Site directory name (tongji, shiyan-taihe, xiangyang)
        
    Returns:
        str: Site abbreviation (TJ, SYTH, XY)
    """
    site_map = {
        'tongji': 'TJ',
        'shiyan-taihe': 'SYTH',
        'xiangyang': 'XY'
    }
    return site_map.get(site_dirname, site_dirname)


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse site_id and pid from filename.
    
    Args:
        filename (str): Filename in format {site_id}_{pid}.nii.gz
        
    Returns:
        Tuple[str, str]: (site_id, pid)
    """
    match = re.match(r'([^_]+)_([^_]+)\.nii\.gz', filename)
    if match:
        return match.group(1), match.group(2)
    return '', ''


def scan_dataset(root_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Scan the dataset directory structure and collect all relevant image files.
    
    Args:
        root_dir (Union[str, Path]): Root directory path containing site directories
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing file information
    """
    root_path = Path(root_dir)
    file_info_list: List[Dict[str, Any]] = []
    
    image_transforms: Dict[str, np.ndarray] = {}
    
    # Define expected site directories
    expected_sites = ['tongji', 'shiyan-taihe', 'xiangyang']
    
    # Scan site directories
    for site_dir in tqdm(expected_sites, desc='Processing sites'):
        site_path = root_path / site_dir
        if not site_path.exists():
            continue
        
        site_abbr = get_site_abbreviation(site_dir)
        
        # Scan phase directories (pre-therapy, post-therapy)
        phases = ['pre-therapy', 'post-therapy']
        for phase in tqdm(phases, desc=f'  {site_dir} - Phases', leave=False):
            phase_path = site_path / phase
            if not phase_path.exists():
                continue
            
            # Scan content directories (images, masks)
            content_dirs = [('images', 'v3d'), ('masks', 'm3d')]
            for content_dirname, file_type in tqdm(content_dirs, desc=f'    {phase} - Content', leave=False):
                content_path = phase_path / content_dirname
                if not content_path.exists():
                    continue
                
                # Process all NIfTI files
                nii_files = sorted(content_path.glob('*.nii.gz'))
                for nii_file in tqdm(nii_files, desc=f'      {content_dirname}', leave=False):
                    filename = nii_file.name
                    site_id, pid = parse_filename(filename)
                    
                    if not site_id or not pid:
                        continue
                    
                    # Extract metadata
                    metadata = get_image_metadata(str(nii_file))
                    
                    # Calculate transform difference if it's a mask
                    diff_f_norm = ''
                    if file_type == 'm3d':
                        # Get corresponding image file path
                        image_file_path = phase_path / 'images' / filename
                        if image_file_path.exists():
                            # Get image transform
                            image_metadata = get_image_metadata(str(image_file_path))
                            diff_f_norm = calculate_transform_f_norm_diff(
                                metadata['transform'],
                                image_metadata['transform']
                            )
                    
                    # Save image transform for future mask comparison
                    if file_type == 'v3d':
                        key = f'{site_id}_{pid}'
                        image_transforms[key] = metadata['transform']
                    
                    # Create file info dictionary
                    rel_path = nii_file.relative_to(root_path)
                    # Map phase to short name
                    phase_short = 'pre' if phase == 'pre-therapy' else 'post'
                    file_info = {
                        'file_path': str(rel_path.as_posix()),
                        'site': site_abbr,
                        'phase': phase_short,
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
    formatter = {'float_kind': lambda x: f'{x:>14.8f}'}
    matrix_str = np.array2string(transform, formatter=formatter, separator='')
    return matrix_str


def load_site_metadata(root_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load metadata from data_meta_{site}.xlsx files in root directory.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing data_meta_{site}.xlsx files
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with site abbreviations as keys and metadata DataFrames as values
    """
    root_path = Path(root_dir)
    metadata_dict: Dict[str, pd.DataFrame] = {}
    
    # Define site mappings
    site_map = {
        'tongji': 'TJ',
        'shiyan-taihe': 'SYTH',
        'xiangyang': 'XY'
    }
    
    for site_name, site_abbr in site_map.items():
        metadata_file = root_path / f'data_meta_{site_name}.xlsx'
        if metadata_file.exists():
            try:
                # Load metadata file
                df_meta = pd.read_excel(metadata_file, dtype={'PID': str})
                
                # Check if PID column exists
                if 'PID' not in df_meta.columns:
                    print(f"Warning: 'PID' column not found in {metadata_file.name}")
                    continue
                
                # Remove PID from metadata columns and set as index
                df_meta = df_meta.set_index('PID')
                
                # Store metadata
                metadata_dict[site_abbr] = df_meta
                print(f"Loaded metadata for {site_abbr} from {metadata_file.name}: {len(df_meta)} records")
                
            except Exception as e:
                print(f"Error loading metadata from {metadata_file.name}: {str(e)}")
    
    return metadata_dict


def generate_manifest_excel(file_info_list: List[Dict[str, Any]], 
                           output_path: Union[str, Path], 
                           metadata_dict: Dict[str, pd.DataFrame], 
                           sheet_name: str = 'Manifest') -> None:
    """
    Generate Excel manifest file from collected file information.
    
    Args:
        file_info_list (List[Dict[str, Any]]): List of dictionaries containing file metadata
        output_path (Union[str, Path]): Output path for the Excel file
        metadata_dict (Dict[str, pd.DataFrame]): Dictionary with site abbreviations as keys and metadata DataFrames as values
        sheet_name (str): Name for the Excel worksheet
    """
    df_data: List[Dict[str, Any]] = []
    
    for file_info in tqdm(file_info_list, desc='Generating Excel rows'):
        row = {
            'file_path': file_info['file_path'],
            'site': file_info['site'],
            'phase': file_info['phase'],
            'pid': file_info['pid'],
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
    df = df.sort_values(by=['site', 'phase', 'pid'], ascending=True)
    
    # Merge metadata if available
    if metadata_dict:
        print(f"\nMerging metadata for {len(metadata_dict)} sites...")
        
        # Create an empty DataFrame to hold all metadata
        df_all_meta = pd.DataFrame()
        
        # Combine metadata from all sites
        for site_abbr, df_meta in metadata_dict.items():
            # Add site column to metadata
            df_meta_with_site = df_meta.copy()
            df_meta_with_site['site'] = site_abbr
            df_meta_with_site = df_meta_with_site.reset_index()  # Reset to get PID back as column
            df_all_meta = pd.concat([df_all_meta, df_meta_with_site], ignore_index=True)
        
        if not df_all_meta.empty:
            # Merge metadata with main DataFrame on site and pid
            df = df.merge(
                df_all_meta, 
                left_on=['site', 'pid'], 
                right_on=['site', 'PID'], 
                how='left'
            )
            
            # Remove redundant PID column from metadata
            if 'PID' in df.columns:
                df = df.drop('PID', axis=1)
            
            print(f"Metadata merged successfully")
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    """
    Main function to orchestrate the manifest generation process.
    """
    args = parse_args()
    
    # Load site metadata
    print(f"Loading site metadata from: {args.root_dir}")
    metadata_dict = load_site_metadata(args.root_dir)
    
    # Scan dataset
    print(f"\nScanning dataset from: {args.root_dir}")
    file_info_list = scan_dataset(args.root_dir)
    print(f"Found {len(file_info_list)} image files")
    
    # Generate manifest
    print(f"\nGenerating manifest Excel file: {args.output_manifest_file}")
    generate_manifest_excel(file_info_list, args.output_manifest_file, metadata_dict, args.sheet_name)
    
    print("\nManifest generation completed successfully!")


if __name__ == '__main__':
    main()
