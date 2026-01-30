# -*- coding: utf-8 -*-
"""
Data Archive Manifest Generator for VerSe Dataset

This script generates a comprehensive manifest Excel file for the VerSe dataset, extracting metadata from NIfTI image files including dimensions,
spacings, orientations, and transformation matrices. It also supports integrating labeled metadata from an Excel file if provided.

Parameters:
    -r, --root_dir: Root directory of the dataset containing VerSe* subdirectories
    -o, --output_manifest_file: Output path for the generated Excel manifest file
    -s, --sheet_name: Optional sheet name for the Excel worksheet
    -m, --labeled_data_meta: Optional path to Excel file containing labeled data metadata including subject, split, CT_image_series, 
        verse_2019, verse_2020, sex, age. This metadata will be integrated into the manifest.

Usage Examples:
    python 01_gen_archive_manifest.py -r /path/to/VerSe -o /path/to/archive_manifest.xlsx
    python 01_gen_archive_manifest.py --root_dir /path/to/VerSe --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name Manifest
    python 01_gen_archive_manifest.py -r /path/to/VerSe -o /path/to/archive_manifest.xlsx -m /path/to/meta.xlsx
"""

import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Tuple, Any, Optional
from tqdm import tqdm
from monai.transforms import LoadImage


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    """
    parser = argparse.ArgumentParser(
        description='Generate data archive manifest for VerSe dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/VerSe -o /path/to/archive_manifest.xlsx -m /path/to/meta.xlsx
  %(prog)s --root_dir /path/to/VerSe --output_manifest_file /path/to/archive_manifest.xlsx --sheet_name Manifest
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing VerSe* subdirectories'
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
        help='Path to Excel file containing labeled data metadata including subject, split, CT_image_series, verse_2019, verse_2020, sex, age'
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


def get_image_metadata(nii_path: str or Path) -> Dict[str, Any]:
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


def format_transform_for_excel(transform: np.ndarray or str) -> str:
    """
    Format 4x4 transformation matrix for Excel output.
    
    Args:
        transform (np.ndarray or str): 4x4 affine transformation matrix or empty string
        
    Returns:
        str: Formatted string representation of the matrix
    """
    formatter = {'float_kind': lambda x: f'{x:>16.8f}'}
    matrix_str = np.array2string(transform, formatter=formatter, separator='')
    return matrix_str


def parse_subset_from_dirname(dirname: str) -> str:
    """
    Parse subset name (train/val/test) from directory name.
    
    Args:
        dirname (str): Directory name (e.g., dataset-*training, dataset-*validation, dataset-*test)
        
    Returns:
        str: Subset name ('train', 'val', or 'test')
    """
    if 'training' in dirname:
        return 'train'
    elif 'validation' in dirname:
        return 'val'
    elif 'test' in dirname:
        return 'test'
    else:
        return ''


def parse_file_info(filename: str) -> Dict[str, str]:
    """
    Parse file information from filename.
    
    Args:
        filename (str): Filename (e.g., sub-<subject>_<suffix>.nii.gz or sub-<subject>_split-<split>_<suffix>.nii.gz)
        
    Returns:
        dict: Dictionary containing subject, split, and suffix
    """
    # Pattern for files without split
    pattern1 = r'sub-(\w+)_(.+)\.nii\.gz'
    # Pattern for files with split
    pattern2 = r'sub-(\w+)_split-(\w+)_(.+)\.nii\.gz'

    match = re.match(pattern2, filename)
    if match:
        return {
            'subject': match.group(1),
            'split': match.group(2),
            'suffix': match.group(3)
        }

    match = re.match(pattern1, filename)
    if match:
        return {
            'subject': match.group(1),
            'split': '',
            'suffix': match.group(2)
        }

    return {
        'subject': '',
        'split': '',
        'suffix': ''
    }


def scan_dataset(root_dir: str or Path) -> List[Dict[str, Any]]:
    """
    Scan the dataset directory structure and collect all relevant image files.
    
    Args:
        root_dir (str or Path): Root directory path containing VerSe* subdirectories
        
    Returns:
        list: List of dictionaries containing file information
    """
    root_path = Path(root_dir)
    file_info_list: List[Dict[str, Any]] = []

    # Track image transforms for F-norm calculation
    image_transforms: Dict[str, np.ndarray] = {}

    # Find all VerSe* subdirectories
    verse_subdirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith('VerSe')])

    for verse_subdir in tqdm(verse_subdirs, desc='Processing VerSe subdirectories', leave=False):
        # Extract subdataset ID from directory name (everything after 'VerSe')
        verse_id: str = verse_subdir.name  # VerSe19, VerSe20

        # Find all dataset-* subdirectories
        dataset_subdirs = sorted([d for d in verse_subdir.iterdir() if d.is_dir() and d.name.startswith('dataset-')])

        for dataset_subdir in tqdm(dataset_subdirs, desc=f'  Processing {verse_subdir.name} datasets', leave=False):
            dirname = dataset_subdir.name
            subset = parse_subset_from_dirname(dirname)

            if not subset:
                continue

            # Process rawdata (images) directory
            rawdata_dir = dataset_subdir / 'rawdata'
            if rawdata_dir.exists():
                process_resource_directory(rawdata_dir, subset, 'v3d', root_path, file_info_list, image_transforms,
                                           verse_id)

            # Process derivatives (masks) directory
            derivatives_dir = dataset_subdir / 'derivatives'
            if derivatives_dir.exists():
                process_resource_directory(derivatives_dir, subset, 'm3d', root_path, file_info_list, image_transforms,
                                           verse_id)

    return file_info_list


def process_resource_directory(resource_dir: Path, subset: str, resource_type: str, root_path: Path, 
                               file_info_list: List[Dict[str, Any]], image_transforms: Dict[str, np.ndarray],
                               verse_id: str) -> None:
    """
    Process a resource directory (rawdata or derivatives) and collect file information.
    
    Args:
        resource_dir (Path): Path to resource directory
        subset (str): Dataset subset (train/val/test)
        resource_type (str): Resource type (v3d for images, m3d for masks)
        root_path (Path): Root directory path
        file_info_list (list): List to append file information to
        image_transforms (dict): Dictionary to track image transforms
        verse_id (str): Subdataset ID extracted from VerSe* directory name
    """
    # Find all sub-<subject> directories
    sample_dirs = sorted([d for d in resource_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    for sample_dir in tqdm(sample_dirs, desc=f'    Processing samples in {resource_dir}', leave=False):
        # Find all NIfTI files in the sample directory
        nii_files = sorted(sample_dir.glob('*.nii.gz'))
        
        for nii_file in tqdm(nii_files, desc=f'      Processing files in {sample_dir.name}', leave=False):
            filename = nii_file.name
            file_info = parse_file_info(filename)

            if not file_info['subject']:
                continue

            # Calculate relative path from root directory
            rel_path = nii_file.relative_to(root_path)

            # Get image metadata
            metadata = get_image_metadata(str(nii_file))

            # Calculate transform F-norm difference for masks
            diff_f_norm = ''
            if resource_type == 'm3d':
                # Create a key to find corresponding image
                key_parts = [file_info['subject']]
                if file_info['split']:
                    key_parts.append(file_info['split'])
                image_key = '_'.join(key_parts)

                if image_key in image_transforms:
                    diff_f_norm = calculate_transform_f_norm_diff(
                        metadata['transform'],
                        image_transforms[image_key]
                    )
            else:
                # Store image transform for mask comparison
                key_parts = [file_info['subject']]
                if file_info['split']:
                    key_parts.append(file_info['split'])
                image_key = '_'.join(key_parts)
                image_transforms[image_key] = metadata['transform']

            # Create complete file info dictionary
            complete_file_info = {
                'file_path': str(rel_path.as_posix()),
                'archive': verse_id,
                'subset': subset,
                'subject': file_info['subject'],
                'split': file_info['split'],
                'suffix': file_info['suffix'],
                'type': resource_type,
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

            file_info_list.append(complete_file_info)


def load_labeled_metadata(meta_path: Union[str, Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Load labeled metadata from Excel file and create a lookup dictionary.
    
    Args:
        meta_path (str or Path): Path to the Excel file containing labeled metadata
        
    Returns:
        dict: Dictionary with (subject, split) tuples as keys and metadata as values
    """
    meta_df = pd.read_excel(meta_path)
    metadata_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    # Collect metadata for all subjects
    all_subjects = {}
    for _, row in meta_df.iterrows():
        subject = str(row['subject'])
        split = str(row['split']) if pd.notna(row['split']) else ''
        
        # Store all metadata for each subject
        if subject not in all_subjects:
            all_subjects[subject] = []
        
        all_subjects[subject].append({
            'split': split,
            'CT_image_series': str(row['CT_image_series']) if pd.notna(row['CT_image_series']) else '',
            'verse_2019': int(row['verse_2019']) if pd.notna(row['verse_2019']) else 0,
            'verse_2020': int(row['verse_2020']) if pd.notna(row['verse_2020']) else 0,
            'sex': int(row['sex (0= f, 1= m)']) if pd.notna(row['sex (0= f, 1= m)']) else None,
            'age': int(row['age']) if pd.notna(row['age']) else None
        })
    
    # Process each subject to fill missing sex and age from CT_image_series with num_cur=1
    for subject, entries in all_subjects.items():
        # Find the entry with CT_image_series starting with '1 of '
        reference_entry = None
        for entry in entries:
            if entry['CT_image_series'].startswith('1 of '):
                reference_entry = entry
                break
        
        # Fill missing sex and age for all entries of this subject
        for entry in entries:
            if entry['sex'] is None and reference_entry:
                entry['sex'] = reference_entry['sex']
            if entry['age'] is None and reference_entry:
                entry['age'] = reference_entry['age']
        
        # Add to metadata_dict
        for entry in entries:
            # Convert sex to M/F
            if entry['sex'] == 0:
                entry['sex_str'] = 'F'
            elif entry['sex'] == 1:
                entry['sex_str'] = 'M'
            else:
                entry['sex_str'] = ''
            
            # Create key and add to dictionary
            key = (subject, entry['split'])
            metadata_dict[key] = entry
    
    return metadata_dict


def generate_manifest_excel(file_info_list: List[Dict[str, Any]], output_path: str or Path, sheet_name: str = 'Manifest', 
                           labeled_metadata: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None) -> None:
    """
    Generate Excel manifest file from collected file information.
    
    Args:
        file_info_list (list): List of dictionaries containing file metadata
        output_path (str or Path): Output path for the Excel file
        sheet_name (str): Name for the Excel worksheet
        labeled_metadata (dict, optional): Dictionary with (subject, split) tuples as keys and metadata as values
            for integrating labeled data metadata into the manifest
    """
    df_data: List[Dict[str, Any]] = []

    for file_info in tqdm(file_info_list, desc='Generating Excel rows'):
        # Get metadata from labeled data if available
        ct_image_series = ''
        in_verse19 = 0
        in_verse20 = 0
        sex = ''
        age = ''
        
        if labeled_metadata:
            subject = file_info['subject']
            split = file_info['split']
            key = (subject, split)
            
            if key in labeled_metadata:
                meta = labeled_metadata[key]
                ct_image_series = meta['CT_image_series']
                in_verse19 = meta['verse_2019']
                in_verse20 = meta['verse_2020']
                sex = meta['sex_str']
                age = meta['age']
        
        row = {
            'file_path': file_info['file_path'],
            'archive': file_info['archive'],
            'subset': file_info['subset'],
            'subject': file_info['subject'],
            'split': file_info['split'],
            'suffix': file_info['suffix'],
            'type': file_info['type'],
            'CT_image_series': ct_image_series,
            'in_verse19': in_verse19,
            'in_verse20': in_verse20,
            'sex': sex,
            'age': age,
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
    df = df.sort_values(by='file_path', ascending=True)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    """
    Main function to orchestrate the manifest generation process.
    """
    args = parse_args()

    print(f"Scanning dataset from: {args.root_dir}")
    file_info_list = scan_dataset(args.root_dir)
    print(f"Found {len(file_info_list)} image files")

    # Load labeled metadata if provided
    labeled_metadata = None
    if args.labeled_data_meta:
        print(f"Loading labeled metadata from: {args.labeled_data_meta}")
        labeled_metadata = load_labeled_metadata(args.labeled_data_meta)
        print(f"Loaded metadata for {len(labeled_metadata)} subject-split pairs")

    print(f"Generating manifest Excel file: {args.output_manifest_file}")
    generate_manifest_excel(file_info_list, args.output_manifest_file, args.sheet_name, labeled_metadata)

    print("Manifest generation completed successfully!")


if __name__ == '__main__':
    main()
