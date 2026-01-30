# -*- coding: utf-8 -*-
"""
Data Pair Confirmation and Reorganization Script for VerSe Dataset

This script confirms and reorganizes image-mask pairs from the VerSe dataset, ensuring metadata consistency
and organizing files into a standardized directory structure.

Parameters:
    -r, --root_dir: Root directory of the dataset containing VerSe* subdirectories
    -am, --archive_manifest: Path to Excel manifest file containing dataset metadata
    -o, --output_dir: Output directory where reorganized data will be saved

Usage Examples:
    python 02_confirm_pairs.py -r /path/to/VerSe -am /path/to/archive_manifest.xlsx -o /path/to/VerSe/grouped
    python 02_confirm_pairs.py --root_dir /path/to/VerSe --archive_manifest /path/to/archive_manifest.xlsx --output_dir /path/to/VerSe/grouped
"""

import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
from monai.transforms import LoadImage, SaveImage


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir, archive_manifest, and output_dir
    """
    parser = argparse.ArgumentParser(
        description='Confirm and reorganize image-mask pairs for VerSe dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/VerSe -am /path/to/archive_manifest.xlsx -o /path/to/VerSe/grouped
  %(prog)s --root_dir /path/to/VerSe --archive_manifest /path/to/archive_manifest.xlsx --output_dir /path/to/VerSe/grouped
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing VerSe* subdirectories'
    )

    parser.add_argument(
        '-am', '--archive_manifest',
        type=str,
        required=True,
        help='Path to Excel manifest file containing dataset metadata'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Output directory where reorganized data will be saved'
    )

    return parser.parse_args()


def load_archive_manifest(manifest_path: str or Path) -> Tuple[Dict[Tuple[str, str, str, str], Dict[str, Optional[str]]], Dict[Tuple[str, str, str, str], Dict[str, str]]]:
    """
    Load archive manifest Excel file and create file information mapping for VerSe dataset.
    
    Args:
        manifest_path (str or Path): Path to Excel manifest file
        
    Returns:
        tuple: (file_info_mapping, metadata_mapping)
            - file_info_mapping: Dictionary mapping (archive, subset, subject, split) to file information
            - metadata_mapping: Dictionary mapping (archive, subset, subject, split) to sample metadata
    """
    manifest_file = Path(manifest_path)

    if not manifest_file.exists():
        raise FileNotFoundError(f"Archive manifest file not found: {manifest_path}")

    df = pd.read_excel(manifest_file, engine='openpyxl', dtype=str)

    required_columns = ['file_path', 'archive', 'subset', 'subject', 'split', 'suffix', 'type']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found in archive manifest: {manifest_path}")

    file_info_mapping: Dict[Tuple[str, str, str, str], Dict[str, Optional[str]]] = {}
    metadata_mapping: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}

    # Define allowed suffix values
    allowed_suffixes = ['ct', 'vert_msk']
    
    for _, row in df.iterrows():
        file_path = str(row['file_path']) if pd.notna(row['file_path']) else ''
        archive = str(row['archive']) if pd.notna(row['archive']) else ''
        subset = str(row['subset']) if pd.notna(row['subset']) else ''
        subject = str(row['subject']) if pd.notna(row['subject']) else ''
        split = str(row['split']) if pd.notna(row['split']) else ''
        suffix = str(row['suffix']) if pd.notna(row['suffix']) else ''
        file_type = str(row['type']).lower() if pd.notna(row['type']) else ''
        
        # Filter rows with allowed suffix - keep if suffix contains any of the allowed_suffixes
        if not any(allowed_suffix in suffix for allowed_suffix in allowed_suffixes):
            continue

        key = (archive, subset, subject, split)
        if key not in file_info_mapping:
            file_info_mapping[key] = {'image_path': None, 'mask_path': None}
        
        if file_type == 'v3d':
            file_info_mapping[key]['image_path'] = file_path
        elif file_type == 'm3d':
            file_info_mapping[key]['mask_path'] = file_path

        if key not in metadata_mapping:
            metadata = {}
            for col in df.columns:
                value = row[col]
                metadata[col] = str(value) if pd.notna(value) else ''
            metadata_mapping[key] = metadata

    return file_info_mapping, metadata_mapping


def find_image_mask_pairs(root_dir: Path, file_info_mapping: Dict[Tuple[str, str, str, str], Dict[str, Optional[str]]]) -> Dict[Tuple[str, str], List[Tuple[str, str, Optional[Path], Optional[Path], Tuple[str, str, str, str]]]]:
    """
    Find matching image and mask file pairs from archive manifest.
    
    Args:
        root_dir (Path): Root directory path
        file_info_mapping (dict): Dictionary mapping (archive, subset, subject, split) to file information
        
    Returns:
        dict: Dictionary mapping (archive, subset) to list of tuples (subject, split, image_path, mask_path, metadata_key)
    """
    root_path = Path(root_dir)
    
    pairs_by_group: Dict[Tuple[str, str], List[Tuple[str, str, Optional[Path], Optional[Path], Tuple[str, str, str, str]]]] = {}
    
    for (archive, subset, subject, split), file_info in file_info_mapping.items():
        image_path = file_info['image_path']
        mask_path = file_info['mask_path']
        
        if image_path is None and mask_path is None:
            continue
        
        full_image_path = root_path / image_path if image_path else None
        full_mask_path = root_path / mask_path if mask_path else None
        
        if full_image_path and not full_image_path.exists():
            full_image_path = None
        if full_mask_path and not full_mask_path.exists():
            full_mask_path = None
        
        if full_image_path is None and full_mask_path is None:
            continue
        
        group_key = (archive, subset)
        if group_key not in pairs_by_group:
            pairs_by_group[group_key] = []
        
        metadata_key = (archive, subset, subject, split)
        pairs_by_group[group_key].append((subject, split, full_image_path, full_mask_path, metadata_key))
    
    for group_key in pairs_by_group:
        pairs_by_group[group_key] = sorted(pairs_by_group[group_key], key=lambda x: (x[0], x[1]))
    
    return pairs_by_group


def check_metadata_consistency(image_meta: Dict[str, Any], mask_meta: Dict[str, Any]) -> bool:
    """
    Check if image and mask metadata (spatial_shape) are consistent.
    
    Args:
        image_meta (dict): Image metadata dictionary
        mask_meta (dict): Mask metadata dictionary
        
    Returns:
        bool: True if metadata is consistent, False otherwise
    """
    image_shape = image_meta['spatial_shape']
    mask_shape = mask_meta['spatial_shape']

    if not (image_shape == mask_shape).all():
        return False

    image_affine = image_meta['affine']
    mask_affine = mask_meta['affine']

    if not (image_affine == mask_affine).all():
        return False

    return True


def copy_metadata_to_mask(image_meta: Dict[str, Any], mask_data: Any) -> Tuple[Any, Dict[str, Any]]:
    """
    Copy image metadata to mask to ensure consistency.
    
    Args:
        image_meta (dict): Image metadata dictionary
        mask_data: Mask image data array
        
    Returns:
        tuple: (mask_data_with_meta, updated_meta)
    """
    updated_meta = image_meta.copy()
    mask_data.meta = updated_meta
    return mask_data, updated_meta


def save_metadata_yaml(output_dir: Path, subject: str, split: str, metadata: Dict[str, Any]) -> None:
    """
    Save sample metadata to YAML file.
    
    Args:
        output_dir (Path): Output directory for the YAML file
        subject (str): Sample subject ID
        split (str): Sample split ID (may be empty)
        metadata (dict): Metadata dictionary
    """
    # Generate YAML filename using new ID format
    yaml_filename = f'{subject}-{split[5:]}_info.yaml' if split else f'{subject}_info.yaml'
    yaml_file = output_dir / yaml_filename
    
    # Prepare inclusion list based on in_verse19 and in_verse20
    inclusion = []
    if metadata.get('in_verse19') == '1':
        inclusion.append('VerSe19')
    if metadata.get('in_verse20') == '1':
        inclusion.append('VerSe20')
    
    # Prepare YAML data
    yaml_metadata: Dict[str, Any] = {
        'archive': metadata.get('archive', '') if metadata.get('archive', '') else '',
        'subset': metadata.get('subset', ''),
        'subject': subject,
        'split': split[5:] if split.startswith('split-') else split,
        'CT_image_series': metadata.get('CT_image_series', ''),
        'inclusion': inclusion,
        'sex': metadata.get('sex', ''),
        'age': int(metadata.get('age', -1))
    }
    
    # Filter out empty string values
    yaml_metadata = {k: v for k, v in yaml_metadata.items() if v or v == []}
    
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_metadata, f, default_flow_style=False, allow_unicode=True)


def process_pairs(root_dir: str or Path, output_dir: str or Path, 
                 file_info_mapping: Dict[Tuple[str, str, str, str], Dict[str, Optional[str]]], 
                 metadata_mapping: Dict[Tuple[str, str, str, str], Dict[str, str]]) -> None:
    """
    Process all image-mask pairs and reorganize them into output directory.
    
    Args:
        root_dir (str or Path): Root directory containing VerSe* subdirectories
        output_dir (str or Path): Output directory for reorganized data
        file_info_mapping (dict): Dictionary mapping (archive, subset, subject, split) to file information
        metadata_mapping (dict): Dictionary mapping (archive, subset, subject, split) to sample metadata
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    pairs_by_group = find_image_mask_pairs(root_path, file_info_mapping)

    total_issues = 0
    total_pairs = 0
    total_images_only = 0
    total_masks_only = 0

    loader = LoadImage(image_only=False, dtype=None)

    for (sub_dataset, subset), pairs in tqdm(pairs_by_group.items(), desc='Processing groups'):
        output_subdir = output_path / sub_dataset / subset
        output_subdir.mkdir(parents=True, exist_ok=True)

        for subject, split, image_path, mask_path, metadata_key in tqdm(pairs, desc=f'  {sub_dataset}/{subset}', leave=False):
            # Create sample directory name based on split presence
            if split:
                # use 3-digit split code
                split_suffix = split[5:] if len(split) >= 5 else split
                sample_dir_name = f'{subject}-{split_suffix}'
                sample_id = sample_dir_name
            else:
                sample_dir_name = f'{subject}'
                sample_id = sample_dir_name
            
            sample_dir = output_subdir / sample_dir_name
            sample_dir.mkdir(parents=True, exist_ok=True)

            if image_path and mask_path:
                try:
                    image_data, image_meta = loader(str(image_path))
                    mask_data, mask_meta = loader(str(mask_path))

                    if not check_metadata_consistency(image_meta, mask_meta):
                        print(f"Warning: Metadata mismatch for {sample_id}")
                        print(f"  Image shape: {image_meta['spatial_shape']}, Mask shape: {mask_meta['spatial_shape']}")
                        print(f"  Copying image metadata to mask...")
                        mask_data, mask_meta = copy_metadata_to_mask(image_meta, mask_data)
                        total_issues += 1

                    volume_filename = f'{sample_id}_volume.nii.gz'
                    mask_filename = f'{sample_id}_mask.nii.gz'

                    saver_image = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.float32)
                    saver_mask = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.uint8)

                    saver_image(image_data, meta_data=image_meta, filename=sample_dir / volume_filename.replace('.nii.gz', ''))
                    saver_mask(mask_data, meta_data=mask_meta, filename=sample_dir / mask_filename.replace('.nii.gz', ''))

                    save_metadata_yaml(sample_dir, subject, split, metadata_mapping[metadata_key])

                    total_pairs += 1

                except Exception as e:
                    print(f"Error processing {sample_id}: {str(e)}")
                    total_issues += 1
                    continue

            elif image_path and not mask_path:
                try:
                    image_data, image_meta = loader(str(image_path))

                    volume_filename = f'{sample_id}_volume.nii.gz'

                    saver_image = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.float32)
                    saver_image(image_data, meta_data=image_meta, filename=sample_dir / volume_filename.replace('.nii.gz', ''))

                    save_metadata_yaml(sample_dir, subject, split, metadata_mapping[metadata_key])

                    total_images_only += 1
                    print(f"Notice: Only image found for {sample_id}, no mask available")

                except Exception as e:
                    print(f"Error processing image {sample_id}: {str(e)}")
                    total_issues += 1
                    continue

            elif mask_path and not image_path:
                try:
                    mask_data, mask_meta = loader(str(mask_path))

                    mask_filename = f'{sample_id}_mask.nii.gz'

                    saver_mask = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.uint8)
                    saver_mask(mask_data, meta_data=mask_meta, filename=sample_dir / mask_filename.replace('.nii.gz', ''))

                    save_metadata_yaml(sample_dir, subject, split, metadata_mapping[metadata_key])

                    total_masks_only += 1
                    print(f"Notice: Only mask found for {sample_id}, no image available")

                except Exception as e:
                    print(f"Error processing mask {sample_id}: {str(e)}")
                    total_issues += 1
                    continue

    print(f"\nProcessing completed!")
    print(f"Total pairs processed: {total_pairs}")
    print(f"Total images only: {total_images_only}")
    print(f"Total masks only: {total_masks_only}")
    print(f"Total issues encountered: {total_issues}")


def main() -> None:
    """
    Main function to orchestrate pair confirmation and reorganization process.
    """
    args = parse_args()

    print(f"Processing dataset from: {args.root_dir}")
    print(f"Archive manifest: {args.archive_manifest}")
    print(f"Output directory: {args.output_dir}")

    file_info_mapping, metadata_mapping = load_archive_manifest(args.archive_manifest)
    print(f"Loaded {len(file_info_mapping)} file entries from manifest")

    process_pairs(args.root_dir, args.output_dir, file_info_mapping, metadata_mapping)

    print("Pair confirmation and reorganization completed successfully!")


if __name__ == '__main__':
    main()
