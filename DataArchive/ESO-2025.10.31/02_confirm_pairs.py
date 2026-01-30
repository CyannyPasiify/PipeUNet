# -*- coding: utf-8 -*-
"""
接收-r/--root_dir指定数据集根目录，此根目录下有多中心site目录，每个中心子目录下有pre-therapy治疗前和post-therapy治疗后时期phase目录，时期目录下包含images图像目录和masks蒙版目录，图像目录下包含{site_id}_{pid}.nii.gz命名格式的图像资源文件，蒙版目录下包含{site_id}_{pid}.nii.gz命名格式的同名蒙版资源文件。-o/--output_dir指定导出目录。-am/--archive_manifest指定数据存档清单Excel文件路径。清单文件中： 
 file_path：指定资源文件相对于root_dir的路径 
 site：记录文件所属的中心，取值为[TJ,SYTH,XY]。 
 phase：记录文件所属的时期，[pre,post]。 
 pid：记录文件所属样本的编号（str）。 
 type：images图像记录为v3d，masks蒙版记录为m3d。 
将(site,phase,pid)相关的资源文件导出到{output_dir}/{site}/{phase}/{site}_{pid}_{phase}样本目录下。使用MONAI的LoadImage加载具有相同编号的图像和蒙版文件（图像或蒙版文件可能不存在），将图像文件的元信息拷贝到蒙版确保二者一致，使用SaveImage按照{site}_{pid}_{phase}_volume.nii.gz形式保存图像文件，按照{site}_{pid}_{phase}_mask.nii.gz形式保存蒙版文件。如果遇到图像和蒙版规格不一致的情况，则在控制台报告问题，并跳过此文件；图像或蒙版单独缺失时不检查规格一致性。导出样本元信息文件{site}_{pid}_{phase}_info.yaml，从清单信息表中提取site, phase, pid, Sex, Age, Chemotherapeutic Drugs, Immunotherapeutic Drugs, Location, cT Stage, cN Stage, cTNM, Neotime - Date of First, Neoadjuvant Therapy, Surgicaltime - Date of Surgery, Hypertension, Diabetes Mellitus, Alcohol Consumption, Smoking Status, Neoadjuvant Cycles, Height, Weight, BMI, Operation Duration, Minimally Invasive Surgery, Surgical Type (1=Three-incision 2=Left Thoracotomy Approach 3=Two-incision), Thoracic Duct Ligation, R0, RRECIST, Differentiation Grade (0=Gx 1=G1 2=G2 3=G3), Tumor Regression Grade (TRG), Major Pathological Response (MPR), pT Stage, pN Stage, Pathological Complete Response (pCR), Lymph Node Metastasis, Adjuvant Therapy, Adjuvant Therapy Regimen, Recurrence Pattern, Recurrence: Cut-off Date 2025-07-28, Death: Cut-off Date 2025-07-28, Time to Recurrence, Overall Survival Time, With Any CECT，其中pid按照str格式读取和保存，按照以上属性顺序保存到yaml。
"""

"""
Data Pair Confirmation and Reorganization Script for ESO Dataset

This script confirms and reorganizes image-mask pairs from the ESO dataset, ensuring metadata consistency
and organizing files into a standardized directory structure grouped by site and phase.

Parameters:
    -r, --root_dir: Root directory of the dataset containing site directories
    -o, --output_dir: Output directory where reorganized data will be saved
    -am, --archive_manifest: Path to Excel manifest file containing dataset metadata

Usage Examples:
    python 02_confirm_pairs.py -r /path/to/ESO -o /path/to/ESO/output -am /path/to/archive_manifest.xlsx
    python 02_confirm_pairs.py --root_dir /path/to/ESO --output_dir /path/to/ESO/output --archive_manifest /path/to/archive_manifest.xlsx
"""

import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
from monai.transforms import LoadImage, SaveImage
from typing import Dict, List, Tuple, Any, Optional, Union


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir, output_dir, and archive_manifest
    """
    parser = argparse.ArgumentParser(
        description='Confirm and reorganize image-mask pairs for ESO dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/ESO -o /path/to/ESO/output -am /path/to/archive_manifest.xlsx
  %(prog)s --root_dir /path/to/ESO --output_dir /path/to/ESO/output --archive_manifest /path/to/archive_manifest.xlsx
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing site directories'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Output directory where reorganized data will be saved'
    )

    parser.add_argument(
        '-am', '--archive_manifest',
        type=str,
        required=True,
        help='Path to Excel manifest file containing dataset metadata'
    )

    return parser.parse_args()


def load_archive_manifest(manifest_path: Union[str, Path]) -> Tuple[
    Dict[Tuple[str, str, str], Dict[str, Optional[str]]], Dict[Tuple[str, str, str], Dict[str, str]]]:
    """
    Load archive manifest Excel file and create file information mapping.
    
    Args:
        manifest_path (Union[str, Path]): Path to Excel manifest file
        
    Returns:
        Tuple[Dict[Tuple[str, str, str], Dict[str, Optional[str]]], Dict[Tuple[str, str, str], Dict[str, str]]]: 
            - file_info_mapping: Dictionary mapping (site, phase, pid) to file information
            - metadata_mapping: Dictionary mapping (site, phase, pid) to sample metadata
    """
    manifest_file = Path(manifest_path)

    if not manifest_file.exists():
        raise FileNotFoundError(f"Archive manifest file not found: {manifest_path}")

    df = pd.read_excel(manifest_file, engine='openpyxl', dtype={'pid': str})

    required_columns = ['file_path', 'site', 'phase', 'pid', 'type']
    metadata_columns = ['site', 'phase', 'pid', 'Sex', 'Age', 'Chemotherapeutic Drugs',
                        'Immunotherapeutic Drugs', 'Location', 'cT Stage', 'cN Stage', 'cTNM',
                        'Neotime - Date of First Neoadjuvant Therapy', 'Surgicaltime - Date of Surgery',
                        'Hypertension', 'Diabetes Mellitus', 'Alcohol Consumption', 'Smoking Status',
                        'Neoadjuvant Cycles', 'Height', 'Weight', 'BMI', 'Operation Duration',
                        'Minimally Invasive Surgery',
                        'Surgical Type (1=Three-incision 2=Left Thoracotomy Approach 3=Two-incision)',
                        'Thoracic Duct Ligation', 'R0', 'RRECIST', 'Differentiation Grade (0=Gx 1=G1 2=G2 3=G3)',
                        'Tumor Regression Grade (TRG)', 'Major Pathological Response (MPR)', 'pT Stage',
                        'pN Stage', 'Pathological Complete Response (pCR)', 'Lymph Node Metastasis',
                        'Adjuvant Therapy', 'Adjuvant Therapy Regimen', 'Recurrence Pattern',
                        'Recurrence: Cut-off Date 2025-07-28', 'Death: Cut-off Date 2025-07-28',
                        'Time to Recurrence', 'Overall Survival Time', 'With Any CECT']

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found in archive manifest: {manifest_path}")

    file_info_mapping: Dict[Tuple[str, str, str], Dict[str, Optional[str]]] = {}
    metadata_mapping: Dict[Tuple[str, str, str], Dict[str, str]] = {}

    for _, row in df.iterrows():
        file_path = str(row['file_path'])
        site = str(row['site'])
        phase = str(row['phase'])
        pid = str(row['pid'])
        file_type = str(row['type']).lower() if pd.notna(row['type']) else ''

        key = (site, phase, pid)
        if key not in file_info_mapping:
            file_info_mapping[key] = {'image_path': None, 'mask_path': None}

        if file_type == 'v3d':
            file_info_mapping[key]['image_path'] = file_path
        elif file_type == 'm3d':
            file_info_mapping[key]['mask_path'] = file_path

        if key not in metadata_mapping:
            metadata = {}
            for col in metadata_columns:
                if col in df.columns:
                    value = row[col]
                    if col in ['Neotime - Date of First Neoadjuvant Therapy', 'Surgicaltime - Date of Surgery']:
                        metadata[col] = pd.to_datetime(value).strftime('%Y-%m-%d')
                    else:
                        metadata[col] = value if pd.notna(value) else 'UND'
                else:
                    metadata[col] = 'UND'
            metadata_mapping[key] = metadata

    return file_info_mapping, metadata_mapping


def find_image_mask_pairs(root_dir: Union[str, Path],
                          file_info_mapping: Dict[Tuple[str, str, str], Dict[str, Optional[str]]]) -> Dict[
    Tuple[str, str], List[Tuple[str, Optional[Path], Optional[Path]]]]:
    """
    Find matching image and mask file pairs from archive manifest.
    
    Args:
        root_dir (Union[str, Path]): Root directory path
        file_info_mapping (Dict[Tuple[str, str, str], Dict[str, Optional[str]]]): Dictionary mapping (site, phase, pid) to file information
        
    Returns:
        Dict[Tuple[str, str], List[Tuple[str, Optional[Path], Optional[Path]]]]: Dictionary mapping (site, phase) to list of tuples (pid, image_path, mask_path)
    """
    root_path = Path(root_dir)

    pairs_by_group: Dict[Tuple[str, str], List[Tuple[str, Optional[Path], Optional[Path]]]] = {}

    for (site, phase, pid), file_info in file_info_mapping.items():
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

        group_key = (site, phase)
        if group_key not in pairs_by_group:
            pairs_by_group[group_key] = []
        pairs_by_group[group_key].append((pid, full_image_path, full_mask_path))

    for group_key in pairs_by_group:
        pairs_by_group[group_key] = sorted(pairs_by_group[group_key], key=lambda x: x[0])

    return pairs_by_group


def check_metadata_consistency(image_meta: Dict[str, Any], mask_meta: Dict[str, Any]) -> bool:
    """
    Check if image and mask metadata (spatial_shape and affine) are consistent.
    
    Args:
        image_meta (Dict[str, Any]): Image metadata dictionary
        mask_meta (Dict[str, Any]): Mask metadata dictionary
        
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
        image_meta (Dict[str, Any]): Image metadata dictionary
        mask_data (Any): Mask image data array
        
    Returns:
        Tuple[Any, Dict[str, Any]]: (mask_data_with_meta, updated_meta)
    """
    updated_meta = image_meta.copy()
    mask_data.meta = updated_meta
    return mask_data, updated_meta


def save_metadata_yaml(output_dir: Path, site: str, phase: str, pid: str, metadata: Dict[str, str]) -> None:
    """
    Save sample metadata to YAML file.
    
    Args:
        output_dir (Path): Output directory for the YAML file
        site (str): Site identifier
        phase (str): Phase identifier
        pid (str): Patient ID
        metadata (Dict[str, str]): Metadata dictionary
    """
    yaml_file = output_dir / f'{site}_{pid}_{phase}_info.yaml'

    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def process_pairs(root_dir: Union[str, Path],
                  output_dir: Union[str, Path],
                  file_info_mapping: Dict[Tuple[str, str, str], Dict[str, Optional[str]]],
                  metadata_mapping: Dict[Tuple[str, str, str], Dict[str, str]]) -> None:
    """
    Process all image-mask pairs and reorganize them into output directory.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing site directories
        output_dir (Union[str, Path]): Output directory for reorganized data
        file_info_mapping (Dict[Tuple[str, str, str], Dict[str, Optional[str]]]): Dictionary mapping (site, phase, pid) to file information
        metadata_mapping (Dict[Tuple[str, str, str], Dict[str, str]]): Dictionary mapping (site, phase, pid) to sample metadata
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    pairs_by_group = find_image_mask_pairs(root_path, file_info_mapping)

    total_issues: int = 0
    total_pairs: int = 0
    total_images_only: int = 0
    total_masks_only: int = 0

    loader: LoadImage = LoadImage(image_only=False, dtype=None)

    for (site, phase), pairs in tqdm(pairs_by_group.items(), desc='Processing groups'):
        output_subdir = output_path / site / phase
        output_subdir.mkdir(parents=True, exist_ok=True)

        for pid, image_path, mask_path in tqdm(pairs, desc=f'  {site}/{phase}', leave=False):
            sample_id = f'{site}_{pid}_{phase}'
            pair_dir = output_subdir / sample_id
            pair_dir.mkdir(parents=True, exist_ok=True)

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

                    volume_filestem = pair_dir / f'{sample_id}_volume'
                    mask_filestem = pair_dir / f'{sample_id}_mask'

                    saver_image = SaveImage(output_dir=str(pair_dir), output_postfix='', output_dtype=np.float32)
                    saver_mask = SaveImage(output_dir=str(pair_dir), output_postfix='', output_dtype=np.uint8)

                    saver_image(image_data, meta_data=image_meta, filename=volume_filestem)
                    saver_mask(mask_data, meta_data=mask_meta, filename=mask_filestem)

                    save_metadata_yaml(pair_dir, site, phase, pid, metadata_mapping[(site, phase, pid)])

                    total_pairs += 1

                except Exception as e:
                    print(f"Error processing {sample_id}: {str(e)}")
                    total_issues += 1
                    continue

            elif image_path and not mask_path:
                try:
                    image_data, image_meta = loader(str(image_path))

                    volume_filestem = pair_dir / f'{sample_id}_volume'

                    saver_image = SaveImage(output_dir=str(pair_dir), output_postfix='', output_dtype=np.float32)
                    saver_image(image_data, meta_data=image_meta, filename=volume_filestem)

                    save_metadata_yaml(pair_dir, site, phase, pid, metadata_mapping[(site, phase, pid)])

                    total_images_only += 1

                except Exception as e:
                    print(f"Error processing image {sample_id}: {str(e)}")
                    total_issues += 1
                    continue

            elif mask_path and not image_path:
                try:
                    mask_data, mask_meta = loader(str(mask_path))

                    mask_filestem = pair_dir / f'{sample_id}_mask'

                    saver_mask = SaveImage(output_dir=str(pair_dir), output_postfix='', output_dtype=np.uint8)
                    saver_mask(mask_data, meta_data=mask_meta, filename=mask_filestem)

                    save_metadata_yaml(pair_dir, site, phase, pid, metadata_mapping[(site, phase, pid)])

                    total_masks_only += 1

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
    print(f"Output directory: {args.output_dir}")
    print(f"Archive manifest: {args.archive_manifest}")

    file_info_mapping, metadata_mapping = load_archive_manifest(args.archive_manifest)
    print(f"Loaded {len(file_info_mapping)} file entries from manifest")

    process_pairs(args.root_dir, args.output_dir, file_info_mapping, metadata_mapping)

    print("Pair confirmation and reorganization completed successfully!")


if __name__ == '__main__':
    main()
