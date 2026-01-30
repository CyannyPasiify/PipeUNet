#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参照 `d:\CBIB\Storages\DevelopmentSoftwares\Trae\PipeUNet\DataArchive\ESO-2025.10.31\02_confirm_pairs.py` 生成02_confirm_pairs.py。接收-r/--root_dir指定数据存档根目录，此根目录下有图像Images和蒙版Masks目录，Images子目录下有形如Breast_MRI_{sid:03d}的图像样本目录，样本pid=Breast_MRI_{sid:03d}，图像样本目录下包含若干图像文件；Masks子目录下包含形如Breast_MRI_{sid:03d}的样本目录，样本目录下包含若干蒙版文件，形如Segmentation_{pid}_{seg_type}.nii.gz，读取-am/--archive_manifest指定的数据存档清单xlsx表格，它包含以下信息： 
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
   diff_trans_f_norm：只对蒙版文件进行记录，其它留空。记录蒙版文件与其对应同pid的首选图像文件的transform空间变换矩阵的差矩阵的F范数。 
 对蒙版文件而言，其seg_type格式为({label_index_origin}_{label_name_origin})+。seg_type包含以下类型： 
 1_Breast：整体乳房蒙版，其中乳房部分的label_index_origin为1。 
 2_Fibroglandular_Tissue：乳腺纤维组织蒙版，其中乳腺纤维组织部分的label_index_origin为2。 
 3_Blood_Vessel：乳房中的血管，其中血管部分的label_index_origin为3，背景为0。 
 1_Breast_Remained_2_Fibroglandular_Tissue：乳房残余和乳腺纤维组织复合蒙版，其中乳房残余指整体乳房蒙版去除乳腺纤维组织和血管后的残余部分，其label_index_origin为1，乳腺纤维组织部分的label_index_origin为2。 
 2_Fibroglandular_Tissue_3_Blood_Vessel：乳腺纤维组织和血管的复合蒙版，其中乳腺纤维组织部分的label_index_origin为2，血管部分的label_index_origin为3。 
 1_Breast_Remained_2_Fibroglandular_Tissue_3_Blood_Vessel：乳房残余、乳腺纤维组织和血管的复合蒙版，其中乳房残余的label_index_origin为1，乳腺纤维组织部分的label_index_origin为2，血管部分的label_index_origin为3。 
 对于每个蒙版文件，解析其seg_type为多键值字典，其中label_index_origin转换为int，label_name_origin变更为小写并将下划线替换为空格。 -e/--label_explanation指定label_map.yaml文件，读取其中的full_form_label_map和short_form_index_map，它们都是字典，full_form_label_map按照{label_name}: {label_index}记录了标签名和标签值，short_form_index_map按照{label_index}: {label_name_short}记录了标签值和标签名简称。遍历数据存档清单表格，对每个pid选中seg_type为1_Breast和1_Breast_Remained_2_Fibroglandular_Tissue_3_Blood_Vessel的2个蒙版。按照full_form_label_map匹配label_name_origin和字典中的label_name，获取对应的label_index，并构建label_index_origin到label_index的索引映射；特别的，Breast_Remained去除Remained后缀后再用于匹配full_form_label_map键名。按照索引映射修改蒙版中的标签值，构造两个映射后的蒙版备用。 
 -o/--output_dir指定导出目录，根据清单表格中存在的subset值在导出目录下创建子集目录，然后在子集目录下创建样本目录，目录命名为{pid}。对每个pid，根据pid将图像和2个映射后的蒙版配对，将首选图像源条目记录中的pid和split属性记录到{pid}_info.yaml，然后将首选图像源文件的数据类型转换到float32输出为{pid}_volume.nii.gz，两个映射后的蒙版文件数据类型转换到uint8，如果其transform与首选图像源不一致，则在控制台报告此不一致性，并将首选图像源的transform拷贝到蒙版，再将蒙版输出，1_Breast对应的蒙版输出名称为{pid}_mask_mass.nii.gz，1_Breast_Remained_2_Fibroglandular_Tissue_3_Blood_Vessel蒙版输出为{pid}_mask.nii.gz。 
   使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。 使用tqdm显示进度，并在进度条前的desc中展示当前正在处理样本的pid。 
   接收可选参数-s/--sheet_name，如果指定则将工作表重命名为指定名称。 
"""

"""
Data Pair Confirmation and Reorganization Script for Duke-Breast-FGT-Segmentation Dataset

This script confirms and reorganizes image-mask pairs from the Duke-Breast-FGT-Segmentation dataset,
ensuring metadata consistency, remapping mask labels according to provided label maps, and organizing
files into a standardized directory structure grouped by subset and patient ID.

Parameters:
    -r, --root_dir: Root directory of the dataset containing Images and Masks directories
    -o, --output_dir: Output directory where reorganized data will be saved
    -am, --archive_manifest: Path to Excel manifest file containing dataset metadata
    -e, --label_explanation: Path to label_map.yaml file containing label mapping information
    -s, --sheet_name: Optional sheet name to rename worksheet to

Usage Examples:
    python 02_confirm_pairs.py -r /path/to/Duke-Breast -o /path/to/output -am /path/to/manifest.xlsx -e /path/to/label_map.yaml
    python 02_confirm_pairs.py --root_dir /path/to/Duke-Breast --output_dir /path/to/output --archive_manifest /path/to/manifest.xlsx --label_explanation /path/to/label_map.yaml --sheet_name "Processed Data"
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
        argparse.Namespace: Parsed arguments containing root_dir, output_dir, archive_manifest, label_explanation, and sheet_name
    """
    parser = argparse.ArgumentParser(
        description='Confirm and reorganize image-mask pairs for Duke-Breast-FGT-Segmentation dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/Duke-Breast -o /path/to/output -am /path/to/manifest.xlsx -e /path/to/label_map.yaml
  %(prog)s --root_dir /path/to/Duke-Breast --output_dir /path/to/output --archive_manifest /path/to/manifest.xlsx --label_explanation /path/to/label_map.yaml --sheet_name "Processed Data"
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing Images and Masks directories'
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

    parser.add_argument(
        '-e', '--label_explanation',
        type=str,
        required=True,
        help='Path to label_map.yaml file containing label mapping information'
    )

    parser.add_argument(
        '-s', '--sheet_name',
        type=str,
        default=None,
        help='Optional sheet name to rename worksheet to'
    )

    return parser.parse_args()


def load_label_map(label_map_path: Union[str, Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load label map from YAML file.
    
    Args:
        label_map_path (Union[str, Path]): Path to label_map.yaml file
        
    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: (full_form_label_map, short_form_index_map)
    """
    label_map_file = Path(label_map_path)
    
    if not label_map_file.exists():
        raise FileNotFoundError(f"Label map file not found: {label_map_path}")
    
    with open(label_map_file, 'r', encoding='utf-8') as f:
        label_map = yaml.safe_load(f)
    
    full_form_label_map = label_map.get('full_form_label_map', {})
    full_form_label_map.update({'breast': full_form_label_map.pop('breast residue')})
    short_form_index_map = label_map.get('short_form_index_map', {})
    
    return full_form_label_map, short_form_index_map


def parse_seg_type(seg_type: str) -> Dict[int, str]:
    """
    Parse segmentation type string into a dictionary of original label index to label name.
    
    Args:
        seg_type (str): Segmentation type string in format ({label_index_origin}_{label_name_origin})+
        
    Returns:
        Dict[int, str]: Dictionary mapping original label indices to label names (lowercase with spaces)
    """
    label_map = {}
    segments = seg_type.split('_')
    
    i = 0
    while i < len(segments):
        if segments[i].isdigit():
            label_index = int(segments[i])
            label_name_parts = []
            i += 1
            
            while i < len(segments) and not segments[i].isdigit():
                label_name_parts.append(segments[i])
                i += 1
            
            if label_name_parts:
                label_name = '_'.join(label_name_parts)
                # Convert to lowercase and replace underscores with spaces
                label_name = label_name.lower().replace('_', ' ')
                label_map[label_index] = label_name
        else:
            i += 1
    
    return label_map


def create_label_remap(original_label_map: Dict[int, str], full_form_label_map: Dict[str, int]) -> Dict[int, int]:
    """
    Create label remapping from original indices to new indices based on full_form_label_map.
    
    Args:
        original_label_map (Dict[int, str]): Original label index to label name mapping
        full_form_label_map (Dict[str, int]): Full form label map from YAML file
        
    Returns:
        Dict[int, int]: Label remapping dictionary
    """
    remap = {}
    
    for orig_idx, orig_name in original_label_map.items():
        # Special handling for Breast_Remained
        if 'breast remained' in orig_name:
            # Remove 'remained' suffix and try to match
            base_name = 'breast'
            if base_name in full_form_label_map:
                remap[orig_idx] = full_form_label_map[base_name]
        else:
            # Direct matching
            if orig_name in full_form_label_map:
                remap[orig_idx] = full_form_label_map[orig_name]
    
    return remap


def remap_mask_labels(mask_data: np.ndarray, remap_dict: Dict[int, int]) -> np.ndarray:
    """
    Remap labels in mask data according to remap dictionary.
    
    Args:
        mask_data (np.ndarray): Original mask data
        remap_dict (Dict[int, int]): Label remapping dictionary
        
    Returns:
        np.ndarray: Mask data with remapped labels
    """
    remapped = np.zeros_like(mask_data, dtype=np.uint8)
    
    for orig_idx, new_idx in remap_dict.items():
        remapped[mask_data == orig_idx] = new_idx
    
    return remapped


def load_archive_manifest(manifest_path: Union[str, Path]) -> Tuple[
    Dict[str, Dict[str, Optional[str]]], Dict[str, Dict[str, Any]]
]:
    """
    Load archive manifest Excel file and create file information mapping.
    
    Args:
        manifest_path (Union[str, Path]): Path to Excel manifest file
        
    Returns:
        Tuple[Dict[str, Dict[str, Optional[str]]], Dict[str, Dict[str, Any]]]: 
            - file_info_mapping: Dictionary mapping pid to file information
            - metadata_mapping: Dictionary mapping pid to sample metadata
    """
    manifest_file = Path(manifest_path)

    if not manifest_file.exists():
        raise FileNotFoundError(f"Archive manifest file not found: {manifest_path}")

    df = pd.read_excel(manifest_file, engine='openpyxl', dtype={'pid': str})

    required_columns = ['file_path', 'pid', 'type', 'subset', 'primary']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found in archive manifest: {manifest_path}")

    file_info_mapping: Dict[str, Dict[str, Optional[str]]] = {}
    metadata_mapping: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        file_path = str(row['file_path'])
        pid = str(row['pid'])
        file_type = str(row['type']).lower() if pd.notna(row['type']) else ''
        subset = str(row['subset']) if pd.notna(row['subset']) else ''
        primary = int(row['primary']) if pd.notna(row['primary']) else 2

        if pid not in file_info_mapping:
            file_info_mapping[pid] = {
                'primary_image': None,
                'mask_1_breast': None,
                'mask_combined': None,
                'subset': subset
            }

        # Store metadata for primary images
        if file_type == 'v3d' and primary == 1:
            file_info_mapping[pid]['primary_image'] = file_path
            metadata_mapping[pid] = {
                'pid': pid,
                'subset': subset
            }
        elif file_type == 'm3d':
            # Extract seg_type from filename
            # Filename format: Segmentation_{pid}_{seg_type}.nii.gz
            # Since pid may contain underscores, we need to use the known pid to extract seg_type
            file_name = Path(file_path).name
            if file_name.startswith('Segmentation_'):
                # Remove 'Segmentation_' prefix
                remaining = file_name[len('Segmentation_'):]
                # Remove '.nii.gz' suffix
                remaining = remaining.replace('.nii.gz', '')
                # Remove the pid prefix to get seg_type
                if remaining.startswith(pid):
                    seg_type = remaining[len(pid):].lstrip('_')
                    if seg_type == '1_Breast':
                        file_info_mapping[pid]['mask_1_breast'] = file_path
                    elif seg_type == '1_Breast_Remained_2_Fibroglandular_Tissue_3_Blood_Vessel':
                        file_info_mapping[pid]['mask_combined'] = file_path

    return file_info_mapping, metadata_mapping


def check_metadata_consistency(image_meta: Dict[str, Any], mask_meta: Dict[str, Any]) -> bool:
    """
    Check if image and mask metadata (spatial_shape and affine) are consistent.
    
    Args:
        image_meta (Dict[str, Any]): Image metadata dictionary
        mask_meta (Dict[str, Any]): Mask metadata dictionary
        
    Returns:
        bool: True if metadata is consistent, False otherwise
    """
    if 'spatial_shape' not in image_meta or 'spatial_shape' not in mask_meta:
        return False
    
    if 'affine' not in image_meta or 'affine' not in mask_meta:
        return False
    
    image_shape = image_meta['spatial_shape']
    mask_shape = mask_meta['spatial_shape']

    if not (image_shape == mask_shape).all():
        return False

    image_affine = image_meta['affine']
    mask_affine = mask_meta['affine']

    if not (image_affine == mask_affine).all():
        return False

    return True


def process_pairs(root_dir: Union[str, Path],
                  output_dir: Union[str, Path],
                  archive_manifest: Union[str, Path],
                  label_explanation: Union[str, Path],
                  sheet_name: Optional[str] = None) -> None:
    """
    Process all image-mask pairs and reorganize them into output directory.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing Images and Masks directories
        output_dir (Union[str, Path]): Output directory for reorganized data
        archive_manifest (Union[str, Path]): Path to Excel manifest file
        label_explanation (Union[str, Path]): Path to label_map.yaml file
        sheet_name (Optional[str]): Optional sheet name to rename worksheet to
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    # Load label maps
    full_form_label_map, short_form_index_map = load_label_map(label_explanation)
    print(f"Loaded label maps with {len(full_form_label_map)} full form labels")

    # Load archive manifest
    file_info_mapping, metadata_mapping = load_archive_manifest(archive_manifest)
    print(f"Loaded {len(file_info_mapping)} sample entries from manifest")

    # Filter samples that have primary image and both required masks
    valid_samples = [pid for pid, info in file_info_mapping.items() 
                    if info['primary_image'] and info['mask_1_breast'] and info['mask_combined']]
    print(f"Found {len(valid_samples)} valid samples with primary image and both required masks")

    # Process each valid sample
    total_processed = 0
    total_issues = 0

    loader: LoadImage = LoadImage(image_only=False, dtype=None)

    with tqdm(total=len(valid_samples), desc='Processing samples') as pbar:
        for pid in valid_samples:
            pbar.set_description(f'Processing {pid}')
            
            info = file_info_mapping[pid]
            subset = info['subset']
            primary_image_path = root_path / info['primary_image']
            mask_1_breast_path = root_path / info['mask_1_breast']
            mask_combined_path = root_path / info['mask_combined']

            # Create output directory structure
            subset_dir = output_path / subset
            sample_dir = subset_dir / pid
            sample_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Load primary image
                image_data, image_meta = loader(str(primary_image_path))

                # Process 1_Breast mask
                mask_1_data, mask_1_meta = loader(str(mask_1_breast_path))
                
                # Parse seg_type and create label remap
                # Filename format: Segmentation_{pid}_{seg_type}.nii.gz
                # Since pid may contain underscores, we need to use the known pid to extract seg_type
                mask_1_filename = Path(mask_1_breast_path).name
                if mask_1_filename.startswith('Segmentation_'):
                    remaining = mask_1_filename[len('Segmentation_'):]
                    remaining = remaining.replace('.nii.gz', '')
                    if remaining.startswith(pid):
                        seg_type_1 = remaining[len(pid):].lstrip('_')
                        orig_label_map_1 = parse_seg_type(seg_type_1)
                        label_remap_1 = create_label_remap(orig_label_map_1, full_form_label_map)
                
                # Remap labels
                remapped_mask_1 = remap_mask_labels(mask_1_data, label_remap_1)

                # Process combined mask
                mask_combined_data, mask_combined_meta = loader(str(mask_combined_path))
                
                # Parse seg_type and create label remap
                # Filename format: Segmentation_{pid}_{seg_type}.nii.gz
                mask_combined_filename = Path(mask_combined_path).name
                if mask_combined_filename.startswith('Segmentation_'):
                    remaining = mask_combined_filename[len('Segmentation_'):]
                    remaining = remaining.replace('.nii.gz', '')
                    if remaining.startswith(pid):
                        seg_type_combined = remaining[len(pid):].lstrip('_')
                        orig_label_map_combined = parse_seg_type(seg_type_combined)
                        label_remap_combined = create_label_remap(orig_label_map_combined, full_form_label_map)
                
                # Remap labels
                remapped_mask_combined = remap_mask_labels(mask_combined_data, label_remap_combined)

                # Check metadata consistency for masks
                if not check_metadata_consistency(image_meta, mask_1_meta):
                    print(f"Warning: Metadata mismatch for {pid} - 1_Breast mask")
                    print(f"  Image shape: {image_meta['spatial_shape']}, Mask shape: {mask_1_meta['spatial_shape']}")
                    print(f"  Copying image metadata to mask...")
                    mask_1_meta = image_meta.copy()
                    total_issues += 1

                if not check_metadata_consistency(image_meta, mask_combined_meta):
                    print(f"Warning: Metadata mismatch for {pid} - combined mask")
                    print(f"  Image shape: {image_meta['spatial_shape']}, Mask shape: {mask_combined_meta['spatial_shape']}")
                    print(f"  Copying image metadata to mask...")
                    mask_combined_meta = image_meta.copy()
                    total_issues += 1

                # Save primary image as volume
                volume_path = sample_dir / f'{pid}_volume.nii.gz'
                saver_image = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.float32)
                saver_image(image_data, meta_data=image_meta, filename=str(volume_path).replace('.nii.gz', ''))

                # Save 1_Breast mask as mask_mass
                mask_mass_path = sample_dir / f'{pid}_mask_mass.nii.gz'
                saver_mask_1 = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.uint8)
                saver_mask_1(remapped_mask_1, meta_data=mask_1_meta, filename=str(mask_mass_path).replace('.nii.gz', ''))

                # Save combined mask as mask
                mask_path = sample_dir / f'{pid}_mask.nii.gz'
                saver_mask_combined = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.uint8)
                saver_mask_combined(remapped_mask_combined, meta_data=mask_combined_meta, filename=str(mask_path).replace('.nii.gz', ''))

                # Save info.yaml
                info_yaml_path = sample_dir / f'{pid}_info.yaml'
                with open(info_yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(metadata_mapping[pid], f, default_flow_style=False, allow_unicode=True)

                total_processed += 1

            except Exception as e:
                print(f"Error processing {pid}: {str(e)}")
                total_issues += 1

            pbar.update(1)

    print(f"\nProcessing completed!")
    print(f"Total samples processed: {total_processed}")
    print(f"Total issues encountered: {total_issues}")


def main() -> None:
    """
    Main function to orchestrate pair confirmation and reorganization process.
    """
    args = parse_args()

    print(f"Processing dataset from: {args.root_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Archive manifest: {args.archive_manifest}")
    print(f"Label explanation: {args.label_explanation}")
    if args.sheet_name:
        print(f"Sheet name: {args.sheet_name}")

    process_pairs(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        archive_manifest=args.archive_manifest,
        label_explanation=args.label_explanation,
        sheet_name=args.sheet_name
    )

    print("Pair confirmation and reorganization completed successfully!")


if __name__ == '__main__':
    main()
