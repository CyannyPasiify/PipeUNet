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
(images)(Tr|Vd|Ts)目录下包含按照NME-Seg_{seq}.{pid}.nii.gz命名的图像文件，其中ID={seq}.{pid}为每个样本的唯一标识符。(labels)(Tr|Vd|Ts)目录下包含按照NME-Seg_{seq}.{pid}.nii.gz命名的蒙版文件。 
-o/--output_dir指定导出目录。-am/--archive_manifest指定数据存档清单Excel文件路径。清单文件中： 
    file_path：记录文件的路径，从数据集根目录开始。 
    collection：记录文件所属的选集。 
    subset：记录文件所属的子集。将(Tr|Vd|Ts)记录为train|val|test。 
    site：记录文件所属的中心。 
    seq：记录文件的所属样本的前置序号（str）。 
    pid：记录文件所属样本的编号（str）。 
    type：图像记录为v3d，蒙版记录为m3d。 
将(site,collection,seq,pid)相关的资源文件导出到{output_dir}/{site}/{collection}/{subset}/{seq}_{pid}样本目录下。使用MONAI的LoadImage加载具有相同编号的图像和蒙版文件（图像或蒙版文件可能不存在），将图像文件的元信息拷贝到蒙版确保二者一致，使用SaveImage按照{seq}_{pid}_volume.nii.gz形式保存图像文件，按照{seq}_{pid}_mask.nii.gz形式保存蒙版文件。如果遇到图像和蒙版规格不一致的情况，则在控制台报告问题，并跳过此文件；图像或蒙版单独缺失时不检查规格一致性。导出样本元信息文件{seq}_{pid}_info.yaml，从清单信息表中提取collection, subset, site, seq, pid，其中seq和pid按照str格式读取和保存，按照以上属性顺序保存到yaml。 
   使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。使用tqdm在进度条前方显示正在处理的样本ID。 
   接收可选参数-s/--sheet_name，如果指定则将工作表重命名为指定名称。 
   为所有变量和函数参数添加类型注解。 
   除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Data Pair Confirmation and Reorganization Script for NME-Seg-2025.8.25 Dataset

This script confirms and reorganizes image-mask pairs from the NME-Seg-2025.8.25 dataset,
ensuring metadata consistency and organizing files into a standardized directory structure.

Parameters:
    -r, --root_dir: Root directory of the dataset containing collection directories
    -o, --output_dir: Output directory where reorganized data will be saved
    -am, --archive_manifest: Path to Excel manifest file containing dataset metadata
    -s, --sheet_name: Optional sheet name for the Excel worksheet (default: Manifest)

Usage Examples:
    python 02_confirm_pairs.py -r /path/to/NME-Seg-2025.8.25 -o /path/to/output -am /path/to/archive_manifest.xlsx
    python 02_confirm_pairs.py --root_dir /path/to/NME-Seg-2025.8.25 --output_dir /path/to/output --archive_manifest /path/to/archive_manifest.xlsx --sheet_name Manifest
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
        argparse.Namespace: Parsed arguments containing root_dir, output_dir, archive_manifest, and sheet_name
    """
    parser = argparse.ArgumentParser(
        description='Confirm and reorganize image-mask pairs for NME-Seg-2025.8.25 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/NME-Seg-2025.8.25 -o /path/to/output -am /path/to/archive_manifest.xlsx
  %(prog)s --root_dir /path/to/NME-Seg-2025.8.25 --output_dir /path/to/output --archive_manifest /path/to/archive_manifest.xlsx --sheet_name Manifest
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing collection directories'
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
        '-s', '--sheet_name',
        type=str,
        default='Manifest',
        help='Optional sheet name for the Excel worksheet (default: Manifest)'
    )

    return parser.parse_args()


def map_subset_name(subset: str) -> str:
    """
    Map subset name from (Tr|Vd|Ts) format to (train|val|test) format.
    
    Args:
        subset (str): Original subset name (e.g., imagesTr, labelsVd, labelsTs)
        
    Returns:
        str: Mapped subset name (e.g., train, val, test)
    """
    if 'Tr' in subset:
        return 'train'
    elif 'Vd' in subset:
        return 'val'
    elif 'Ts' in subset:
        return 'test'
    else:
        return subset


def load_archive_manifest(manifest_path: Union[str, Path], sheet_name: str = 'Manifest') -> Tuple[
    Dict[Tuple[str, str, str, str], Dict[str, Optional[str]]], Dict[Tuple[str, str, str, str], Dict[str, str]]]:
    """
    Load archive manifest Excel file and create file information mapping.
    
    Args:
        manifest_path (Union[str, Path]): Path to Excel manifest file
        sheet_name (str): Sheet name in the Excel file
        
    Returns:
        Tuple[Dict[Tuple[str, str, str, str], Dict[str, Optional[str]]], Dict[Tuple[str, str, str, str], Dict[str, str]]]: 
            - file_info_mapping: Dictionary mapping (site, collection, seq, pid) to file information
            - metadata_mapping: Dictionary mapping (site, collection, seq, pid) to sample metadata
    """
    manifest_file = Path(manifest_path)

    if not manifest_file.exists():
        raise FileNotFoundError(f"Archive manifest file not found: {manifest_path}")

    df = pd.read_excel(manifest_file, sheet_name=sheet_name, engine='openpyxl', dtype={'seq': str, 'pid': str})

    required_columns = ['file_path', 'site', 'collection', 'subset', 'seq', 'pid', 'type']
    metadata_columns = ['collection', 'subset', 'site', 'seq', 'pid']

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found in archive manifest: {manifest_path}")

    file_info_mapping: Dict[Tuple[str, str, str, str], Dict[str, Optional[str]]] = {}
    metadata_mapping: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}

    for _, row in df.iterrows():
        file_path = str(row['file_path'])
        site = str(row['site'])
        collection = str(row['collection'])
        subset = str(row['subset'])
        seq = str(row['seq'])
        pid = str(row['pid'])
        file_type = str(row['type']).lower() if pd.notna(row['type']) else ''

        # Map subset name to train/val/test
        mapped_subset = map_subset_name(subset)

        key = (site, collection, seq, pid)
        if key not in file_info_mapping:
            file_info_mapping[key] = {'image_path': None, 'mask_path': None, 'subset': mapped_subset}

        if file_type == 'v3d':
            file_info_mapping[key]['image_path'] = file_path
        elif file_type == 'm3d':
            file_info_mapping[key]['mask_path'] = file_path

        if key not in metadata_mapping:
            metadata = {}
            for col in metadata_columns:
                if col == 'subset':
                    metadata[col] = mapped_subset
                elif col in df.columns:
                    value = row[col]
                    metadata[col] = str(value) if pd.notna(value) else ''
                else:
                    metadata[col] = ''
            metadata_mapping[key] = metadata

    return file_info_mapping, metadata_mapping


def find_image_mask_pairs(root_dir: Union[str, Path],
                          file_info_mapping: Dict[Tuple[str, str, str, str], Dict[str, Optional[str]]]) -> Dict[
    Tuple[str, str, str], List[Tuple[str, str, Optional[Path], Optional[Path]]]]:
    """
    Find matching image and mask file pairs from archive manifest.
    
    Args:
        root_dir (Union[str, Path]): Root directory path
        file_info_mapping (Dict[Tuple[str, str, str, str], Dict[str, Optional[str]]]): Dictionary mapping (site, collection, seq, pid) to file information
        
    Returns:
        Dict[Tuple[str, str, str], List[Tuple[str, str, Optional[Path], Optional[Path]]]]: Dictionary mapping (site, collection, subset) to list of tuples (seq, pid, image_path, mask_path)
    """
    root_path = Path(root_dir)

    pairs_by_group: Dict[Tuple[str, str, str], List[Tuple[str, str, Optional[Path], Optional[Path]]]] = {}

    for (site, collection, seq, pid), file_info in file_info_mapping.items():
        image_path = file_info['image_path']
        mask_path = file_info['mask_path']
        subset = file_info['subset']

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

        group_key = (site, collection, subset)
        if group_key not in pairs_by_group:
            pairs_by_group[group_key] = []
        pairs_by_group[group_key].append((seq, pid, full_image_path, full_mask_path))

    for group_key in pairs_by_group:
        pairs_by_group[group_key] = sorted(pairs_by_group[group_key], key=lambda x: (x[0], x[1]))

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


def save_metadata_yaml(output_dir: Path, metadata: Dict[str, str]) -> None:
    """
    Save sample metadata to YAML file.
    
    Args:
        output_dir (Path): Output directory for the YAML file
        metadata (Dict[str, str]): Metadata dictionary containing collection, subset, site, seq, pid
    """
    seq = metadata.get('seq', '')
    pid = metadata.get('pid', '')
    yaml_file = output_dir / f'{seq}_{pid}_info.yaml'

    # Reorder metadata to match required order: collection, subset, site, seq, pid
    ordered_metadata = {
        'collection': metadata.get('collection', ''),
        'subset': metadata.get('subset', ''),
        'site': metadata.get('site', ''),
        'seq': metadata.get('seq', ''),
        'pid': metadata.get('pid', '')
    }

    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(ordered_metadata, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def process_pairs(root_dir: Union[str, Path],
                  output_dir: Union[str, Path],
                  file_info_mapping: Dict[Tuple[str, str, str, str], Dict[str, Optional[str]]],
                  metadata_mapping: Dict[Tuple[str, str, str, str], Dict[str, str]]) -> None:
    """
    Process all image-mask pairs and reorganize them into output directory.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing collection directories
        output_dir (Union[str, Path]): Output directory for reorganized data
        file_info_mapping (Dict[Tuple[str, str, str, str], Dict[str, Optional[str]]]): Dictionary mapping (site, collection, seq, pid) to file information
        metadata_mapping (Dict[Tuple[str, str, str, str], Dict[str, str]]): Dictionary mapping (site, collection, seq, pid) to sample metadata
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    pairs_by_group = find_image_mask_pairs(root_path, file_info_mapping)

    total_issues: int = 0
    total_pairs: int = 0
    total_images_only: int = 0
    total_masks_only: int = 0

    loader: LoadImage = LoadImage(image_only=False, dtype=None)

    for (site, collection, subset), pairs in tqdm(pairs_by_group.items(), desc='Processing groups'):
        output_subdir = output_path / site / collection / subset
        output_subdir.mkdir(parents=True, exist_ok=True)

        for seq, pid, image_path, mask_path in tqdm(pairs, desc=f'  {site}/{collection}/{subset}', leave=False):
            sample_id = f'{seq}_{pid}'
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

                    save_metadata_yaml(pair_dir, metadata_mapping[(site, collection, seq, pid)])

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

                    save_metadata_yaml(pair_dir, metadata_mapping[(site, collection, seq, pid)])

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

                    save_metadata_yaml(pair_dir, metadata_mapping[(site, collection, seq, pid)])

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
    print(f"Sheet name: {args.sheet_name}")

    file_info_mapping, metadata_mapping = load_archive_manifest(args.archive_manifest, args.sheet_name)
    print(f"Loaded {len(file_info_mapping)} file entries from manifest")

    process_pairs(args.root_dir, args.output_dir, file_info_mapping, metadata_mapping)

    print("Pair confirmation and reorganization completed successfully!")


if __name__ == '__main__':
    main()
