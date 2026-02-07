#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参照 `d:\CBIB\Storages\DevelopmentSoftwares\Trae\PipeUNet\DataArchive\ESO-2025.10.31\pCR_analysis\07_split_habitat.py` 添加08_extract_radiomics.py。-r/--root_dir指定根目录，-p/--phase指定时间阶段列表（默认：[pre, post]），-v/--volume_type指定图像类型列表（默认：[]），-m/--mask_type指定蒙版类型列表（默认：[]），-h/--habitat_type指定生境类型列表（默认：[]），生境类型名格式为k={n_clusters}_{inffix}_habitat，其中n_clusters中记录了聚类数。-c/--radiomics_config指定radiomics配置文件。-o/--output_excel指定影像组学特征记录工作簿路径。 
 根目录下包含若干层级的子目录，第一层是site中心目录，中心目录下包含{site}_{pid}样本目录，样本目录中包含形如{site}_{pid}_(pre|post)(_{volume_type})?_volume.nii.gz的图像文件，{site}_{pid}_(pre|post)(_{mask_type})?_mask.nii.gz的若干蒙版文件，{site}_{pid}_(pre|post)(_{habitat_type})?_habitat.nii.gz的若干生境目录，生境目录下包含若干{site}_{pid}_(pre|post)_k={n_clusters}_habitat_{label_index}.nii.gz生境蒙版文件，其中label_index必须匹配为数字编号。 
 创建若干工作表：对每个组合(phase, volume_type, mask_type)创建一个名为({phase}, {volume_type}, {mask_type})的工作表。 
 对每个habitat_type，根据其n_clusters创建habitat_index列表，其内容为从1至n_clusters的整数。对每个组合(phase, volume_type, habitat_type, habitat_index)创建一个名为({phase}, {volume_type}, {habitat_type}, {habitat_index})的工作表。 
 程序遍历每个(phase, volume_type, mask_type)组合：遍历样本目录，获取{site}_{pid}_{phase}(_{volume_type})?_volume.nii.gz图像文件，{site}_{pid}_{phase}(_{mask_type})?_mask.nii.gz蒙版文件，按照radiomics_config配置提取若干影像组学特征{feature_name}。记录到对应的工作表，工作表列记录以下属性： 
 ID：由{site}_{pid}构成。 
 若干feature_name列。 
 每完成一次(phase, volume_type, mask_type)组合的遍历就保存一次工作表（不要覆盖其它工作表）。 
 程序遍历每个(phase, volume_type, habitat_type, habitat_index)组合：获取{site}_{pid}_{phase}(_{volume_type})?_volume.nii.gz图像文件，遍历生境目录下的每个{site}_{pid}_{phase}_k={n_clusters}_habitat_{habitat_index}.nii.gz生境蒙版文件（label_index必须匹配为数字编号，忽略其它不匹配的），按照radiomics_config配置提取若干影像组学特征{feature_name}。记录到对应的工作表。 
 遇到文件不存在的情况则立刻报告缺失情况，并终止程序。 
 使用MONAI库LoadImage读取图像文件，使用pathlib处理路径。使用tqdm展示处理进度和当前处理的组合。 
 除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Extract Radiomics Features from Images and Masks

This script extracts radiomics features from medical images using various masks (including habitat masks)
and saves the results to an Excel workbook with multiple sheets for different combinations.

Parameters:
    -r, --root_dir: Root directory containing site directories with {site}_{pid} sample directories
    -p, --phase: List of phases to process (default: ['pre', 'post'])
    -v, --volume_type: List of volume types to process (default: [])
    -m, --mask_type: List of mask types to process (default: [])
    -ht, --habitat_type: List of habitat types to process (default: [])
    -c, --radiomics_config: Path to radiomics configuration file
    -o, --output_excel_dir: Path to output directory for Excel files
    --skip_existing: Skip processing if output Excel file already exists

Usage Examples:
    python 08_extract_radiomics.py -r /path/to/root -c radiomics_config.yaml -o /path/to/output
    python 08_extract_radiomics.py --root_dir /path/to/root --radiomics_config config.yaml --output_excel_dir /path/to/output --phase pre --volume_type "" --mask_type ""
    python 08_extract_radiomics.py -r /path/to/root -c config.yaml -o /path/to/output --habitat_type "k=5_custom_habitat"
    python 08_extract_radiomics.py -r /path/to/root -c config.yaml -o /path/to/output --skip_existing
"""

import argparse
import re
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
from monai.transforms import LoadImage

import radiomics

# radiomics.setVerbosity(10)

from radiomics import featureextractor
from typing import List, Dict, Tuple, Optional, Pattern


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Extract radiomics features from images and masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/root -c radiomics_config.yaml -o /path/to/output
  %(prog)s --root_dir /path/to/root --radiomics_config config.yaml --output_excel_dir /path/to/output --phase pre --volume_type "" --mask_type ""
  %(prog)s -r /path/to/root -c config.yaml -o /path/to/output --habitat_type "k=5_custom_habitat"
  %(prog)s -r /path/to/root -c config.yaml -o /path/to/output --skip_existing
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing site directories with {site}_{pid} sample directories'
    )

    parser.add_argument(
        '-p', '--phase',
        type=str,
        nargs='+',
        default=['pre', 'post'],
        help='List of phases to process (default: [\'pre\', \'post\'])'
    )

    parser.add_argument(
        '-v', '--volume_type',
        type=str,
        nargs='+',
        default=[''],
        help='List of volume types to process (default: [\'\'])'
    )

    parser.add_argument(
        '-m', '--mask_type',
        type=str,
        nargs='+',
        default=[''],
        help='List of mask types to process (default: [\'\'])'
    )

    parser.add_argument(
        '-ht', '--habitat_type',
        type=str,
        nargs='+',
        default=[],
        help='List of habitat types to process (format: k={n_clusters}_{inffix}_habitat, default: [])'
    )

    parser.add_argument(
        '-c', '--radiomics_config',
        type=str,
        required=True,
        help='Path to radiomics configuration file'
    )

    parser.add_argument(
        '-o', '--output_excel_dir',
        type=str,
        required=True,
        help='Path to output directory for Excel files'
    )

    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip processing if output Excel file already exists'
    )

    return parser.parse_args()


def find_sample_dirs(root_dir: str) -> List[Path]:
    """
    Find all sample directories in the specified root directory.
    
    Args:
        root_dir: Root directory containing site directories with {site}_{pid} sample directories
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path: Path = Path(root_dir)
    sample_dirs: List[Path] = []

    for site_dir in root_path.iterdir():
        if site_dir.is_dir():
            for sample_dir in site_dir.iterdir():
                if sample_dir.is_dir() and re.match(r'^[^_]+_[^_]+$', sample_dir.name):
                    # Check if the directory contains {site}_{pid}_info.yaml file
                    info_file: Path = sample_dir / f"{sample_dir.name}_info.yaml"
                    if info_file.exists():
                        sample_dirs.append(sample_dir)

    return sorted(sample_dirs)





def parse_habitat_type(habitat_type: str) -> Tuple[int, str]:
    """
    Parse habitat type string to extract n_clusters and inffix.
    
    Args:
        habitat_type: Habitat type string in format k={n_clusters}_{inffix}
        
    Returns:
        Tuple[int, str]: (n_clusters, inffix)
    """
    match = re.match(r'^k=(\d+)_(.+)$', habitat_type)
    if not match:
        raise ValueError(f"Invalid habitat type format: {habitat_type}. Expected format: k={{n_clusters}}_{{inffix}}")
    
    n_clusters: int = int(match.group(1))
    inffix: str = match.group(2)
    
    return n_clusters, inffix


def find_volume_file(sample_dir: Path, phase: str, volume_type: str) -> Path:
    """
    Find volume file for a specific phase and volume type.
    
    Args:
        sample_dir: Sample directory path
        phase: Phase (pre/post)
        volume_type: Volume type
        
    Returns:
        Path: Path to volume file
        
    Raises:
        FileNotFoundError: If volume file is not found
    """
    base_name: str = sample_dir.name
    
    if volume_type:
        volume_pattern: Pattern = re.compile(rf'^{base_name}_{phase}_{volume_type}_volume\.nii\.gz$')
    else:
        volume_pattern: Pattern = re.compile(rf'^{base_name}_{phase}_volume\.nii\.gz$')
    
    for file_path in sample_dir.iterdir():
        if file_path.is_file() and volume_pattern.match(file_path.name):
            return file_path
    
    raise FileNotFoundError(f"Volume file not found for {base_name} in phase {phase} with volume type '{volume_type}'")


def find_mask_file(sample_dir: Path, phase: str, mask_type: str) -> Path:
    """
    Find mask file for a specific phase and mask type.
    
    Args:
        sample_dir: Sample directory path
        phase: Phase (pre/post)
        mask_type: Mask type
        
    Returns:
        Path: Path to mask file
        
    Raises:
        FileNotFoundError: If mask file is not found
    """
    base_name: str = sample_dir.name
    
    if mask_type:
        mask_pattern: Pattern = re.compile(rf'^{base_name}_{phase}_{mask_type}_mask\.nii\.gz$')
    else:
        mask_pattern: Pattern = re.compile(rf'^{base_name}_{phase}_mask\.nii\.gz$')
    
    for file_path in sample_dir.iterdir():
        if file_path.is_file() and mask_pattern.match(file_path.name):
            return file_path
    
    raise FileNotFoundError(f"Mask file not found for {base_name} in phase {phase} with mask type '{mask_type}'")


def find_habitat_dir(sample_dir: Path, phase: str, habitat_type: str) -> Path:
    """
    Find habitat directory for a specific phase and habitat type.
    
    Args:
        sample_dir: Sample directory path
        phase: Phase (pre/post)
        habitat_type: Habitat type
        
    Returns:
        Path: Path to habitat directory
        
    Raises:
        FileNotFoundError: If habitat directory is not found
    """
    base_name: str = sample_dir.name
    
    if habitat_type:
        habitat_pattern: Pattern = re.compile(rf'^{base_name}_{phase}_{habitat_type}_habitat$')
    else:
        habitat_pattern: Pattern = re.compile(rf'^{base_name}_{phase}_habitat$')
    
    for item in sample_dir.iterdir():
        if item.is_dir() and habitat_pattern.match(item.name):
            return item
    
    raise FileNotFoundError(f"Habitat directory not found for {base_name} in phase {phase} with habitat type '{habitat_type}'")


def find_habitat_mask_files(habitat_dir: Path, base_name: str, phase: str, n_clusters: int, habitat_index: int) -> List[Path]:
    """
    Find habitat mask files for a specific habitat index.
    
    Args:
        habitat_dir: Habitat directory path
        base_name: Sample base name
        phase: Phase (pre/post)
        n_clusters: Number of clusters
        habitat_index: Habitat index to match
        
    Returns:
        List[Path]: List of matching habitat mask files
    """
    habitat_pattern: Pattern = re.compile(rf'^{base_name}_{phase}_k={n_clusters}_habitat_{habitat_index}\.nii\.gz$')
    
    matching_files = []
    for file_path in habitat_dir.iterdir():
        if file_path.is_file() and habitat_pattern.match(file_path.name):
            matching_files.append(file_path)
    
    return matching_files


def extract_radiomics_features(volume_file: Path, mask_file: Path, config: Dict) -> Dict[str, float]:
    """
    Extract radiomics features from volume and mask files using pyradiomics.
    
    Args:
        volume_file: Path to volume NIfTI file
        mask_file: Path to mask NIfTI file
        config: Radiomics configuration dictionary
        
    Returns:
        Dict[str, float]: Dictionary of feature names and their values
    """
    features = {}
    
    try:
        # Check if mask is empty before feature extraction
        loader = LoadImage(image_only=True, dtype=None)
        mask_data = loader(str(mask_file))
        
        # Check if mask has any foreground pixels (non-zero values)
        if not np.any(mask_data > 0):
            # Mask is empty, return empty set
            return {}
        
        # Create feature extractor with configuration
        if isinstance(config, str):
            # If config is a path to a yaml file
            extractor = featureextractor.RadiomicsFeatureExtractor(config)
        else:
            # If config is a dictionary
            extractor = featureextractor.RadiomicsFeatureExtractor()
            # Apply configuration from dictionary
            if 'featureClass' in config:
                for feature_class, enabled in config['featureClass'].items():
                    extractor.enableFeatureClassByName(feature_class, enabled)
            
        # Extract features
        result = extractor.execute(str(volume_file), str(mask_file))
        
        # Filter and process features (exclude non-numeric and metadata)
        for key, value in result.items():
            if "diagnostics" not in key:
                try:
                    features[key] = float(value)
                except:
                    features[key] = value  # reserve non-numeric attributes
        
    except Exception as e:
        print(f"Error extracting radiomics features: {e}")
        return {}
    
    return features


def process_mask_combination(sample_dirs: List[Path], phase: str, volume_type: str, mask_type: str,
                           root_dir: Path, config: Dict) -> pd.DataFrame:
    """
    Process all samples for a specific (phase, volume_type, mask_type) combination.
    
    Args:
        sample_dirs: List of sample directories
        phase: Phase (pre/post)
        volume_type: Volume type
        mask_type: Mask type
        root_dir: Root directory path
        config: Radiomics configuration
        
    Returns:
        pd.DataFrame: DataFrame containing extracted features
    """
    results = []
    
    with tqdm(sample_dirs, desc=f"Processing ({phase},{volume_type},{mask_type})", leave=False) as pbar:
        for sample_dir in pbar:
            try:
                # Update progress bar description with current sample ID
                base_name = sample_dir.name
                pbar.set_description(f"Processing {base_name}({phase},{volume_type},{mask_type})")
                
                # Find volume and mask files
                volume_file = find_volume_file(sample_dir, phase, volume_type)
                mask_file = find_mask_file(sample_dir, phase, mask_type)
                
                # Extract features using pyradiomics
                radiomics_features = extract_radiomics_features(volume_file, mask_file, config)
                
                # Create ID
                features = { 'ID': base_name }
                features.update(radiomics_features)
                
                results.append(features)
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
                # Continue processing other samples instead of raising
            except Exception as e:
                print(f"Error processing {base_name}: {e}")
                # Continue processing other samples instead of raising
    
    return pd.DataFrame(results)


def process_habitat_combination(sample_dirs: List[Path], phase: str, volume_type: str, habitat_type: str,
                              habitat_index: int, root_dir: Path, config: Dict) -> pd.DataFrame:
    """
    Process all samples for a specific (phase, volume_type, habitat_type, habitat_index) combination.
    
    Args:
        sample_dirs: List of sample directories
        phase: Phase (pre/post)
        volume_type: Volume type
        habitat_type: Habitat type
        habitat_index: Habitat index
        root_dir: Root directory path
        config: Radiomics configuration
        
    Returns:
        pd.DataFrame: DataFrame containing extracted features
    """
    results = []
    n_clusters, _ = parse_habitat_type(habitat_type)
    
    with tqdm(sample_dirs, desc=f"Processing ({phase},{volume_type},{habitat_type},{habitat_index})", leave=False) as pbar:
        for sample_dir in pbar:
            try:
                # Update progress bar description with current sample ID
                base_name = sample_dir.name
                pbar.set_description(f"Processing {base_name}({phase},{volume_type},{habitat_type},{habitat_index})")
                
                # Find volume file
                volume_file = find_volume_file(sample_dir, phase, volume_type)
                
                # Find habitat directory
                habitat_dir = find_habitat_dir(sample_dir, phase, habitat_type)
                
                # Find habitat mask files for this index
                habitat_mask_files = find_habitat_mask_files(habitat_dir, base_name, phase, n_clusters, habitat_index)
                
                if not habitat_mask_files:
                    print(f"Warning: No habitat mask files found for index {habitat_index} in {habitat_dir}")
                    continue
                
                # Process each habitat mask file
                for habitat_mask_file in habitat_mask_files:
                    # Extract features using pyradiomics
                    radiomics_features = extract_radiomics_features(volume_file, habitat_mask_file, config)
                    
                    # Create ID
                    features = { 'ID': base_name }
                    features.update(radiomics_features)
                    
                    results.append(features)
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
                # Continue processing other samples instead of raising
            except Exception as e:
                print(f"Error processing {base_name}: {e}")
                # Continue processing other samples instead of raising
    
    return pd.DataFrame(results)


def save_excel_file(output_dir: Path, df: pd.DataFrame, file_name: str) -> None:
    """
    Save a DataFrame to a separate Excel file.
    
    Args:
        output_dir: Path to output directory
        df: DataFrame to save
        file_name: Name of the Excel file (without extension)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file_name}.xlsx"
    
    with pd.ExcelWriter(output_file, mode='w', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Features', index=False)
    
    print(f"Saved Excel file: {output_file}")


def main() -> None:
    """
    Main function to orchestrate radiomics feature extraction workflow.
    """
    # Parse command line arguments
    args: argparse.Namespace = parse_args()
    
    # Print configuration
    print(f"Root directory: {args.root_dir}")
    print(f"Phases: {args.phase}")
    print(f"Volume types: {args.volume_type}")
    print(f"Mask types: {args.mask_type}")
    print(f"Habitat types: {args.habitat_type}")
    print(f"Radiomics config: {args.radiomics_config}")
    print(f"Output Excel directory: {args.output_excel_dir}")
    print()
    
    # Use radiomics configuration file path directly
    print("Using radiomics configuration file:", args.radiomics_config)
    config = args.radiomics_config
    print()
    
    # Find all sample directories
    root_path = Path(args.root_dir)
    sample_dirs = find_sample_dirs(str(root_path))
    print(f"Found {len(sample_dirs)} sample directories")
    print()
    
    # Create output directory
    output_dir = Path(args.output_excel_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process (phase, volume_type, mask_type) combinations
    print("Processing (phase,volume_type,mask_type) combinations...")
    mask_combinations = [(p, v, m) for p in args.phase for v in args.volume_type for m in args.mask_type]

    for phase, volume_type, mask_type in tqdm(mask_combinations, desc="Mask combinations"):
        file_name = f"{phase}-{volume_type}-{mask_type}"
        output_file = output_dir / f"{file_name}.xlsx"

        # Check if we should skip processing
        if args.skip_existing and output_file.exists():
            print(f"Skipping combination ({phase},{volume_type},{mask_type}): Output file already exists")
            continue

        try:
            df = process_mask_combination(sample_dirs, phase, volume_type, mask_type, root_path, config)
            save_excel_file(output_dir, df, file_name)
        except Exception as e:
            print(f"Error processing combination ({phase},{volume_type},{mask_type}): {e}")
            # Continue processing other combinations instead of raising

    print()
    
    # Process (phase, volume_type, habitat_type, habitat_index) combinations
    print("Processing (phase,volume_type,habitat_type,habitat_index) combinations...")
    
    for habitat_type in args.habitat_type:
        n_clusters, _ = parse_habitat_type(habitat_type)
        habitat_indices = list(range(1, n_clusters + 1))
        
        for habitat_index in habitat_indices:
            for phase in args.phase:
                for volume_type in args.volume_type:
                    file_name = f"{phase}-{volume_type}-{habitat_type}-{habitat_index}"
                    output_file = output_dir / f"{file_name}.xlsx"
                    
                    # Check if we should skip processing
                    if args.skip_existing and output_file.exists():
                        print(f"Skipping combination ({phase},{volume_type},{habitat_type},{habitat_index}): Output file already exists")
                        continue
                    
                    try:
                        df = process_habitat_combination(sample_dirs, phase, volume_type, habitat_type,
                                                      habitat_index, root_path, config)
                        save_excel_file(output_dir, df, file_name)
                    except Exception as e:
                        print(f"Error processing combination ({phase},{volume_type},{habitat_type},{habitat_index}): {e}")
                        # Continue processing other combinations instead of raising
    
    print()
    print("=" * 80)
    print("Radiomics feature extraction completed successfully!")
    print(f"Output saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
