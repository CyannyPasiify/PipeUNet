#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
接收-r/--root_dir指定根目录，根目录下包含{site}/{site}_{pid}样本目录。对每个样本，加载其图像文件{site}_{pid}_(pre|post)_volume.nii.gz和对应的蒙版文件{site}_{pid}_(pre|post)_mask.nii.gz，使用pyradiomics提取voxel-based特征图。-c/--radiomics_config指定运行pyradiomics的配置文件。在解析radiomics配置文件时，读取voxelSetting属性下的kernelRadius属性，用于特征图回填时的边界框扩展。提取结果保存在样本目录下的新目录中，文件名为{site}_{pid}_(pre|post)_{对应的radiomic feature name}.nii.gz，radiomic feature name例如original_firstorder_Entropy，其meta保持与图像一致。
 使用MONAI库LoadImage和SaveImage读取和保存图像文件，使用pathlib处理路径。使用tqdm显示进度。 
 对全部变量和函数参数添加类型注解。 
  除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Voxel-based Radiomics Feature Extraction Script for ESO-2025.10.31 Dataset

This script extracts voxel-based radiomics features from pre and post-treatment volume images
using pyradiomics.

Parameters:
    -r, --root_dir: Root directory containing {site}/{site}_{pid} sample directories
    -c, --radiomics_config: Path to pyradiomics configuration file in YAML format
    -v, --volume_type: Type of volume to process (default: "")
    -m, --mask_type: Type of mask to process (default: "")
    --skip_existing: Skip processing if output voxel radiomics directory already exists and contains files

Usage Examples:
    python 04_extract_voxel_based_radiomics.py -r /path/to/root -c radiomics_voxel_based_config.yaml
    python 04_extract_voxel_based_radiomics.py --root_dir /path/to/root --radiomics_config config.yaml
"""

import argparse
import re
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Pattern, Any, Tuple
from tqdm import tqdm
from monai.transforms import LoadImage, SaveImage
from monai.data import MetaTensor
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

radiomics.setVerbosity(10)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Extract voxel-based radiomics features from volume images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/root -c radiomics_voxel_based_config.yaml
  %(prog)s --root_dir /path/to/root --radiomics_config config.yaml
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing {site}/{site}_{pid} sample directories'
    )

    parser.add_argument(
        '-c', '--radiomics_config',
        type=str,
        required=True,
        help='Path to pyradiomics configuration file in YAML format'
    )

    parser.add_argument(
        '-v', '--volume_type',
        type=str,
        default='',
        help='Type of volume to process (default: "")'
    )

    parser.add_argument(
        '-m', '--mask_type',
        type=str,
        default='',
        help='Type of mask to process (default: "")'
    )

    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip processing if output voxel radiomics directory already exists and contains files'
    )

    return parser.parse_args()


def generate_expected_filenames(base_name: str, phase: str, output_dir: Path, radiomics_config: Dict[str, Any],
                                volume_type: str = '', mask_type: str = '') -> List[Path]:
    """
    Generate a list of expected filenames based on the radiomics configuration.
    
    Args:
        base_name: Sample base name (format: {site}_{pid})
        phase: Phase (pre or post)
        output_dir: Output directory for voxel radiomics features
        radiomics_config: Parsed radiomics configuration
        volume_type: Type of volume used (default: "")
        mask_type: Type of mask used (default: "")
        
    Returns:
        List[Path]: List of expected file paths
    """
    expected_files: List[Path] = []
    image_types: List[str] = radiomics_config.get('image_types', [])
    features: Dict[str, List[str]] = radiomics_config.get('features', {})

    for image_type in image_types:
        for feature_class, feature_list in features.items():
            for feature in feature_list:
                filename: str = f"{base_name}_{phase}_{image_type}_{feature_class}_{feature}.nii.gz"
                expected_files.append(output_dir / filename)

    return expected_files


def parse_radiomics_config(config_path: Path) -> Dict[str, Any]:
    """
    Parse the radiomics YAML configuration file and extract image types, feature classes, features, and kernelRadius.
    
    Args:
        config_path: Path to the radiomics YAML configuration file
        
    Returns:
        Dict[str, Any]: Dictionary containing image types, feature classes, features, and kernelRadius.
            Format: {'image_types': [...], 'feature_classes': [...], 'features': {'class1': [...], 'class2': [...]}, 'kernelRadius': int}
    """
    with open(config_path, 'r') as f:
        config: Dict = yaml.safe_load(f)

    # Extract image types and convert to lowercase
    image_types: List[str] = []
    if 'imageType' in config:
        image_types = [img_type.lower() for img_type in config['imageType'].keys()]

    # Extract feature classes and features
    feature_classes: List[str] = []
    features: Dict[str, List[str]] = {}
    if 'featureClass' in config:
        feature_classes = list(config['featureClass'].keys())
        for feature_class, feature_list in config['featureClass'].items():
            features[feature_class] = feature_list

    # Extract kernelRadius from voxelSetting
    kernel_radius: int = 1  # Default to 1 if not specified
    if 'voxelSetting' in config and 'kernelRadius' in config['voxelSetting']:
        kernel_radius = config['voxelSetting']['kernelRadius']

    return {
        'image_types': image_types,
        'feature_classes': feature_classes,
        'features': features,
        'kernelRadius': kernel_radius
    }


def find_sample_dirs(root_dir: str) -> List[Path]:
    """
    Find all sample directories in the specified root directory.
    
    Args:
        root_dir: Root directory containing {site}/{site}_{pid} sample directories
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path: Path = Path(root_dir)
    sample_dirs: List[Path] = []

    # Iterate through site directories
    for site_dir in root_path.iterdir():
        if site_dir.is_dir():
            # Iterate through sample directories in site directory
            for sample_dir in site_dir.iterdir():
                if sample_dir.is_dir() and re.match(r'^[^_]+_[^_]+$', sample_dir.name):
                    sample_dirs.append(sample_dir)

    return sorted(sample_dirs)


def extract_voxel_based_radiomics(image_path: Path, mask_path: Path, config_path: Path) -> Dict[str, np.ndarray]:
    """
    Extract voxel-based radiomics features using pyradiomics.
    
    Args:
        image_path: Path to the input image NIfTI file
        mask_path: Path to the input mask NIfTI file
        config_path: Path to pyradiomics configuration file
        
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping feature names to their voxel-based feature maps
    """
    # Load the ROI mask
    load_image = LoadImage(image_only=False, dtype=None)
    mask_data: MetaTensor
    mask_data, _ = load_image(str(mask_path))
    full_shape = mask_data.shape

    # Get coordinates of non-zero pixels
    coords = np.argwhere(mask_data)
    if coords.size == 0:
        raise ValueError("The mask is empty")

    # Get bounding box coordinates
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    # Parse radiomics config to get kernelRadius
    radiomics_config: Dict = parse_radiomics_config(config_path)
    kernel_radius: int = radiomics_config.get('kernelRadius', 1)

    # Initialize pyradiomics feature extractor with configuration
    extractor: featureextractor.RadiomicsFeatureExtractor = featureextractor.RadiomicsFeatureExtractor(str(config_path))

    # Extract features using rectangular mask
    result: Dict = extractor.execute(str(image_path), str(mask_path), voxelBased=True)

    # Filter and return only the voxel-based feature maps
    feature_maps: Dict[str, np.ndarray] = {}
    for key, value in result.items():
        if isinstance(value, sitk.Image):
            # Convert SimpleITK image to numpy array and permute as (x, y, z)
            bb_feature_array: np.ndarray = sitk.GetArrayFromImage(value).transpose((2, 1, 0))

            # Restore feature map from bounding box size to full image size
            full_feature_array = np.zeros(full_shape, dtype=bb_feature_array.dtype)

            # Place the bounding box region back into the full array
            # Use kernelRadius from config for expansion
            full_feature_array[
                x_min - kernel_radius:x_max + 1 + kernel_radius,
                y_min - kernel_radius:y_max + 1 + kernel_radius,
                z_min - kernel_radius:z_max + 1 + kernel_radius] = bb_feature_array

            feature_maps[key] = full_feature_array

    return feature_maps


def process_sample_dir(sample_dir: Path, config_path: Path, skip_existing: bool, volume_type: str,
                       mask_type: str) -> int:
    """
    Process a single sample directory to extract voxel-based radiomics features for all volume images.
    
    Args:
        sample_dir: Path to the sample directory
        config_path: Path to pyradiomics configuration file
        skip_existing: If True, skip processing if all expected output files already exist
        volume_type: Type of volume to process
        mask_type: Type of mask to process
        
    Returns:
        int: Number of voxel-based radiomics feature maps created
    """
    feature_maps_created: int = 0

    # Initialize MONAI components
    load_image = LoadImage(image_only=False)

    # Find pre and post volume images and their corresponding masks
    # Handle empty volume_type and mask_type by omitting the infixes
    pre_volume_suffix = f'_pre{"_" + volume_type if volume_type else ""}_volume.nii.gz'
    post_volume_suffix = f'_post{"_" + volume_type if volume_type else ""}_volume.nii.gz'
    pre_mask_suffix = f'_pre{"_" + mask_type if mask_type else ""}_mask.nii.gz'
    post_mask_suffix = f'_post{"_" + mask_type if mask_type else ""}_mask.nii.gz'

    pre_volume_pattern: Pattern = re.compile(rf'^([^_]+_[^_]+){pre_volume_suffix}$')
    post_volume_pattern: Pattern = re.compile(rf'^([^_]+_[^_]+){post_volume_suffix}$')
    pre_mask_pattern: Pattern = re.compile(rf'^([^_]+_[^_]+){pre_mask_suffix}$')
    post_mask_pattern: Pattern = re.compile(rf'^([^_]+_[^_]+){post_mask_suffix}$')

    pre_volume: Optional[Path] = None
    post_volume: Optional[Path] = None
    pre_mask: Optional[Path] = None
    post_mask: Optional[Path] = None
    base_name: str = sample_dir.name

    for file_path in sample_dir.iterdir():
        if file_path.is_file():
            if pre_volume is None:
                pre_match = pre_volume_pattern.match(file_path.name)
                if pre_match:
                    pre_volume = file_path
            if post_volume is None:
                post_match = post_volume_pattern.match(file_path.name)
                if post_match:
                    post_volume = file_path
            if pre_mask is None:
                pre_match = pre_mask_pattern.match(file_path.name)
                if pre_match:
                    pre_mask = file_path
            if post_mask is None:
                post_match = post_mask_pattern.match(file_path.name)
                if post_match:
                    post_mask = file_path

    # Validate that each volume has a corresponding mask
    pre_has_both = pre_volume is not None and pre_mask is not None
    post_has_both = post_volume is not None and post_mask is not None

    if not pre_has_both and not post_has_both:
        print(f"No valid volume+mask pairs found for {sample_dir.name}")
        return 0

    if pre_volume is not None and pre_mask is None:
        print(f"Warning: Pre volume found but no corresponding mask for {sample_dir.name}")
    if post_volume is not None and post_mask is None:
        print(f"Warning: Post volume found but no corresponding mask for {sample_dir.name}")
    if pre_mask is not None and pre_volume is None:
        print(f"Warning: Pre mask found but no corresponding volume for {sample_dir.name}")
    if post_mask is not None and post_volume is None:
        print(f"Warning: Post mask found but no corresponding volume for {sample_dir.name}")

    # Parse radiomics configuration
    radiomics_config: Dict = parse_radiomics_config(config_path)

    # Process pre volume if both volume and mask are found
    if pre_volume and pre_mask:
        try:
            # Check if output directory already exists and contains files
            # Handle empty volume_type and mask_type by omitting the infixes
            pre_volume_infix = f"_{volume_type}" if volume_type else ""
            pre_mask_infix = f"_{mask_type}" if mask_type else ""
            output_dir: Path = (
                    sample_dir /
                    (f"{base_name}_pre_voxel_radiomics" if pre_volume_infix == '' and pre_mask_infix == ''
                     else f"{base_name}_pre{pre_volume_infix}_volume{pre_mask_infix}_mask_voxel_radiomics")
            )
            if skip_existing:
                # Generate all expected filenames based on radiomics config
                expected_files: List[Path] = generate_expected_filenames(base_name, 'pre', output_dir, radiomics_config,
                                                                         volume_type, mask_type)

                # Check if all expected files exist
                all_files_exist: bool = True
                missing_files: List[Path] = []

                for expected_file in expected_files:
                    if not expected_file.exists():
                        all_files_exist = False
                        missing_files.append(expected_file)

                if all_files_exist:
                    print(f"Skipping pre volume for {sample_dir.name}: All expected files already exist")
                else:
                    print(f"Pre volume for {sample_dir.name}: Missing {len(missing_files)} files, will extract features")
                    print(f"Processing pre volume for {sample_dir.name}...")

                    # Extract voxel-based radiomics features
                    feature_maps: Dict[str, np.ndarray] = extract_voxel_based_radiomics(pre_volume, pre_mask, config_path)

                    # Create output directory (already defined, just ensure it exists)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Load original volume metadata for saving
                    meta_data: Dict
                    _, meta_data = load_image(str(pre_volume))

                    # Save each feature map
                    for feature_name, feature_array in feature_maps.items():
                        # Create output filestem
                        output_filestem: Path = output_dir / f"{base_name}_pre_{feature_name}"

                        # Create SaveImage instance with custom filename
                        save_image = SaveImage(
                            output_dir=str(output_dir),
                            output_format="nii.gz",
                            separate_folder=False,
                            print_log=True
                        )

                        # Save using MONAI with custom filename
                        save_image(feature_array, meta_data=meta_data, filename=str(output_filestem))
                        feature_maps_created += 1

                    print(f"Extracted {len(feature_maps)} voxel-based radiomics features for pre volume")

        except Exception as e:
            print(f"Error processing pre volume for {sample_dir.name}: {e}")

    # Process post volume if both volume and mask are found
    if post_volume and post_mask:
        try:
            # Check if output directory already exists and contains files
            # Handle empty volume_type and mask_type by omitting the infixes
            post_volume_infix = f"_{volume_type}" if volume_type else ""
            post_mask_infix = f"_{mask_type}" if mask_type else ""
            output_dir: Path = (
                    sample_dir /
                    (f"{base_name}_post_voxel_radiomics" if post_volume_infix == '' and post_mask_infix == ''
                     else f"{base_name}_post{post_volume_infix}_volume{post_mask_infix}_mask_voxel_radiomics")
            )
            if skip_existing:
                # Generate all expected filenames based on radiomics config
                expected_files: List[Path] = generate_expected_filenames(base_name, 'post', output_dir,
                                                                         radiomics_config, volume_type, mask_type)

                # Check if all expected files exist
                all_files_exist: bool = True
                missing_files: List[Path] = []

                for expected_file in expected_files:
                    if not expected_file.exists():
                        all_files_exist = False
                        missing_files.append(expected_file)

                if all_files_exist:
                    print(f"Skipping post volume for {sample_dir.name}: All expected files already exist")
                else:
                    print(f"Post volume for {sample_dir.name}: Missing {len(missing_files)} files, will extract features")
                    print(f"Processing post volume for {sample_dir.name}...")

                    # Extract voxel-based radiomics features
                    feature_maps: Dict[str, np.ndarray] = extract_voxel_based_radiomics(post_volume, post_mask, config_path)

                    # Create output directory (already defined, just ensure it exists)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Load original volume metadata for saving
                    meta_data: Dict
                    _, meta_data = load_image(str(post_volume))

                    # Save each feature map
                    for feature_name, feature_array in feature_maps.items():
                        # Create output filestem
                        output_filestem: Path = output_dir / f"{base_name}_post_{feature_name}"

                        # Create SaveImage instance with custom filename
                        save_image = SaveImage(
                            output_dir=str(output_dir),
                            output_format="nii.gz",
                            separate_folder=False,
                            print_log=True
                        )

                        # Save using MONAI with custom filename
                        save_image(feature_array, meta_data=meta_data, filename=str(output_filestem))
                        feature_maps_created += 1

                    print(f"Extracted {len(feature_maps)} voxel-based radiomics features for post volume")

        except Exception as e:
            print(f"Error processing post volume for {sample_dir.name}: {e}")

    return feature_maps_created


def main() -> None:
    """
    Main function to orchestrate the voxel-based radiomics feature extraction process.
    """
    args: argparse.Namespace = parse_args()

    print(f"Finding sample directories in: {args.root_dir}")
    sample_dirs: List[Path] = find_sample_dirs(args.root_dir)

    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Using radiomics configuration: {args.radiomics_config}")
    print(f"Skip existing files: {args.skip_existing}")
    print(f"Using volume type: {args.volume_type}")
    print(f"Using mask type: {args.mask_type}")

    total_feature_maps: int = 0

    with tqdm(sample_dirs, desc="Processing samples", unit="sample") as pbar:
        for sample_dir in pbar:
            pbar.set_description(f"Processing sample: {sample_dir.name}")
            feature_maps_created: int = process_sample_dir(sample_dir, Path(args.radiomics_config), args.skip_existing,
                                                           args.volume_type, args.mask_type)
            total_feature_maps += feature_maps_created

    print(f"Voxel-based radiomics feature extraction completed!")
    print(f"Total feature maps created: {total_feature_maps}")


if __name__ == '__main__':
    main()
