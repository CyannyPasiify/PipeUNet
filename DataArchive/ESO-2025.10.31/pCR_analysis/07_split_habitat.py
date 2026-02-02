#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取-m/--model_parameter指定的K-Means的模型yaml文件，其中anchors属性以列表形式记录了K-Means模型每个锚特征向量，使用此配置构造K-Means模型作为预训练模型使用。
-r/--root_dir指定根目录，子目录第一层是site中心目录，中心目录下包含包含{site}_{pid}样本目录，样本目录中包含形如{site}_{pid}_(pre|post)(_{radiomics_type})?_voxel_radiomics的治疗前|后影像组学特征目录，{site}_{pid}_(pre|post)(_{mask_type})?_mask.nii.gz的若干蒙版文件，其中(_{mask_type})?和(_{radiomics_type})?部分表示对应的内容是可选的。
{site}_{pid}_(pre|post)(_{radiomics_type})?_voxel_radiomics目录下包含若干{site}_{pid}_(pre|post)_{radiomics_name}.nii.gz的影像组学Voxel特征图。
-m/--mask_type用于指定选中的蒙版文件的extra_type列表，默认值为['']。
-p/--phase用于指定阶段列表，默认为['pre','post']。
-rt/--radiomics_type用于指定影像组学类型后缀，默认值为''。
--skip_existing用于指定是否跳过已经存在的生境文件处理，默认值为False。
遍历mask_type中的每个元素，对每个mask_type：遍历每个样本目录，遍历每个phase对应的mask文件和voxel_radiomics目录，对voxel_radiomics目录下的全部影像组学特征图文件按radiomics_name进行排序；然后顺序读取影像组学特征图并合成一个多通道特征图feat，选用对应mask_type的mask资源文件，使用预训练的K-Means模型划分mask前景，划分label_index从1开始递增。
将结果输出到样本目录下的子目录中，子目录命名格式为{site}_{pid}_(pre|post)_k={n_clusters}(_{mask_type})?_mask(_{radiomics_type})?_voxel_radiomics_habitat。
在该目录下首先输出uint8多值蒙版{site}_{pid}_(pre|post)_k={n_clusters}_habitat_comb.nii.gz，然后从中导出所有标签值（除了0）对应的uint8二值蒙版，保存为{site}_{pid}_(pre|post)_k={n_clusters}_habitat_{label_index}.nii.gz。
"""

"""
Apply Pre-trained K-Means Model for Habitat Classification

This script applies a pre-trained K-Means model to voxel-based radiomics features
for habitat classification and saves the results as NIfTI files.

Parameters:
    -mp, --model_parameter: Path to pre-trained K-Means model YAML file
    -r, --root_dir: Root directory containing site directories with {site}_{pid} sample directories
    -p, --phase: List of phases to process (default: ['pre', 'post'])
    -mt, --mask_type: List of mask types to process (default: [''])
    -rt, --radiomics_type: Radiomics type suffix for directory name (default: '')
    -srt, --small_region_thresh: Threshold for removing small connected regions (default: 10 voxels)
    --skip_existing: Skip processing if output habitat directory already exists and contains files

Usage Examples:
    python 06_split_habitat.py --model_parameter 5_means_anchors.yaml --root_dir /path/to/root
    python 06_split_habitat.py --model_parameter 5_means_anchors.yaml --root_dir /path/to/root --phase pre --mask_type "" "peritumor"
    python 06_split_habitat.py --model_parameter 5_means_anchors.yaml --root_dir /path/to/root --skip_existing
    python 06_split_habitat.py --model_parameter 5_means_anchors.yaml --root_dir /path/to/root --phase pre --mask_type "" --radiomics_type custom
    python 06_split_habitat.py --model_parameter 5_means_anchors.yaml --root_dir /path/to/root --small_region_thresh 10
"""

import argparse
import re
import numpy as np
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Pattern
from tqdm import tqdm
from sklearn.cluster import KMeans
from monai.transforms import LoadImage, SaveImage
from scipy.ndimage import label


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Apply pre-trained K-Means model for habitat classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model_parameter 5_means_anchors.yaml --root_dir /path/to/root
  %(prog)s --model_parameter 5_means_anchors.yaml --root_dir /path/to/root --phase pre --mask_type "" "peritumor"
  %(prog)s --model_parameter 5_means_anchors.yaml --root_dir /path/to/root --skip_existing
        """
    )

    parser.add_argument(
        '-mp', '--model_parameter',
        type=str,
        required=True,
        help='Path to pre-trained K-Means model YAML file'
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
        help='List of phases to process (default: [\'pre\', post])'
    )

    parser.add_argument(
        '-mt', '--mask_type',
        type=str,
        nargs='+',
        default=[''],
        help='List of mask types to process (default: [\'\'])'
    )

    parser.add_argument(
        '-rt', '--radiomics_type',
        type=str,
        default='',
        help='Radiomics type suffix for directory name (default: \'\')'
    )

    parser.add_argument(
        '-srt', '--small_region_thresh',
        type=int,
        default=10,
        help='Threshold for removing small connected regions (default: 10 voxels)'
    )

    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip processing if output habitat directory already exists and contains files'
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
                    sample_dirs.append(sample_dir)

    return sorted(sample_dirs)


def load_kmeans_model(model_path: str) -> KMeans:
    """
    Load a K-Means model from a YAML file.
    
    Args:
        model_path: Path to the pre-trained K-Means model YAML file
        
    Returns:
        KMeans: Reconstructed K-Means model with pre-trained cluster centers
    """
    model_path: Path = Path(model_path)

    # Read YAML file
    with open(model_path, 'r') as f:
        data: Dict = yaml.safe_load(f)

    # Extract anchors (cluster centers)
    if 'anchors' not in data:
        raise ValueError(f"Model file {model_path} does not contain 'anchors' attribute")

    anchors: np.ndarray = np.array(data['anchors'])
    n_clusters: int = anchors.shape[0]

    # Create KMeans model with the appropriate number of clusters
    kmeans: KMeans = KMeans(n_clusters=n_clusters, random_state=0)

    # Set the cluster centers (this makes the model ready for prediction)
    kmeans.cluster_centers_ = np.array(anchors, dtype=np.float32)
    kmeans._n_threads = 1  # Set default n_threads to avoid warnings
    kmeans._n_features_in = anchors.shape[1]  # Set number of features
    kmeans.labels_ = np.array([])  # Initialize labels array
    kmeans.inertia_ = 0.0  # Initialize inertia

    return kmeans


def remove_small_regions(habitat_map: np.ndarray, threshold: int) -> np.ndarray:
    """
    Remove small connected regions from a habitat map.
    
    Args:
        habitat_map: Habitat map array
        threshold: Minimum number of voxels for a region to be kept
        
    Returns:
        np.ndarray: Habitat map with small regions removed
    """
    # Create a copy of the habitat map
    cleaned_map = habitat_map.copy()
    
    # Get unique labels (excluding 0 which is background)
    unique_labels = np.unique(habitat_map)
    unique_labels = unique_labels[unique_labels > 0]
    
    # Process each label separately
    for label in unique_labels:
        # Create binary mask for current label
        label_mask = (habitat_map == label).astype(np.uint8)
        
        # Label connected regions in the binary mask
        labeled_mask, num_regions = label(label_mask)
        
        # Remove small regions
        for i in range(1, num_regions + 1):
            region = (labeled_mask == i)
            if np.sum(region) < threshold:
                cleaned_map[region] = 0
                print(f"Removed small region {i} with {np.sum(region)} voxels for label {label}")
    
    return cleaned_map


def load_sample_features(sample_dir: Path, phase: str, mask_type: str, radiomics_type: str = '') -> Tuple[
    np.ndarray, np.ndarray, Dict]:
    """
    Load features and mask for a sample.
    
    Args:
        sample_dir: Path to the sample directory
        phase: Phase (pre/post)
        mask_type: Mask type
        radiomics_type: Radiomics type suffix
        
    Returns:
        Tuple[np.ndarray, np.ndarray, Dict]: Multi-channel feature array, binary mask, and mask metadata
    """
    base_name: str = sample_dir.name

    # Find voxel_radiomics directory
    voxel_radiomics_dir: Optional[Path] = None
    if radiomics_type:
        radiomics_dir_pattern: Pattern = re.compile(rf'^{base_name}_{phase}_{radiomics_type}_voxel_radiomics$')
    else:
        radiomics_dir_pattern: Pattern = re.compile(rf'^{base_name}_{phase}_voxel_radiomics$')

    for item in sample_dir.iterdir():
        if item.is_dir() and radiomics_dir_pattern.match(item.name):
            voxel_radiomics_dir = item
            break

    if not voxel_radiomics_dir:
        raise FileNotFoundError(f"Voxel radiomics directory not found for {base_name} in phase {phase}")

    # Find mask file
    mask_file: Optional[Path] = None
    if mask_type:
        mask_pattern: Pattern = re.compile(rf'^{base_name}_{phase}_{mask_type}_mask\.nii\.gz$')
    else:
        mask_pattern: Pattern = re.compile(rf'^{base_name}_{phase}_mask\.nii\.gz$')

    for file_path in sample_dir.iterdir():
        if file_path.is_file() and mask_pattern.match(file_path.name):
            mask_file = file_path
            break

    if not mask_file:
        raise FileNotFoundError(f"Mask file not found for {base_name} in phase {phase} with mask type '{mask_type}'")

    # Load mask
    load_image = LoadImage(image_only=False, dtype=None)
    mask_data, mask_meta = load_image(str(mask_file))
    binary_mask: np.ndarray = (mask_data > 0).astype(bool)

    # Load and sort radiomics feature files
    radiomics_files: List[Path] = list(voxel_radiomics_dir.glob(f"{base_name}_{phase}_*.nii.gz"))

    radiomics_info = []
    for file_path in radiomics_files:
        match = re.match(rf'^{base_name}_{phase}_(.+)\.nii\.gz$', file_path.name)
        if match:
            radiomics_name = match.group(1)
            radiomics_info.append((radiomics_name, file_path))

    # Sort by radiomics name
    radiomics_info.sort(key=lambda x: x[0])
    sorted_files: List[Path] = [file_path for _, file_path in radiomics_info]

    # Create multi-channel feature array
    features_list = []
    for file_path in sorted_files:
        feature_data, _ = load_image(str(file_path))
        features_list.append(feature_data)
        print(f'Loaded voxel-radiomics {file_path} with shape {tuple(feature_data.shape)}')

    features: np.ndarray = np.stack(features_list, axis=3)

    return features, binary_mask, mask_meta


def generate_habitat_map(kmeans: KMeans, features: np.ndarray, binary_mask: np.ndarray, mask_meta: Dict,
                         output_path: Path, small_region_thresh: int = 10) -> np.ndarray:
    """
    Generate and save a habitat map using a pre-trained K-Means model.
    
    Args:
        kmeans: Pre-trained K-Means model
        features: Multi-channel feature array (shape: x, y, z, channels)
        binary_mask: Binary mask array indicating foreground voxels
        mask_meta: Metadata from the mask file
        output_path: Path to save the habitat map
        small_region_thresh: Threshold for removing small connected regions
        
    Returns:
        np.ndarray: Generated habitat map as uint8 array
    """
    # Get image dimensions
    x_dim, y_dim, z_dim, channels = features.shape

    # Flatten features and mask for K-Means prediction
    features_flat: np.ndarray = features.reshape(-1, channels)
    mask_flat: np.ndarray = binary_mask.reshape(-1)

    # Get foreground voxels indices
    foreground_indices: np.ndarray = np.where(mask_flat)[0]

    # Predict clusters for foreground voxels
    if len(foreground_indices) > 0:
        foreground_features: np.ndarray = features_flat[foreground_indices]
        foreground_labels: np.ndarray = kmeans.predict(foreground_features)

        # Convert labels to start from 1 (not 0)
        foreground_labels += 1

    # Create empty habitat map
    habitat_map: np.ndarray = np.zeros_like(mask_flat, dtype=np.uint8)

    # Assign labels to foreground voxels
    if len(foreground_indices) > 0:
        habitat_map[foreground_indices] = foreground_labels.astype(np.uint8)

    # Reshape back to original image shape
    habitat_map: np.ndarray = habitat_map.reshape(x_dim, y_dim, z_dim)

    # Remove small connected regions
    if small_region_thresh > 0:
        habitat_map = remove_small_regions(habitat_map, small_region_thresh)
        print(f"Removed small connected regions with threshold: {small_region_thresh}")

    # Save habitat map using MONAI's SaveImage
    save_image = SaveImage(
        output_dir=str(output_path.parent),
        output_format="nii.gz",
        separate_folder=False,
        print_log=True,
        output_postfix="",
        output_dtype=np.uint8
    )

    # Save with custom filename
    save_image(habitat_map, meta_data=mask_meta, filename=str(output_path).replace('.nii.gz', ''))
    print(f"Saved habitat map to: {output_path}")
    
    return habitat_map


def process_mask_type(root_dir: Path, mask_type: str, phases: List[str], kmeans: KMeans,
                      radiomics_type: str = '', skip_existing: bool = False, small_region_thresh: int = 10) -> None:
    """
    Process all samples for a specific mask type.
    
    Args:
        root_dir: Root directory containing site directories
        mask_type: Mask type to process
        phases: List of phases to process
        kmeans: Pre-trained K-Means model
        radiomics_type: Radiomics type suffix
        skip_existing: Skip processing if output habitat directory already exists and contains files
        small_region_thresh: Threshold for removing small connected regions
    """
    print(f"Processing mask type: '{mask_type}'")

    # Find all sample directories
    sample_dirs: List[Path] = find_sample_dirs(str(root_dir))
    print(f"Found {len(sample_dirs)} sample directories")

    with tqdm(total=len(sample_dirs) * len(phases), desc="Generating habitat maps", unit="sample-phase") as pbar:
        for sample_dir in sample_dirs:
            for phase in phases:
                try:
                    # Update progress bar description with current sample and phase
                    pbar.set_description(f"Processing: {sample_dir.name} ({phase})")
                    
                    # Create output directory and filenames according to new format
                    base_name: str = sample_dir.name
                    n_clusters: int = kmeans.n_clusters
                    
                    # Create subdirectory with mask and radiomics type
                    subdir_parts: List[str] = [
                        f"{base_name}_{phase}",
                        f"k={n_clusters}"
                    ]
                    
                    if mask_type:
                        subdir_parts.append(f"{mask_type}_mask")
                    else:
                        subdir_parts.append("mask")
                    
                    if radiomics_type:
                        subdir_parts.append(f"{radiomics_type}_voxel_radiomics")
                    else:
                        subdir_parts.append("voxel_radiomics")
                    
                    subdir_parts.append("habitat")
                    subdir_name: str = "_".join(subdir_parts)
                    output_dir: Path = sample_dir / subdir_name
                    
                    # Create combined habitat map path
                    comb_filename: str = f"{base_name}_{phase}_k={n_clusters}_habitat_comb.nii.gz"
                    comb_output_path: Path = output_dir / comb_filename
                    
                    # Check if we should skip processing
                    if skip_existing and output_dir.exists():
                        # Check if output directory contains the expected files
                        
                        # Check for combined habitat map
                        if comb_output_path.exists():
                            # Check for at least one habitat binary mask
                            habitat_masks = list(output_dir.glob(f"{base_name}_{phase}_k={n_clusters}_habitat_*.nii.gz"))
                            if len(habitat_masks) > n_clusters:  # Should have at least the comb file and n_clusters habitat masks
                                print(f"Skipping {sample_dir.name} in phase {phase}: "
                                      f"Output directory already contains habitat files ({len(habitat_masks)})")
                                pbar.update(1)
                                continue
                    
                    # Create output directory if it doesn't exist
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Load sample features and mask
                    features, binary_mask, mask_meta = load_sample_features(sample_dir, phase, mask_type, radiomics_type)
                    
                    # Generate and save combined habitat map
                    habitat_map: np.ndarray = generate_habitat_map(kmeans, features, binary_mask, mask_meta, comb_output_path, small_region_thresh)
                    
                    # Export binary masks for each label (excluding 0)
                    unique_labels: np.ndarray = np.arange(1, n_clusters + 1)
                    label_indices: np.ndarray = unique_labels[unique_labels > 0]
                    
                    print(f"Exporting binary masks for labels: {label_indices}")
                    
                    for label_idx in label_indices:
                        # Create binary mask
                        binary_label_mask: np.ndarray = (habitat_map == label_idx).astype(np.uint8)
                        
                        # Create output path
                        binary_filename: str = f"{base_name}_{phase}_k={n_clusters}_habitat_{label_idx}.nii.gz"
                        binary_output_path: Path = output_dir / binary_filename
                        
                        # Save binary mask
                        save_image = SaveImage(
                            output_dir=str(binary_output_path.parent),
                            output_format="nii.gz",
                            separate_folder=False,
                            print_log=True,
                            output_postfix="",
                            output_dtype=np.uint8
                        )
                        
                        save_image(binary_label_mask, meta_data=mask_meta, filename=str(binary_output_path).replace('.nii.gz', ''))
                        print(f"Saved binary mask for label {label_idx} to: {binary_output_path}")

                except Exception as e:
                    print(f"Error processing {sample_dir.name} in phase {phase}: {e}")
                finally:
                    pbar.update(1)


def main() -> None:
    """
    Main function to orchestrate the habitat classification workflow.
    """
    # Parse command line arguments
    args: argparse.Namespace = parse_args()

    # Print configuration
    print(f"Model parameter file: {args.model_parameter}")
    print(f"Root directory: {args.root_dir}")
    print(f"Mask types: {args.mask_type}")
    print(f"Phases: {args.phase}")
    print(f"Radiomics type: '{args.radiomics_type}'")
    print(f"Small region threshold: {args.small_region_thresh} voxels")

    # Load K-Means model
    print("\nLoading K-Means model...")
    kmeans: KMeans = load_kmeans_model(args.model_parameter)
    print(f"Loaded K-Means model with {kmeans.cluster_centers_.shape[0]} clusters")

    # Process each mask type
    print("\nStarting habitat classification...")
    for mask_type in args.mask_type:
        process_mask_type(Path(args.root_dir), mask_type, args.phase, kmeans, args.radiomics_type, args.skip_existing, args.small_region_thresh)

    print("\nHabitat classification completed successfully!")


if __name__ == '__main__':
    main()
