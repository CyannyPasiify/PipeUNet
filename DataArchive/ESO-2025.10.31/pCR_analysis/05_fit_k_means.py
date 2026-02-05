#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 -r/--root_dir指定根目录，根目录下包含若干层级的子目录，第一层是site中心目录，中心目录下包含包含{site}_{pid}样本目录，样本目录中包含形如{site}_{pid}_(pre|post)_voxel_radiomics的治疗前|后影像组学特征目录，{site}_{pid}_(pre|post)(_{mask_type})?_mask.nii.gz的若干蒙版文件，其中(_{mask_type})?部分表示_{mask_type}内容是可选的。{site}_{pid}_(pre|post)_voxel_radiomics目录下包含若干{site}_{pid}_(pre|post)_{radiomics_name}.nii.gz的影像组学Voxel特征图。 
 -m/--mask_type用于指定选中的蒙版文件的extra_type列表，默认值为['']。 
 -p/--phase用于指定阶段列表，默认为['pre','post']。 
 -rt/--radiomics_type用于指定影像组学类型后缀，默认值为''。 
 遍历mask_type中的每个元素，对每个mask_type：遍历每个样本，选中所有phase指定阶段的voxel_radiomics，对voxel_radiomics目录下的全部影像组学特征图文件按radiomics_name进行排序；然后顺序读取影像组学特征图并合成一个多通道特征图feat，选用对应mask_type的mask资源文件，收集feat中由mask指定前景体素点的特征向量；拟合k_means模型，-k/--k_cluster指定2个数，表示所尝试k值的范围，默认[2,12]；-it/--iter指定迭代步数，默认为None，如果为None时则根据单步迭代差异自动决定收敛时机，否则迭代固定步数后停止；-o/--output_model_dir指定拟合完成的K-Means模型参数保存目录，对于每个k值保存一个{k}_means_anchors.yaml文件，其中anchors属性以列表形式记录了K-Means模型每个锚特征向量，计算当前K值下的SSE、Silhouette、CH、DB值并记录为属性；-sp/--select_prob指定取用每个点作为KMeans样本的概率（float），默认0.05=5%，在collect_voxel_features时就进行选取；-rs/--random_seed指定numpy和KMeans的随机数种子，默认0。
"""

"""
K-Means Clustering for Voxel-Based Radiomics Features

This script fits K-Means models to voxel-based radiomics features extracted from medical images.

Parameters:
    -r, --root_dir: Root directory containing site directories with {site}_{pid} sample directories
    -p, --phase: List of phases to process (default: ['pre', 'post'])
    -m, --mask_type: List of mask types to process (default: [''])
    -rt, --radiomics_type: Radiomics type suffix for directory name (default: '')
    -k, --k_cluster: Range of k values to try (default: [2, 12])
    -it, --iter: Number of iterations (default: 300)
    -o, --output_model_dir: Directory to save K-Means model parameters
    -g, --gpu: GPU device ID to use for cuML (default: None, uses CPU)
    -sp, --select_prob: Probability to select each voxel as KMeans sample (default: 0.05 = 5%)

Usage Examples:
    python 04_fit_k_means.py -r /path/to/root
    python 04_fit_k_means.py --root_dir /path/to/root --phase pre --mask_type "" "peritumor" --k_cluster 2 8 --iter 100 --output_model_dir /path/to/models
    python 04_fit_k_means.py --root_dir /path/to/root --phase pre --mask_type ""  --radiomics_type custom --k_cluster 2 8 --output_model_dir /path/to/models
"""

import argparse
import re
import numpy as np
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Pattern
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from monai.transforms import LoadImage

# Try to import cuML for GPU acceleration
try:
    from cuml.cluster import KMeans as CuKMeans
    import cupy as cp

    has_cuml = True
    print("cuML imported successfully - using GPU acceleration")
except ImportError:
    from sklearn.cluster import KMeans

    has_cuml = False
    print("cuML not available - falling back to scikit-learn KMeans")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Fit K-Means models to voxel-based radiomics features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/root
  %(prog)s --root_dir /path/to/root --phase pre --mask_type "" "peritumor_3.0mm" --k_cluster 2 8 --iter 100 --output_model_dir /path/to/models
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
        '-m', '--mask_type',
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
        '-k', '--k_cluster',
        type=int,
        nargs=2,
        default=[2, 12],
        help='Range of k values to try (default: [2, 12])'
    )

    parser.add_argument(
        '-it', '--iter',
        type=int,
        default=300,
        help='Number of iterations (default: 300)'
    )

    parser.add_argument(
        '-o', '--output_model_dir',
        type=str,
        required=True,
        help='Directory to save K-Means model parameters'
    )

    parser.add_argument(
        '-g', '--gpu',
        type=int,
        default=None,
        help='GPU device ID to use for cuML (default: None, uses CPU)'
    )

    parser.add_argument(
        '-sp', '--select_prob',
        type=float,
        default=0.05,
        help='Probability to select each voxel as KMeans sample (default: 0.05 = 5%%)'
    )

    parser.add_argument(
        '-rs', '--random_seed',
        type=int,
        default=0,
        help='Random seed for numpy and KMeans (default: 0)'
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


def collect_voxel_features(sample_dir: Path, phase: str, mask_type: str, radiomics_type: str = '', select_prob: float = 0.05, random_seed: int = 0) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Collect voxel-based radiomics features from a sample directory.
    
    Args:
        sample_dir: Path to the sample directory
        phase: Phase (pre/post)
        mask_type: Mask type
        radiomics_type: Radiomics type suffix
        select_prob: Probability to select each voxel as KMeans sample (default: 0.05 = 5%)
        random_seed: Random seed for numpy and KMeans (default: 0)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and their corresponding mask
    """
    base_name: str = sample_dir.name

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

    # load_image = LoadImage(image_only=False, reader='ITKReader')  # NibabelReader will encounter NaN, we have no choice but using ITKReader (as 0.0)
    load_image = LoadImage(image_only=False)
    mask_data, mask_meta = load_image(str(mask_file))
    binary_mask: np.ndarray = (mask_data > 0).astype(bool)
    print(f'Loaded binary_mask with shape {binary_mask.shape}')

    radiomics_files: List[Path] = list(voxel_radiomics_dir.glob(f"{base_name}_{phase}_*.nii.gz"))

    radiomics_info = []
    for file_path in radiomics_files:
        match = re.match(rf'^{base_name}_{phase}_(.+)\.nii\.gz$', file_path.name)
        if match:
            radiomics_name = match.group(1)
            radiomics_info.append((radiomics_name, file_path))

    radiomics_info.sort(key=lambda x: x[0])
    sorted_files: List[Path] = [file_path for _, file_path in radiomics_info]

    features_list = []
    for file_path in sorted_files:
        feature_data, feature_meta = load_image(str(file_path))
        # Extract foreground immediately to reduce memory usage
        feature_foreground = feature_data[binary_mask]
        print(f'Loaded voxel-radiomics {file_path} with shape {tuple(feature_data.shape)}, '
              f'foreground voxels: {len(feature_foreground)}')
        feature_nan = np.isnan(feature_foreground)
        nan_count = feature_nan.sum()
        if nan_count > 0:
            feature_foreground = feature_foreground[np.logical_not(feature_nan)]
            print(f'NaN found {feature_nan.sum()}: {file_path}\n'
                  f'now foreground voxels: {len(feature_foreground)}')

        features_list.append(feature_foreground)
        # Free memory
        del feature_data

    # Stack foreground features directly (n_voxels, channels)
    features_in_mask: np.ndarray = np.stack(features_list, axis=1)
    print(f'{sample_dir.name}_{phase} Features shape: {features_in_mask.shape}')

    # Apply sample selection based on select_prob
    if select_prob < 1.0:
        n_voxels = features_in_mask.shape[0]
        # Generate random mask for selection
        selection_mask = np.random.rand(n_voxels) < select_prob
        # Ensure at least some voxels are selected
        if np.sum(selection_mask) == 0:
            # Select at least 100 voxels or all if fewer than 100
            n_select = min(100, n_voxels)
            selection_indices = np.random.choice(n_voxels, n_select, replace=False)
            selection_mask[selection_indices] = True
        
        selected_features = features_in_mask[selection_mask]
        print(f'Selected {selected_features.shape[0]} voxels out of {n_voxels} based on probability {select_prob}')
        return selected_features, binary_mask
    else:
        # Use all voxels if select_prob >= 1.0
        return features_in_mask, binary_mask


def fit_k_means(features: np.ndarray, k_range: List[int], max_iter: Optional[int], mask_output_dir: Path,
                gpu: Optional[int] = None, random_seed: int = 0) -> Dict[int, Dict]:
    """
    Fit K-Means models for a range of k values and save results immediately.
    
    Args:
        features: Feature array
        k_range: Range of k values
        max_iter: Maximum number of iterations
        mask_output_dir: Directory to save K-Means results
        gpu: GPU device ID to use for cuML (default: None, uses default GPU)
        random_seed: Random seed for numpy and KMeans (default: 0)
        
    Returns:
        Dict[int, Dict]: Results for each k value
    """
    results = {}

    with tqdm(total=k_range[1] - k_range[0] + 1, desc="Fitting K-Means", unit="k") as pbar:
        for k in range(k_range[0], k_range[1] + 1):
            pbar.set_postfix_str(f"k={k}")

            print(f"\nFitting K-Means for k={k}...")

            # Use cuML KMeans if available, otherwise use scikit-learn
            if has_cuml:
                print(f"  Attempting to use cuML (GPU {gpu if gpu is not None else 'default'})...")
                kmeans = CuKMeans(n_clusters=k, random_state=random_seed, max_iter=max_iter, verbose=1)
                using_gpu = True
            else:
                print("  Using scikit-learn (CPU)...")
                kmeans = KMeans(n_clusters=k, random_state=random_seed, max_iter=max_iter, verbose=1)
                using_gpu = False

            kmeans.fit(features)

            sse: float = float(kmeans.inertia_)

            # Get labels - need to convert to numpy array for cuML
            labels = kmeans.labels_

            silhouette: float = float(silhouette_score(features, labels))
            ch: float = float(calinski_harabasz_score(features, labels))
            db: float = float(davies_bouldin_score(features, labels))

            # Get cluster centers - need to convert to numpy array for cuML
            cluster_centers = kmeans.cluster_centers_

            result = {
                'anchors': cluster_centers.tolist(),
                'sse': sse,
                'silhouette': silhouette,
                'ch': ch,
                'db': db
            }
            results[k] = result

            # Save immediately
            output_path: Path = mask_output_dir / f"{k}_means_anchors.yaml"
            with open(output_path, 'w') as f:
                yaml.dump(result, f, sort_keys=False)

            # Print information
            print(f"K={k} completed:")
            print(f"  SSE: {sse:.6f}")
            print(f"  Silhouette: {silhouette:.6f}")
            print(f"  Calinski-Harabasz: {ch:.6f}")
            print(f"  Davies-Bouldin: {db:.6f}")
            print(f"  Results saved to: {output_path}")
            if using_gpu:
                print(f"  Using cuML (GPU {gpu if gpu is not None else 'default'})")
            else:
                print("  Using scikit-learn (CPU)")

            pbar.update(1)

    return results


def process_mask_type(root_dir: Path, mask_type: str, phases: List[str], k_range: List[int], max_iter: Optional[int],
                      output_dir: Path, radiomics_type: str = '', gpu: Optional[int] = None, select_prob: float = 0.05, random_seed: int = 0) -> None:
    """
    Process all samples for a specific mask type.
    
    Args:
        root_dir: Root directory
        mask_type: Mask type to process
        phases: Phases to process
        k_range: Range of k values
        max_iter: Maximum number of iterations
        output_dir: Output directory for models
        radiomics_type: Radiomics type suffix
        gpu: GPU device ID to use for cuML (default: None, uses default GPU)
        select_prob: Probability to select each voxel as KMeans sample (default: 0.05 = 5%)
        random_seed: Random seed for numpy and KMeans (default: 0)
    """
    print(f"Processing mask type: '{mask_type}'")

    sample_dirs: List[Path] = find_sample_dirs(str(root_dir))
    print(f"Found {len(sample_dirs)} sample directories")

    all_features = []

    with tqdm(total=len(sample_dirs) * len(phases), desc="Collecting features", unit="sample-phase") as pbar:
        for sample_dir in sample_dirs:
            for phase in phases:
                try:
                    features, _ = collect_voxel_features(sample_dir, phase, mask_type, radiomics_type, select_prob, random_seed)
                    all_features.append(features)
                except Exception as e:
                    print(f"Error processing {sample_dir.name} in phase {phase}: {e}")
                finally:
                    pbar.update(1)

    if not all_features:
        print(f"No features collected for mask type '{mask_type}'. Skipping...")
        return

    combined_features: np.ndarray = np.concatenate(all_features, axis=0)
    print(f"Total features collected: {combined_features.shape[0]}")
    print(f"Feature dimension: {combined_features.shape[1]}")

    mask_output_dir: Path = (
            output_dir / '_'.join([
        f"{mask_type}_mask" if mask_type else "mask",
        f"{radiomics_type}_voxel_radiomics" if radiomics_type else "voxel_radiomics"]
    ))
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {mask_output_dir}")

    print(f"Fitting K-Means models for k values: {k_range[0]} to {k_range[1]}")
    results: Dict[int, Dict] = fit_k_means(combined_features, k_range, max_iter, mask_output_dir, gpu, random_seed)

    print(f"All K-Means models completed and saved!")


def main() -> None:
    """
    Main function to orchestrate the K-Means fitting process.
    """
    args: argparse.Namespace = parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    print(f"Set random seed to: {args.random_seed}")

    if args.gpu:
        # Try to import cuML for GPU acceleration
        try:
            from cuml.cluster import KMeans as CuKMeans
            import cupy as cp

            has_cuml = True
            print("cuML imported successfully - using GPU acceleration")
        except ImportError:
            from sklearn.cluster import KMeans

            has_cuml = False
            print("cuML not available - falling back to scikit-learn KMeans")
    else:
        from sklearn.cluster import KMeans

        has_cuml = False
        print("command using CPU - scikit-learn KMeans")

    
    if has_cuml and args.gpu:
        cp.cuda.Device(args.gpu).use()

    root_dir: Path = Path(args.root_dir)
    output_model_dir: Path = Path(args.output_model_dir)

    print(f"Root directory: {root_dir}")
    print(f"Mask types: {args.mask_type}")
    print(f"Phases: {args.phase}")
    print(f"K range: {args.k_cluster}")
    print(f"Max iterations: {args.iter}")
    print(f"Output model directory: {output_model_dir}")
    print(f"Radiomics type: '{args.radiomics_type}'")
    print(f"GPU device ID: {args.gpu if args.gpu is not None else 'default'}")
    print(f"Sample selection probability: {args.select_prob}")
    print(f"Random seed: {args.random_seed}")

    for mask_type in args.mask_type:
        process_mask_type(root_dir, mask_type, args.phase, args.k_cluster, args.iter, output_model_dir,
                          args.radiomics_type, args.gpu, args.select_prob, args.random_seed)

    print("K-Means fitting process completed!")


if __name__ == '__main__':
    main()
