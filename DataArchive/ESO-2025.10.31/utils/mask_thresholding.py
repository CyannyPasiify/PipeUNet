# -*- coding: utf-8 -*-
"""
参照，生成mask_thresholding.py。-r/--root_dir指定根目录，根目录下包含若干层级的子目录，最低层为site中心目录，中心目录下包含phase阶段目录，即治疗前pre和治疗后post目录，阶段目录中包含{site}_{pid}_{phase}样本目录，每个中包含形如{site}_{pid}_{phase}_volume.nii.gz的图像文件，{site}_{pid}_{phase}_mask.nii.gz的蒙版文件。-t/--thresh指定2个float数（默认-140.0, 1000.0），表示有效值域下限和上限，对于在蒙版文件中标注为前景的点，如果其图像值在thresh之外，则将这些点值设置为0背景。指定-o/--output_dir输出目录，如果蒙版发生了更新，在控制台报告此修正，并将新蒙版文件输出到此目录，output_dir保持目录树结构与root_dir一致。
 对全部变量和函数参数添加类型注解。
 除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Mask Thresholding Script for ESO-2025 Dataset

This script processes medical image masks by thresholding based on corresponding image values.
For foreground pixels in the mask, if the corresponding image pixel value is outside the specified
threshold range, the mask pixel is set to background (0).

Parameters:
    -r, --root_dir: Root directory containing nested subdirectories with sample data
    -t, --thresh: Two float values specifying the lower and upper threshold bounds (default: -140.0, 1000.0)
    -o, --output_dir: Output directory to save updated masks with the same directory structure as root_dir

Usage Examples:
    python mask_thresholding.py -r /path/to/root -o /path/to/output
    python mask_thresholding.py --root_dir /path/to/root --thresh -140.0 1000.0 --output_dir /path/to/output
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
from monai.transforms import LoadImage, SaveImage
from monai.data import MetaTensor
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir, thresh, and output_dir
    """
    parser = argparse.ArgumentParser(
        description='Threshold masks based on corresponding image values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/root -o /path/to/output
  %(prog)s --root_dir /path/to/root --thresh -140.0 1000.0 --output_dir /path/to/output
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing nested subdirectories with sample data'
    )

    parser.add_argument(
        '-t', '--thresh',
        type=float,
        nargs=2,
        default=[-140.0, 1000.0],
        help='Two float values specifying the lower and upper threshold bounds (default: -140.0, 1000.0)'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Output directory to save updated masks with the same directory structure as root_dir'
    )

    return parser.parse_args()


def process_mask(image_data: MetaTensor, mask_data: MetaTensor, threshold: Tuple[float, float]) \
        -> Tuple[MetaTensor, int]:
    """
    Process mask by thresholding based on corresponding image values.
    
    Args:
        image_data (MetaTensor): Image data array
        mask_data (MetaTensor): Mask data array
        threshold (Tuple[float, float]): Lower and upper threshold bounds
        
    Returns:
        Tuple[MetaTensor, int]: Updated mask data and number of pixels modified
    """
    lower, upper = threshold

    # Create mask of foreground pixels
    foreground_mask = (mask_data > 0)

    # Create mask of pixels outside threshold range
    outside_threshold = (image_data < lower) | (image_data > upper)

    # Combine masks to find foreground pixels outside threshold
    pixels_to_correct = foreground_mask & outside_threshold

    # Count number of pixels to correct
    num_corrected = np.sum(pixels_to_correct)

    # Correct the mask
    updated_mask = mask_data.clone()
    updated_mask[pixels_to_correct] = 0

    return updated_mask, num_corrected


def process_sample_dir(sample_dir: Path, threshold: Tuple[float, float], output_dir: Path) -> Tuple[str, int]:
    """
    Process all image-mask pairs in a sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        threshold (Tuple[float, float]): Lower and upper threshold bounds
        output_dir (Path): Output directory root
        
    Returns:
        Tuple[str, int]: Sample ID and number of pixels modified
    """
    sample_name: str = sample_dir.name
    match = re.match(r'([^_]+)_([^_]+)_(pre|post)', sample_name)
    if not match:
        return '', 0

    site_id: str = match.group(1)
    pid: str = match.group(2)
    phase: str = match.group(3)
    sample_id: str = f'{site_id}_{pid}_{phase}'

    # Find image and mask files
    image_files: List[Path] = sorted(sample_dir.glob(f'{sample_id}_volume.nii.gz'))
    mask_files: List[Path] = sorted(sample_dir.glob(f'{sample_id}_mask.nii.gz'))

    if not image_files or not mask_files:
        return sample_id, 0

    image_file: Path = image_files[0]
    mask_file: Path = mask_files[0]

    # Load image and mask
    loader: LoadImage = LoadImage(image_only=False, dtype=np.float32)
    image_data, image_meta = loader(str(image_file))

    mask_loader: LoadImage = LoadImage(image_only=False, dtype=np.uint8)
    mask_data, mask_meta = mask_loader(str(mask_file))

    # Process mask
    updated_mask: MetaTensor
    updated_mask, num_corrected = process_mask(image_data, mask_data, threshold)

    # Save updated mask if changes were made
    if num_corrected > 0:
        # Create output directory structure
        relative_path: Path = sample_dir.relative_to(sample_dir.parents[2])  # Get path relative to root_dir/site/phase
        output_sample_dir: Path = output_dir / relative_path
        output_sample_dir.mkdir(parents=True, exist_ok=True)

        # Save updated mask
        output_mask_file: Path = output_sample_dir / mask_file.name
        saver: SaveImage = SaveImage(output_dir=str(output_sample_dir), output_postfix='', output_dtype=np.uint8)
        saver(updated_mask, meta_data=mask_meta, filename=str(output_mask_file).replace('.nii.gz', ''))

    return sample_id, num_corrected


def find_sample_dirs(root_dir: Path) -> List[Path]:
    """
    Recursively find all sample directories in the root directory.
    
    Args:
        root_dir (Path): Root directory to search
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    sample_dirs: List[Path] = []

    # Find all directories matching {site}_{pid}_{phase} pattern
    for path in root_dir.rglob('*'):
        if path.is_dir() and re.match(r'[^_]+_[^_]+_(pre|post)', path.name):
            sample_dirs.append(path)

    return sorted(sample_dirs)


def process_root_dir(root_dir: Path, threshold: Tuple[float, float], output_dir: Path) -> None:
    """
    Process all sample directories in the root directory.
    
    Args:
        root_dir (Path): Root directory containing sample data
        threshold (Tuple[float, float]): Lower and upper threshold bounds
        output_dir (Path): Output directory to save updated masks
    """
    if not root_dir.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    sample_dirs: List[Path] = find_sample_dirs(root_dir)

    if not sample_dirs:
        print(f"Warning: No sample directories found in {root_dir}")
        return

    total_corrected: int = 0
    samples_corrected: int = 0

    for sample_dir in tqdm(sample_dirs, desc='Processing samples'):
        sample_id, num_corrected = process_sample_dir(sample_dir, threshold, output_dir)

        if num_corrected > 0:
            total_corrected += num_corrected
            samples_corrected += 1
            print(f"Corrected {num_corrected} pixels in sample: {sample_id}")

    print(f"\nProcessing completed!")
    print(f"Total samples processed: {len(sample_dirs)}")
    print(f"Total samples corrected: {samples_corrected}")
    print(f"Total pixels corrected: {total_corrected}")
    print(f"Output directory: {output_dir}")


def main() -> None:
    """
    Main function to orchestrate the mask thresholding process.
    """
    args: argparse.Namespace = parse_args()

    root_path: Path = Path(args.root_dir)
    output_path: Path = Path(args.output_dir)
    threshold: Tuple[float, float] = tuple(args.thresh)

    print(f"Processing data from: {root_path}")
    print(f"Threshold range: {threshold[0]} to {threshold[1]}")
    print(f"Output directory: {output_path}")

    process_root_dir(root_path, threshold, output_path)

    print("Mask thresholding completed successfully!")


if __name__ == '__main__':
    main()
