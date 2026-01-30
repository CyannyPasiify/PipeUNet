# -*- coding: utf-8 -*-
"""
 -r/--root_dir指定根目录，根目录下包含若干层级的子目录，最低层为site中心目录，中心目录下包含phase阶段目录，即治疗前pre和治疗后post目录，阶段目录中包含{site}_{pid}_{phase}样本目录，每个中包含形如{site}_{pid}_{phase}_mask.nii.gz的蒙版文件。
 -e/--label_explanation指定label_map.yaml文件，读取其中的short_form_index_map，这是一个字典，按照<label_index>整型数: "<label_name>"记录了标签值及其解释。对每个样本蒙版文件，使用LoadImage读取其内容，将其中的所有标签值（从0开始）对应的蒙版分离出来，转换成多个01二值蒙版，然后使用SaveImage输出，文件保存在相同二级子目录下，文件名为{site}_{pid}_{phase}_mask_{label_index:02d}_{label_name}.nii.gz。在处理途中同时检查mask文件中的实际标签值类别，如果有不在short_form_index_map范围内的，则报告错误但不影响程序继续执行。
  对全部变量和函数参数添加类型注解。 
  除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Binary Mask Extraction Script for ESO-2025 Dataset

This script extracts binary masks from multi-label mask files, separating each label into individual
binary masks for easier processing and analysis.

Parameters:
    -r, --root_dir: Root directory containing nested subdirectories with sample data
    -e, --label_explanation: Path to label_map.yaml file containing label definitions
    --skip_existing: Skip processing if binary mask files already exist

Usage Examples:
    python 03_extract_binary_masks.py -r /path/to/root -e /path/to/label_map.yaml
    python 03_extract_binary_masks.py --root_dir /path/to/root --label_explanation /path/to/label_map.yaml
    python 03_extract_binary_masks.py -r /path/to/root -e /path/to/label_map.yaml --skip_existing
"""

import argparse
import re
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from monai.transforms import LoadImage, SaveImage
from typing import List, Dict, Tuple, Optional, Union


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir and label_explanation
    """
    parser = argparse.ArgumentParser(
        description='Extract binary masks from multi-label mask files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/root -e /path/to/label_map.yaml
  %(prog)s --root_dir /path/to/root --label_explanation /path/to/label_map.yaml
  %(prog)s -r /path/to/root -e /path/to/label_map.yaml --skip_existing
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing nested subdirectories with sample data'
    )

    parser.add_argument(
        '-e', '--label_explanation',
        type=str,
        required=True,
        help='Path to label_map.yaml file containing label definitions'
    )

    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip processing if binary mask files already exist'
    )

    return parser.parse_args()


def load_label_definitions(label_map_path: Union[str, Path]) -> Dict[int, str]:
    """
    Load label definitions from label_map.yaml file.
    
    Args:
        label_map_path (Union[str, Path]): Path to label_map.yaml file
        
    Returns:
        Dict[int, str]: Dictionary mapping label numbers (int) to label explanations (str)
    """
    label_map_file = Path(label_map_path)

    if not label_map_file.exists():
        raise FileNotFoundError(f"Label map YAML file not found: {label_map_path}")

    with open(label_map_file, 'r', encoding='utf-8') as f:
        label_map = yaml.safe_load(f)

    if 'short_form_index_map' not in label_map:
        raise ValueError(f"'short_form_index_map' key not found in label_map.yaml: {label_map_path}")

    short_form_index_map = label_map['short_form_index_map']

    label_definitions: Dict[int, str] = {}
    for label_num, label_name in short_form_index_map.items():
        if not isinstance(label_num, int):
            print(f"Warning: Skipping invalid label key '{label_num}' (not an integer)")
            continue
        label_definitions[label_num] = re.sub(r'[\s/\\]', '_', label_name)

    if not label_definitions:
        raise ValueError(f"No valid label definitions found in label_map.yaml: {label_map_path}")

    return label_definitions


def extract_binary_masks(mask_data: np.ndarray, label_definitions: Dict[int, str]) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Extract binary masks from multi-label mask data.
    
    Args:
        mask_data (np.ndarray): Multi-label mask data array
        label_definitions (Dict[int, str]): Dictionary mapping label numbers to label explanations
        
    Returns:
        Tuple[List[np.ndarray], List[int], List[str]]: (list of binary mask arrays, list of label numbers, list of label explanations)
    """
    binary_masks: List[np.ndarray] = []
    label_numbers: List[int] = []
    label_names: List[str] = []

    for label_num in sorted(label_definitions.keys()):
        label_name = label_definitions[label_num].replace(' ', '_')
        binary_mask = (mask_data == label_num).astype(np.uint8)
        binary_masks.append(binary_mask)
        label_numbers.append(label_num)
        label_names.append(label_name)

    return binary_masks, label_numbers, label_names


def find_sample_dirs(root_dir: Union[str, Path]) -> List[Path]:
    """
    Recursively find all sample directories matching {site}_{pid}_{phase} pattern.
    
    Args:
        root_dir (Union[str, Path]): Root directory to search
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path = Path(root_dir)
    sample_dirs: List[Path] = []
    
    for path in root_path.rglob('*_*_*'):
        if path.is_dir() and re.match(r'[A-Z]+_[0-9]+_(pre|post)', path.name):
            sample_dirs.append(path)
    
    return sorted(sample_dirs)


def process_sample_dir(sample_dir: Path, label_definitions: Dict[int, str], skip_existing: bool = False) -> Tuple[int, int, int]:
    """
    Process a single sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        label_definitions (Dict[int, str]): Dictionary mapping label numbers to label explanations
        skip_existing (bool): Skip processing if binary mask files already exist
        
    Returns:
        Tuple[int, int, int]: (masks_processed, binary_masks_extracted, error_count)
    """
    loader: LoadImage = LoadImage(image_only=False, dtype=None)

    sample_name: str = sample_dir.name

    match = re.match(r'([A-Z]+)_([0-9]+)_(pre|post)', sample_name)
    if not match:
        return 0, 0, 0

    site_id: str = match.group(1)
    pid: str = match.group(2)
    phase: str = match.group(3)
    sample_id: str = f'{site_id}_{pid}_{phase}'

    mask_files: List[Path] = list(sample_dir.glob(f'{sample_id}_mask.nii.gz'))

    if not mask_files:
        return 0, 0, 0

    mask_file: Path = mask_files[0]

    masks_processed: int = 0
    binary_masks_extracted: int = 0
    error_count: int = 0

    # Check if all binary mask files already exist
    if skip_existing:
        all_exist = True
        for label_num in sorted(label_definitions.keys()):
            label_name = label_definitions[label_num].replace(' ', '_')
            output_file = sample_dir / f'{sample_id}_mask_{label_num:02d}_{label_name}.nii.gz'
            if not output_file.exists():
                all_exist = False
                break
        
        if all_exist:
            print(f"Skipping {mask_file.name}: All binary mask files already exist")
            return masks_processed, binary_masks_extracted, error_count

    try:
        mask_data, mask_meta = loader(str(mask_file))

        unique_labels: np.ndarray = np.unique(mask_data)
        valid_labels: set = set(label_definitions.keys())
        invalid_labels: List[int] = [label for label in unique_labels if label not in valid_labels]

        if invalid_labels:
            print(f"Error: {mask_file.name} contains labels not in label_map.yaml: {invalid_labels}")
            error_count += 1

        binary_masks, label_numbers, label_names = extract_binary_masks(mask_data, label_definitions)

        saver: SaveImage = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.uint8)

        for binary_mask, label_num, label_name in zip(binary_masks, label_numbers, label_names):
            output_filestem = sample_dir / f'{sample_id}_mask_{label_num:02d}_{label_name}'

            saver(binary_mask, meta_data=mask_meta, filename=output_filestem)
            binary_masks_extracted += 1

        masks_processed += 1

    except Exception as e:
        print(f"Error processing {mask_file.name}: {str(e)}")
        error_count += 1

    return masks_processed, binary_masks_extracted, error_count


def process_root_dir(root_dir: Union[str, Path], label_definitions: Dict[int, str], skip_existing: bool = False) -> None:
    """
    Process all sample directories in the root directory recursively.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing sample directories at any depth
        label_definitions (Dict[int, str]): Dictionary mapping label numbers to label explanations
        skip_existing (bool): Skip processing if binary mask files already exist
    """
    root_path: Path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    sample_dirs: List[Path] = find_sample_dirs(root_path)

    if not sample_dirs:
        print(f"Warning: No sample directories found in {root_dir}")
        return

    total_masks_processed: int = 0
    total_binary_masks_extracted: int = 0
    total_errors: int = 0

    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Label definitions: {label_definitions}")
    print(f"Number of labels: {len(label_definitions)}\n")

    for sample_dir in tqdm(sample_dirs, desc='Processing samples'):
        masks_processed, binary_masks_extracted, errors = process_sample_dir(sample_dir, label_definitions, skip_existing)
        total_masks_processed += masks_processed
        total_binary_masks_extracted += binary_masks_extracted
        total_errors += errors

    print(f"\nProcessing completed!")
    print(f"Total mask files processed: {total_masks_processed}")
    print(f"Total binary masks extracted: {total_binary_masks_extracted}")
    print(f"Total errors encountered: {total_errors}")


def main() -> None:
    """
    Main function to orchestrate the binary mask extraction process.
    """
    args: argparse.Namespace = parse_args()

    print(f"Processing data from: {args.root_dir}")
    print(f"Loading label definitions from: {args.label_explanation}")

    try:
        label_definitions: Dict[int, str] = load_label_definitions(args.label_explanation)
    except Exception as e:
        print(f"Error loading label definitions: {str(e)}")
        return

    process_root_dir(args.root_dir, label_definitions, args.skip_existing)

    print("Binary mask extraction completed successfully!")


if __name__ == '__main__':
    main()