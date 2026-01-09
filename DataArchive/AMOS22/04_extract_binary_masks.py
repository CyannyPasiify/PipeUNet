# -*- coding: utf-8 -*-
"""
参照02_confirm_pairs.py生成二值蒙版提取脚本。使用argparse。接收-r/--root_dir指定根目录，-e/--label_explanation指定AMOS22源数据存档中的dataset.json文件，读取其中的labels，这是一个字典，按照"<label_num>": "<label_explanation>"记录了标签值及其解释，标签值需要转换为整型数值以使用。根目录下包含多个子集子目录，子集子目录下包含以amos_xxxx命名的二级子目录，从每个二级子目录中匹配形如amos_xxxx_mask.nii.gz的蒙版文件，使用LoadImage读取其内容，将其中的所有标签值（从0开始）对应的蒙版分离出来，转换成多个01二值蒙版，然后使用SaveImage输出，文件保存在相同二级子目录下，文件名为amos_xxxx_mask后接对应的label_explanation后缀。在处理途中同时检查mask文件中的实际标签值类别，如果有不在dataset.json所记录labels范围内的，则报告错误但不影响程序继续执行。
"""

"""
Binary Mask Extraction Script for AMOS22 Dataset

This script extracts binary masks from multi-label mask files, separating each label into individual
binary masks for easier processing and analysis.

Parameters:
    -r, --root_dir: Root directory containing sample directories at any depth
    -e, --label_explanation: Path to label_map.yaml file containing label definitions

Usage Examples:
    python 04_extract_binary_masks.py -r /path/to/grouped -e /path/to/label_map.yaml
    python 04_extract_binary_masks.py --root_dir /path/to/grouped --label_explanation /path/to/label_map.yaml
"""

import argparse
import re
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from monai.transforms import LoadImage, SaveImage


def parse_args():
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
  %(prog)s -r /path/to/grouped -e /path/to/label_map.yaml
  %(prog)s --root_dir /path/to/grouped --label_explanation /path/to/label_map.yaml
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing sample directories at any depth'
    )

    parser.add_argument(
        '-e', '--label_explanation',
        type=str,
        required=True,
        help='Path to label_map.yaml file containing label definitions'
    )

    return parser.parse_args()


def load_label_definitions(label_map_path):
    """
    Load label definitions from label_map.yaml file.
    
    Args:
        label_map_path (str or Path): Path to label_map.yaml file
        
    Returns:
        dict: Dictionary mapping label numbers (int) to label explanations (str)
    """
    label_map_file = Path(label_map_path)

    if not label_map_file.exists():
        raise FileNotFoundError(f"Label map YAML file not found: {label_map_path}")

    with open(label_map_file, 'r', encoding='utf-8') as f:
        label_map = yaml.safe_load(f)

    if 'full_form_index_map' not in label_map:
        raise ValueError(f"'full_form_index_map' key not found in label_map.yaml: {label_map_path}")

    full_form_index_map = label_map['full_form_index_map']

    label_definitions = {}
    for label_num, label_name in full_form_index_map.items():
        if not isinstance(label_num, int):
            print(f"Warning: Skipping invalid label key '{label_num}' (not an integer)")
            continue
        label_definitions[label_num] = re.sub(r'[\s/\\]', '_', label_name)

    if not label_definitions:
        raise ValueError(f"No valid label definitions found in label_map.yaml: {label_map_path}")

    return label_definitions


def extract_binary_masks(mask_data, label_definitions):
    """
    Extract binary masks from multi-label mask data.
    
    Args:
        mask_data (np.ndarray): Multi-label mask data array
        label_definitions (dict): Dictionary mapping label numbers to label explanations
        
    Returns:
        tuple: (list of binary mask arrays, list of label numbers, list of label explanations)
    """
    binary_masks = []
    label_numbers = []
    label_names = []

    for label_num in sorted(label_definitions.keys()):
        label_name = label_definitions[label_num].replace(' ', '_')
        binary_mask = (mask_data == label_num).astype(np.uint8)
        binary_masks.append(binary_mask)
        label_numbers.append(label_num)
        label_names.append(label_name)

    return binary_masks, label_numbers, label_names


def find_sample_dirs(root_dir):
    """
    Recursively find all sample directories named amos_xxxx*.
    
    Args:
        root_dir (Path): Root directory to search
        
    Returns:
        list: List of Path objects pointing to sample directories
    """
    root_path = Path(root_dir)
    sample_dirs = []
    
    for path in root_path.rglob('amos_*'):
        if path.is_dir():
            sample_dirs.append(path)
    
    return sorted(sample_dirs)


def process_sample_dir(sample_dir, label_definitions):
    """
    Process a single sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        label_definitions (dict): Dictionary mapping label numbers to label explanations
        
    Returns:
        tuple: (masks_processed, binary_masks_extracted, error_count)
    """
    loader = LoadImage(image_only=False, dtype=None)

    sample_name = sample_dir.name

    match = re.match(r'amos_(\d{4})', sample_name)
    if not match:
        return 0, 0, 0

    sample_id = match.group(1)

    mask_files = list(sample_dir.glob('*_mask.nii.gz'))

    if not mask_files:
        return 0, 0, 0

    mask_file = mask_files[0]

    masks_processed = 0
    binary_masks_extracted = 0
    error_count = 0

    try:
        mask_data, mask_meta = loader(str(mask_file))

        unique_labels = np.unique(mask_data)
        valid_labels = set(label_definitions.keys())
        invalid_labels = [label for label in unique_labels if label not in valid_labels]

        if invalid_labels:
            print(f"Error: {mask_file.name} contains labels not in label_map.yaml: {invalid_labels}")
            error_count += 1

        binary_masks, label_numbers, label_names = extract_binary_masks(mask_data, label_definitions)

        saver = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.uint8)

        for binary_mask, label_num, label_name in zip(binary_masks, label_numbers, label_names):
            output_filestem = sample_dir / f'amos_{sample_id}_mask_{label_num:02d}_{label_name}'

            saver(binary_mask, meta_data=mask_meta, filename=output_filestem)
            binary_masks_extracted += 1

        masks_processed += 1

    except Exception as e:
        print(f"Error processing {mask_file.name}: {str(e)}")
        error_count += 1

    return masks_processed, binary_masks_extracted, error_count


def process_root_dir(root_dir, label_definitions):
    """
    Process all sample directories in the root directory recursively.
    
    Args:
        root_dir (str or Path): Root directory containing sample directories at any depth
        label_definitions (dict): Dictionary mapping label numbers to label explanations
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    sample_dirs = find_sample_dirs(root_path)

    if not sample_dirs:
        print(f"Warning: No sample directories (amos_xxxx*) found in {root_dir}")
        return

    total_masks_processed = 0
    total_binary_masks_extracted = 0
    total_errors = 0

    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Label definitions: {label_definitions}")
    print(f"Number of labels: {len(label_definitions)}\n")

    for sample_dir in tqdm(sample_dirs, desc='Processing samples'):
        masks_processed, binary_masks_extracted, errors = process_sample_dir(sample_dir, label_definitions)
        total_masks_processed += masks_processed
        total_binary_masks_extracted += binary_masks_extracted
        total_errors += errors

    print(f"\nProcessing completed!")
    print(f"Total mask files processed: {total_masks_processed}")
    print(f"Total binary masks extracted: {total_binary_masks_extracted}")
    print(f"Total errors encountered: {total_errors}")


def main():
    """
    Main function to orchestrate the binary mask extraction process.
    """
    args = parse_args()

    print(f"Processing data from: {args.root_dir}")
    print(f"Loading label definitions from: {args.label_explanation}")

    try:
        label_definitions = load_label_definitions(args.label_explanation)
    except Exception as e:
        print(f"Error loading label definitions: {str(e)}")
        return

    process_root_dir(args.root_dir, label_definitions)

    print("Binary mask extraction completed successfully!")


if __name__ == '__main__':
    main()
