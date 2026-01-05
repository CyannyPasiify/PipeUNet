# -*- coding: utf-8 -*-
"""
参照02_confirm_pairs.py生成二值蒙版提取脚本。使用argparse。接收-r/--root_dir指定根目录，-e/--label_explanation是一个str列表
指定每个标签值的释义，默认值为['Bg', 'RV', 'Myo', 'LV']。根目录下包含多个子集子目录，子集子目录下包含以patientxxx_frameyy命名
的二级子目录，从每个二级子目录中匹配形如patientxxx_frameyy_mask.nii.gz的蒙版文件，使用LoadImage读取其内容，将其中的所有标签值
（从0开始）对应的蒙版分离出来，转换成多个01二值蒙版，然后使用SaveImage输出，文件保存在相同二级子目录下，文件名为
patientxxx_frameyy_mask后接对应的label_explanation后缀。
"""

"""
Binary Mask Extraction Script for ACDC Dataset

This script extracts binary masks from multi-label mask files, separating each label into individual
binary masks for easier processing and analysis.

Parameters:
    -r, --root_dir: Root directory containing subset subdirectories (e.g., train, test)
    -e, --label_explanation: List of label explanations for each label value (default: ['Bg', 'RV', 'Myo', 'LV'])

Usage Examples:
    python 03_extract_binary_masks.py -r /path/to/grouped
    python 03_extract_binary_masks.py --root_dir /path/to/grouped --label_explanation Background RV_Cavity Myocardium LV_Cavity
"""

import argparse
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
  %(prog)s -r /path/to/grouped
  %(prog)s --root_dir /path/to/grouped --label_explanation Background RV_Cavity Myocardium LV_Cavity
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing subset subdirectories (e.g., train, test)'
    )

    parser.add_argument(
        '-e', '--label_explanation',
        type=str,
        nargs='+',
        default=['Bg', 'RV', 'Myo', 'LV'],
        help='List of label explanations for each label value (default: Bg RV Myo LV)'
    )

    return parser.parse_args()


def extract_binary_masks(mask_data, num_labels):
    """
    Extract binary masks from multi-label mask data.
    
    Args:
        mask_data (np.ndarray): Multi-label mask data array
        num_labels (int): Number of labels to extract
        
    Returns:
        list: List of binary mask arrays, one for each label
    """
    binary_masks = []

    for label in range(num_labels):
        binary_mask = (mask_data == label).astype(np.uint8)
        binary_masks.append(binary_mask)

    return binary_masks


def process_subset_dir(subset_dir, label_explanations):
    """
    Process all mask files in a subset directory.
    
    Args:
        subset_dir (Path): Path to the subset directory
        label_explanations (list): List of label explanations for naming output files
        
    Returns:
        tuple: (total_masks_processed, total_binary_masks_extracted)
    """
    num_labels = len(label_explanations)
    loader = LoadImage(image_only=False, dtype=None)

    total_masks_processed = 0
    total_binary_masks_extracted = 0

    pair_dirs = sorted(subset_dir.glob('patient*_frame*'))

    for pair_dir in tqdm(pair_dirs, desc=f'  Processing {subset_dir.name}', leave=False):
        identifier = pair_dir.name.split('_', maxsplit=2)[:2]
        combined_identifier = '_'.join(identifier)
        mask_file = pair_dir / f'{combined_identifier}_mask.nii.gz'

        if not mask_file.exists():
            print(f"Warning: Mask file not found: {mask_file}")
            continue

        try:
            mask_data, mask_meta = loader(str(mask_file))

            binary_masks = extract_binary_masks(mask_data, num_labels)

            saver = SaveImage(output_dir=str(pair_dir), output_postfix='', output_dtype=np.uint8)

            for label_idx, binary_mask in enumerate(binary_masks):
                label_name = label_explanations[label_idx]
                output_filestem = pair_dir / f'{combined_identifier}_mask_{label_name}'

                saver(binary_mask, meta_data=mask_meta, filename=output_filestem)
                total_binary_masks_extracted += 1

            total_masks_processed += 1

        except Exception as e:
            print(f"Error processing {mask_file.name}: {str(e)}")
            continue

    return total_masks_processed, total_binary_masks_extracted


def process_root_dir(root_dir, label_explanations):
    """
    Process all subset directories in the root directory.
    
    Args:
        root_dir (str or Path): Root directory containing subset subdirectories
        label_explanations (list): List of label explanations for naming output files
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    subset_dirs = [d for d in root_path.iterdir() if d.is_dir()]

    if not subset_dirs:
        print(f"Warning: No subset directories found in {root_dir}")
        return

    total_masks_processed = 0
    total_binary_masks_extracted = 0

    print(f"Found {len(subset_dirs)} subset directories")
    print(f"Label explanations: {label_explanations}")
    print(f"Number of labels: {len(label_explanations)}\n")

    for subset_dir in tqdm(subset_dirs, desc='Processing subsets'):
        masks_processed, binary_masks_extracted = process_subset_dir(subset_dir, label_explanations)
        total_masks_processed += masks_processed
        total_binary_masks_extracted += binary_masks_extracted

    print(f"\nProcessing completed!")
    print(f"Total mask files processed: {total_masks_processed}")
    print(f"Total binary masks extracted: {total_binary_masks_extracted}")


def main():
    """
    Main function to orchestrate the binary mask extraction process.
    """
    args = parse_args()

    print(f"Processing data from: {args.root_dir}")
    print(f"Label explanations: {args.label_explanation}")

    process_root_dir(args.root_dir, args.label_explanation)

    print("Binary mask extraction completed successfully!")


if __name__ == '__main__':
    main()
