 # -*- coding: utf-8 -*-
"""
Binary Mask Extraction Script for VerSe Dataset

This script extracts binary masks from multi-label mask files, separating each label into individual
binary masks for easier processing and analysis.

Parameters:
    -r, --root_dir: Root directory containing sample directories at any depth
    -e, --label_explanation: Path to label_map.yaml file containing label definitions

Usage Examples:
    python 04_extract_binary_masks.py -r /path/to/Verse -e /path/to/label_map.yaml
    python 04_extract_binary_masks.py --root_dir /path/to/Verse --label_explanation /path/to/label_map.yaml
"""

import argparse
import re
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any
from tqdm import tqdm
import numpy as np
from monai.transforms import LoadImage, SaveImage


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir and label_explanation
    """
    parser = argparse.ArgumentParser(
        description='Extract binary masks from multi-label mask files for VerSe dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/Verse -e /path/to/label_map.yaml
  %(prog)s --root_dir /path/to/Verse --label_explanation /path/to/label_map.yaml
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


def load_label_definitions(label_map_path: Union[str, Path]) -> Dict[int, str]:
    """
    Load label definitions from label_map.yaml file.
    
    Args:
        label_map_path (Union[str, Path]): Path to label_map.yaml file
        
    Returns:
        Dict[int, str]: Dictionary mapping label indices (int) to label names (str)
    """
    label_map_file = Path(label_map_path)

    if not label_map_file.exists():
        raise FileNotFoundError(f"Label map YAML file not found: {label_map_path}")

    with open(label_map_file, 'r', encoding='utf-8') as f:
        label_map = yaml.safe_load(f)

    if 'short_form_index_map' not in label_map:
        raise ValueError(f"'short_form_index_map' key not found in label_map.yaml: {label_map_path}")

    short_form_index_map = label_map['short_form_index_map']

    label_definitions = {}
    for label_index, label_name in short_form_index_map.items():
        if not isinstance(label_index, int):
            print(f"Warning: Skipping invalid label key '{label_index}' (not an integer)")
            continue
        # Replace spaces in label_name with underscores
        label_definitions[label_index] = re.sub(r'[\s/]', '_', label_name)

    if not label_definitions:
        raise ValueError(f"No valid label definitions found in label_map.yaml: {label_map_path}")

    return label_definitions


def extract_binary_masks(mask_data: np.ndarray, label_definitions: Dict[int, str]) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Extract binary masks from multi-label mask data.
    
    Args:
        mask_data (np.ndarray): Multi-label mask data array
        label_definitions (Dict[int, str]): Dictionary mapping label indices to label names
        
    Returns:
        Tuple[List[np.ndarray], List[int], List[str]]: (list of binary mask arrays, list of label indices, list of label names)
    """
    binary_masks = []
    label_indices = []
    label_names = []

    for label_index in sorted(label_definitions.keys()):
        label_name = label_definitions[label_index]
        binary_mask = (mask_data == label_index).astype(np.uint8)
        binary_masks.append(binary_mask)
        label_indices.append(label_index)
        label_names.append(label_name)

    return binary_masks, label_indices, label_names


def find_sample_dirs(root_dir: Path) -> List[Path]:
    """
    Find sample directories with specified structure: Root → VerSe* → train/val/test → Sample directory.
    
    Args:
        root_dir (Path): Root directory to search
        
    Returns:
        List[Path]: List of Path objects pointing to sample directories
    """
    root_path: Path = Path(root_dir)
    sample_dirs: List[Path] = []
    
    # Iterate through first-level VerSe* subdataset directories
    for sub_dataset_dir in root_path.glob('VerSe*'):
        if not sub_dataset_dir.is_dir():
            continue
        
        # Iterate through second-level train/val/test subset directories
        for subset_dir in sub_dataset_dir.glob('*'):
            if not subset_dir.is_dir():
                continue
            
            # Iterate through third-level sample directories
            for sample_dir in subset_dir.glob('*'):
                if sample_dir.is_dir():
                    # Check if the directory contains mask files
                    mask_files = list(sample_dir.glob('*_mask.nii.gz'))
                    if mask_files:
                        sample_dirs.append(sample_dir)
    
    return sorted(sample_dirs)


def process_sample_dir(sample_dir: Path, label_definitions: Dict[int, str]) -> Tuple[int, int, int]:
    """
    Process a single sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        label_definitions (Dict[int, str]): Dictionary mapping label indices to label names
        
    Returns:
        Tuple[int, int, int]: (masks_processed, binary_masks_extracted, error_count)
    """
    loader: LoadImage = LoadImage(image_only=False, dtype=None)

    masks_processed: int = 0
    binary_masks_extracted: int = 0
    error_count: int = 0

    # Find all <ID>_mask.nii.gz format mask files
    mask_files: List[Path] = list(sample_dir.glob('*_mask.nii.gz'))

    if not mask_files:
        return 0, 0, 0

    for mask_file in mask_files:
        try:
            mask_data: np.ndarray
            mask_meta: Dict[str, Any]
            mask_data, mask_meta = loader(str(mask_file))

            # Check actual label values in mask file
            unique_labels: np.ndarray = np.unique(mask_data)
            valid_labels: set = set(label_definitions.keys())
            invalid_labels: List[int] = [label for label in unique_labels if label not in valid_labels]

            if invalid_labels:
                print(f"Error: {mask_file.name} contains labels not in label_map.yaml: {invalid_labels}")
                error_count += 1

            # Extract binary masks
            binary_masks: List[np.ndarray]
            label_indices: List[int]
            label_names: List[str]
            binary_masks, label_indices, label_names = extract_binary_masks(mask_data, label_definitions)

            # Create saver
            saver: SaveImage = SaveImage(output_dir=str(sample_dir), output_postfix='', output_dtype=np.uint8)

            # Get <ID> part
            mask_filename = mask_file.name
            id_part = mask_filename[:-len('_mask.nii.gz')]

            # Save each binary mask
            for binary_mask, label_index, label_name in zip(binary_masks, label_indices, label_names):
                # Generate output filename: <ID>_mask_<label_index>_<label_name>.nii.gz
                output_filestem = sample_dir / f'{id_part}_mask_{label_index:02d}_{label_name}'

                saver(binary_mask, meta_data=mask_meta, filename=output_filestem)
                binary_masks_extracted += 1

            masks_processed += 1

        except Exception as e:
            print(f"Error processing {mask_file.name}: {str(e)}")
            error_count += 1

    return masks_processed, binary_masks_extracted, error_count


def process_root_dir(root_dir: Union[str, Path], label_definitions: Dict[int, str]) -> None:
    """
    Process all sample directories in the root directory recursively.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing sample directories at any depth
        label_definitions (Dict[int, str]): Dictionary mapping label indices to label names
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    sample_dirs = find_sample_dirs(root_path)

    if not sample_dirs:
        print(f"Warning: No sample directories containing mask files found in {root_dir}")
        return

    total_masks_processed: int = 0
    total_binary_masks_extracted: int = 0
    total_errors: int = 0

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


def main() -> None:
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