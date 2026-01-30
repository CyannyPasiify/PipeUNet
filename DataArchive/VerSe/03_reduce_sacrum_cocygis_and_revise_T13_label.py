# -*- coding: utf-8 -*-
"""
Label Processing Script for VerSe Dataset

This script processes mask files to:
1. Reduce sacrum (26) and coccyx (27) labels to background (0)
2. Revise T13 (28) label to (26)

Parameters:
    -r, --root_dir: Root directory containing sample directories at any depth

Usage Examples:
    python 03_reduce_sacrum_cocygis_and_revise_T13_label.py -r /path/to/Verse
    python 03_reduce_sacrum_cocygis_and_revise_T13_label.py --root_dir /path/to/Verse
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
import numpy as np
from monai.transforms import LoadImage, SaveImage


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir
    """
    parser = argparse.ArgumentParser(
        description='Reduce sacrum/coccyx labels and revise T13 label',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/Verse
  %(prog)s --root_dir /path/to/Verse
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing sample directories at any depth'
    )

    return parser.parse_args()


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


def process_mask_file(mask_file: Path) -> Tuple[bool, bool, str]:
    """
    Process a single mask file to reduce sacrum/coccyx and revise T13 label.
    
    Args:
        mask_file (Path): Path to the mask file
        
    Returns:
        Tuple[bool, bool, str]: (success: bool, modified: bool, message: str)
    """
    loader = LoadImage(image_only=False, dtype=None)
    saver = SaveImage(output_dir=str(mask_file.parent), output_postfix='', output_dtype=np.uint8)

    try:
        mask_data, mask_meta = loader(str(mask_file))
    except Exception as e:
        return False, False, f"Error loading mask file: {str(e)}"

    # Check if there are target labels
    has_26 = np.any(mask_data == 26)
    has_27 = np.any(mask_data == 27)
    has_28 = np.any(mask_data == 28)

    # Collect findings
    findings = []
    if has_26:
        findings.append("label 26 (sacrum)")
    if has_27:
        findings.append("label 27 (coccyx)")
    if has_28:
        findings.append("label 28 (T13)")

    modified = False
    if has_26 or has_27 or has_28:
        # Report findings
        sample_id = mask_file.stem.replace('_mask', '')
        finding_msg = f"Found {', '.join(findings)} in {sample_id}"
        print(f"  {finding_msg}")

        # Modify label values
        if has_26 or has_27:
            # Set 26 and 27 to 0
            mask_data[mask_data == 26] = 0
            mask_data[mask_data == 27] = 0
            modified = True
        
        if has_28:
            # Set 28 to 26
            mask_data[mask_data == 28] = 26
            modified = True

        if modified:
            try:
                # Resave the file
                saver(mask_data, meta_data=mask_meta, filename=str(mask_file).replace('.nii.gz', ''))
                return True, True, f"Modified {sample_id}: {' and '.join(findings)} processed"
            except Exception as e:
                return False, False, f"Error saving modified mask: {str(e)}"

    return True, False, f"No target labels found in {mask_file.stem.replace('_mask', '')}"


def process_sample_dir(sample_dir: Path) -> Tuple[bool, bool, str]:
    """
    Process a single sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        
    Returns:
        Tuple[bool, bool, str]: (success: bool, modified: bool, message: str)
    """
    # Find all <ID>_mask.nii.gz format mask files
    mask_files = list(sample_dir.glob('*_mask.nii.gz'))

    if not mask_files:
        return True, False, f"No mask files found in {sample_dir.name}"

    # Process each mask file
    total_success = 0
    total_modified = 0
    errors = []

    mask_files: List[Path] = list(sample_dir.glob('*_mask.nii.gz'))
    
    if not mask_files:
        return True, False, f"No mask files found in {sample_dir.name}"

    # Process each mask file
    total_success: int = 0
    total_modified: int = 0
    errors: List[str] = []

    for mask_file in mask_files:
        success, modified, message = process_mask_file(mask_file)
        if success:
            total_success += 1
            if modified:
                total_modified += 1
        else:
            errors.append(message)

    if errors:
        return False, False, f"{len(errors)} errors in {sample_dir.name}: {'; '.join(errors)}"
    elif total_modified > 0:
        return True, True, f"Processed {total_success} mask file(s) in {sample_dir.name}, modified {total_modified}"
    else:
        return True, False, f"Processed {total_success} mask file(s) in {sample_dir.name}, no modifications"



def process_root_dir(root_dir: Union[str, Path]) -> None:
    """
    Process all sample directories in the root directory recursively.
    
    Args:
        root_dir (Union[str, Path]): Root directory containing sample directories at any depth
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    sample_dirs = find_sample_dirs(root_path)

    if not sample_dirs:
        print(f"Warning: No sample directories containing mask files found in {root_dir}")
        return

    total_processed = 0
    total_modified = 0
    total_errors = 0

    print(f"Found {len(sample_dirs)} sample directories\n")

    for sample_dir in tqdm(sample_dirs, desc='Processing samples'):
        success, modified, message = process_sample_dir(sample_dir)

        if success:
            total_processed += 1
            if modified:
                total_modified += 1
        else:
            total_errors += 1
            print(f"  Error: {message}")

    print(f"\nProcessing completed!")
    print(f"Total sample directories processed: {total_processed}")
    print(f"Total sample directories with modifications: {total_modified}")
    print(f"Total errors encountered: {total_errors}")



def main() -> None:
    """
    Main function to orchestrate the label processing.
    """
    args = parse_args()

    print(f"Processing data from: {args.root_dir}")

    process_root_dir(args.root_dir)

    print("Label processing completed successfully!")


if __name__ == '__main__':
    main()