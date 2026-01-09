# -*- coding: utf-8 -*-
"""
Split prostate/uterus label based on patient gender. Use argparse to receive -r/--root_dir to specify root directory. Root directory contains multiple subset subdirectories, each subset subdirectory contains secondary subdirectories named amos_xxxx. From each secondary subdirectory, match mask files named amos_xxxx_mask.nii.gz, use LoadImage to read their content, process the mask regions with label value 15 (representing prostate/uterus). Read the Excel file specified by -m/--archive_manifest, match records based on seq column and xxxx number, then get the sex column attribute value to determine gender (M/F). If F (female), replace the label value of this mask region with 16 (specifically representing uterus); if M (male), keep the original value 15 (representing prostate) without modification. Use SaveImage to output the modified mask to overwrite the original file.
"""

"""
Split Prostate/uterus Label Script for AMOS22 Dataset

This script processes mask files to split label 15 (prostate/uterus) into separate labels based on patient gender.
Female patients (F) will have label 15 changed to 16 (uterus), while male patients (M) keep label 15 (prostate).

Parameters:
    -r, --root_dir: Root directory containing subset subdirectories (e.g., train, val, test)

Usage Examples:
    python 03_split_prostate_uterus_label.py -r /path/to/grouped
    python 03_split_prostate_uterus_label.py --root_dir /path/to/grouped
"""

import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from monai.transforms import LoadImage, SaveImage


def parse_args():
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing root_dir
    """
    parser = argparse.ArgumentParser(
        description='Split prostate/uterus label based on patient gender',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/grouped
  %(prog)s --root_dir /path/to/grouped
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory containing subset subdirectories (e.g., train, val, test)'
    )

    return parser.parse_args()


def process_mask_file(mask_file):
    """
    Process a single mask file to split prostate/uterus label based on gender.
    
    Args:
        mask_file (Path): Path to the mask file
        
    Returns:
        tuple: (success: bool, modified: bool, message: str)
    """
    loader = LoadImage(image_only=False, dtype=None)
    saver = SaveImage(output_dir=str(mask_file.parent), output_postfix='', output_dtype=np.uint8)

    try:
        mask_data, mask_meta = loader(str(mask_file))
    except Exception as e:
        return False, False, f"Error loading mask file: {str(e)}"

    sample_dir = mask_file.parent
    yaml_files = list(sample_dir.glob('*_info.yaml'))

    if not yaml_files:
        return False, False, f"YAML metadata file not found in {sample_dir}"

    yaml_file = yaml_files[0]

    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
    except Exception as e:
        return False, False, f"Error loading YAML file: {str(e)}"

    if 'sex' not in metadata:
        return False, False, f"'sex' field not found in YAML metadata"

    sex = str(metadata['sex']).upper()

    if sex == 'F':
        if np.any(mask_data == 15):
            mask_data = np.where(mask_data == 15, 16, mask_data)
            try:
                saver(mask_data, meta_data=mask_meta, filename=str(mask_file).replace('.nii.gz', ''))
                return True, True, f"Modified label 15 to 16 (uterus) for female patient {sample_dir.name}"
            except Exception as e:
                return False, False, f"Error saving modified mask: {str(e)}"
        else:
            return True, False, f"No label 15 found in mask for {sample_dir.name}"
    elif sex == 'M':
        return True, False, f"Kept label 15 (prostate) for male patient {sample_dir.name}"
    else:
        return False, False, f"Unknown sex value '{sex}' for {sample_dir.name}"


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


def process_sample_dir(sample_dir):
    """
    Process a single sample directory.
    
    Args:
        sample_dir (Path): Path to the sample directory
        
    Returns:
        tuple: (success: bool, modified: bool, message: str)
    """
    mask_files = list(sample_dir.glob('*_mask.nii.gz'))

    if not mask_files:
        return False, False, f"Mask file not found in {sample_dir}"

    mask_file = mask_files[0]

    return process_mask_file(mask_file)


def process_root_dir(root_dir):
    """
    Process all sample directories in the root directory recursively.
    
    Args:
        root_dir (str or Path): Root directory containing sample directories at any depth
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return

    sample_dirs = find_sample_dirs(root_path)

    if not sample_dirs:
        print(f"Warning: No sample directories (amos_xxxx*) found in {root_dir}")
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
                print(f"  {message}")
        else:
            total_errors += 1
            print(f"  Error: {message}")

    print(f"\nProcessing completed!")
    print(f"Total mask files processed: {total_processed}")
    print(f"Total mask files modified (F->16): {total_modified}")
    print(f"Total errors encountered: {total_errors}")


def main():
    """
    Main function to orchestrate the prostate/uterus label splitting process.
    """
    args = parse_args()

    print(f"Processing data from: {args.root_dir}")

    process_root_dir(args.root_dir)

    print("Prostate/uterus label splitting completed successfully!")


if __name__ == '__main__':
    main()
