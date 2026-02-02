#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Remove {pid}.nii.gz files from sample directories under grouped subdirectories.

This script recursively traverses the grouped directory structure and removes
{pid}.nii.gz files from sample directories (Breast_MRI_*).

Usage:
    python foo.py
"""

import os
from pathlib import Path
from tqdm import tqdm


def find_sample_dirs(root_dir: Path) -> list:
    """
    Recursively find all sample directories under root_dir.
    
    Args:
        root_dir: Root directory to start searching from
        
    Returns:
        List of Path objects pointing to sample directories
    """
    sample_dirs = []
    
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name.startswith('Breast_MRI_'):
                sample_dirs.append(Path(root) / dir_name)
    
    return sample_dirs


def remove_pid_nii_gz(sample_dir: Path) -> bool:
    """
    Remove {pid}.nii.gz file from sample directory.
    
    Args:
        sample_dir: Path to sample directory
        
    Returns:
        True if file was removed, False otherwise
    """
    pid = sample_dir.name
    target_file = sample_dir / f"{pid}_mask_mass_re.nii.gz"
    
    if target_file.exists():
        try:
            target_file.unlink()
            return True
        except Exception as e:
            print(f"Error removing {target_file}: {e}")
            return False
    
    return False


def main() -> None:
    """
    Main function to orchestrate the file removal process.
    """
    root_dir = Path("G:\Datasets\Duke-Breast-FGT-Segmentation-2025.4.10\grouped")
    
    if not root_dir.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        return
    
    print(f"Searching for sample directories under: {root_dir}")
    sample_dirs = find_sample_dirs(root_dir)
    
    print(f"Found {len(sample_dirs)} sample directories")
    
    removed_count = 0
    
    with tqdm(total=len(sample_dirs), desc="Processing sample directories") as pbar:
        for sample_dir in sample_dirs:
            pbar.set_description(f"Processing {sample_dir.name}")
            
            if remove_pid_nii_gz(sample_dir):
                removed_count += 1
            
            pbar.update(1)
    
    print(f"\nProcessing completed!")
    print(f"Total sample directories processed: {len(sample_dirs)}")
    print(f"Total {removed_count} files removed: {removed_count}")


if __name__ == '__main__':
    main()
