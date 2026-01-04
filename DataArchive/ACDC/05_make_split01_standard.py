# -*- coding: utf-8 -*-
"""
参照04_gen_dataset_manifest.py生成数据集划分脚本。使用argparse。接收-m/--manifest_file指定的Excel清单文件路径，
-s指定主要清单的工作表名称，默认为Manifest，-spn/--split_name指定划分方案名称，默认为split01_standard，
在清单中增加名称为split_name的新工作表，拷贝主要工作表内容，并在第一列插入名称为split_name的新列，
其中记录值为train或test，读取主表的volume列，将所有首级目录为train的样本记录为train，所有首级目录为test的样本记录为test，而后保存工作簿。
"""

"""
Dataset Split Generation Script for ACDC Dataset

This script creates a new worksheet in the manifest Excel file with split information (train/test)
based on the first-level directory of the volume file path.

Parameters:
    -m, --manifest_file: Path to the Excel manifest file
    -s, --sheet_name: Name of the main manifest worksheet (default: Manifest)
    -spn, --split_name: Name of the split scheme (default: split01_standard)

Usage Examples:
    python 05_make_split01_standard.py -m /path/to/dataset_manifest.xlsx
    python 05_make_split01_standard.py --manifest_file /path/to/dataset_manifest.xlsx -s Manifest -spn split01_standard
    python 05_make_split01_standard.py -m /path/to/dataset_manifest.xlsx --split_name split01_standard
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing manifest_file, sheet_name, and split_name
    """
    parser = argparse.ArgumentParser(
        description='Create split worksheet in manifest Excel file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m /path/to/dataset_manifest.xlsx
  %(prog)s --manifest_file /path/to/dataset_manifest.xlsx -s Manifest -spn split01_standard
  %(prog)s -m /path/to/dataset_manifest.xlsx --sheet_name Manifest --split_name split01_standard
        """
    )

    parser.add_argument(
        '-m', '--manifest_file',
        type=str,
        required=True,
        help='Path to the Excel manifest file'
    )

    parser.add_argument(
        '-s', '--sheet_name',
        type=str,
        default='Manifest',
        help='Name of the main manifest worksheet (default: Manifest)'
    )

    parser.add_argument(
        '-spn', '--split_name',
        type=str,
        default='split01_standard',
        help='Name of the split scheme (default: split01_standard)'
    )

    return parser.parse_args()


def determine_split(volume_path):
    """
    Determine split (train/test) based on the first-level directory of the volume path.
    
    Args:
        volume_path (str): Volume file path relative to dataset root
        
    Returns:
        str: 'train' or 'test' based on the first-level directory
    """
    if not volume_path:
        return ''

    first_dir = volume_path.split('/')[0].lower()

    if first_dir == 'train':
        return 'train'
    elif first_dir == 'test':
        return 'test'
    else:
        return ''


def create_split_worksheet(manifest_file, sheet_name, split_name):
    """
    Create a new worksheet with split information in the manifest Excel file.
    
    Args:
        manifest_file (str or Path): Path to the Excel manifest file
        sheet_name (str): Name of the main manifest worksheet to copy
        split_name (str): Name of the new split worksheet
    """
    manifest_path = Path(manifest_file)

    if not manifest_path.exists():
        print(f"Error: Manifest file does not exist: {manifest_file}")
        return

    print(f"Reading manifest file: {manifest_file}")
    print(f"Main worksheet: {sheet_name}")
    print(f"Creating split worksheet: {split_name}")

    try:
        df = pd.read_excel(manifest_path, sheet_name=sheet_name, dtype={'ID': str, 'patient': str, 'frame': str})

        if 'volume' not in df.columns:
            print(f"Error: 'volume' column not found in worksheet '{sheet_name}'")
            return

        df_copy = df.copy()

        df_copy.insert(0, split_name, df_copy['volume'].apply(determine_split))

        train_count = (df_copy[split_name] == 'train').sum()
        test_count = (df_copy[split_name] == 'test').sum()
        unknown_count = (df_copy[split_name] == '').sum()

        print(f"\nSplit statistics:")
        print(f"  Train samples: {train_count}")
        print(f"  Test samples: {test_count}")
        print(f"  Unknown samples: {unknown_count}")

        if unknown_count > 0:
            unknown_samples = df_copy[df_copy[split_name] == '']
            print(f"\nWarning: {unknown_count} samples could not be classified:")
            for idx, row in unknown_samples.head(10).iterrows():
                print(f"  - {row.get('volume', 'N/A')}")
            if unknown_count > 10:
                print(f"  ... and {unknown_count - 10} more")

        with pd.ExcelWriter(manifest_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df_copy.to_excel(writer, index=False, sheet_name=split_name)

        print(f"\nSplit worksheet '{split_name}' created successfully!")
        print(f"Total samples: {len(df_copy)}")

    except Exception as e:
        print(f"Error processing manifest file: {str(e)}")
        return


def main():
    """
    Main function to orchestrate the split worksheet creation process.
    """
    args = parse_args()

    print(f"Processing manifest file: {args.manifest_file}")
    print(f"Main worksheet name: {args.sheet_name}")
    print(f"Split name: {args.split_name}")

    create_split_worksheet(args.manifest_file, args.sheet_name, args.split_name)

    print("Split worksheet generation completed successfully!")


if __name__ == '__main__':
    main()
