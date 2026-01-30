# -*- coding: utf-8 -*-
"""
参照05_gen_dataset_manifest.py生成数据集划分脚本。使用argparse。接收-m/--manifest_file指定的Excel清单文件路径，
-s指定主要清单的工作表名称，默认为Manifest，-spn/--split_name指定划分方案名称，默认为split01_standard，
添加-i/--inclusion参数用于指定导出哪些子数据集，默认值为[VerSe19, VerSe20]。检查-m表格的in_verse19和in_verse20列，
其值为1表示该样本在对应的VerSe19/VerSe20子数据集中，只保留-i指定的子数据集样本。
在清单中增加名称为split_name的新工作表，拷贝主要工作表内容，并在第一列插入名称为split_name的新列，
其中记录值为train/val/test，读取主表的subset列，将值为train的样本记录为train，值为val的样本记录为val，
值为test的样本记录为test，而后保存工作簿。
"""

"""
Dataset Split Generation Script for VerSe Dataset

This script creates a new worksheet in the manifest Excel file with split information (train/val/test)
based on the subset column value and filters samples by specified VerSe sub-datasets.

Parameters:
    -m, --manifest_file: Path to the Excel manifest file
    -s, --sheet_name: Name of the main manifest worksheet (default: Manifest)
    -spn, --split_name: Name of the split scheme (default: split01_standard)
    -i, --inclusion: List of VerSe sub-datasets to include (default: ['VerSe19', 'VerSe20'])

Usage Examples:
    python 06_make_split.py -m /path/to/dataset_manifest.xlsx
    python 06_make_split.py --manifest_file /path/to/dataset_manifest.xlsx -s Manifest -spn split01_standard
    python 06_make_split.py -m /path/to/dataset_manifest.xlsx --split_name split01_standard -i VerSe19 VerSe20
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing manifest_file, sheet_name, split_name, and inclusion
    """
    parser = argparse.ArgumentParser(
        description='Create split worksheet in manifest Excel file with sub-dataset filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m /path/to/dataset_manifest.xlsx
  %(prog)s --manifest_file /path/to/dataset_manifest.xlsx -s Manifest -spn split01_standard
  %(prog)s -m /path/to/dataset_manifest.xlsx --split_name split01_standard -i VerSe19 VerSe20
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

    parser.add_argument(
        '-i', '--inclusion',
        type=str,
        nargs='+',
        default=['VerSe19', 'VerSe20'],
        help='List of VerSe sub-datasets to include (default: VerSe19 VerSe20)'
    )

    return parser.parse_args()


def determine_split(subset_value):
    """
    Determine split (train/val/test) based on the subset column value.
    
    Args:
        subset_value (str): Subset value from the manifest
        
    Returns:
        str: 'train', 'val', or 'test' based on the subset value
    """
    if pd.isna(subset_value) or subset_value == '':
        return ''

    subset_lower = str(subset_value).lower()

    if subset_lower == 'train':
        return 'train'
    elif subset_lower == 'val':
        return 'val'
    elif subset_lower == 'test':
        return 'test'
    else:
        return ''


def filter_by_inclusion(df, inclusion):
    """
    Filter samples by inclusion in specified VerSe sub-datasets.
    
    Args:
        df (pd.DataFrame): DataFrame containing manifest data
        inclusion (list): List of VerSe sub-datasets to include (e.g., ['VerSe19', 'VerSe20'])
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only samples from specified sub-datasets
    """
    filtered_df = df.copy()
    
    # Create mask for samples to include
    include_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
    
    # Check each inclusion parameter
    for dataset in inclusion:
        if dataset.lower() == 'verse19' and 'in_verse19' in filtered_df.columns:
            include_mask = include_mask | (filtered_df['in_verse19'] == 1)
        elif dataset.lower() == 'verse20' and 'in_verse20' in filtered_df.columns:
            include_mask = include_mask | (filtered_df['in_verse20'] == 1)
    
    return filtered_df[include_mask]


def create_split_worksheet(manifest_file, sheet_name, split_name, inclusion):
    """
    Create a new worksheet with split information in the manifest Excel file,
    filtered by specified VerSe sub-datasets.
    
    Args:
        manifest_file (str or Path): Path to the Excel manifest file
        sheet_name (str): Name of the main manifest worksheet to copy
        split_name (str): Name of the new split worksheet
        inclusion (list): List of VerSe sub-datasets to include
    """
    manifest_path = Path(manifest_file)

    if not manifest_path.exists():
        print(f"Error: Manifest file does not exist: {manifest_file}")
        return

    print(f"Reading manifest file: {manifest_file}")
    print(f"Main worksheet: {sheet_name}")
    print(f"Creating split worksheet: {split_name}")
    print(f"Including sub-datasets: {', '.join(inclusion)}")

    try:
        df = pd.read_excel(
            manifest_path, 
            sheet_name=sheet_name, 
            dtype={'ID': str, 'seq': str, 'birth_date': str, 'acquisition_date': str}
            )

        # Check required columns
        if 'subset' not in df.columns:
            print(f"Error: 'subset' column not found in worksheet '{sheet_name}'")
            return
            
        required_columns = []
        if 'VerSe19' in inclusion:
            required_columns.append('in_verse19')
        if 'VerSe20' in inclusion:
            required_columns.append('in_verse20')
        
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: '{col}' column not found in worksheet '{sheet_name}'")
                return

        # Filter by inclusion
        filtered_df = filter_by_inclusion(df, inclusion)
        
        if filtered_df.empty:
            print(f"Error: No samples found matching the inclusion criteria: {inclusion}")
            return
            
        print(f"Filtered to {len(filtered_df)} samples from {len(df)} total samples")

        # Create copy with split information
        df_copy = filtered_df.copy()
        df_copy.insert(0, split_name, df_copy['subset'].apply(determine_split))

        train_count = (df_copy[split_name] == 'train').sum()
        val_count = (df_copy[split_name] == 'val').sum()
        test_count = (df_copy[split_name] == 'test').sum()
        unknown_count = (df_copy[split_name] == '').sum()

        print(f"\nSplit statistics:")
        print(f"  Train samples: {train_count}")
        print(f"  Val samples: {val_count}")
        print(f"  Test samples: {test_count}")
        print(f"  Unknown samples: {unknown_count}")

        if unknown_count > 0:
            unknown_samples = df_copy[df_copy[split_name] == '']
            print(f"\nWarning: {unknown_count} samples could not be classified:")
            for idx, row in unknown_samples.head(10).iterrows():
                print(f"  - ID: {row.get('ID', 'N/A')}, subset: {row.get('subset', 'N/A')}")
            if unknown_count > 10:
                print(f"  ... and {unknown_count - 10} more")

        with pd.ExcelWriter(manifest_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df_copy.to_excel(writer, index=False, sheet_name=split_name)

        print(f"\nSplit worksheet '{split_name}' created successfully!")
        print(f"Total samples in split: {len(df_copy)}")

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
    print(f"Inclusion: {args.inclusion}")

    create_split_worksheet(args.manifest_file, args.sheet_name, args.split_name, args.inclusion)

    print("Split worksheet generation completed successfully!")


if __name__ == '__main__':
    main()
