# -*- coding: utf-8 -*-
"""
参照05_make_split01_standard.py生成数据集划分脚本。使用argparse。接收-m/--manifest_file指定的Excel清单文件路径，
-s指定主要清单的工作表名称，默认为Manifest，-spn/--split_name指定划分方案名称，默认为split02_t8v2s，
-val_ratio指定验证集比例，默认0.2。在清单中增加名称为split_name的新工作表，拷贝主要工作表内容，
并在第一列插入名称为split_name的新列，其中记录值为train、val或test，读取主表的volume列，
将所有首级目录为train的样本按比例划分为train和val，所有首级目录为test的样本记录为test，而后保存工作簿。
"""

"""
Dataset Split Generation Script for ACDC Dataset (Train/Val/Test Split)

This script creates a new worksheet in the manifest Excel file with split information (train/val/test)
based on the first-level directory of the volume file path and a validation ratio.

Parameters:
    -m, --manifest_file: Path to the Excel manifest file
    -s, --sheet_name: Name of the main manifest worksheet (default: Manifest)
    -spn, --split_name: Name of the split scheme (default: split02_t8v2s)
    -v, --val_ratio: Validation set ratio (default: 0.2)
    -r, --random_seed: Random seed for reproducible splitting (default: 0)
    -b, --bind_col: Column name for group binding (default: patient)

Usage Examples:
    python 05_make_split02_train_val_fixed_test.py -m /path/to/dataset_manifest.xlsx --sheet_name Manifest
    python 05_make_split02_train_val_fixed_test.py --manifest_file /path/to/dataset_manifest.xlsx -s Manifest -spn split02_t8v2s
    python 05_make_split02_train_val_fixed_test.py -m /path/to/dataset_manifest.xlsx --sheet_name Manifest --val_ratio 0.2 --random_seed 0 --bind_col patient
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing manifest_file, sheet_name, split_name, and val_ratio
    """
    parser = argparse.ArgumentParser(
        description='Create split worksheet with train/val/test in manifest Excel file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m /path/to/dataset_manifest.xlsx
  %(prog)s --manifest_file /path/to/dataset_manifest.xlsx -s Manifest -spn split02_t8v2s
  %(prog)s -m /path/to/dataset_manifest.xlsx --sheet_name Manifest --val_ratio 0.2
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
        default='split02_t8v2s',
        help='Name of the split scheme (default: split02_t8v2s)'
    )

    parser.add_argument(
        '-v', '--val_ratio',
        type=float,
        default=0.2,
        help='Validation set ratio (default: 0.2)'
    )

    parser.add_argument(
        '-r', '--random_seed',
        type=int,
        default=0,
        help='Random seed for reproducible splitting (default: 0)'
    )

    parser.add_argument(
        '-b', '--bind_col',
        type=str,
        default='patient',
        help='Column name for group binding (default: patient)'
    )

    return parser.parse_args()


def create_split_worksheet(manifest_file, sheet_name, split_name, val_ratio, random_seed, bind_col):
    """
    Create a new worksheet with split information (train/val/test) in the manifest Excel file.
    
    Args:
        manifest_file (str or Path): Path to the Excel manifest file
        sheet_name (str): Name of the main manifest worksheet to copy
        split_name (str): Name of the new split worksheet
        val_ratio (float): Ratio of training samples to use as validation
        random_seed (int): Random seed for reproducible splitting
        bind_col (str): Column name for group binding
    """
    manifest_path = Path(manifest_file)

    if not manifest_path.exists():
        print(f"Error: Manifest file does not exist: {manifest_file}")
        return

    print(f"Reading manifest file: {manifest_file}")
    print(f"Main worksheet: {sheet_name}")
    print(f"Creating split worksheet: {split_name}")
    print(f"Validation ratio: {val_ratio}")
    print(f"Random seed: {random_seed}")
    print(f"Bind column: {bind_col}")

    try:
        df = pd.read_excel(manifest_path, sheet_name=sheet_name, dtype={'ID': str, 'patient': str, 'frame': str})

        if 'volume' not in df.columns:
            print(f"Error: 'volume' column not found in worksheet '{sheet_name}'")
            return

        if bind_col not in df.columns:
            print(f"Error: Bind column '{bind_col}' not found in worksheet '{sheet_name}'")
            return

        df_copy = df.copy()

        train_mask = df_copy['volume'].apply(lambda x: x.split('/')[0].lower() if x else '') == 'train'
        test_mask = df_copy['volume'].apply(lambda x: x.split('/')[0].lower() if x else '') == 'test'

        train_df = df_copy[train_mask].copy()
        test_df = df_copy[test_mask].copy()

        print(f"\nChecking group consistency within each {bind_col}...")

        train_bind_groups = train_df.groupby(bind_col)
        inconsistent_binds = []

        for bind_name, bind_df in train_bind_groups:
            unique_groups = bind_df['group'].unique()
            if len(unique_groups) > 1:
                inconsistent_binds.append((bind_name, unique_groups))
                print(f"  Error: {bind_col} '{bind_name}' has inconsistent group values: {unique_groups}")

        if inconsistent_binds:
            print(f"\nError: Found {len(inconsistent_binds)} {bind_col}(s) with inconsistent group values.")
            print("Please ensure each sample within the same {bind_col} has the same group value.")
            return

        print(f"  All {bind_col}s have consistent group values.")

        print(f"\nBuilding sample groups DataFrame...")

        train_group_data = []
        for bind_name, bind_df in train_bind_groups:
            group_value = bind_df['group'].iloc[0]
            indexes = bind_df.index.tolist()
            train_group_data.append({
                bind_col: bind_name,
                'indexes': indexes,
                'group': group_value
            })

        train_groups_df = pd.DataFrame(train_group_data)

        print(f"  Total sample groups: {len(train_groups_df)}")
        print(f"  Group distribution:")
        group_dist = train_groups_df['group'].value_counts()
        for group, count in group_dist.items():
            print(f"    {group}: {count} groups")

        np.random.seed(random_seed)

        train_groups_df[split_name] = ''

        group_categories = train_groups_df.groupby('group')

        for group_name, group_df in group_categories:
            group_indices = group_df.index.tolist()
            np.random.shuffle(group_indices)

            val_count = int(len(group_indices) * val_ratio)
            val_indices = group_indices[:val_count]
            train_indices = group_indices[val_count:]

            train_groups_df.loc[train_indices, split_name] = 'train'
            train_groups_df.loc[val_indices, split_name] = 'val'

        test_df[split_name] = 'test'
        test_indices = test_df.index.tolist()
        df_copy.loc[test_indices, split_name] = 'test'

        print(f"\nAssigning samples to splits based on sample groups...")

        for idx, row in train_groups_df.iterrows():
            split_value = row[split_name]
            sample_indexes = row['indexes']
            df_copy.loc[sample_indexes, split_name] = split_value

        df_copy = df_copy[[split_name] + [col for col in df_copy.columns if col != split_name]]

        train_count = (df_copy[split_name] == 'train').sum()
        val_count = (df_copy[split_name] == 'val').sum()
        test_count = (df_copy[split_name] == 'test').sum()
        unknown_count = (df_copy[split_name] == '').sum()

        print(f"\nSplit statistics:")
        print(f"  Train samples: {train_count} ({train_count/len(df_copy)*100:.1f}%)")
        print(f"  Validation samples: {val_count} ({val_count/len(df_copy)*100:.1f}%)")
        print(f"  Test samples: {test_count} ({test_count/len(df_copy)*100:.1f}%)")
        print(f"  Unknown samples: {unknown_count}")

        print(f"\nSample group distribution by split:")
        group_split_stats = train_groups_df.groupby('group')[split_name].value_counts().unstack(fill_value=0)
        print(group_split_stats.to_string())

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
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Random seed: {args.random_seed}")
    print(f"Bind column: {args.bind_col}")

    create_split_worksheet(args.manifest_file, args.sheet_name, args.split_name, args.val_ratio, args.random_seed, args.bind_col)

    print("Split worksheet generation completed successfully!")


if __name__ == '__main__':
    main()
