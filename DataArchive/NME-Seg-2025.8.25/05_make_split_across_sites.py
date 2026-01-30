# -*- coding: utf-8 -*-
"""
接收-m/--manifest_file指定的Excel清单文件路径，
-s指定主要清单的工作表名称，默认为Manifest，-spn/--split_name指定划分方案名称，默认为split01_Tongji。移除-st参数。
由-ts/--test_ratio指定测试集占比，默认0.3，-v/--val_ratio指定验证集在训练验证集中的占比，默认0.2。-rs/--random_seed指定随机数种子（默认0）。
Tongji site的划分遵循subset定义，其它中心则按照--test_ratio和--val_ratio的比例进行划分。
在清单中增加名称为split_name的新工作表，拷贝主要工作表内容，并在第一列插入名称为split_name的新列，其中记录划分值train/val/test，而后保存工作簿。
  对全部变量和函数参数添加类型注解。
  除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Dataset Cross-Site Split Generation Script for NME-Seg-2025.8.25 Dataset

This script creates a new worksheet in the manifest Excel file with split information
(train/val/test). For Tongji site, split follows the subset definition. For other sites,
stratified split is performed based on Pathological Complete Response (pCR) attribute.

Parameters:
    -m, --manifest_file: Path to the Excel manifest file
    -s, --sheet_name: Name of the main manifest worksheet (default: Manifest)
    -spn, --split_name: Name of the split scheme (default: split07_Mix)
    -ts, --test_ratio: Test set ratio (default: 0.3)
    -v, --val_ratio: Validation set ratio within train+val set (default: 0.2)
    -rs, --random_seed: Random seed for reproducibility (default: 0)

Usage Examples:
    python 05_make_split_across_sites.py -m /path/to/dataset_manifest.xlsx
    python 05_make_split_across_sites.py --manifest_file /path/to/dataset_manifest.xlsx --test_ratio 0.3 --val_ratio 0.2
    python 05_make_split_across_sites.py -m /path/to/dataset_manifest.xlsx --split_name split02_custom --random_seed 42
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set, Union


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all command line parameters
    """
    parser = argparse.ArgumentParser(
        description='Create cross-site split worksheet in manifest Excel file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m /path/to/dataset_manifest.xlsx
  %(prog)s --manifest_file /path/to/dataset_manifest.xlsx --test_ratio 0.3 --val_ratio 0.2
  %(prog)s -m /path/to/dataset_manifest.xlsx --split_name split02_custom --random_seed 42
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
        default='split07_Mix',
        help='Name of the split scheme (default: split07_Mix)'
    )

    parser.add_argument(
        '-ts', '--test_ratio',
        type=float,
        default=0.3,
        help='Test set ratio (default: 0.3)'
    )

    parser.add_argument(
        '-v', '--val_ratio',
        type=float,
        default=0.2,
        help='Validation set ratio within train+val set (default: 0.2)'
    )

    parser.add_argument(
        '-rs', '--random_seed',
        type=int,
        default=0,
        help='Random seed for reproducibility (default: 0)'
    )

    return parser.parse_args()


def perform_split(df: pd.DataFrame, test_ratio: float, val_ratio: float, random_seed: int) -> Dict[str, str]:
    """
    Perform split based on ratios using numpy.
    
    Args:
        df (pd.DataFrame): DataFrame containing non-Tongji samples
        test_ratio (float): Test set ratio
        val_ratio (float): Validation set ratio within train+val set
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Dict[str, str]: Dictionary mapping IDs to split values (train/val/test)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create a dictionary to store split results
    split_dict: Dict[str, str] = {}
    
    # Get all IDs and shuffle them
    ids = df['ID'].tolist()
    np.random.shuffle(ids)
    
    total = len(ids)
    
    # Step 1: Split into test and train_val sets
    if test_ratio == 1.0:
        # All samples go to test set
        for id in ids:
            split_dict[id] = 'test'
        return split_dict
    elif test_ratio == 0.0:
        # All samples go to train_val set
        train_val_ids = ids.copy()
    else:
        # Calculate split sizes
        test_size = int(total * test_ratio)
        test_ids = ids[:test_size]
        train_val_ids = ids[test_size:]
        
        # Assign test set
        for id in test_ids:
            split_dict[id] = 'test'
    
    # Step 2: Split train_val into train and val sets
    if test_ratio != 1.0:
        train_val_size = len(train_val_ids)
        
        if val_ratio == 1.0:
            # All train_val samples go to val set
            for id in train_val_ids:
                split_dict[id] = 'val'
        elif val_ratio == 0.0:
            # All train_val samples go to train set
            for id in train_val_ids:
                split_dict[id] = 'train'
        else:
            # Calculate split sizes
            val_size = int(train_val_size * val_ratio)
            val_ids = train_val_ids[:val_size]
            train_ids = train_val_ids[val_size:]
            
            # Assign train and val sets
            for id in train_ids:
                split_dict[id] = 'train'
            
            for id in val_ids:
                split_dict[id] = 'val'
    
    return split_dict


def create_split_worksheet(
    manifest_file: Union[str, Path],
    sheet_name: str,
    split_name: str,
    test_ratio: float,
    val_ratio: float,
    random_seed: int
) -> None:
    """
    Create a new worksheet with split information in the manifest Excel file.
    
    Args:
        manifest_file (Union[str, Path]): Path to the Excel manifest file
        sheet_name (str): Name of the main manifest worksheet to copy
        split_name (str): Name of the new split worksheet
        test_ratio (float): Test set ratio
        val_ratio (float): Validation set ratio within train+val set
        random_seed (int): Random seed for reproducibility
    """
    manifest_path = Path(manifest_file)

    if not manifest_path.exists():
        print(f"Error: Manifest file does not exist: {manifest_file}")
        return

    print(f"Reading manifest file: {manifest_file}")
    print(f"Main worksheet: {sheet_name}")
    print(f"Creating split worksheet: {split_name}")
    print(f"Test ratio: {test_ratio}, Validation ratio: {val_ratio}")
    print(f"Random seed: {random_seed}")

    try:
        # Read the manifest file
        df = pd.read_excel(
            manifest_path, 
            sheet_name=sheet_name,
            dtype={'ID': str, 'site': str, 'pid': str}
        )

        # Check required columns
        required_columns = ['ID', 'site', 'subset']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: '{col}' column not found in worksheet '{sheet_name}'")
                return

        print(f"Total samples: {len(df)}")

        # Create a copy of the dataframe to add split information
        df_copy = df.copy()
        
        # Process Tongji site: use subset column
        tongji_mask = df_copy['site'] == 'Tongji'
        df_copy.loc[tongji_mask, split_name] = df_copy.loc[tongji_mask, 'subset']
        
        # Process other sites: perform split based on IDs
        df_other = df[df['site'] != 'Tongji'].copy()
        if not df_other.empty:
            print(f"Number of non-Tongji samples: {len(df_other)}")

            # Perform split for non-Tongji sites
            split_dict = perform_split(df_other, test_ratio, val_ratio, random_seed)

            # Map split values to non-Tongji rows
            non_tongji_mask = df_copy['site'] != 'Tongji'
            df_copy.loc[non_tongji_mask, split_name] = df_copy.loc[non_tongji_mask, 'ID'].map(split_dict)
            
            # Fill any missing values with empty string
            df_copy.loc[non_tongji_mask, split_name] = df_copy.loc[non_tongji_mask, split_name].fillna('')

        # Insert split column as the first column
        df_copy.insert(0, split_name, df_copy.pop(split_name))

        # Calculate and print split statistics
        train_count = (df_copy[split_name] == 'train').sum()
        val_count = (df_copy[split_name] == 'val').sum()
        test_count = (df_copy[split_name] == 'test').sum()

        print(f"\nSplit statistics:")
        print(f"  Train samples: {train_count}")
        print(f"  Val samples: {val_count}")
        print(f"  Test samples: {test_count}")
        print(f"  Total: {len(df_copy)}")

        # Write the split dataframe to a new worksheet
        with pd.ExcelWriter(
            manifest_path, 
            mode='a', 
            engine='openpyxl', 
            if_sheet_exists='replace'
        ) as writer:
            df_copy.to_excel(writer, index=False, sheet_name=split_name)

        print(f"\nSplit worksheet '{split_name}' created successfully!")
        print(f"Excel file updated: {manifest_file}")

    except Exception as e:
        print(f"Error processing manifest file: {str(e)}")
        import traceback
        traceback.print_exc()
        return


def main() -> None:
    """
    Main function to orchestrate the split worksheet creation process.
    """
    args = parse_args()

    print(f"Processing manifest file: {args.manifest_file}")
    print(f"Main worksheet name: {args.sheet_name}")
    print(f"Split name: {args.split_name}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Random seed: {args.random_seed}")

    # Validate ratio inputs
    if not (0.0 <= args.test_ratio <= 1.0):
        print(f"Error: Test ratio must be between 0 and 1, got {args.test_ratio}")
        return
    
    if not (0.0 <= args.val_ratio <= 1.0):
        print(f"Error: Validation ratio must be between 0 and 1, got {args.val_ratio}")
        return

    create_split_worksheet(
        args.manifest_file,
        args.sheet_name,
        args.split_name,
        args.test_ratio,
        args.val_ratio,
        args.random_seed
    )

    print("\nSplit worksheet generation completed successfully!")


if __name__ == '__main__':
    main()