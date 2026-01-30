# -*- coding: utf-8 -*-
"""
 接收-m/--manifest_file指定的Excel清单文件路径，-s指定主要清单的工作表名称，默认为Manifest，-spn/--split_name指定划分方案名称，默认为split01_TJ。-s/--site指定中心编号列表list(str)，默认['TJ']，读取主表的site列，筛选包含在--site中的样本。由-ts/--test_ratio指定测试集占比，默认0.4，-v/--val_ratio指定验证集在训练验证集中的占比，默认0.2。-rs/--random_seed指定随机数种子（默认0）。 
 如果-ts不为0或1，则按比例划分测试集和训练验证集；如果-ts为0，则全部作为训练验证集；如果-ts为1，则全部作为测试集。-ts不为1时，继续处理训练验证集，如果-v不为0或1，则按比例划分训练集和验证集；如果-v为1，则全部作为验证集；如果-v为0，则全部作为训练集。 
 在划分时，首先以site和pid为键分组收集具有相同site和pid的表项，检查每组内表项的'Pathological Complete Response (pCR)'属性，如果不一致则报告错误并终止；检查通过后，按照'Pathological Complete Response (pCR)'对site和pid键组进行分层抽样。完成train/val/test抽样后，在清单中增加名称为split_name的新工作表，拷贝主要工作表内容，并在第一列插入名称为split_name的新列，其中记录划分值train/val/test。 
  对全部变量和函数参数添加类型注解。 
  除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Dataset Split Generation Script for ESO-2025 Dataset

This script creates a new worksheet in the manifest Excel file with stratified split information
(train/val/test) based on site, pid, and Pathological Complete Response (pCR) attribute.

Parameters:
    -m, --manifest_file: Path to the Excel manifest file
    -s, --sheet_name: Name of the main manifest worksheet (default: Manifest)
    -spn, --split_name: Name of the split scheme (default: split01_TJ)
    -st, --site: List of site IDs to include (default: ['TJ'])
    -ts, --test_ratio: Test set ratio (default: 0.4)
    -v, --val_ratio: Validation set ratio within train+val set (default: 0.2)
    -rs, --random_seed: Random seed for reproducibility (default: 0)

Usage Examples:
    python 05_make_split.py -m /path/to/dataset_manifest.xlsx
    python 05_make_split.py --manifest_file /path/to/dataset_manifest.xlsx --site TJ BJ
    python 05_make_split.py -m /path/to/dataset_manifest.xlsx -ts 0.3 -v 0.1 -rs 42
    python 05_make_split.py -m /path/to/dataset_manifest.xlsx --split_name split02_custom
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
        description='Create stratified split worksheet in manifest Excel file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m /path/to/dataset_manifest.xlsx
  %(prog)s --manifest_file /path/to/dataset_manifest.xlsx --site TJ BJ
  %(prog)s -m /path/to/dataset_manifest.xlsx -ts 0.3 -v 0.1 -rs 42
  %(prog)s -m /path/to/dataset_manifest.xlsx --split_name split02_custom
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
        default='split01_TJ',
        help='Name of the split scheme (default: split01_TJ)'
    )

    parser.add_argument(
        '-st', '--site',
        type=str,
        nargs='+',
        default=['TJ'],
        help='List of site IDs to include (default: ["TJ"])'
    )

    parser.add_argument(
        '-ts', '--test_ratio',
        type=float,
        default=0.4,
        help='Test set ratio (default: 0.4)'
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


def check_pcr_consistency(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Check consistency of 'Pathological Complete Response (pCR)' within each (site, pid) group.
    
    Args:
        df (pd.DataFrame): DataFrame containing the manifest data
        
    Returns:
        Tuple[bool, Optional[str]]: (True if consistent, None if no error; False if inconsistent, error message)
    """
    pcr_column = 'Pathological Complete Response (pCR)'
    
    if pcr_column not in df.columns:
        return False, f"Error: '{pcr_column}' column not found in the dataframe"
    
    # Group by site and pid
    grouped = df.groupby(['site', 'pid'])
    
    for (site, pid), group in grouped:
        pcr_values = group[pcr_column].unique()
        valid_pcr_values = [val for val in pcr_values if pd.notna(val) and val != '']
        
        if len(valid_pcr_values) > 1:
            error_msg = f"Error: Inconsistent pCR values for site={site}, pid={pid}: {valid_pcr_values}"
            return False, error_msg
    
    return True, None


def create_grouped_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a grouped dataframe with unique (site, pid) pairs and their pCR values.
    
    Args:
        df (pd.DataFrame): Original dataframe with all samples
        
    Returns:
        pd.DataFrame: Grouped dataframe with unique (site, pid) pairs
    """
    # Group by site and pid, then take the first pCR value (which should be consistent)
    grouped_df = df.groupby(['site', 'pid']).first().reset_index()
    return grouped_df[['site', 'pid', 'Pathological Complete Response (pCR)']]


def stratified_split_indices(group: pd.DataFrame, test_ratio: float, random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform stratified split on a single group.
    
    Args:
        group (pd.DataFrame): Grouped dataframe
        test_ratio (float): Test set ratio
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Indices for train_val and test sets
    """
    np.random.seed(random_seed)
    indices = np.arange(len(group))
    np.random.shuffle(indices)
    
    test_size = int(len(indices) * test_ratio)
    test_indices = indices[:test_size]
    train_val_indices = indices[test_size:]
    
    return train_val_indices, test_indices

def perform_split(grouped_df: pd.DataFrame, test_ratio: float, val_ratio: float, random_seed: int) -> Dict[Tuple[str, str], str]:
    """
    Perform stratified split based on pCR values and ratios using numpy.
    
    Args:
        grouped_df (pd.DataFrame): Grouped dataframe with unique (site, pid) pairs
        test_ratio (float): Test set ratio
        val_ratio (float): Validation set ratio within train+val set
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Dict[Tuple[str, str], str]: Dictionary mapping (site, pid) pairs to split values (train/val/test)
    """
    pcr_column = 'Pathological Complete Response (pCR)'
    
    # Create a dictionary to store split results
    split_dict: Dict[Tuple[str, str], str] = {}
    
    # Step 1: Split into test and train_val sets
    if test_ratio == 1.0:
        # All samples go to test set
        for _, row in grouped_df.iterrows():
            split_dict[(row['site'], row['pid'])] = 'test'
        return split_dict
    elif test_ratio == 0.0:
        # All samples go to train_val set
        train_val_df = grouped_df.copy()
    else:
        # Perform stratified split between test and train_val
        train_val_indices = []
        test_indices = []
        
        # Group by pCR values and perform stratified split
        for pcr_value, group in grouped_df.groupby(pcr_column, dropna=False):
            # Generate a unique seed for this group based on the main seed
            group_seed = hash(f"{random_seed}_{pcr_value}") % (2**32)
            
            # Split the group into train_val and test
            group_train_val_idx, group_test_idx = stratified_split_indices(group, test_ratio, group_seed)
            
            # Convert group indices to original dataframe indices
            train_val_indices.extend(group.index[group_train_val_idx].tolist())
            test_indices.extend(group.index[group_test_idx].tolist())
        
        # Create train_val and test dataframes
        train_val_df = grouped_df.loc[train_val_indices]
        test_df = grouped_df.loc[test_indices]
        
        # Assign test set
        for _, row in test_df.iterrows():
            split_dict[(row['site'], row['pid'])] = 'test'
    
    # Step 2: Split train_val into train and val sets
    if test_ratio != 1.0:
        if val_ratio == 1.0:
            # All train_val samples go to val set
            for _, row in train_val_df.iterrows():
                split_dict[(row['site'], row['pid'])] = 'val'
        elif val_ratio == 0.0:
            # All train_val samples go to train set
            for _, row in train_val_df.iterrows():
                split_dict[(row['site'], row['pid'])] = 'train'
        else:
            # Perform stratified split between train and val
            train_indices = []
            val_indices = []
            
            # Group by pCR values and perform stratified split
            for pcr_value, group in train_val_df.groupby(pcr_column, dropna=False):
                # Generate a unique seed for this group based on the main seed
                group_seed = hash(f"{random_seed}_val_{pcr_value}") % (2**32)
                
                # Split the group into train and val
                group_train_idx, group_val_idx = stratified_split_indices(group, val_ratio, group_seed)
                
                # Convert group indices to original dataframe indices
                train_indices.extend(group.index[group_train_idx].tolist())
                val_indices.extend(group.index[group_val_idx].tolist())
            
            # Create train and val dataframes
            train_df = train_val_df.loc[train_indices]
            val_df = train_val_df.loc[val_indices]
            
            # Assign train and val sets
            for _, row in train_df.iterrows():
                split_dict[(row['site'], row['pid'])] = 'train'
            
            for _, row in val_df.iterrows():
                split_dict[(row['site'], row['pid'])] = 'val'
    
    return split_dict


def create_split_worksheet(
    manifest_file: Union[str, Path],
    sheet_name: str,
    split_name: str,
    sites: List[str],
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
        sites (List[str]): List of site IDs to include
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
    print(f"Including sites: {sites}")
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
        required_columns = ['ID', 'site', 'pid', 'Pathological Complete Response (pCR)']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: '{col}' column not found in worksheet '{sheet_name}'")
                return

        # Filter by sites
        df_filtered = df[df['site'].isin(sites)].copy()
        
        if df_filtered.empty:
            print(f"Error: No samples found for sites {sites}")
            return
        
        print(f"Total samples before filtering: {len(df)}")
        print(f"Total samples after filtering by sites {sites}: {len(df_filtered)}")

        # Check pCR consistency within (site, pid) groups
        is_consistent, error_msg = check_pcr_consistency(df_filtered)
        if not is_consistent:
            print(error_msg)
            return

        # Create grouped dataframe for split
        grouped_df = create_grouped_dataframe(df_filtered)
        print(f"Number of unique (site, pid) pairs: {len(grouped_df)}")

        # Perform split
        split_dict = perform_split(grouped_df, test_ratio, val_ratio, random_seed)

        # Create a copy of the filtered dataframe and add the split column
        df_copy = df_filtered.copy()
        
        # Map split values to each row based on (site, pid)
        df_copy[split_name] = df_copy.apply(
            lambda row: split_dict.get((row['site'], row['pid']), ''),
            axis=1
        )

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
    print(f"Sites: {args.site}")
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
        args.site,
        args.test_ratio,
        args.val_ratio,
        args.random_seed
    )

    print("\nSplit worksheet generation completed successfully!")


if __name__ == '__main__':
    main()