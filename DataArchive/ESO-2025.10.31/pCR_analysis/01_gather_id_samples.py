#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
接收-r/--root_dir指定根目录，-m/--manifest_file指定的Excel清单文件路径，-s/--sheet_name指定工作表名称（默认Manifest）。-o/--output_dir指定输出根目录。工作表中包含以下关键信息列： 
 ID：记录文件所属样本的ID（{site}_{pid}_{phase}）。 
 site：记录样本的中心编号 str。
 phase：记录样本的时期，pre/post。
 pid：记录样本受试者编号 str。
 valid_labels：记录不为空的二值蒙版对应的标签列表，每个标签按照{label_index}_{label_name}格式记录，逗号','分隔。
 info：记录元信息文件的路径，相对于数据集根，使用POSIX路径连接符。
 volume：记录图像文件的路径，相对于数据集根，使用POSIX路径连接符。
 mask：记录蒙版文件的路径，相对于数据集根，使用POSIX路径连接符。
 按照site和pid分组，检查每组内是否分别包含phase为pre和post的2个表项，如果不符合则在控制台报告异常。 
 对检查通过的组，在-o输出根目录下创建{site}/{site}_{pid}样本目录。提取2个表项的共同属性[site,pid]，以及各自的[info,volume,mask]属性，读取2个info YAML文件的内容，保留其中值相等的属性并输出到新样本目录，文件名为{site}_{pid}_info.yaml。2个表项对应的volume,mask相对路径对应的文件直接拷贝到新样本目录，文件名不变。 
  对全部变量和函数参数添加类型注解。 
  除了此聊天内容以外，代码其它部分全部使用英文注释。首先，将此聊天内容以注释形式添加到代码文件开头。第二，添加脚本功能说明，参数释义和用法示例。第三，添加parse_args函数，其中添加argparse的实例化和参数解析。第四，添加具体业务逻辑的若干函数。最后，创建main主函数和函数入口。
"""

"""
Sample Gathering Script for ESO-2025.10.31 Dataset

This script processes an Excel manifest file, groups samples by site and PID, and creates a new directory structure
with consolidated information and copied files for valid pre/post sample pairs.

Parameters:
    -r, --root_dir: Root directory of the dataset
    -m, --manifest_file: Path to the Excel manifest file
    -s, --sheet_name: Name of the worksheet containing the manifest (default: Manifest)
    -o, --output_dir: Output root directory for the organized samples

Usage Examples:
    python 01_gather_id_samples.py -r /path/to/dataset -m /path/to/manifest.xlsx -o /path/to/output
    python 01_gather_id_samples.py --root_dir /path/to/dataset --manifest_file /path/to/manifest.xlsx --sheet_name Manifest --output_dir /path/to/output
"""

import argparse
import shutil
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Gather and organize samples based on Excel manifest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/dataset -m /path/to/manifest.xlsx -o /path/to/output
  %(prog)s --root_dir /path/to/dataset --manifest_file /path/to/manifest.xlsx --sheet_name Manifest --output_dir /path/to/output
        """)
    
    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset'
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
        help='Name of the worksheet containing the manifest (default: Manifest)'
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Output root directory for the organized samples'
    )
    
    return parser.parse_args()

def read_manifest(manifest_file: str, sheet_name: str) -> pd.DataFrame:
    """
    Read the Excel manifest file and return a DataFrame.
    
    Args:
        manifest_file: Path to the Excel manifest file
        sheet_name: Name of the worksheet containing the manifest
        
    Returns:
        pd.DataFrame: DataFrame containing the manifest data
    """
    try:
        df = pd.read_excel(manifest_file, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"Error reading manifest file: {e}")
        raise

def group_and_validate_samples(df: pd.DataFrame) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Group samples by site and pid, and validate that each group has both pre and post phases.
    
    Args:
        df: DataFrame containing the manifest data
        
    Returns:
        Dict[Tuple[str, str], List[Dict]]: Dictionary mapping (site, pid) tuples to lists of sample dictionaries
    """
    # Group samples by site and pid
    grouped = df.groupby(['site', 'pid'])
    
    valid_groups: Dict[Tuple[str, str], List[Dict]] = {}
    invalid_groups: List[Tuple[str, str, List[str]]] = []
    
    for (site, pid), group_df in grouped:
        phases: List[str] = group_df['phase'].unique()
        
        if len(phases) == 2 and set(phases) == {'pre', 'post'}:
            # Valid group with both pre and post phases
            valid_groups[(site, pid)] = group_df.to_dict('records')
        else:
            # Invalid group
            invalid_groups.append((site, pid, phases))
    
    # Report invalid groups
    if invalid_groups:
        print("Invalid groups found:")
        for site, pid, phases in invalid_groups:
            print(f"  Site: {site}, PID: {pid}, Phases: {list(phases)}")
    
    return valid_groups

def extract_common_attributes(yaml_path1: str, yaml_path2: str) -> Dict[str, Any]:
    """
    Extract common attributes from two YAML files.
    
    Args:
        yaml_path1: Path to the first YAML file
        yaml_path2: Path to the second YAML file
        
    Returns:
        Dict[str, Any]: Dictionary containing common attributes with matching values
    """
    try:
        with open(yaml_path1, 'r', encoding='utf-8') as f:
            yaml1: Dict[str, Any] = yaml.safe_load(f) or {}
        
        with open(yaml_path2, 'r', encoding='utf-8') as f:
            yaml2: Dict[str, Any] = yaml.safe_load(f) or {}
        
        # Find common keys with matching values
        common_attributes: Dict[str, Any] = {}
        for key in yaml1:
            if key in yaml2 and yaml1[key] == yaml2[key]:
                common_attributes[key] = yaml1[key]
        
        return common_attributes
    except Exception as e:
        print(f"Error extracting common attributes from YAML files: {e}")
        raise

def process_sample_group(
    group: List[Dict[str, str]],
    root_dir: str,
    output_dir: str
) -> None:
    """
    Process a single valid sample group (one pre and one post).
    
    Args:
        group: List of sample dictionaries (should be length 2)
        root_dir: Root directory of the dataset
        output_dir: Output root directory
    """
    if len(group) != 2:
        return
    
    # Extract common attributes
    site: str = group[0]['site']
    pid: str = group[0]['pid']
    
    # Create output directory structure: {output_dir}/{site}/{site}_{pid}
    sample_output_dir: Path = Path(output_dir) / site / f"{site}_{pid}"
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find pre and post samples
    pre_sample: Dict[str, str] = next(sample for sample in group if sample['phase'] == 'pre')
    post_sample: Dict[str, str] = next(sample for sample in group if sample['phase'] == 'post')
    
    # Process info files
    pre_info_path: Path = Path(root_dir) / pre_sample['info'].replace('/', '\\')
    post_info_path: Path = Path(root_dir) / post_sample['info'].replace('/', '\\')
    
    common_attributes: Dict[str, Any] = extract_common_attributes(str(pre_info_path), str(post_info_path))
    
    # Write common attributes to new info file
    info_output_path: Path = sample_output_dir / f"{site}_{pid}_info.yaml"
    with open(info_output_path, 'w', encoding='utf-8') as f:
        yaml.dump(common_attributes, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Copy volume and mask files for both pre and post samples
    files_to_copy: List[Tuple[str, str]] = [
        (pre_sample['volume'], 'volume'),
        (pre_sample['mask'], 'mask'),
        (post_sample['volume'], 'volume'),
        (post_sample['mask'], 'mask')
    ]
    
    for file_path, file_type in files_to_copy:
        src_path: Path = Path(root_dir) / file_path.replace('/', '\\')
        dst_path: Path = sample_output_dir / Path(file_path).name
        
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {file_type} file {src_path} to {dst_path}: {e}")

def main() -> None:
    """
    Main function to orchestrate the sample gathering process.
    """
    args: argparse.Namespace = parse_args()
    
    print(f"Reading manifest file: {args.manifest_file}")
    df: pd.DataFrame = read_manifest(args.manifest_file, args.sheet_name)
    
    print("Grouping and validating samples...")
    valid_groups: Dict[Tuple[str, str], List[Dict[str, str]]] = group_and_validate_samples(df)
    
    print(f"Found {len(valid_groups)} valid groups with both pre and post phases")
    
    print(f"Processing samples to output directory: {args.output_dir}")
    with tqdm(valid_groups.items(), desc="Processing groups", unit="group") as pbar:
        for (site, pid), group in pbar:
            site: str
            pid: str
            group: List[Dict[str, str]]
            # Update progress bar description to show current site and pid
            pbar.set_description(f"Processing group: Site {site}, PID {pid}")
            process_sample_group(group, args.root_dir, args.output_dir)
    
    print("Sample gathering completed successfully!")

if __name__ == '__main__':
    main()
