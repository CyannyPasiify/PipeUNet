# -*- coding: utf-8 -*-
"""
参照01_gen_manifest.py生成数据配对确认脚本。使用argparse。接收-r/--root_dir指定根目录，根目录下包含training和
testing两个子目录，-o/--output_dir指定导出目录，然后创建train和test两个子目录。遍历training和testing中的二级子
目录，从每个二级子目录中匹配形如patientxxx_frameyy.nii.gz的图像文件和对应的patientxxx_frameyy_gt.nii.gz蒙版文
件，每个二级子目录中可有多组，使用MONAI的LoadImage加载图像和蒙版文件，将图像文件的元信息拷贝到蒙版确保二者一致，然
后根据其所处子目录training或testing，在对应的导出子目录train或test中创建形如patientxxx_frameyy的子目录，使用
SaveImage按照patientxxx_frameyy_volume.nii.gz形式保存图像文件，按照patientxxx_frameyy_mask.nii.gz形式保存蒙
版文件。如果遇到无法匹配的图像或蒙版，以及图像和蒙版规格不一致的情况，则在控制台报告问题，并跳过此文件。
"""

"""
Data Pair Confirmation and Reorganization Script for ACDC Dataset

This script confirms and reorganizes image-mask pairs from the ACDC dataset, ensuring metadata consistency
and organizing files into a standardized directory structure with patient information.

Parameters:
    -r, --root_dir: Root directory of the dataset containing 'training' and 'testing' subdirectories
    -o, --output_dir: Output directory where reorganized data will be saved

Usage Examples:
    python 02_confirm_pairs.py -r /path/to/ACDC -o /path/to/grouped
    python 02_confirm_pairs.py --root_dir /path/to/ACDC --output_dir /path/to/grouped
"""

import re
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from monai.transforms import LoadImage, SaveImage


def parse_args():
    """
    Parse command line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments containing root_dir and output_dir
    """
    parser = argparse.ArgumentParser(
        description='Confirm and reorganize image-mask pairs for ACDC dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r /path/to/ACDC -o /path/to/grouped
  %(prog)s --root_dir /path/to/ACDC --output_dir /path/to/grouped
        """
    )

    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Root directory of the dataset containing training and testing subdirectories'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Output directory where reorganized data will be saved'
    )

    return parser.parse_args()


def find_image_mask_pairs(patient_dir):
    """
    Find matching image and mask file pairs in a patient directory.

    Args:
        patient_dir (Path): Path to the patient directory

    Returns:
        tuple: (matched_pairs, unmatched_images, unmatched_masks)
            - matched_pairs: List of tuples (image_path, mask_path, frame_id)
            - unmatched_images: List of image paths without matching masks
            - unmatched_masks: List of mask paths without matching images
    """
    patient_match = re.match(r'patient(\d{3})', patient_dir.name)
    if not patient_match:
        return [], [], []

    patient_id = patient_match.group(1)

    nii_files = list(patient_dir.glob('*.nii.gz'))

    image_files = {}
    mask_files = {}

    for nii_file in nii_files:
        filename = nii_file.name

        image_match = re.match(rf'patient{patient_id}_frame(\d{{2}})\.nii\.gz', filename)
        mask_match = re.match(rf'patient{patient_id}_frame(\d{{2}})_gt\.nii\.gz', filename)

        if image_match:
            frame_id = image_match.group(1)
            image_files[frame_id] = nii_file
        elif mask_match:
            frame_id = mask_match.group(1)
            mask_files[frame_id] = nii_file

    matched_pairs = []
    unmatched_images = []
    unmatched_masks = []

    all_frame_ids = set(image_files.keys()) | set(mask_files.keys())

    for frame_id in all_frame_ids:
        if frame_id in image_files and frame_id in mask_files:
            matched_pairs.append((image_files[frame_id], mask_files[frame_id], frame_id))
        elif frame_id in image_files:
            unmatched_images.append(image_files[frame_id])
        elif frame_id in mask_files:
            unmatched_masks.append(mask_files[frame_id])

    return matched_pairs, unmatched_images, unmatched_masks


def check_metadata_consistency(image_meta, mask_meta):
    """
    Check if image and mask metadata are consistent.

    Args:
        image_meta (dict): Image metadata dictionary
        mask_meta (dict): Mask metadata dictionary

    Returns:
        bool: True if metadata is consistent, False otherwise
    """
    image_shape = image_meta['spatial_shape']
    mask_shape = mask_meta['spatial_shape']

    if not (image_shape == mask_shape).all():
        return False

    image_affine = image_meta['affine']
    mask_affine = mask_meta['affine']

    if not (image_affine == mask_affine).all():
        return False

    return True


def copy_metadata_to_mask(image_meta, mask_data):
    """
    Copy image metadata to mask to ensure consistency.

    Args:
        image_meta (dict): Image metadata dictionary
        mask_data: Mask image data array

    Returns:
        tuple: (mask_data_with_meta, updated_meta)
    """
    updated_meta = image_meta.copy()
    mask_data.meta = updated_meta
    return mask_data, updated_meta


def parse_info_cfg(info_cfg_path):
    """
    Parse Info.cfg file to extract patient information.

    Args:
        info_cfg_path (str or Path): Path to the Info.cfg file

    Returns:
        dict: Dictionary containing ED, ES, Group, Height, NbFrame, Weight information
    """
    info = {
        'ED': None,
        'ES': None,
        'Group': None,
        'Height': None,
        'NbFrame': None,
        'Weight': None
    }

    try:
        with open(info_cfg_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().split('#')[0].strip()

                    if key in info:
                        if key in ['Height', 'Weight']:
                            try:
                                info[key] = float(value)
                            except ValueError:
                                pass
                        elif key in ['ED', 'ES', 'NbFrame']:
                            try:
                                info[key] = int(value)
                            except ValueError:
                                pass
                        else:
                            info[key] = value
    except Exception as e:
        pass

    return info


def determine_phase(frame_id, ed, es):
    """
    Determine the cardiac phase (ED/ES/UND) based on frame number.

    Args:
        frame_id (str): Frame number as string
        ed (int): End Diastolic frame number
        es (int): End Systolic frame number

    Returns:
        str: 'ED', 'ES', or 'UND' based on frame number
    """
    try:
        frame_num = int(frame_id)
        if ed is not None and frame_num == ed:
            return 'ED'
        elif es is not None and frame_num == es:
            return 'ES'
        else:
            return 'UND'
    except (ValueError, TypeError):
        return 'UND'


def process_pairs(root_dir, output_dir):
    """
    Process all image-mask pairs and reorganize them into the output directory.

    Args:
        root_dir (str or Path): Root directory containing training and testing subdirectories
        output_dir (str or Path): Output directory for reorganized data
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    train_dir = output_path / 'train'
    test_dir = output_path / 'test'

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    subdirs = ['training', 'testing']

    total_issues = 0
    total_pairs = 0

    loader = LoadImage(image_only=False, dtype=None)

    for subdir in tqdm(subdirs, desc='Processing subsets'):
        subdir_path = root_path / subdir
        if not subdir_path.exists():
            print(f"Warning: {subdir} directory not found, skipping...")
            continue

        output_subdir = train_dir if subdir == 'training' else test_dir

        patient_dirs = sorted(subdir_path.glob('patient*'))

        for patient_dir in tqdm(patient_dirs, desc=f'  {subdir} patients', leave=False):
            patient_match = re.match(r'patient(\d{3})', patient_dir.name)
            if not patient_match:
                continue

            patient_id = patient_match.group(1)

            info_cfg_path = patient_dir / 'Info.cfg'
            patient_info = parse_info_cfg(info_cfg_path)

            matched_pairs, unmatched_images, unmatched_masks = find_image_mask_pairs(patient_dir)

            for image_path, mask_path, frame_id in matched_pairs:
                phase = determine_phase(frame_id, patient_info['ED'], patient_info['ES'])
                group = patient_info['Group'] if patient_info['Group'] else 'Unknown'

                pair_dir = output_subdir / f'patient{patient_id}_frame{frame_id}_{phase}_{group}'
                pair_dir.mkdir(parents=True, exist_ok=True)

                try:
                    image_data, image_meta = loader(str(image_path))
                    mask_data, mask_meta = loader(str(mask_path))

                    if not check_metadata_consistency(image_meta, mask_meta):
                        print(f"Warning: Metadata mismatch for {image_path.name} and {mask_path.name}")
                        print(f"  Image shape: {image_meta['spatial_shape']}, Mask shape: {mask_meta['spatial_shape']}")
                        print(f"  Copying image metadata to mask...")
                        mask_data, mask_meta = copy_metadata_to_mask(image_meta, mask_data)
                        total_issues += 1

                    volume_filestem: Path = pair_dir / f'patient{patient_id}_frame{frame_id}_volume'
                    mask_filestem: Path = pair_dir / f'patient{patient_id}_frame{frame_id}_mask'

                    saver_image = SaveImage(output_dir=str(pair_dir), output_postfix='', output_dtype=np.float32)
                    saver_mask = SaveImage(output_dir=str(pair_dir), output_postfix='', output_dtype=np.uint8)

                    saver_image(image_data, meta_data=image_meta, filename=volume_filestem)
                    saver_mask(mask_data, meta_data=mask_meta, filename=mask_filestem)

                    info_data = {
                        'phase': phase,
                        'group': patient_info['Group'],
                        'tot_frame': patient_info['NbFrame'],
                        'height': patient_info['Height'],
                        'weight': patient_info['Weight']
                    }

                    info_yaml_path = pair_dir / f'patient{patient_id}_frame{frame_id}_info.yaml'
                    with open(info_yaml_path, 'w', encoding='utf-8') as f:
                        yaml.dump(info_data, f, default_flow_style=False, allow_unicode=True)

                    total_pairs += 1

                except Exception as e:
                    print(f"Error processing {image_path.name} and {mask_path.name}: {str(e)}")
                    total_issues += 1
                    continue

            for unmatched_image in unmatched_images:
                print(f"Warning: Unmatched image file: {unmatched_image}")
                total_issues += 1

            for unmatched_mask in unmatched_masks:
                print(f"Warning: Unmatched mask file: {unmatched_mask}")
                total_issues += 1

    print(f"\nProcessing completed!")
    print(f"Total pairs processed: {total_pairs}")
    print(f"Total issues encountered: {total_issues}")


def main():
    """
    Main function to orchestrate the pair confirmation and reorganization process.
    """
    args = parse_args()

    print(f"Processing dataset from: {args.root_dir}")
    print(f"Output directory: {args.output_dir}")

    process_pairs(args.root_dir, args.output_dir)

    print("Pair confirmation and reorganization completed successfully!")


if __name__ == '__main__':
    import numpy as np

    main()
