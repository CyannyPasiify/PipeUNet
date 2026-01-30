# Duke-Breast-FGT-Segmentation Data Preprocess Tutorial

## Steps

### 1️⃣Download Duke-Breast-FGT-Segmentation-2025.4.10 data archive from [this link (Baidu Drive)](https://pan.baidu.com/s/1Xx_6o4KzTK61zRUo2KAA4Q?pwd=vugf)

Place the `archive` into an empty `root_dir`, e.g., `root_dir=/path/to/Duke-Breast-FGT-Segmentation-2025.4.10 `.

### 2️⃣Scan Duke-Breast archive and manual check metadatas: Run routine 01

```sh
python 01_gen_archive_manifest.py --root_dir root_dir/archive --output_manifest_file root_dir/descriptor/archive_manifest.xlsx --dataset_split_json root_dir/archive/dataset.json
```

Routine 01 recursively scans all NIfTI `nii.gz` files in the directory `root_dir/archive`, filters valid 3D volume and segmentation mask files, and generates a manifest file named `archive_manifest.xlsx` in a newly created subdirectory `descriptor` under `root_dir`.

All filtered NIfTI files are recorded in `archive_manifest.xlsx`. The file contains a single worksheet titled **Manifest**, which includes the following attributes:

- `file_path`: Relative path (with respect to `root_dir`) of the NIfTI `nii.gz` file.

- `type`: Content type of the file, with two possible values: `v3d` (3D volume) or `m3d` (3D mask).
- `subset`: Dataset split label (`train`, `val`, or `test`), consistent with the split definitions recorded in `dataset.json`.
- `primary`: Binary indicator where 1 denotes the file is the primary image source for the associated `pid` sample, and 0 denotes it is not.
- `szx`: Volume dimension size along the X-axis.
- `szy`: Volume dimension size along the Y-axis.
- `szz`: Volume dimension size along the Z-axis.
- `spx`: Voxel spacing along the X-axis (unit: mm).
- `spy`: Voxel spacing along the Y-axis (unit: mm).
- `spz`: Voxel spacing along the Z-axis (unit: mm).
- `orientation_from`: Orientation code for the start side, defined under the L-R (Left-Right) / P-A (Posterior-Anterior) / I-S (Inferior-Superior) coordinate system, may be oblique (e.g., `LPI`, `Oblique(closest to RAI)`).
- `orientation_to`: Orientation code for the end side, defined under the L-R / P-A / I-S coordinate system, may be oblique (e.g., `RAS`, `Oblique(closest to LPS)`).
- `dtype`: Data type of elements within the NIfTI file (e.g., `uint8`, `int16`, `int32`).
- `transform`: 4×4 affine transformation matrix of the NIfTI file.
- `diff_trans_f_norm`: For entries sharing the same `seq` value, this metric represents the Frobenius norm (F-norm) of the differential affine transformation matrix between the volume (`v3d`) and mask (`m3d`) files, calculated as `m3d−v3d`. This value is recorded **exclusively for mask (`m3d`) files**. An F-norm of 0 indicates full spatial consistency between the corresponding primary volume and mask files.

### 3️⃣Confirm and reorganize volume-mask pairs: Run routine 02

```sh
python 02_confirm_pairs.py --root_dir root_dir/archive --output_dir root_dir/grouped --archive_manifest root_dir/descriptor/archive_manifest.xlsx --label_explanation root_dir/descriptor/label_map.yaml
```

Routine 02 loads the archive manifest file to retrieve dataset metadata. The script filters all image files labeled as the primary image source, as well as all mask files with a suffix in `{1_Breast, 1_Breast_Remained_2_Fibroglandular_Tissue_3_Blood_Vessel}`. It then verifies and reorganizes valid image-mask pairs from the Duke-Breast-FGT-Segmentation dataset, ensures metadata consistency, remaps mask labels according to the provided label maps, and organizes all files into a standardized directory structure grouped by subset and patient ID. Mask files with the suffix `1_Breast` are saved as `{ID}_mask_mass.nii.gz`, while mask files with the suffix `1_Breast_Remained_2_Fibroglandular_Tissue_3_Blood_Vessel` are saved as the multi-label mask `{ID}_mask.nii.gz`.

A dataset directory is created at `root_dir/grouped`, with three subset subdirectories (`train`, `val`, `test`) established within it. Inside each subset directory, an individual subdirectory is created for each sample, named in the format `Breast_MRI_{sid}` (e.g., `Breast_MRI_002`) — this name also serves as the unique sample ID. Each sample directory contains the following files:

- `{ID}_info.yaml`: Information file for the sample (`pid`, `subset`).
- `{ID}_volume.nii.gz`: 3D volume file (data type: `float32`), saved after volume metadata correction.
- `{ID}_mask.nii.gz`: Multi-label segmentation mask file (data type: `uint8`), with metadata aligned to the corresponding volume file.
- `{ID}_mask_mass.nii.gz`: Whole breast segmentation mask file (data type: `uint8`), with metadata aligned to the corresponding volume file.

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - train
    - Breast_MRI_002
      - Breast_MRI_002_info.yaml
      - Breast_MRI_002_volume.nii.gz
      - Breast_MRI_002_mask.nii.gz
      - Breast_MRI_002_mask_mass.nii.gz
  - val
  - test
```

### [Additional] Mask denoising and quality optimization

```sh
python utils/small_connected_region_detect.py --root_dir root_dir/grouped --region_volume_thresh 100.0 --output_manifest root_dir/descriptor/small_connected_region_100_manifest.xlsx
```

`utils/small_connected_region_detect.py` detects small connected regions in binary mask files and reports their centroid coordinates and corresponding physical volumes. Only regions with a physical volume below a predefined threshold are included in the output. Based on empirical settings, we adopt **100 mm³** as threshold to filter small 3D connected regions.

```sh
python utils/union_multi_mask_fill_holes_main_part.py --root_dir root_dir/grouped
--overwrite --hole_thresh 40
```

`utils/union_multi_mask_fill_holes_main_part.py` processes multi-label mask files: sets foreground pixels to 1, fills slice-wise holes smaller than 40 voxels, retains the largest connected component, and reconstructs the multi-label mask by overlaying `label_index > 1` regions from the original mask onto the processed binary mask. A detailed procedure is provided below:

1. Extract the multi-label mask and set all foreground labels to 1.
2. Perform hole filling along the last axis with the largest voxel spacing (typically the Z-axis). First, invert the 2D slice and conduct connected component analysis; fill regions containing fewer than 40 voxels with label index 1.
3. Perform 3D connected component analysis on the mask. Reserve only the largest region and save the processed mask as the new `mask_mass` file.
4. Extract regions with `label_index > 1` from the original multi-label mask and overlay them onto `mask_mass` (confined to the foreground region of `mask_mass`). Save the processed mask as a new multi-label mask.

```sh
python utils/small_connected_region_detect.py --root_dir root_dir/grouped --region_volume_thresh 100.0 --output_manifest root_dir/descriptor/small_connected_region_100_after_modification_manifest.xlsx
```

You may re-run `utils/small_connected_region_detect.py` with a threshold of **100 mm³** to verify the mask state after denoising and quality optimization. This step is **optional**.

### 4️⃣Extract binary masks from multi-label mask files: Run routine 03

```sh
python 03_extract_binary_masks.py --root_dir root_dir/grouped --label_explanation root_dir/descriptor/label_map.yaml
```

Routine 03 recursively scans all NIfTI `{ID}_mask.nii.gz` mask files in the `root_dir/grouped` directory, and splits each multi-label segmentation mask into a set of binary segmentation masks. The label values in the original masks are defined in `label_map.yaml` as follows: 

> Format: `{label_index}` = `{label_name}` (`short_form`)

- 0 = background (Bg)
- 1 = breast residue (BR)
- 2 = fibroglandular tissue (FGT)
- 3 = blood vessel (VSL)

These binary masks are then saved in the same subdirectory as the original mask file, with the respective suffixes `{label_index:02d}_{label_name.short_form}` appended to the filenames.

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - train
    - Breast_MRI_002
      - Breast_MRI_002_info.yaml
      - Breast_MRI_002_volume.nii.gz
      - Breast_MRI_002_mask.nii.gz
      - Breast_MRI_002_mask_mass.nii.gz
      - Breast_MRI_002_mask_00_Bg.nii.gz
      - Breast_MRI_002_mask_01_BR.nii.gz
      - Breast_MRI_002_mask_02_FGT.nii.gz
      - Breast_MRI_002_mask_03_VSL.nii.gz
    - ...
  - val
  - test
```

### 5️⃣Generate comprehensive manifest for dataset: Run routine 04

```sh
python 04_gen_dataset_manifest.py --root_dir root_dir/grouped --output_manifest_file root_dir/grouped/dataset_manifest.xlsx
```

Routine 04 recursively scans all NIfTI `nii.gz` files in the `root_dir/grouped` directory. For each sample in the subdirectories, it collates the corresponding volume file, multi-label mask file, whole breast mask file (`mask_mass`) and individual binary mask files, then generates a manifest file named `dataset_manifest.xlsx` that indexes all files pertaining to each individual sample.

This routine raises an error if inconsistencies are detected in the **spatial size, affine transformation matrix, voxel spacing, or origin coordinates** across all NIfTI files for a single sample. An error is also raised if the data types of all mask NIfTI files associated with the same sample are mismatched.

Sample files for the Duke-Breast-FGT-Segmentation dataset is documented in manifest file `grouped/dataset_manifest.xlsx`. The manifest file contains a primary worksheet titled **Manifest**, with the following attributes included:

- `ID`: Sample identifier, formatted as `Breast_MRI_{sid}` (e.g., `Breast_MRI_002`).

- `subset`: Dataset split label (`train`, `val`, or `test`), consistent with the split definitions recorded in `dataset.json`.

- `valid_labels`: All label indexes the mask contains.

- `info`: Sample information YAML file path relative to `root_dir`.

- `volume`: Volume NIfTI `nii.gz` file path relative to `root_dir`.

- `mask`: Multi-label mask NIfTI `nii.gz` file path relative to `root_dir`.

- `mask_mass`: Binary mask for the whole breast NIfTI `nii.gz` file path relative to `root_dir`.

- `mask_{label_index}_{label_name}`: NIfTI `nii.gz` file path relative to `root_dir`, representing binary mask with `label_index` `label_name`.

- `mask_{label_index}_{label_name}_existence`: Existence status of the corresponding ROI. Value definition: `0` = no such ROI present in the mask, `1` = ROI present, `<empty>` = mask file is unavailable.

- `szx`: Volume size of X dimension for all NIfTI files associated with this sample.

- `szy`: Volume size of Y dimension for all NIfTI files associated with this sample.

- `szz`: Volume size of Z dimension for all NIfTI files associated with this sample.

- `spx`: Spacing (unit: mm) of X dimension for all NIfTI files associated with this sample.

- `spy`: Spacing (unit: mm) of Y dimension for all NIfTI files associated with this sample.

- `spz`: Spacing (unit: mm) of Z dimension for all NIfTI files associated with this sample.

- `orientation_from`: Orientation code for all NIfTI files associated with this sample from the start side, defined under the L-R (Left-Right) / P-A (Posterior-Anterior) / I-S (Inferior-Superior) coordinate system, may be oblique (e.g., `LPI`, `Oblique(closest to RAI)`).

- `orientation_to`: Orientation code for all NIfTI files associated with this sample from the start side, defined under the L-R / P-A / I-S coordinate system, may be oblique (e.g., `RAS`, `Oblique(closest to LPS)`).

- `vol_dtype`: Data type of the elements within the volume NIfTI file (e.g., `float32`).

  > Note: The data type of volumes is standardized to `float32` for all samples in `grouped`.

- `mask_dtype`: Data type of the elements within all mask NIfTI files associated with this sample (e.g., `uint8`).

  > Note: The data type of multi-label masks and binary masks is standardized to `uint8` for all samples in `grouped`.

- `transform`: Affine transformation matrix (4×4) for all NIfTI files associated with this sample.

### 6️⃣Split dataset and derive worksheet in dataset manifest: Run routine 05

```sh
python 05_make_split01_standard.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split01_standard
```

This script generates the **split01_standard** worksheet in the `grouped/dataset_manifest.xlsx` Excel manifest file, which stores the train/val/test split assignments for the dataset. The script copies all attributes from the primary **Manifest** worksheet, appends a dedicated column (titled `split01_standard`) for this split, and populates each entry with a `train`, `val`, or `test` label based on the sample's `subset` attribute — this aligns with the predefined standard dataset splits in the original data archive.

### 7️⃣Extracts subset-specific index manifests from dataset manifest: Run routine 06

Given a split-specific worksheet within the dataset manifest file, Routine 06 extracts subset-specific index manifests based on the split labels (e.g., train, val, test). Index manifests for individual subsets (train, val, test) or combined subsets (e.g., train+val) can be generated as separate Excel files; for instance, the train index manifest contains only entries labeled as `train` in the target split worksheet.

Prior to running routine 06, ensure the desired split is included as a dedicated worksheet in the dataset manifest file. For example, if `root_dir/grouped/dataset_manifest.xlsx` contains the worksheet `split01_standard`, you may specify the option `--split_name split01_standard` when executing routine 06 to generate the corresponding subset-specific index manifests.

The following routine generates index manifests for the split `split01_standard`. Manifest files `split01_standard_train_val.xlsx`, `split01_standard_train.xlsx`, `split01_standard_val.xlsx` and `split01_standard_test.xlsx` are generated in the directory `root_dir/grouped/splits/split01_standard`.

```sh
python 06_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split01_standard --output_index_manifest_dir root_dir/grouped/splits/split01_standard --subsets train val --subsets train --subsets val --subsets test
```

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - splits
    - split01_standard
      - split01_standard_train_val.xlsx
      - split01_standard_train.xlsx
      - split01_standard_val.xlsx
      - split01_standard_test.xlsx
```

You may modify the `--subsets` option by adding, removing, or revising subset values to control the generation of subset-specific index manifests. Note that specifying a subset (e.g., `--subsets val`) for which no corresponding `val` label exists in the target split worksheet will result in an overall failure, where all extraction tasks are aborted. You may refer to error messages for more information.

## End

You may now utilize the data files, dataset manifest, and subset-specific index manifests located in `root_dir/grouped` for downstream tasks, such as neural network training and model evaluation.
