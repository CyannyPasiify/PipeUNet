# VerSe Data Preprocess Tutorial

## Steps

### 1️⃣Download VerSe data archive from [this link (Baidu Drive)](https://pan.baidu.com/s/1p8AwLs-dsmQ7tE1KByBTlQ?pwd=caxh)

Place the `archive` into an empty `root_dir`, e.g., `root_dir=/path/to/VerSe `.

### 2️⃣Scan VerSe archive and manual check metadatas: Run routine 01

```sh
python 01_gen_archive_manifest.py --root_dir root_dir/archive --output_manifest_file root_dir/descriptor/archive_manifest.xlsx
```

Routine 01 recursively scans all NIfTI `nii.gz` files in the directory `root_dir/archive`, filters valid 3D volume and segmentation mask files, and generates a manifest file named `archive_manifest.xlsx` in a newly created subdirectory `descriptor` under `root_dir`.

All filtered NIfTI files are recorded in `archive_manifest.xlsx`. The file contains a single worksheet titled **Manifest**, which includes the following attributes:

- `file_path`: Relative path (with respect to `root_dir`) of the NIfTI `nii.gz` file. `root_dir=/path/to/VerSe` denotes a relative or absolute path to the root directory of the VerSe data archive.

- `archive`: Denotes the source archive of the sample, with values restricted to `VerSe19` or `VerSe20`.

  > Note: A sample annotated as belonging to the VerSe20 archive does not preclude its inclusion in VerSe19, given the partial overlap between the two datasets. While the official released archive of VerSe20 was intended to contain only extension samples not present in VerSe19, the actual VerSe20 dataset includes overlapping samples from VerSe19.

- `subset`: Dataset split label (`train`, `val`, or `test`), consistent with the original folder structure.

- `subject`: Primary sample identifier, formatted as a letter prefix followed by a 3-digit numerical value (e.g., gl003, verse004).

- `split`: Optional split-specific identifier for samples sharing the same `subject`, formatted as a letter prefix followed by a 3-digit numerical value (e.g., verse090). This field exists because a single patient’s vertebral span may cover multiple scan acquisitions, resulting in multiple paired volume and mask files for one patient.

- `suffix`: Suffix of the resource file name, which may indicate the file type and directional attributes of the scan data.

- `type`: Content type of the file, with two possible values: `v3d` (3D volume) or `m3d` (3D mask).

- `CT_image_series`: Internal index of the sample among all cases sharing the same `subject`, formatted as `{index} of {total}` (e.g., `1 of 2`). Here, `total` denotes the number of samples associated with the same `subject`.

- `in_verse19`: Binary indicator of whether the sample is included in VerSe19; `1` = included, `0` = not included.

- `in_verse20`: Binary indicator of whether the sample is included in VerSe20; `1` = included, `0` = not included.

- `sex`: Patient's biological sex (M = Male, F = Female).

- `age`: Patient's age at the time of examination.

- `szx`: Volume dimension size along the X-axis.

- `szy`: Volume dimension size along the Y-axis.

- `szz`: Volume dimension size along the Z-axis.

- `spx`: Voxel spacing along the X-axis (unit: mm).

- `spy`: Voxel spacing along the Y-axis (unit: mm).

- `spz`: Voxel spacing along the Z-axis (unit: mm).

- `orientation_from`: Orientation code for the start side, defined under the L-R (Left-Right) / P-A (Posterior-Anterior) / I-S (Inferior-Superior) coordinate system, may be oblique (e.g., `RAI`, `Oblique(closest to ASL)`).

- `orientation_to`: Orientation code for the end side, defined under the L-R / P-A / I-S coordinate system, may be oblique (e.g., `LPS`, `Oblique(closest to PIR)`).

- `dtype`: Data type of elements within the NIfTI file (e.g., `uint8`, `int16`, `int32`, `float32`, `float64`).

- `transform`: 4×4 affine transformation matrix of the NIfTI file.

- `diff_trans_f_norm`: For entries sharing the same (`subject`, `split`) key, this metric represents the Frobenius norm (F-norm) of the differential affine transformation matrix between the volume (`v3d`) and mask (`m3d`) files, calculated as `m3d−v3d`. This value is recorded **exclusively for mask (`m3d`) files**. An F-norm of 0 indicates full spatial consistency between the corresponding volume and mask files.

### 3️⃣Confirm and reorganize volume-mask pairs: Run routine 02

```sh
python 02_confirm_pairs.py --root_dir root_dir/archive --output_dir root_dir/grouped --archive_manifest root_dir/descriptor/archive_manifest.xlsx
```

Routine 02 loads the archive manifest file to retrieve dataset metadata. The script filters samples to retain only those whose `suffix` contains any of the substrings in the set `{ct, vert_msk}`. It then scans the NIfTI `nii.gz` files in the `root_dir/archive` directory, filters the relevant files, and pairs each 3D volume file with its corresponding segmentation mask file to form valid sample pairs.

Sub-dataset directories (`VerSe19` and `VerSe20`) are created under `root_dir/grouped`, with three subset subdirectories (`train`, `val`, `test`) created under each sub-dataset directory. Within each subset directory, subdirectories are created for each sample, name of which are formatted as `{subject}` or `{subject}-{num of split}` (e.g., gl003, verse004, verse401-201). We only use the latter format when the sample has `split`, and we reserve the last 3-digit number of the `split` to formulate the directory name which also serves as the unique sample identifier `ID`. Each sample directory contains the following files:

- `{ID}_info.yaml`: Information file for the sample (`subset`, `inclusion`, `sex`, `age`, etc).
- `{ID}_volume.nii.gz`: 3D volume file (data type: `float32`), saved after volume metadata correction.
- `{ID}_mask.nii.gz`: Segmentation mask file (data type: `uint8`), with metadata aligned to the corresponding volume file.

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - VerSe19
    - train
      - verse004
        - verse004_info.yaml
        - verse004_volume.nii.gz
        - verse004_mask.nii.gz
      - verse401-201
        - verse401-201_info.yaml
        - verse401-201_volume.nii.gz
        - verse401-201_mask.nii.gz
    - val
    - test
  - VerSe20
    - ...
```

### 4️⃣Reduce unused labels: Run routine 03

```sh
python 03_reduce_sacrum_cocygis_and_revise_T13_label.py --root_dir root_dir/grouped
```

Routine 03 recursively scans all sample directories under the `root_dir/grouped` directory, reduces sacrum (26) and coccyx (27) labels to background (0), and revises the T13 label (28) to index 26 — this ensures the continuity of label indices.

### 5️⃣Extract binary masks from multi-label mask files: Run routine 04

```sh
python 04_extract_binary_masks.py --root_dir root_dir/grouped --label_explanation root_dir/descriptor/label_map.yaml
```

Routine 04 recursively scans all NIfTI `nii.gz` mask files in the `root_dir/grouped` directory, and splits each multi-label segmentation mask into a set of binary segmentation masks. The label values in the original masks are defined in `label_map.yaml` as follows: 

> Format: `{label_index}` = `{label_name}` (`short_form`)

- 0 = background (Bg)
- 1 = cervical spine 1 (C1)
- 2 = cervical spine 2 (C2)
- 3 = cervical spine 3 (C3)
- 4 = cervical spine 4 (C4)
- 5 = cervical spine 5 (C5)
- 6 = cervical spine 6 (C6)
- 7 = cervical spine 7 (C7)
- 8 = thoracic spine 1 (T1)
- 9 = thoracic spine 2 (T2)
- 10 = thoracic spine 3 (T3)
- 11 = thoracic spine 4 (T4)
- 12 = thoracic spine 5 (T5)
- 13 = thoracic spine 6 (T6)
- 14 = thoracic spine 7 (T7)
- 15 = thoracic spine 8 (T8)
- 16 = thoracic spine 9 (T9)
- 17 = thoracic spine 10 (T10)
- 18 = thoracic spine 11 (T11)
- 19 = thoracic spine 12 (T12)
- 20 = lumbar spine 1 (L1)
- 21 = lumbar spine 2 (L2)
- 22 = lumbar spine 3 (L3)
- 23 = lumbar spine 4 (L4)
- 24 = lumbar spine 5 (L5)
- 25 = lumbar spine 6 (L6)
- 26 = additional 13th thoracic vertebra (T13)

These binary masks are then saved in the same subdirectory as the original mask file, with the respective suffixes `{label_index:02d}_{label_name.short_form}` appended to the filenames.

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - VerSe19
    - train
      - verse004
        - verse004_info.yaml
        - verse004_volume.nii.gz
        - verse004_mask.nii.gz
        - verse004_mask_00_Bg.nii.gz
        - verse004_mask_01_C1.nii.gz
        - verse004_mask_02_C2.nii.gz
        - verse004_mask_03_C3.nii.gz
        - verse004_mask_04_C4.nii.gz
        - verse004_mask_05_C5.nii.gz
        - verse004_mask_06_C6.nii.gz
        - verse004_mask_07_C7.nii.gz
        - verse004_mask_08_T1.nii.gz
        - verse004_mask_09_T2.nii.gz
        - verse004_mask_10_T3.nii.gz
        - verse004_mask_11_T4.nii.gz
        - verse004_mask_12_T5.nii.gz
        - verse004_mask_13_T6.nii.gz
        - verse004_mask_14_T7.nii.gz
        - verse004_mask_15_T8.nii.gz
        - verse004_mask_16_T9.nii.gz
        - verse004_mask_17_T10.nii.gz
        - verse004_mask_18_T11.nii.gz
        - verse004_mask_19_T12.nii.gz
        - verse004_mask_20_L1.nii.gz
        - verse004_mask_21_L2.nii.gz
        - verse004_mask_22_L3.nii.gz
        - verse004_mask_23_L4.nii.gz
        - verse004_mask_24_L5.nii.gz
        - verse004_mask_25_L6.nii.gz
        - verse004_mask_26_T13.nii.gz
      - verse401-201
        - ...
    - val
    - test
  - VerSe20
    - ...
```

### [Additional] Mask denoising and quality optimization

Firstly, manually correct label `L3` near (299,262,101) and (352,239,101), `L5` near (273,304,44) of sample `gl279`.

```sh
python utils/small_connected_region_detect.py --root_dir root_dir/grouped --region_volume_thresh 100.0 --output_manifest root_dir/descriptor/small_connected_region_100_manifest.xlsx
```

`utils/small_connected_region_detect.py` detects small connected regions in binary mask files and reports their centroid coordinates and corresponding physical volumes. Only regions with a physical volume below a predefined threshold are included in the output. Based on empirical settings, we adopt **100 mm³** as threshold to filter small 3D connected regions.

```sh
python utils/morphological_close_on_slice_small_region.py --root_dir root_dir/grouped --output_dir root_dir/grouped --kernel_size 5 5 --filter_halfedge 3 --region_area_thresh 2.0
```

`utils/morphological_close_on_slice_small_region.py` performs slice-wise morphological closing operation on binary masks, then reconstructs the multi-label mask from the processed binary masks. Voxel label conflicts are resolved by quantifying the number of foreground voxels within a local filtering neighborhood. First, independent morphological closing operation is applied to each original slice. Concurrently, connected component analysis is implemented on the background regions of each binary mask slice to identify small connected regions (< `region_area_thresh` mm²). Small background regions adopt the results of morphological closing (hole filling), and left regions retain the original slice content. For voxels with no assigned foreground labels or with multiple foreground labels after this processing step, the number of foreground support voxels within a square neighborhood of size [2×`filter_halfedge`+1, 2×`filter_halfedge`+1] (defined by `filter_halfedge`) is computed for each binary mask on the same slice; the valid label corresponding to the maximum count of support voxels is then assigned to the target voxel. A detailed procedural description is provided as follows:

Small connected region masks (holes) for the **background** regions of the binary mask corresponding to each label: 27 labels map to 27 individual binary masks.
$$
SC^i_{m-} = \operatorname{SmallComp_{<10mm^2}} \left(\neg mask_i\right),~ i \in \{0,1,...,26\}
$$
Non-small connected region masks of the binary mask corresponding to each label (these large regions remain unmodified):
$$
SC^i_{res} = \neg SC^i_{m-}
$$
Morphological opening and closing operations are executed using a 5×5 all-ones structural element. Large regions are preserved, with the optimized mask derived by **background** regions after morphological closing (hole filling):
$$
\operatorname{Optimize} \left(mask_i\right) = SC^i_{res} \odot mask_i + SC^i_{m-} \odot \operatorname{Close_{5×5}} \left(mask_i\right)
$$
Label index sequence across the 27 labels for each individual voxel at spatial position (x,y,z):
$$
\operatorname{Idx} \left(x,y,z\right) = \left[mask_0\left(x,y,z\right),mask_1\left(x,y,z\right),...,mask_{26}\left(x,y,z\right)\right]
$$
The 27 processed binary masks are fused into a single multi-label mask, where each voxel is assigned a value corresponding to its label index in the range of 0 to 26.

- If the label index sequence is one-hot encoded, the index of the component with a value of 1 is designated as the voxel's mask value.
- If the label index sequence is **not** one-hot encoded, two distinct cases are addressed:
  1. If the label index sequence is a zero vector, this denotes that the voxel has no valid foreground labels post-processing. We calculate the number of foreground voxels (support voxels) within a 7×7 local neighborhood across all binary masks, and assign the label index with the highest count of support voxels to the voxel (equivalent to a modal value selection).
  2. If multiple components in the label index sequence equal 1, this indicates label overlap (voxel ambiguity) at the target position post-processing. We calculate the number of foreground voxels (support voxels) within a 7×7 local neighborhood for each binary mask with a value of 1 at this voxel, and assign the label index with the highest count of support voxels to resolve the ambiguity.

$$
mask\left(x,y,z\right) = \begin{cases}
   \operatorname{Argmax}\left(\operatorname{Idx} \left(x,y,z\right)\right) &\text{if } \operatorname{Idx} \left(x,y,z\right) \text{ is one-hot} \\
   \operatorname{Argmax}\left(\left[\operatorname{Sum^{7×7}_0}\left(x,y,z\right),\operatorname{Sum^{7×7}_1}\left(x,y,z\right),...,\operatorname{Sum^{7×7}_{26}}\left(x,y,z\right)\right]\right) &\text{if } \operatorname{Idx} \left(x,y,z\right) \text{ is all-zero} \\
   \operatorname{Argmax}\left(\left[\operatorname{Sum^{7×7}_0}\left(x,y,z\right),\operatorname{Sum^{7×7}_1}\left(x,y,z\right),...,\operatorname{Sum^{7×7}_{26}}\left(x,y,z\right)\right] \odot \operatorname{Idx} \left(x,y,z\right)\right) &\text{otherwise (multiple 1 comp.)} \\
   \end{cases}
$$

**Note** that this segmentation mask rectification procedure is not **PERFECT**, which may erroneously fill bona fide holes, as no single algorithm can comprehensively address all labeling artifacts. Nevertheless, this correction procedure reduced the total number of small 3D connected regions (< 100 mm³) from 3282 to 2635 (indicated by `root_dir/descriptor/small_connected_region_100_manifest.xlsx` above and `root_dir/descriptor/small_connected_region_100_after_modification_manifest.xlsx` below) — a substantial improvement that indicates the remediation of extensive labeling errors and omissions in the original masks.

```sh
python utils/small_connected_region_detect.py --root_dir root_dir/grouped --region_volume_thresh 100.0 --output_manifest root_dir/descriptor/small_connected_region_100_after_modification_manifest.xlsx
```

You may re-run `utils/small_connected_region_detect.py` with a threshold of **100 mm³** to verify the mask state after denoising and quality optimization. This step is **optional**.

### 6️⃣Generate comprehensive manifest for dataset: Run routine 05

```sh
python 05_gen_dataset_manifest.py --root_dir root_dir/grouped --output_manifest_file root_dir/grouped/dataset_manifest.xlsx
```

Routine 05 recursively scans all NIfTI `nii.gz` files in the `root_dir/grouped` directory. For each sample in the subdirectories, it collates the corresponding volume file, multi-label mask file and individual binary mask files, then generates a manifest file named `dataset_manifest.xlsx` that indexes all files pertaining to each individual sample.

This routine raises an error if inconsistencies are detected in the **spatial size, affine transformation matrix, voxel spacing, or origin coordinates** across all NIfTI files for a single sample. An error is also raised if the data types of all mask NIfTI files associated with the same sample are mismatched.

Sample files for the VerSe dataset is documented in manifest file `grouped/dataset_manifest.xlsx`. The manifest file contains a primary worksheet titled **Manifest**, with the following attributes included:

- `ID`: Sample identifier, formatted as `{subject}` or `{subject}-{num of split}` (e.g., gl003, verse004, verse401-201). We only use the latter format when the sample has `split`, and we reserve the last 3-digit number of the `split` to formulate sample identifier.

- `archive`: Denotes the source archive of the sample, with values restricted to `VerSe19` or `VerSe20`.

  > Note: A sample annotated as belonging to the VerSe20 archive does not preclude its inclusion in VerSe19, given the partial overlap between the two datasets. While the official released archive of VerSe20 was intended to contain only extension samples not present in VerSe19, the actual VerSe20 dataset includes overlapping samples from VerSe19.

- `subset`: Dataset split label (`train`, `val`, or `test`), consistent with the split definitions recorded in `dataset.json`.

- `subject`: Primary sample identifier, formatted as a letter prefix followed by a 3-digit numerical value (e.g., gl003, verse004).

- `split`: Optional split-specific identifier for samples sharing the same `subject`, formatted as a letter prefix followed by a 3-digit numerical value (e.g., verse090). This field exists because a single patient’s vertebral span may cover multiple scan acquisitions, resulting in multiple paired volume and mask files for one patient.

- `in_verse19`: Binary indicator of whether the sample is included in VerSe19; `1` = included, `0` = not included.

- `in_verse20`: Binary indicator of whether the sample is included in VerSe20; `1` = included, `0` = not included.

- `CT_image_series`: Internal index of the sample among all cases sharing the same `subject`, formatted as `{index} of {total}` (e.g., `1 of 2`). Here, `total` denotes the number of samples associated with the same `subject`.

- `sex`: Patient's biological sex (M = Male, F = Female).

- `age`: Patient's age at the time of examination.

- `valid_labels`: All label indexes the mask contains.

- `info`: Sample information YAML file path relative to `root_dir`.

- `volume`: Volume NIfTI `nii.gz` file path relative to `root_dir`.

- `mask`: Multi-label mask NIfTI `nii.gz` file path relative to `root_dir`.

- `mask_{label_index}_{label_name}`: NIfTI `nii.gz` file path relative to `root_dir`, representing binary mask with `label_index` `label_name`.

- `mask_{label_index}_{label_name}_existence`: Existence status of the corresponding ROI. Value definition: `0` = no such ROI present in the mask, `1` = ROI present.

- `szx`: Volume size of X dimension for all NIfTI files associated with this sample.

- `szy`: Volume size of Y dimension for all NIfTI files associated with this sample.

- `szz`: Volume size of Z dimension for all NIfTI files associated with this sample.

- `spx`: Spacing (unit: mm) of X dimension for all NIfTI files associated with this sample.

- `spy`: Spacing (unit: mm) of Y dimension for all NIfTI files associated with this sample.

- `spz`: Spacing (unit: mm) of Z dimension for all NIfTI files associated with this sample.

- `orientation_from`: Orientation code for all NIfTI files associated with this sample from the start side, defined under the L-R (Left-Right) / P-A (Posterior-Anterior) / I-S (Inferior-Superior) coordinate system, may be oblique (e.g., `RAI`, `Oblique(closest to ASL)`).

- `orientation_to`: Orientation code for all NIfTI files associated with this sample from the start side, defined under the L-R / P-A / I-S coordinate system, may be oblique (e.g., `LPS`, `Oblique(closest to PIR)`).

- `vol_dtype`: Data type of the elements within the volume NIfTI file (e.g., `float32`).

  > Note: The data type of volumes is standardized to `float32` for all samples in `grouped`.

- `mask_dtype`: Data type of the elements within all mask NIfTI files associated with this sample (e.g., `uint8`).

  > Note: The data type of multi-label masks and binary masks is standardized to `uint8` for all samples in `grouped`.

- `transform`: Affine transformation matrix (4×4) for all NIfTI files associated with this sample.

### 7️⃣Split dataset and derive worksheet in dataset manifest: Run routine 06

```sh
python 06_make_split.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split01_standard
python 06_make_split.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split02_verse19 --inclusion VerSe19
python 06_make_split.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split03_verse20 --inclusion VerSe20
```

This script generates dedicated worksheets **split01_standard**, **split02_verse19**, and **split03_verse20** in the `grouped/dataset_manifest.xlsx` Excel manifest file. These worksheets store the official training/validation/test split assignments for the full VerSe dataset, as well as the VerSe19 and VerSe20 sub-datasets respectively. The script copies all attributes from the primary **Manifest** worksheet, appends a dedicated column for each split (titled `split01_standard`, `split02_verse19`, and `split03_verse20`), and populates each entry with a `train`, `val`, or `test` label based on the sample's `subset` attribute — this aligns with the official standard dataset splits defined in the original VerSe archive.

### 8️⃣Extracts subset-specific index manifests from dataset manifest: Run routine 07

Given a split-specific worksheet within the dataset manifest file, Routine 07 extracts subset-specific index manifests based on the split labels (e.g., train, val, test). Index manifests for individual subsets (train, val, test) or combined subsets (e.g., train+val) can be generated as separate Excel files; for instance, the train index manifest contains only entries labeled as `train` in the target split worksheet.

Prior to running routine 07, ensure the desired split is included as a dedicated worksheet in the dataset manifest file. For example, if `root_dir/grouped/dataset_manifest.xlsx` contains the worksheet `split01_standard`, you may specify the option `--split_name split01_standard` when executing routine 07 to generate the corresponding subset-specific index manifests.

The following routine generates index manifests for the split `split01_standard`. Manifest files `split01_standard_train_val.xlsx`, `split01_standard_train.xlsx`, `split01_standard_val.xlsx` and `split01_standard_test.xlsx` are generated in the directory `root_dir/grouped/splits/split01_standard`.

```sh
python 07_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split01_standard --output_index_manifest_dir root_dir/grouped/splits/split01_standard --subsets train val --subsets train --subsets val --subsets test
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
  - VerSe19  # sample directories are stored here 
  - VerSe20  # sample directories are stored here 
```

You may modify the `--subsets` option by adding, removing, or revising subset values to control the generation of subset-specific index manifests. Note that specifying a subset (e.g., `--subsets val`) for which no corresponding `val` label exists in the target split worksheet will result in an overall failure, where all extraction tasks are aborted. You may refer to error messages for more information.

Routine 07 is executed analogously for the VerSe19 and VerSe20 sub-datasets with the following commands:

```sh
python 07_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split02_verse19 --output_index_manifest_dir root_dir/grouped/splits/split02_verse19 --subsets train val --subsets train --subsets val --subsets test
python 07_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split03_verse20 --output_index_manifest_dir root_dir/grouped/splits/split03_verse20 --subsets train val --subsets train --subsets val --subsets test
```

This generates the corresponding directory structures for the VerSe19 and VerSe20 sub-datasets as shown below:

```sh
root_dir
- grouped
  - splits
    - split02_verse19
      - split02_verse19_train_val.xlsx
      - split02_verse19_train.xlsx
      - split02_verse19_val.xlsx
      - split02_verse19_test.xlsx
    - split03_verse20
      - split03_verse20_train_val.xlsx
      - split03_verse20_train.xlsx
      - split03_verse20_val.xlsx
      - split03_verse20_test.xlsx
  - VerSe19  # sample directories are stored here 
  - VerSe20  # sample directories are stored here 
```

## End

You may now utilize the data files, dataset manifest, and subset-specific index manifests located in `root_dir/grouped` for downstream tasks, such as neural network training and model evaluation.

The VerSe dataset can be rearranged into two sub-datasets: VerSe19 and VerSe20. Please refer to `grouped/dataset_manifest.xlsx` for more information.
