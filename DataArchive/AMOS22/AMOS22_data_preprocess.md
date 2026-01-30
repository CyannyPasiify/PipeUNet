# AMOS22 Data Preprocess Tutorial

## Steps

### 1️⃣Download AMOS22 data archive from [this link (Baidu Drive)](https://pan.baidu.com/s/1owJdUTH5--NhuHHEI2QhMg?pwd=qu5p)

Place the `archive` into an empty `root_dir`, e.g., `root_dir=/path/to/AMOS22 `.

### 2️⃣Scan AMOS22 archive and manual check metadatas: Run routine 01

```sh
python 01_gen_archive_manifest.py --root_dir root_dir/archive --output_manifest_file root_dir/descriptor/archive_manifest.xlsx
```

Routine 01 recursively scans all NIfTI `nii.gz` files in the directory `root_dir/archive`, filters valid 3D volume and segmentation mask files, and generates a manifest file named `archive_manifest.xlsx` in a newly created subdirectory `descriptor` under `root_dir`.

All filtered NIfTI files are recorded in `archive_manifest.xlsx`. The file contains a single worksheet titled **Manifest**, which includes the following attributes:

- `file_path`: Relative path (with respect to `root_dir`) of the NIfTI `nii.gz` file. `root_dir=/path/to/AMOS22` denotes a relative or absolute path to the root directory of the AMOS22 data archive.
- `subset`: Dataset split label (`train`, `val`, or `test`), consistent with the split definitions recorded in `dataset.json`.
- `seq`: AMOS unique identifier, represented as a 4-digit numerical value (e.g., 0001).
- `type`: Content type of the file, with two possible values: `v3d` (3D volume) or `m3d` (3D mask).
- `szx`: Volume dimension size along the X-axis.
- `szy`: Volume dimension size along the Y-axis.
- `szz`: Volume dimension size along the Z-axis.
- `spx`: Voxel spacing along the X-axis (unit: mm).
- `spy`: Voxel spacing along the Y-axis (unit: mm).
- `spz`: Voxel spacing along the Z-axis (unit: mm).
- `orientation_from`: Orientation code for the start side, defined under the L-R (Left-Right) / P-A (Posterior-Anterior) / I-S (Inferior-Superior) coordinate system, may be oblique (e.g., `RPI`, `Oblique(closest to LPI)`).
- `orientation_to`: Orientation code for the end side, defined under the L-R / P-A / I-S coordinate system, may be oblique (e.g., `LAS`, `Oblique(closest to RAS)`).
- `dtype`: Data type of elements within the NIfTI file (e.g., `uint8`, `int16`, `int32`, `float64`).
- `transform`: 4×4 affine transformation matrix of the NIfTI file.
- `diff_trans_f_norm`: For entries sharing the same `seq` value, this metric represents the Frobenius norm (F-norm) of the differential affine transformation matrix between the volume (`v3d`) and mask (`m3d`) files, calculated as `m3d−v3d`. This value is recorded **exclusively for mask (`m3d`) files**. An F-norm of 0 indicates full spatial consistency between the corresponding volume and mask files.
- `birth_date`: Patient's date of birth, formatted as `YYYYMMDD`.
- `sex`: Patient's biological sex (M = Male, F = Female).
- `age`: Patient's age at the time of examination, formatted as a 3-digit number followed by **Y** (e.g., `062Y`), or marked as `UND` (undefined).
- `manufacturer_model_name`: Scanner model used for the examination (e.g., `Brilliance16`, `Aquilion ONE`).
- `manufacturer`: Manufacturer of the scanning equipment (e.g., `Philips`, `TOSHIBA`, `SIEMENS`).
- `acquisition_date`: Date of examination acquisition, formatted as `YYYYMMDD`.
- `site`: Clinical site where the examination was acquired.

### 3️⃣Confirm and reorganize volume-mask pairs: Run routine 02

```sh
python AMOS-CT/02_amos_ct_confirm_pairs.py --root_dir root_dir/archive --output_dir root_dir/grouped/AMOS-CT --archive_manifest root_dir/descriptor/archive_manifest.xlsx
python AMOS-MRI/02_amos_mri_confirm_pairs.py --root_dir root_dir/archive --output_dir root_dir/grouped/AMOS-MRI --archive_manifest root_dir/descriptor/archive_manifest.xlsx
```

Routine 02 consists of two scripts, corresponding to the AMOS-CT and AMOS-MRI sub-datasets respectively. Both scripts load the archive manifest file to retrieve dataset metadata. Specifically:

- `AMOS-CT/02_amos_ct_confirm_pairs.py` filters samples to include only those with a `manufacturer_model_name` in the set {`Aquilion ONE`, `Brilliance16`, `SOMATOM Force`, `Optima CT660`, `Optima CT540`} (all CT scanner models).
- `AMOS-MRI/02_amos_mri_confirm_pairs.py` filters samples to include only those with a `manufacturer_model_name` in the set {`Ingenia`, `Prisma`, `SIGNA HDe`, `Achieva`} (all MRI scanner models).

Both scripts scan NIfTI `nii.gz` files in the `root_dir/archive` directory, filter relevant files, and pair each 3D volume file with its corresponding segmentation mask file to form sample pairs. If a mask file does not exist for a volume, only the volume file is retained.

Sub-dataset directories (`AMOS-CT` and `AMOS-MRI`) are created under `root_dir/grouped`, with three subset subdirectories (`train`, `val`, `test`) created under each sub-dataset directory. Within each subset directory, subdirectories are created for each `manufacturer_model_name`, and sample-level directories (formatted as `amos_{seq}_{sex}`) are nested within these scanner model-specific directories. Each sample directory contains the following files:

- `amos_{seq}_info.yaml`: Information file for the sample (`subset`, `seq`, `birth_date`, `sex`, etc).
- `amos_{seq}_volume.nii.gz`: 3D volume file (data type: `float32`), saved after volume metadata correction.
- `amos_{seq}_mask.nii.gz`: Segmentation mask file (data type: `uint8`), with metadata aligned to the corresponding volume file.

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - AMOS-CT
    - train
      - Aquilion ONE
        - amos_0004_F
          - amos_0004_info.yaml
          - amos_0004_volume.nii.gz
          - amos_0004_mask.nii.gz
        - amos_...
      - Brilliance16
      - SOMATOM Force
    - val
    - test
  - AMOS-MRI
    - train
      - Achieva
      - Prisma
        - amos_0507_M
          - amos_0507_info.yaml
          - amos_0507_volume.nii.gz
          - amos_0507_mask.nii.gz
        - amos_...
      - SIGNA HDe
    - val
    - test
```

### 4️⃣Split prostate/uterus label: Run routine 03

```sh
python 03_split_prostate_uterus_label.py --root_dir root_dir/grouped
```

Routine 03 recursively scans all sample directories within the `root_dir/grouped` directory, loads the `amos_{seq}_info.yaml` info file to retrieve the `sex` attribute, and inspects the multi-label segmentation mask file `amos_{seq}_mask.nii.gz`. For samples annotated with `sex=F` (Female), the label index 15 (originally mapped to `uterus`) is re-assigned to index 16 to differentiate it from label index 15 (mapped to `prostate`) in samples annotated with `sex=M` (Male). This script modifies mask files **in-place** without altering the existing folder structure or any other file attributes.

### 5️⃣Extract binary masks from multi-label mask files: Run routine 04

```sh
python 04_extract_binary_masks.py --root_dir root_dir/grouped --label_explanation root_dir/descriptor/label_map.yaml
```

Routine 04 recursively scans all NIfTI `nii.gz` mask files in the `root_dir/grouped` directory, and splits each multi-label segmentation mask into a set of binary segmentation masks. The label values in the original masks are defined in `label_map.yaml` as follows: 

> Format: `{label_index}` = `{label_name}`

- 0 = background
- 1 = spleen
- 2 = right kidney
- 3 = left kidney
- 4 = gall bladder
- 5 = esophagus
- 6 = liver
- 7 = stomach
- 8 = aorta
- 9 = postcava
- 10 = pancreas
- 11 = right adrenal gland
- 12 = left adrenal gland
- 13 = duodenum
- 14 = bladder
- 15 = prostate
- 16 = uterus

These binary masks are then saved in the same subdirectory as the original mask file, with the respective suffixes `{label_index:02d}_{label_name.replace(' ', '_')}` appended to the filenames.

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - AMOS-CT
    - train
      - Aquilion ONE
        - amos_0004_F
          - amos_0004_info.yaml
          - amos_0004_volume.nii.gz
          - amos_0004_mask.nii.gz
          - amos_0004_mask_00_background.nii.gz
          - amos_0004_mask_01_spleen.nii.gz
          - amos_0004_mask_02_right_kidney.nii.gz
          - amos_0004_mask_03_left_kidney.nii.gz
          - amos_0004_mask_04_gall_bladder.nii.gz
          - amos_0004_mask_05_esophagus.nii.gz
          - amos_0004_mask_06_liver.nii.gz
          - amos_0004_mask_07_stomach.nii.gz
          - amos_0004_mask_08_arota.nii.gz
          - amos_0004_mask_09_postcava.nii.gz
          - amos_0004_mask_10_pancreas.nii.gz
          - amos_0004_mask_11_right_adrenal_gland.nii.gz
          - amos_0004_mask_12_left_adrenal_gland.nii.gz
          - amos_0004_mask_13_duodenum.nii.gz
          - amos_0004_mask_14_bladder.nii.gz
          - amos_0004_mask_15_prostate.nii.gz
          - amos_0004_mask_16_uterus.nii.gz
        - amos_...
      - Brilliance16
      - SOMATOM Force
    - val
    - test
  - AMOS-MRI
    - ...
```

### [Additional] Mask denoising and quality optimization

```sh
python utils/small_connected_region_detect.py --root_dir root_dir/grouped --region_volume_thresh 300.0 --output_manifest root_dir/descriptor/small_connected_region_300_manifest.xlsx
python utils/small_connected_region_detect.py --root_dir root_dir/grouped --region_volume_thresh 100.0 --output_manifest root_dir/descriptor/small_connected_region_100_manifest.xlsx
```

`utils/small_connected_region_detect.py` detects small connected regions in binary mask files and reports their centroid coordinates and corresponding physical volumes. Only regions with a physical volume below a predefined threshold are included in the output. Based on empirical settings, we adopt **100 mm³** and **300 mm³** as thresholds to filter small 3D connected regions. Processing with threshold 300 mm³ is **optional**.

```sh
python utils/morphological_open_close_on_slice_small_region.py --root_dir root_dir/grouped --output_dir root_dir/grouped --kernel_size 5 5 --filter_halfedge 3 --region_area_thresh 10.0
```

`utils/morphological_open_close_on_slice_small_region.py` performs slice-wise morphological closing and opening operations on binary masks, then reconstructs the multi-label mask from the processed binary masks. Voxel label conflicts are resolved by quantifying the number of foreground voxels within a local filtering neighborhood. First, independent morphological closing and opening operations are applied to each original slice. Concurrently, connected component analysis is implemented on the foreground and background regions of each binary mask slice to identify small connected regions (< `region_area_thresh` mm²). Small foreground regions adopt the results of morphological opening (isolated island removal), small background regions adopt the results of morphological closing (hole filling), and all non-small regions retain the original slice content. For voxels with no assigned foreground labels or with multiple foreground labels after this processing step, the number of foreground support voxels within a square neighborhood of size [2×`filter_halfedge`+1, 2×`filter_halfedge`+1] (defined by `filter_halfedge`) is computed for each binary mask on the same slice; the valid label corresponding to the maximum count of support voxels is then assigned to the target voxel. A detailed procedural description is provided as follows:

Small connected region masks (isolated islands) for the **foreground** regions of the binary mask corresponding to each label: 17 labels map to 17 individual binary masks.
$$
SC^i_{m+} = \operatorname{SmallComp_{<10mm^2}} \left(mask_i\right),~ i \in \{0,1,...,16\}
$$
Small connected region masks (holes) for the **background** regions of the binary mask corresponding to each label:
$$
SC^i_{m-} = \operatorname{SmallComp_{<10mm^2}} \left(\neg mask_i\right),~ i \in \{0,1,...,16\}
$$
Non-small connected region masks of the binary mask corresponding to each label (these large regions remain unmodified):
$$
SC^i_{res} = \neg(SC^i_{m+} ~|~ SC^i_{m-})
$$
Morphological opening and closing operations are executed using a 5×5 all-ones structural element. Large regions are preserved, with the optimized mask derived by fusing the **foreground** after morphological opening (isolated island suppression) and the **background** after morphological closing (hole filling):
$$
\operatorname{Optimize} \left(mask_i\right) = SC^i_{res} \odot mask_i + SC^i_{m+} \odot \operatorname{Open_{5×5}} \left(mask_i\right) + SC^i_{m-} \odot \operatorname{Close_{5×5}} \left(mask_i\right)
$$
Label index sequence across the 17 labels for each individual voxel at spatial position (x,y,z):
$$
\operatorname{Idx} \left(x,y,z\right) = \left[mask_0\left(x,y,z\right),mask_1\left(x,y,z\right),...,mask_{16}\left(x,y,z\right)\right]
$$
The 17 processed binary masks are fused into a single multi-label mask, where each voxel is assigned a value corresponding to its label index in the range of 0 to 16.

- If the label index sequence is one-hot encoded, the index of the component with a value of 1 is designated as the voxel's mask value.
- If the label index sequence is **not** one-hot encoded, two distinct cases are addressed:
  1. If the label index sequence is a zero vector, this denotes that the voxel has no valid foreground labels post-processing. We calculate the number of foreground voxels (support voxels) within a 7×7 local neighborhood across all binary masks, and assign the label index with the highest count of support voxels to the voxel (equivalent to a modal value selection).
  2. If multiple components in the label index sequence equal 1, this indicates label overlap (voxel ambiguity) at the target position post-processing. We calculate the number of foreground voxels (support voxels) within a 7×7 local neighborhood for each binary mask with a value of 1 at this voxel, and assign the label index with the highest count of support voxels to resolve the ambiguity.

$$
mask\left(x,y,z\right) = \begin{cases}
   \operatorname{Argmax}\left(\operatorname{Idx} \left(x,y,z\right)\right) &\text{if } \operatorname{Idx} \left(x,y,z\right) \text{ is one-hot} \\
   \operatorname{Argmax}\left(\left[\operatorname{Sum^{7×7}_0}\left(x,y,z\right),\operatorname{Sum^{7×7}_1}\left(x,y,z\right),...,\operatorname{Sum^{7×7}_{16}}\left(x,y,z\right)\right]\right) &\text{if } \operatorname{Idx} \left(x,y,z\right) \text{ is all-zero} \\
   \operatorname{Argmax}\left(\left[\operatorname{Sum^{7×7}_0}\left(x,y,z\right),\operatorname{Sum^{7×7}_1}\left(x,y,z\right),...,\operatorname{Sum^{7×7}_{16}}\left(x,y,z\right)\right] \odot \operatorname{Idx} \left(x,y,z\right)\right) &\text{otherwise (multiple 1 comp.)} \\
   \end{cases}
$$

**Note** that this segmentation mask rectification procedure is not **PERFECT**, which may erroneously fill bona fide holes and remove valid isolated regions, as no single algorithm can comprehensively address all labeling artifacts. Nevertheless, this correction procedure reduced the total number of small 3D connected regions (< 100 mm³) from 8768 to 443 (indicated by `root_dir/descriptor/small_connected_region_100_manifest.xlsx` above and `root_dir/descriptor/small_connected_region_100_after_modification_manifest.xlsx` below) — a substantial improvement that indicates the remediation of extensive labeling errors and omissions in the original masks.

```sh
python utils/small_connected_region_detect.py --root_dir root_dir/grouped --region_volume_thresh 100.0 --output_manifest root_dir/descriptor/small_connected_region_100_after_modification_manifest.xlsx
```

You may re-run `utils/small_connected_region_detect.py` with a threshold of **100 mm³** to verify the mask state after denoising and quality optimization. This step is **optional**.

### 6️⃣Generate comprehensive manifest for dataset: Run routine 05

```sh
python 05_gen_dataset_manifest.py --root_dir root_dir/grouped --output_manifest_file root_dir/grouped/dataset_manifest.xlsx
```

Routine 05 recursively scans all NIfTI `nii.gz` files in the `root_dir/grouped` directory. For each sample in the subdirectories, it collates the corresponding volume file, multi-label mask file and individual binary mask files, then generates a manifest file named `dataset_manifest.xlsx` that indexes all files pertaining to each individual sample. Note that samples sourced from the test set contain only a volume file (no corresponding mask files).

This routine raises an error if inconsistencies are detected in the **spatial size, affine transformation matrix, voxel spacing, or origin coordinates** across all NIfTI files for a single sample. An error is also raised if the data types of all mask NIfTI files associated with the same sample are mismatched.

```sh
python 05_gen_dataset_manifest.py --root_dir root_dir/grouped/AMOS-CT --output_manifest_file root_dir/grouped/dataset_manifest_amos_ct.xlsx
python 05_gen_dataset_manifest.py --root_dir root_dir/grouped/AMOS-MRI --output_manifest_file root_dir/grouped/dataset_manifest_amos_mri.xlsx
```

The AMOS22 dataset can be further partitioned into two standalone sub-datasets: AMOS-CT and AMOS-MRI. Routine 05 is additionally executed for each sub-dataset to generate dedicated index manifests, with the root directories specified as `root_dir/grouped/AMOS-CT` and `root_dir/grouped/AMOS-MRI` respectively.

Sample files for the whole AMOS22 dataset, as well as the AMOS-CT and AMOS-MRI sub-datasets, are documented in dedicated manifest files (named with the format `dataset_manifest*.xlsx`) under their respective root directories. Each manifest file contains a primary worksheet titled **Manifest**, with the following attributes included:

- `ID`: Sample identifier, formatted as `amos_{seq}` (e.g., `amos_0001`).

- `seq`: AMOS unique identifier, represented as a 4-digit numerical value (e.g., 0001).

- `sex`: Patient's biological sex (M = Male, F = Female).

- `subset`: Dataset split label (`train`, `val`, or `test`), consistent with the split definitions recorded in `dataset.json`.

- `manufacturer_model_name`: Scanner model used for the examination (e.g., `Brilliance16`, `Aquilion ONE`).

- `manufacturer`: Manufacturer of the scanning equipment (e.g., `Philips`, `TOSHIBA`, `SIEMENS`).

- `birth_date`: Patient's date of birth, formatted as `YYYYMMDD`.

- `age`: Patient's age at the time of examination, formatted as a 3-digit number followed by **Y** (e.g., `062Y`), or marked as `UND` (undefined).

- `acquisition_date`: Date of examination acquisition, formatted as `YYYYMMDD`.

- `site`: Clinical site where the examination was acquired.

- `valid_labels`: All label indexes the mask contains.

- `info`: Sample information YAML file path relative to `root_dir`.

- `volume`: Volume NIfTI `nii.gz` file path relative to `root_dir`.

- `mask`: Multi-label mask NIfTI `nii.gz` file path relative to `root_dir`.

- `mask_{label_index}_{label_name}`: NIfTI `nii.gz` file path relative to `root_dir`, representing binary mask with `label_index` `label_name`.

- `mask_{label_index}_{label_name}_existence`: Existence status of the corresponding ROI. Value definition: `0` = no such ROI present in the mask, `1` = ROI present, `<empty>` = mask file is unavailable.

- `szx`: Volume size of X dimension for all NIfTI files associated with this sample.

- `szy`: Volume size of Y dimension for all NIfTI files associated with this sample.

- `szz`: Volume size of Z dimension for all NIfTI files associated with this sample.

- `spx`: Spacing (unit: mm) of X dimension for all NIfTI files associated with this sample.

- `spy`: Spacing (unit: mm) of Y dimension for all NIfTI files associated with this sample.

- `spz`: Spacing (unit: mm) of Z dimension for all NIfTI files associated with this sample.

- `orientation_from`: Orientation code for all NIfTI files associated with this sample from the start side, defined under the L-R (Left-Right) / P-A (Posterior-Anterior) / I-S (Inferior-Superior) coordinate system, may be oblique (e.g., `RPI`, `Oblique(closest to LPI)`).

- `orientation_to`: Orientation code for all NIfTI files associated with this sample from the start side, defined under the L-R / P-A / I-S coordinate system, may be oblique (e.g., `LAS`, `Oblique(closest to RAS)`).

- `vol_dtype`: Data type of the elements within the volume NIfTI file (e.g., `float32`).

  > Note: The data type of volumes is standardized to `float32` for all samples in `grouped`.

- `mask_dtype`: Data type of the elements within all mask NIfTI files associated with this sample (e.g., `uint8`).

  > Note: The data type of multi-label masks and binary masks is standardized to `uint8` for all samples in `grouped`.

- `transform`: Affine transformation matrix (4×4) for all NIfTI files associated with this sample.

### 7️⃣Split dataset and derive worksheet in dataset manifest: Run routine 06

```sh
python 06_make_split01_standard.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split01_standard
python 06_make_split01_standard.py --manifest_file root_dir/grouped/dataset_manifest_amos_ct.xlsx --split_name split01_standard
python 06_make_split01_standard.py --manifest_file root_dir/grouped/dataset_manifest_amos_mri.xlsx --split_name split01_standard
```

This script can be utilized to create a dedicated worksheet named **split01_standard** in the `dataset_manifest*.xlsx` Excel manifest file for the whole AMOS22 dataset, as well as the AMOS-CT and AMOS-MRI sub-datasets respectively, where the official training/validation/test split assignments are stored. It copies all attributes from the primary **Manifest** worksheet, appends a new attribute column titled `split01_standard`, and assigns a label of `train`, `val`, or `test` to each entry according to the `subset` attribute — this corresponds to the official standard dataset split provided by the original AMOS22 data archive.

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
  - AMOS-CT  # sample directories are stored here 
  - AMOS-MRI  # sample directories are stored here 
```

You may modify the `--subsets` option by adding, removing, or revising subset values to control the generation of subset-specific index manifests. Note that specifying a subset (e.g., `--subsets val`) for which no corresponding `val` label exists in the target split worksheet will result in an overall failure, where all extraction tasks are aborted. You may refer to error messages for more information.

Routine 07 is executed analogously for the AMOS-CT and AMOS-MRI sub-datasets with the following commands:

```sh
python 07_extract_split_index_manifest.py --manifest_file root_dir/grouped/AMOS-CT/dataset_manifest_amos_ct.xlsx --split_name split01_standard --output_index_manifest_dir root_dir/grouped/AMOS-CT/splits/split01_standard --subsets train val --subsets train --subsets val --subsets test
python 07_extract_split_index_manifest.py --manifest_file root_dir/grouped/AMOS-MRI/dataset_manifest_amos_mri.xlsx --split_name split01_standard --output_index_manifest_dir root_dir/grouped/AMOS-MRI/splits/split01_standard --subsets train val --subsets train --subsets val --subsets test
```

This generates the corresponding directory structures for the AMOS-CT and AMOS-MRI sub-datasets as shown below:

```sh
root_dir
- grouped
  - AMOS-CT  # sub-dataset
    - splits
      - split01_standard
        - split01_standard_train_val.xlsx
        - split01_standard_train.xlsx
        - split01_standard_val.xlsx
        - split01_standard_test.xlsx
    - train
    - val
    - test
  - AMOS-MRI  # sub-dataset
    - splits
      - split01_standard
        - split01_standard_train_val.xlsx
        - split01_standard_train.xlsx
        - split01_standard_val.xlsx
        - split01_standard_test.xlsx
    - ...
```

## End

You may now utilize the data files, dataset manifest, and subset-specific index manifests located in `root_dir/grouped` for downstream tasks, such as neural network training and model evaluation.

The whole AMOS22 dataset can be further partitioned into two standalone sub-datasets: AMOS-CT and AMOS-MRI, which are stored in `root_dir/grouped/AMOS-CT` and `root_dir/grouped/AMOS-MRI` respectively.

**Important Note for AMOS-MRI**: The labels `14-bladder`, `15-prostate`, and `16-uterus` may be **excluded from segmentation tasks to ensure stability**, given that the number of samples annotated with these labels is extremely limited. Specifically, the training set contains only 2 samples with the `14-bladder` region, 1 sample with `15-prostate` (compounded by severe labeling errors), and 1 sample with `16-uterus`. No annotated samples for these three labels are available in the validation set for model evaluation purposes.