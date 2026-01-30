# NME-2025.8.25 Data Preprocess Tutorial

## Steps

### 1️⃣Download NME-2025.8.25 data archive from [this link (Baidu Drive)](https://pan.baidu.com/s/1ZRMRcmym64zsGWPsJ1fitw)

This is a private data archive. You may email the maintainer for password.

Place the `archive` into an empty `root_dir`, e.g., `root_dir=/path/to/NME-2025.8.25 `.

### 2️⃣Scan NME-2025.8.25 archive and manual check metadatas: Run routine 01

```sh
python 01_gen_archive_manifest.py --root_dir root_dir/archive --output_manifest_file root_dir/descriptor/archive_manifest.xlsx
```

Routine 01 recursively scans all NIfTI files with the `.nii.gz` extension in the `root_dir/archive` directory, filters out valid 3D volume files and segmentation mask files, and finally generates a manifest file named `archive_manifest.xlsx` in the newly created `descriptor` subdirectory under `root_dir`.

All filtered NIfTI files are recorded in `archive_manifest.xlsx`. The file contains a single worksheet titled **Manifest**, which includes the following attributes:

- `file_path`: NIfTI `nii.gz` file path relative to `root_dir`. 

- `site`: Clinical site where the examination was acquired.
- `collection`: The collection directory of the sample sourced from the data archive (e.g., `train-val`, `test-inner-01`, `test-outer-*`).
- `subset`: The subset directory to which the sample belongs; valid values follow the format `(images|labels)(Tr|Vd|Ts)`.
- `pid`: Patient identifier.
- `type`: Content type the file expressed. There are 2 options: volume 3D (v3d), mask 3D (m3d).
- `szx`: Volume dimension size along the X-axis.
- `szy`: Volume dimension size along the Y-axis.
- `szz`: Volume dimension size along the Z-axis.
- `spx`: Voxel spacing along the X-axis (unit: mm).
- `spy`: Voxel spacing along the Y-axis (unit: mm).
- `spz`: Voxel spacing along the Z-axis (unit: mm).
- `orientation_from`: Orientation code for the start side, defined under the L-R (Left-Right) / P-A (Posterior-Anterior) / I-S (Inferior-Superior) coordinate system (e.g., `RAS`).
- `orientation_to`: Orientation code for the end side, defined under the L-R / P-A / I-S coordinate system, may be oblique (e.g., `LPI`).
- `dtype`: Data type of the elements within the volume NIfTI file (e.g., `uint8`, `float64`).
- `transform`: Affine transformation matrix (4×4) of the NIfTI file.
- `diff_trans_f_norm`: For items with same `patient` and `frame` attributes, this value represents the Frobenius norm (F-norm) of the differential affine transformation matrix between the volume (v3d) and mask (m3d), i.e., \(m3d - v3d\). This metric is exclusively recorded for the mask (m3d). An F-norm of 0 indicates full consistency between the volume and mask files.

### 3️⃣Confirm and reorganize volume-mask pairs: Run routine 02

```sh
python 02_confirm_pairs.py --root_dir root_dir/archive --output_dir root_dir/grouped --archive_manifest root_dir/descriptor/archive_manifest.xlsx
```

Routine 02 recursively scans all NIfTI `.nii.gz` files in the `root_dir/archive` directory, filters for valid files, and pairs each 3D volume file with its corresponding ground truth (gt) mask file to form valid sample pairs.

Four site-specific directories — `Tongji`, `Yunnan`, `Zhongnan`, and `HKU-SZH` — are created under `root_dir/grouped`; these directories are dedicated to storing derived files sourced from `root_dir/archive`.

- `Tongji`: Sourced from `root_dir/archive/NME-Seg-2025.8.25-train-val` and `root_dir/archive/NME-Seg-2025.8.25-test-inner-01`
- `Yunnan`: Sourced from `root_dir/archive/NME-Seg-2025.8.25-test-outer-02`
- `Zhongnan`: Sourced from `root_dir/archive/NME-Seg-2025.8.25-test-outer-03`
- `HKU-SZH`: Sourced from `root_dir/archive/NME-Seg-2025.8.25-test-outer-04`

Subsequently, collection directories `train-val`, `test-inner-01`, and `test-outer-*` are created under each site directory. Nested within each collection directory are subset directories `train`, `val`, and `test`, which are mapped from the suffixes of the original collection directory names in `root_dir/archive/NME-Seg-2025.8.25-*` — with `Tr` mapped to `train`, `Vd` to `val`, and `Ts` to `test`. Sample directories formatted as `{seq}_{pid}` are then created at the lowest level of each subset directory.

The processed files for each sample are saved with the following naming conventions and specifications:

- Info file: `{seq}_{pid}_info.yaml`
- Volume file: `{seq}_{pid}_volume.nii.gz` (data type: `float32`), saved following metadata correction
- Mask file: `{seq}_{pid}_mask.nii.gz` (data type: `uint8`), with metadata aligned to the corresponding volume file

After processing, the resulting directory structure is as follows:

```sh
root_dir
- grouped
  - Tongji
    - train-val
      - train
        - 01A002_60012875311
          - 01A002_60012875311_info.yaml
          - 01A002_60012875311_volume.nii.gz
          - 01A002_60012875311_mask.nii.gz
      - val
    - test-inner-01
      - test
  - Yunnan
    - test-outer-02
      - test
  - Zhongnan
    - test-outer-03
      - test
  - HKU-SZH
    - test-outer-04
      - test
```

### 4️⃣Extract binary masks from multi-label mask files: Run routine 03

```sh
python 03_extract_binary_masks.py --root_dir root_dir/grouped --label_explanation root_dir/descriptor/label_map.yaml
```

Routine 03 recursively scans all NIfTI `{seq}_{pid}_mask.nii.gz` mask files in the `root_dir/grouped` directory, and splits each multi-label segmentation mask into a set of binary segmentation masks. The label values in the original masks are defined in `label_map.yaml` as follows: 

> Format: `{label_index}` = `{label_name}` (`short_form`)

- 0 = background (Bg)
- 1 = non mass enhancement (NME)

These binary masks are then saved in the same subdirectory as the original mask file, with the respective suffixes `{label_index:02d}_{label_name.short_form}` appended to the filenames.

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - Tongji
    - train-val
      - train
        - 01A002_60012875311
          - 01A002_60012875311_info.yaml
          - 01A002_60012875311_volume.nii.gz
          - 01A002_60012875311_mask.nii.gz
          - 01A002_60012875311_mask_00_Bg.nii.gz
          - 01A002_60012875311_mask_01_NME.nii.gz
      - val
    - test-inner-01
      - test
  - Yunnan
    - test-outer-02
      - test
  - Zhongnan
    - test-outer-03
      - test
  - HKU-SZH
    - test-outer-04
      - test
```

### 5️⃣Generate comprehensive manifest for dataset: Run routine 04

```sh
python 04_gen_dataset_manifest.py --root_dir root_dir/grouped --output_manifest_file root_dir/grouped/dataset_manifest.xlsx
```

Routine 04 recursively scans all NIfTI `nii.gz` files in the `root_dir/grouped` directory. For each sample in the subdirectories, it collates the corresponding volume file, multi-label mask file and individual binary mask files, then generates a manifest file named `dataset_manifest.xlsx` that indexes all files pertaining to each individual sample.

This routine raises an error if inconsistencies are detected in the **spatial size, affine transformation matrix, voxel spacing, or origin coordinates** across all NIfTI files for a single sample. An error is also raised if the data types of all mask NIfTI files associated with the same sample are mismatched.

Sample files for the NME-2025.8.25 dataset is documented in manifest file `grouped/dataset_manifest.xlsx`. The manifest file contains a primary worksheet titled **Manifest**, with the following attributes included:

- `ID`: Sample identifier, formatted as `{seq}_{pid}` (e.g., `01A001_1004091054`).

- `site`: Clinical site where the examination was acquired.

- `collection`: The collection directory of the sample sourced from the data archive (e.g., `train-val`, `test-inner-01`, `test-outer-*`).

- `subset`: The subset directory to which the sample belongs. Values from the data archive will be reformat from `*(Tr|Vd|Ts)` as `train|val|test`.

- `pid`: Patient identifier.

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

- `orientation_from`: Orientation code for all NIfTI files associated with this sample from the start side, defined under the L-R (Left-Right) / P-A (Posterior-Anterior) / I-S (Inferior-Superior) coordinate system, may be oblique (e.g., `RAS`).

- `orientation_to`: Orientation code for all NIfTI files associated with this sample from the start side, defined under the L-R / P-A / I-S coordinate system, may be oblique (e.g., `LPI`).

- `vol_dtype`: Data type of the elements within the volume NIfTI file (e.g., `float32`).

  > Note: The data type of volumes is standardized to `float32` for all samples in `grouped`.

- `mask_dtype`: Data type of the elements within all mask NIfTI files associated with this sample (e.g., `uint8`).

  > Note: The data type of multi-label masks and binary masks is standardized to `uint8` for all samples in `grouped`.

- `transform`: Affine transformation matrix (4×4) for all NIfTI files associated with this sample.

### 6️⃣Split dataset and derive worksheet in dataset manifest: Run routine 05

```sh
python 05_make_split_predefined.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split01_Tongji --site Tongji --collection train-val test-inner-01
python 05_make_split_predefined.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split02_Yunnan --site Yunnan --collection test-outer-02
python 05_make_split_predefined.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split03_Zhongnan --site Zhongnan --collection test-outer-03
python 05_make_split_predefined.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split04_HKU-SZH --site HKU-SZH --collection test-outer-04
```

This script generates subset splits for samples from each `site` and `collection`. It appends the **split01_Tongji**, **split02_Yunnan**, **split03_Zhongnan**, and **split04_HKU-SZH** worksheets to the `grouped/dataset_manifest.xlsx` Excel manifest file, where the train/val/test split assignments are stored in accordance with the original `subset` attribute.

```sh
python 05_make_split_predefined.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split05_External --site Yunnan Zhongnan HKU-SZH --collection test-outer-02 test-outer-03 test-outer-04
```

**split05_External** is a split that allocates all samples from `Yunnan`, `Zhongnan`, and `HKU-SZH` to a single test set — this split is a combination of **split02_Yunnan**, **split03_Zhongnan**, and **split04_HKU-SZH**. The script appends the **split05_External** worksheet to the `grouped/dataset_manifest.xlsx` Excel manifest file, which stores the split assignments for the combined external site subset (consisting solely of the test set).

```sh
python 05_make_split_predefined.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split06_Whole --site Tongji Yunnan Zhongnan HKU-SZH --collection train-val test-inner-01 test-outer-02 test-outer-03 test-outer-04
```

**split06_Whole** is a full-dataset split for all site samples, combining **split01_Tongji** and **split05_External**. For this split, samples from the `test-inner-01` collection and all external `test-outer-*` collections are merged into a single unified test set.

```sh
python 05_make_split_across_sites.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split07_Mix --test_ratio 0.3 --val_ratio 0.2 --random_seed 0
```

This script generates a cross-site subset split for all samples across all sites. **split07_Mix** is derived from all site samples: it directly replicates `split01_Tongji` for the Tongji site, and splits samples from all other sites following the two-stage ratios `train+val:test=7:3` (first stage) and `train:val=8:2` (second stage), with the internal/external site definitions disregarded. The script appends the **split07_Mix** worksheet to the `grouped/dataset_manifest.xlsx` Excel manifest file, which stores the split assignments for the entire dataset.

### 7️⃣Extracts subset-specific index manifests from dataset manifest: Run routine 06

Given a split-specific worksheet in the dataset manifest file, Routine 06 extracts subset-specific index manifests based on split labels (e.g., `train`, `val`, `test`). Separate Excel files can be generated for index manifests of individual subsets (`train`, `val`, `test`) or combined subsets (e.g., `train+val`); for example, the training index manifest contains only entries labeled `train` in the target split worksheet.

Prior to running Routine 06, ensure the intended split is included as a dedicated worksheet in the dataset manifest file. For instance, if `root_dir/grouped/dataset_manifest.xlsx` contains the worksheet `split01_Tongji`, the option `--split_name split01_Tongji` can be specified when executing Routine 06 to generate the corresponding subset-specific index manifests.

The following execution of Routine 06 generates index manifests for `split01_Tongji`. The manifest files `split01_Tongji_train_val.xlsx`, `split01_Tongji_train.xlsx`, `split01_Tongji_val.xlsx`, and `split01_Tongji_test.xlsx` are created in the directory `root_dir/grouped/splits/split01_Tongji`.

```sh
python 06_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split01_Tongji --output_index_manifest_dir root_dir/grouped/splits/split01_Tongji --subsets train val --subsets train --subsets val --subsets test
```

For all other splits, launch the script using the same method as follows:

```sh
python 06_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split02_Yunnan --output_index_manifest_dir root_dir/grouped/splits/split02_Yunnan --subsets test
python 06_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split03_Zhongnan --output_index_manifest_dir root_dir/grouped/splits/split03_Zhongnan --subsets test
python 06_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split04_HKU-SZH --output_index_manifest_dir root_dir/grouped/splits/split04_HKU-SZH --subsets test
python 06_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split05_External --output_index_manifest_dir root_dir/grouped/splits/split05_External --subsets test
python 06_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split06_Whole --output_index_manifest_dir root_dir/grouped/splits/split06_Whole --subsets train val --subsets train --subsets val --subsets test
python 06_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split07_Mix --output_index_manifest_dir root_dir/grouped/splits/split07_Mix --subsets train val --subsets train --subsets val --subsets test
```

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - splits
    - split01_Tongji
      - split01_Tongji_train_val.xlsx
      - split01_Tongji_train.xlsx
      - split01_Tongji_val.xlsx
      - split01_Tongji_test.xlsx
    - split02_Yunnan
      - split02_Yunnan_test.xlsx
    - split03_Zhongnan
      - split03_Zhongnan_test.xlsx
    - split04_HKU-SZH
      - split04_HKU-SZHtest.xlsx
    - split05_External
      - split05_External_test.xlsx
    - split06_Whole
      - split06_Whole_train_val.xlsx
      - split06_Whole_train.xlsx
      - split06_Whole_val.xlsx
      - split06_Whole_test.xlsx
    - split07_Mix
      - split07_Mix_train_val.xlsx
      - split07_Mix_train.xlsx
      - split07_Mix_val.xlsx
      - split07_Mix_test.xlsx
```

You may modify the `--subsets` option by adding, removing, or revising subset values to control the generation of subset-specific index manifests. Note that specifying a subset (e.g., `--subsets val`) for which no corresponding `val` label exists in the target split worksheet will result in an overall failure, where all extraction tasks are aborted. You may refer to error messages for more information.

## End

You may now utilize the data files, dataset manifest, and subset-specific index manifests located in `root_dir/grouped` for downstream tasks, such as neural network training and model evaluation.
