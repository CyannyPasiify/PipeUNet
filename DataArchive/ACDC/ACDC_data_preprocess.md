# ACDC Data Preprocess Tutorial

## Steps

### 1️⃣Download ACDC data archive from [this link (Baidu Drive)](https://pan.baidu.com/s/1c0FPOKR4TWmmDk7_3Qs0Tg?pwd=523e)

Place the `archive` into an empty `root_dir`, e.g., `root_dir=/path/to/ACDC`.

### 2️⃣Scan ACDC archive and manual check metadatas: Run routine 01

```sh
python 01_gen_archive_manifest.py --root_dir root_dir/archive --output_manifest_file root_dir/descriptor/archive_manifest.xlsx
```

Routine 01 recursively scans all NIfTI `nii.gz` files in the directory `root_dir/archive`, filters valid 3D volume and ground truth (gt) mask files, and generates a manifest file named `archive_manifest.xlsx` in a newly created subdirectory `descriptor` under `root_dir`.

All filtered NIfTI files are recorded in `archive_manifest.xlsx`. The file contains a single worksheet titled **Manifest**, which includes the following attributes:

- `file_path`: NIfTI `nii.gz` file path relative to `root_dir`.
- `subset`: `training` or `testing` as the 1st level subdir.
- `patient`: Patient identifier, a 3-digit number, e.g., 001.
- `frame`: Frame time-point, a 2-digit number, e.g. 01.
- `type`: Content type the file expressed. There are 2 options: volume 3D (v3d), mask 3D (m3d).
- `phase`: Diastolic or systolic phase of the volume frame.
  - ED = End Diastolic phase
  - ES = End Systolic phase

- `group`: Subgroup of the patient (4 pathological plus 1 healthy subject groups).
  - NOR = normal subjects
  - MINF = patient with previous myocardial infarction (ejection fraction of the left ventricle lower than 40% and several myocardial segments with abnormal contraction)
  - DCM = patient with dilated cardiomyopathy (diastolic left ventricular volume >100 mL/m2 and an ejection fraction of the left ventricle lower than 40%) 
  - HCM = patient with hypertrophic cardiomyopathy (left ventricular cardiac mass high than 110 g/m2, several myocardial segments with a thickness higher than 15 mm in diastole and a normal ejecetion fraction) 
  - RV = patient with abnormal right ventricle (volume of the right ventricular cavity higher than 110 mL/m2 or ejection fraction of the rigth ventricle lower than 40%) 

- `tot_frame`: Total frames of the patient's 4D Cine-MRI.
- `height`: Height of the patient in cm.
- `weight`: Weight of the patient in kg.
- `szx`: Volume size of X dimension.
- `szy`: Volume size of Y dimension.
- `szz`: Volume size of Z dimension.
- `spx`: Spacing of X dimension.
- `spy`: Spacing of Y dimension.
- `spz`: Spacing of Z dimension.
- `orientation_from`: Orientation code from the start side under L-R P-A I-S system, e.g., LPS.
- `orientation_to`: Orientation code from the end side under L-R P-A I-S system, e.g., RAI.
- `dtype`: Data type of the elements within the NIfTI file, e.g., uint8, int16, float32.
- `transform`: Affine transformation matrix (4×4) of the NIfTI file.
- `diff_trans_f_norm`: For items with same `patient` and `frame` attributes, this value represents the Frobenius norm (F-norm) of the differential affine transformation matrix between the volume (v3d) and mask (m3d), i.e., \(m3d - v3d\). This metric is exclusively recorded for the mask (m3d). An F-norm of 0 indicates full consistency between the volume and mask files.

### 3️⃣Confirm and reorganize volume-mask pairs: Run routine 02

```sh
python 02_confirm_pairs.py --root_dir root_dir/archive --output_dir root_dir/grouped
```

Routine 02 recursively scans all NIfTI `nii.gz` files in the `root_dir/archive` directory, filters valid files, and pairs each 3D volume file with its corresponding ground truth (gt) mask file to form sample pairs.

Two subdirectories named `train` and `test` are created under `root_dir/grouped`, which are dedicated to storing derived files sourced from `root_dir/archive/training` and `root_dir/archive/testing`, respectively.

Tertiary subdirectories formatted as `patient{patient}_frame{frame}_{phase}_{group}` (e.g., `patient001_frame01`) are then generated under either `root_dir/grouped/train` or `root_dir/grouped/test`. Within each corresponding tertiary subdirectory:

- The info file is saved as `patient{patient}_frame{frame}_info.yaml`;
- The volume file is saved as `patient{patient}_frame{frame}_volume.nii.gz` (`dtype=float32`) after metadata correction;
- The mask file is saved as `patient{patient}_frame{frame}_mask.nii.gz ` (`dtype=uint8`), with metadata consistent with that of the volume file.

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - train
    - patient001_frame01_ED_DCM
      - patient001_frame01_volume.nii.gz
      - patient001_frame01_mask.nii.gz
    - ...
  - test
    - patient101_frame01_ED_DCM
    - ...
```

### 4️⃣Extract binary masks from multi-label mask files: Run routine 03

```sh
python 03_extract_binary_masks.py --root_dir root_dir/grouped --label_explanation Bg RV Myo LV
```

Routine 03 recursively scans all NIfTI `nii.gz` mask files in the `root_dir/grouped` directory, and splits each multi-label segmentation mask into a set of binary segmentation masks. The label values in the original masks are defined as follows: **0 = Background (Bg)**, **1 = Right Ventricle cavity (RV)**, **2 = Myocardium (Myo)**, and **3 = Left Ventricle cavity (LV)**. These binary masks are then saved in the same subdirectory as the original mask file, with the respective suffixes `label_value` with `Bg`, `RV`, `Myo`, and `LV` appended to the filenames.

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - train
    - patient001_frame01_ED_DCM
      - patient001_frame01_info.yaml
      - patient001_frame01_volume.nii.gz
      - patient001_frame01_mask.nii.gz
      - patient001_frame01_mask_00_Bg.nii.gz
      - patient001_frame01_mask_01_RV.nii.gz
      - patient001_frame01_mask_02_Myo.nii.gz
      - patient001_frame01_mask_03_LV.nii.gz
    - ...
```

### 5️⃣Generate comprehensive manifest for dataset: Run routine 04

```sh
python 04_gen_dataset_manifest.py --root_dir root_dir/grouped --output_manifest_file root_dir/grouped/dataset_manifest.xlsx
```

Routine 04 recursively scans all NIfTI `nii.gz` files in the `root_dir/grouped` directory. For each sample in the `train` and `test` subdirectories, it collates the volume file, multi-label mask file, and corresponding binary mask files, and generates a manifest file named `dataset_manifest.xlsx` that indexes all files associated with each individual sample.

This routine reports an error if the size, transformation matrix, spacing, or origin of all NIfTI files associated with a given sample are inconsistent, or if the data types of all mask NIfTI files associated with the same sample do not match.

All sample files are recorded in `dataset_manifest.xlsx`. The file contains a main worksheet titled **Manifest**, which includes the following attributes:

- `ID`: Sample identifier, formatted as `patient{patient}_frame{frame}` (e.g., `patient001_frame01`).
- `patient`: Patient identifier, a 3-digit number, e.g., 001.
- `frame`: Frame time-point, a 2-digit number, e.g. 01.
- `phase`: Diastolic or systolic phase of the volume frame.
  - ED = End Diastolic
  - ES = End Systolic
- `group`: Subgroup of the patient (4 pathological plus 1 healthy subject groups).
  - NOR = normal subjects
  - MINF = patient with previous myocardial infarction (ejection fraction of the left ventricle lower than 40% and several myocardial segments with abnormal contraction)
  - DCM = patient with dilated cardiomyopathy (diastolic left ventricular volume >100 mL/m2 and an ejection fraction of the left ventricle lower than 40%) 
  - HCM = patient with hypertrophic cardiomyopathy (left ventricular cardiac mass high than 110 g/m2, several myocardial segments with a thickness higher than 15 mm in diastole and a normal ejecetion fraction) 
  - RV = patient with abnormal right ventricle (volume of the right ventricular cavity higher than 110 mL/m2 or ejection fraction of the rigth ventricle lower than 40%) 
- `tot_frame`: Total frames of the patient's 4D Cine-MRI.
- `height`: Height of the patient in cm.
- `weight`: Weight of the patient in kg.
- `info`: Sample information YAML file path relative to `root_dir`.
- `volume`: Volume NIfTI `nii.gz` file path relative to `root_dir`.
- `mask`: Multi-label mask NIfTI `nii.gz` file path relative to `root_dir`.
- `mask_00_Bg`: **Background** binary mask NIfTI `nii.gz` file path relative to `root_dir`.
- `mask_01_RV`:  **Right Ventricle cavity** binary mask NIfTI `nii.gz` file path relative to `root_dir`.
- `mask_02_Myo`: **Myocardium** binary mask NIfTI `nii.gz` file path relative to `root_dir`.
- `mask_03_LV`: **Left Ventricle cavity** binary mask NIfTI `nii.gz` file path relative to `root_dir`.
- `szx`: Volume size of X dimension for all NIfTI files associated with this sample.
- `szy`: Volume size of Y dimension for all NIfTI files associated with this sample.
- `szz`: Volume size of Z dimension for all NIfTI files associated with this sample.
- `spx`: Spacing of X dimension for all NIfTI files associated with this sample.
- `spy`: Spacing of Y dimension for all NIfTI files associated with this sample.
- `spz`: Spacing of Z dimension for all NIfTI files associated with this sample.
- `orientation_from`: Orientation code for all NIfTI files associated with this sample from the start side under L-R P-A I-S system, e.g., LPS.
- `orientation_to`: Orientation code for all NIfTI files associated with this sample from the end side under L-R P-A I-S system, e.g., RAI.
- `vol_dtype`: Data type of the elements within the volume NIfTI file, e.g., int16, float32.
- `mask_dtype`: Data type of the elements within all mask NIfTI files associated with this sample, e.g., uint8, int16.
- `transform`: Affine transformation matrix (4×4) for all NIfTI files associated with this sample.

### 6️⃣Split dataset and derive worksheet in dataset manifest: Run routine 05

There are multiple ways to split this dataset.

```sh
python 05_make_split01_standard.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split01_standard
```

This script creates a worksheet named `split01_standard` in the manifest Excel file `dataset_manifest.xlsx`, where the train/test split information is stored. It copies all attributes from the primary **Manifest** worksheet, appends a new attribute column named `split01_standard`, and assigns a `train` or `test` label to each entry based on the top-level directory (`train` or `test`) in the corresponding volume file path.

```sh
python 05_make_split02_train_val_fixed_test.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split02_t8v2s --val_ratio 0.2 --random_seed 0
```

This script creates a new worksheet in the manifest Excel file to store train/val/test split information, which is determined by both the top-level directory of each volume file path and a predefined validation ratio. It copies all attributes from the primary **Manifest** worksheet and appends a new attribute column named `split02_t8v2s`. For samples in the top-level `train` directory, the script first groups samples by unique patient identifier `ID`, then performs a train/val split with a predefined validation ratio (`val_ratio`) of 0.2. This split operation is executed hierarchically based on the values of the `group` attribute. For samples in the top-level `test` directory, it assigns a `test` label to every corresponding entry.

### 7️⃣Extracts subset-specific index manifests from dataset manifest: Run routine 06

Given a split-specific worksheet within the dataset manifest file, Routine 06 extracts subset-specific index manifests based on the split labels (e.g., train, val, test). Index manifests for individual subsets (train, val, test) or combined subsets (e.g., train+val) can be generated as separate Excel files; for instance, the train index manifest contains only entries labeled as `train` in the target split worksheet.

Prior to running routine 06, ensure the desired split is included as a dedicated worksheet in the dataset manifest file. For example, if `root_dir/grouped/dataset_manifest.xlsx` contains the worksheet `split01_standard`, you may specify the option `--split_name split01_standard` when executing routine 06 to generate the corresponding subset-specific index manifests.

The following routine generates index manifests for the split `split01_standard`. Manifest files `split01_standard_train.xlsx` and `split01_standard_test.xlsx` are generated in the directory `root_dir/grouped/splits/split01_standard`.

```sh
python 06_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split01_standard --output_index_manifest_dir root_dir/grouped/splits/split01_standard --subsets train --subsets test
```

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - splits
    - split01_standard
      - split01_standard_train.xlsx
      - split01_standard_test.xlsx
```

The following routine generates index manifests for the split `split02_t8v2s`. Manifest files `split02_t8v2s_train_val.xlsx`, `split02_t8v2s_train.xlsx`, `split02_t8v2s_val.xlsx` and `split02_t8v2s_test.xlsx` are generated in the directory `root_dir/grouped/splits/split02_t8v2s`.

```sh
python 06_extract_split_index_manifest.py --manifest_file root_dir/grouped/dataset_manifest.xlsx --split_name split02_t8v2s --output_index_manifest_dir root_dir/grouped/splits/split02_t8v2s --subsets train val --subsets train --subsets val --subsets test
```

After processing, the directory structure is as follows:

```sh
root_dir
- grouped
  - splits
    - split02_t8v2s
      - split02_t8v2s_train_val.xlsx
      - split02_t8v2s_train.xlsx
      - split02_t8v2s_val.xlsx
      - split02_t8v2s_test.xlsx
```

You may modify the `--subsets` option by adding, removing, or revising subset values to control the generation of subset-specific index manifests. Note that specifying a subset (e.g., `--subsets val`) for which no corresponding `val` label exists in the target split worksheet will result in an overall failure, where all extraction tasks are aborted. You may refer to error messages for more information.

## End

You may now utilize the data files, dataset manifest, and subset-specific index manifests located in `root_dir/grouped` for downstream tasks, such as neural network training and model evaluation.