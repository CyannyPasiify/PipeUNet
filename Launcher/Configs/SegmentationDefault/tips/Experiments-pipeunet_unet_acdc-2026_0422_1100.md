# Experiments-pipeunet_unet_acdc-2026_0422_1100

## Detect

```sh
-r Experiments
-e pipeunet_unet_acdc
-v detect-2026_0422_1100
--accelerator gpu
--devices 9
--deterministic warn
--wandb_project PipeUNet
--dataset_root_dir Storage/ACDC/grouped
--dataset_manifest_file Storage/ACDC/grouped/splits/split02_t8v2s/split02_t8v2s_train.xlsx
--volume_keys volume
--mask_keys mask_00_Bg mask_01_RV mask_02_Myo mask_03_LV
--batch_size 2
--num_sequence 1
--num_classes 4
detect
--val_dataset_root_dir Storage/ACDC/grouped
--val_dataset_manifest_file Storage/ACDC/grouped/splits/split02_t8v2s/split02_t8v2s_val.xlsx
--val_volume_keys volume
--val_mask_keys mask_00_Bg mask_01_RV mask_02_Myo mask_03_LV
--val_batch_size 1
```

## Fit

```sh
-r Experiments
-e pipeunet_unet_acdc
-v fit-2026_0422_1100
--accelerator gpu
--devices 9
--deterministic warn
--wandb_project PipeUNet
--dataset_root_dir Storage/ACDC/grouped
--dataset_manifest_file Storage/ACDC/grouped/splits/split02_t8v2s/split02_t8v2s_train.xlsx
--volume_keys volume
--mask_keys mask_00_Bg mask_01_RV mask_02_Myo mask_03_LV
--batch_size 2
--num_sequence 1
--num_classes 4
fit
--epochs 200
--accumulate_grad_batches 16
--val_dataset_root_dir Storage/ACDC/grouped
--val_dataset_manifest_file Storage/ACDC/grouped/splits/split02_t8v2s/split02_t8v2s_val.xlsx
--val_volume_keys volume
--val_mask_keys mask_00_Bg mask_01_RV mask_02_Myo mask_03_LV
--val_batch_size 1
--max_lr 0.01
--steps_per_epoch 80
--final_div_factor 1e4
--roi_size 128 128 128
--sw_batch_size 4
--overlap 0.5
```

## Test

```sh
-r Experiments
-e pipeunet_unet_acdc
-v test-2026_0422_1100
--accelerator gpu
--devices 9
--deterministic warn
--wandb_project PipeUNet
--dataset_root_dir Storage/ACDC/grouped
--dataset_manifest_file Storage/ACDC/grouped/splits/split02_t8v2s/split02_t8v2s_test.xlsx
--volume_keys volume
--mask_keys mask_00_Bg mask_01_RV mask_02_Myo mask_03_LV
--batch_size 1
--num_sequence 1
--num_classes 4
```

