#!/bin/bash
# Envs
export WANDB_API_KEY=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

# Basic Dirs
StorageRoot=/mnt/data4/cwy/storage
ProjectRoot=${StorageRoot}/projects
PipeUNetRoot=${ProjectRoot}/PipeUNet
LauncherScript=${PipeUNetRoot}/Launcher/launcher_segmentation_default.py
ExperimentRoot=${PipeUNetRoot}/Experiments

# Experiment Specific (Modify These!)
NetName=PipeUNet_UNet
DatasetName=ACDC
SplitName=split02_t8v2s
ExperimentName=${NetName}_${DatasetName}_${SplitName}
ExperimentTime=2026_0423_1700
export CUDA_VISIBLE_DEVICES=9

# Usage：./launch.sh fit
if [ $# -ne 1 ]; then
    echo "Argument error! Please give 1 arg：fit"
    echo "Example：$0 fit"
    exit 1
fi

RunMode="$1"

if [ "${RunMode}" != "fit" ]; then
    echo "Argument error! Only support [fit], given: ${RunMode}"
    exit 1
fi

# Other Derived
ExperimentVersion=${ExperimentTime}-${RunMode}
DatasetRoot=${StorageRoot}/datasets/${DatasetName}/grouped
TrainManifest=${DatasetRoot}/splits/${SplitName}/${SplitName}_train.xlsx
ValManifest=${DatasetRoot}/splits/${SplitName}/${SplitName}_val.xlsx
LogDir=${ExperimentRoot}/${ExperimentName}/${ExperimentVersion}
LogFile=${LogDir}/log-${ExperimentName}-${ExperimentVersion}.log

mkdir -p "$LogDir"

RoutineArgs=(
  -r "$ExperimentRoot"
  -e "$ExperimentName"
  -v "$ExperimentVersion"
  --accelerator gpu
  --devices 0
  --deterministic warn
  --wandb_project PipeUNet
  --dataset_root_dir "$DatasetRoot"
  --dataset_manifest_file "$TrainManifest"
  --volume_keys volume
  --mask_keys mask_00_Bg mask_01_RV mask_02_Myo mask_03_LV
  --roi_size 128 128 128
  --batch_size 1
  --num_sequence 1
  --num_classes 4
  fit
  --epochs 200
  --accumulate_grad_batches 10
  --val_dataset_root_dir "$DatasetRoot"
  --val_dataset_manifest_file "$ValManifest"
  --val_volume_keys volume
  --val_mask_keys mask_00_Bg mask_01_RV mask_02_Myo mask_03_LV
  --val_batch_size 1
  --max_lr 0.01
  --steps_per_epoch 16
  --final_div_factor 1e4
  --sw_batch_size 1
  --overlap 0.5
)

echo "All prepared, starting routine ..."
FullCmd="nohup python \"${LauncherScript}\" ${TRAIN_ARGS[*]} > \"${LogFile}\" 2>&1 &"
echo FullCmd

# Print current env
echo "Python env activated：${CONDA_PREFIX}"
which python
export PYTHONPATH=${PipeUNetRoot}
nohup python "${LauncherScript}" "${RoutineArgs[@]}" > "$LogFile" 2>&1 &

# Print Info
echo "Process is launched, logging at：${LogFile}"
echo "Process PID：$!"