import torch
from torch import Tensor
import monai.transforms as mT
from typing import Dict, Optional, Any, Union, List, Tuple, Hashable, Literal
import pandas as pd
import pathlib as pl
from typing_extensions import override
from dataclasses import dataclass
from Operator.operator_configurer import ConfigOperatorBase, TLSeq, PathLike
from pandas._typing import DtypeArg
from abc import ABCMeta, abstractmethod


@dataclass
class ConfigOperatorHookStepBase(ConfigOperatorBase, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, ret_dict: Dict[str, Any]) -> Any:
        pass


@dataclass
class ConfigOperatorHookStepDisplayDictKeys(ConfigOperatorHookStepBase):
    tags: TLSeq[str] = ()

    @override
    def is_ready(self) -> bool:
        # Always ok
        return True

    @override
    def init_essentials(self) -> 'ConfigOperatorHookStepDisplayDictKeys':
        return self

    @override
    def __call__(self, ret_dict: Dict[str, Any]) -> None:
        desc: str = ''
        desc += ' '.join(f'[{tag}]' for tag in self.tags)
        desc += f' {ret_dict.keys()}'
        print(desc)


@dataclass
class ConfigOperatorHookStepExportMulticlassPredWithMaskResults(ConfigOperatorHookStepBase):
    export_root_dir: PathLike = "hook_export_pred_results"
    dataset_root_dir: PathLike = ""
    manifest_file: PathLike = "manifest.xlsx"
    column_dtype_map: Optional[DtypeArg] = None
    pred_key: str = "pred"
    id_keys: TLSeq[str] = ("ID",)  # Must specify at least one
    volume_keys: TLSeq[str] = ()  # e.g., ("volume",); ("volume_00", "volume_01")
    mask_keys: TLSeq[str] = ()  # e.g., ("mask_00", "mask_01")
    combined_mask_key: str = "mask"  # Must specify
    # Control what to save
    # volume: save the original volumes, this can be very large
    # mask: save the original masks (binary and combined)
    # pred: save the pred_softmax_logits, pred_mask (binary and combined)
    # diff: save diff mask derived by pred & gt mask
    save_option: TLSeq[Literal["volume", "mask", "pred", "diff"]] = ("mask", "pred", "diff")

    @override
    def init_essentials(self, *args, **kwargs) -> 'ConfigOperatorHookStepExportMulticlassPredWithMaskResults':
        # Assertions
        assert len(self.id_keys) > 0, f"id_keys must have at least one key"
        assert len(self.volume_keys) > 0, f"volume_keys must have at least one key, and it shall include all sequences"
        assert len(self.mask_keys) > 0, f"mask_keys must have at least one key, and it shall include all classes"

        self.id_keys = list(self.id_keys)
        self.volume_keys = list(self.volume_keys)
        self.mask_keys = list(self.mask_keys)

        # Validate dataset_root_dir and manifest_file
        if not pl.Path(self.dataset_root_dir).exists():
            raise ValueError(f"dataset_root_dir not exists: {self.dataset_root_dir}")
        if not pl.Path(self.manifest_file).exists():
            raise ValueError(f"manifest_file not exists: {self.manifest_file}")
        try:
            manifest_df: pd.DataFrame = pd.read_excel(self.manifest_file, dtype=self.column_dtype_map)
        except Exception as e:
            raise ValueError(f"can not load manifest: {str(e)}")

        # Validate required columns
        retrieved_columns: List[str] = (
                self.id_keys +
                self.volume_keys +
                self.mask_keys +
                [self.combined_mask_key]
        )
        for col in retrieved_columns:
            if col not in manifest_df.columns:
                raise ValueError(f"required columns missing: {col}")
        retrieved_manifest_df: pd.DataFrame = manifest_df[retrieved_columns]

        # Convert DataFrame to list of dictionaries
        manifest: List[Dict[str, Any]] = retrieved_manifest_df.to_dict(orient="records")

        all_path_keys: List[str] = self.volume_keys + self.mask_keys + [self.combined_mask_key]
        # Process relative paths to absolute paths
        self.modified_manifest: List[Dict[str, Any]] = []
        for sample in manifest:
            modified_sample: Dict[str, Any] = {}
            for key, value in sample.items():
                if key in all_path_keys:
                    modified_sample[key] = (pl.Path(self.dataset_root_dir) / value).as_posix()
                else:
                    modified_sample[key] = value
            self.modified_manifest.append(modified_sample)

        # Define transforms
        self.tf_load_image: mT.LoadImaged = mT.LoadImaged(
            keys=all_path_keys,
            ensure_channel_first=True,
            allow_missing_keys=True
        )

        self.tf_as_discrete: mT.AsDiscreted = mT.AsDiscreted(
            keys=self.pred_key,
            argmax=True,
            dim=1,
            dtype=torch.int,
            keepdim=False,
            allow_missing_keys=True
        )

        self.tf_save_image_volume: mT.SaveImaged = mT.SaveImaged(
            keys=self.volume_keys,
            output_dir=self.export_root_dir,
            data_root_dir=self.dataset_root_dir,
            output_postfix="",
            output_dtype=torch.float32,
            separate_folder=False
        )

        self.tf_save_image_mask: mT.SaveImaged = mT.SaveImaged(
            keys=self.mask_keys + [self.combined_mask_key],
            output_dir=self.export_root_dir,
            data_root_dir=self.dataset_root_dir,
            output_ext="",
            output_postfix="",
            output_dtype=torch.uint8,
            separate_folder=False
        )

        return self

    @override
    def __call__(self, ret_dict: Dict[str, Any]) -> None:
        self._assert_init_essentials()
        # batch_idx is the only index we can use to identify samples, which shall certainly exist
        assert "batch_idx" in ret_dict, f"Key 'barch_idx' must exist, but not found in {ret_dict}"

        idx: int = int(ret_dict["batch_idx"])

        # Keys: volume, mask
        all_path_keys: List[str] = self.volume_keys + self.mask_keys + [self.combined_mask_key]

        record_dict: Dict[str, Any] = {
            k: v for k, v in self.modified_manifest[idx].items()
            if k in all_path_keys
        }

        # It contains all volumes, all binary masks and the combined mask
        # They shall share the same size
        loaded_dict: Dict[str, Any] = self.tf_load_image(record_dict)  # Tensor shall be (1, X, Y, Z)
        shared_size: Optional[Tuple[int, int, int]] = None
        for k in all_path_keys:
            assert k in loaded_dict, f"Key '{k}' not found in {loaded_dict}"
            ts_size = (loaded_dict[k].size(1), loaded_dict[k].size(2), loaded_dict[k].size(3))
            if shared_size is None:
                shared_size = ts_size
            else:
                assert ts_size == shared_size, f"Size of {k} is not consistent, {k}: {ts_size} != {shared_size}"

        pred_logits = ret_dict[self.pred_key]  # (1, C, X, Y, Z)
        num_classes: int = pred_logits.size(1)
        assert num_classes == len(self.mask_keys), f"num_classes shall be same length with {self.mask_keys}"

        loaded_dict[self.pred_key] = pred_logits
        dis_dict: Dict[Hashable, Any] = self.tf_as_discrete(loaded_dict)
        pred_mask: Tensor = dis_dict[self.pred_key]  # (1, X, Y, Z)
        pred_size: Tuple[int, int, int] = (pred_mask.size(1), pred_mask.size(2), pred_mask.size(3))
        assert pred_size == shared_size, f"Size of {self.pred_key} is not consistent, {self.pred_key}: {pred_size} != {shared_size}"

        # Save all sample files
        if "volume" in self.save_option:
            self.tf_save_image_volume(dis_dict)
        if "mask" in self.save_option:
            self.tf_save_image_mask(dis_dict)

        # Save pred logits
        tf_save_pred_logits: mT.SaveImage = mT.SaveImage(
            output_dir=self.export_root_dir,
            output_ext="",
            output_postfix="",
            output_dtype=torch.float32,
            separate_folder=False
        )
        pred_softmax: Tensor = torch.softmax(pred_logits, dim=1)  # (1, C, X, Y, Z)

        tf_save_pred_mask: mT.SaveImage = mT.SaveImage(
            output_dir=self.export_root_dir,
            output_ext="",
            output_postfix="",
            output_dtype=torch.uint8,
            separate_folder=False
        )
        pred_save_rel_dir: pl.Path = pl.Path(record_dict[self.mask_keys[0]]).relative_to(self.dataset_root_dir).parent
        pred_save_abs_dir: pl.Path = pl.Path(self.export_root_dir) / pred_save_rel_dir
        pred_file_id: str = "_".join([self.modified_manifest[idx][k] for k in self.id_keys])

        # Softmax logits per class
        for i in range(num_classes):
            class_logits: Tensor = pred_softmax[:, i]  # (1, X, Y, Z)
            class_mask: Tensor = pred_mask.cpu() == i
            class_gt: Tensor = dis_dict[self.mask_keys[i]].to(torch.bool)
            tp: Tensor = torch.logical_and(class_mask, class_gt)
            # tn: Tensor = torch.logical_and(torch.logical_not(class_mask), torch.logical_not(class_gt))
            fp: Tensor = torch.logical_and(class_mask, torch.logical_not(class_gt))
            fn: Tensor = torch.logical_and(torch.logical_not(class_mask), class_gt)
            class_diff = fp.to(torch.uint8) + 2 * tp.to(torch.uint8) + 3 * fn.to(torch.uint8)

            postfix: str = self.mask_keys[i].split("_", maxsplit=1)[1]
            meta: Dict[str, Any] = dis_dict[self.mask_keys[i]].meta

            if "pred" in self.save_option:
                # Softmax logits per class
                logits_softmax_class_save_file: pl.Path = pred_save_abs_dir / f"{pred_file_id}_pred_logits_{postfix}.nii.gz"
                tf_save_pred_logits(
                    class_logits,
                    meta_data=meta,
                    filename=logits_softmax_class_save_file
                )

                # pred mask per class
                pred_mask_class_save_file: pl.Path = pred_save_abs_dir / f"{pred_file_id}_pred_{self.mask_keys[i]}.nii.gz"
                tf_save_pred_mask(
                    class_mask,
                    meta_data=meta,
                    filename=pred_mask_class_save_file
                )

            if "diff" in self.save_option:
                # Diff pred-gt mask per class
                # 0-tn 1-fp 2-tp 3-fn
                pred_mask_class_save_file: pl.Path = pred_save_abs_dir / f"{pred_file_id}_diff_{self.mask_keys[i]}.nii.gz"
                tf_save_pred_mask(
                    class_diff,
                    meta_data=meta,
                    filename=pred_mask_class_save_file
                )

        if "pred" in self.save_option:
            # Combined pred
            pred_save_file: pl.Path = pred_save_abs_dir / f"{pred_file_id}_pred_{self.combined_mask_key}.nii.gz"
            tf_save_pred_mask(
                dis_dict[self.pred_key],
                meta_data=dis_dict[self.combined_mask_key].meta,
                filename=pred_save_file
            )


@dataclass
class ConfigOperatorHookStepExportMulticlassPredOnlyResults(ConfigOperatorHookStepBase):
    export_root_dir: PathLike = "hook_export_pred_results"
    dataset_root_dir: PathLike = ""
    manifest_file: PathLike = "manifest.xlsx"
    column_dtype_map: Optional[DtypeArg] = None
    pred_key: str = "pred"
    id_keys: TLSeq[str] = ("ID",)  # Must specify at least one
    volume_keys: TLSeq[str] = ()  # e.g., ("volume",); ("volume_00", "volume_01")
    mask_keys: TLSeq[str] = ()  # e.g., ("mask_00", "mask_01"), we just need the key names, no bother processing
    combined_mask_key: str = "mask"  # Must specify, but we just need the key names, no bother processing
    # Control what to save
    # volume: save the original volumes, this can be very large
    # pred: save the pred_softmax_logits, pred_mask (binary and combined)
    save_option: TLSeq[Literal["volume", "pred"]] = ("pred",)

    @override
    def init_essentials(self, *args, **kwargs) -> 'ConfigOperatorHookStepExportMulticlassPredOnlyResults':
        # Assertions
        assert len(self.id_keys) > 0, f"id_keys must have at least one key"
        assert len(self.volume_keys) > 0, f"volume_keys must have at least one key, and it shall include all sequences"
        assert len(self.mask_keys) > 0, f"mask_keys must have at least one key, and it shall include all classes"

        self.id_keys = list(self.id_keys)
        self.volume_keys = list(self.volume_keys)
        self.mask_keys = list(self.mask_keys)

        # Validate dataset_root_dir and manifest_file
        if not pl.Path(self.dataset_root_dir).exists():
            raise ValueError(f"dataset_root_dir not exists: {self.dataset_root_dir}")
        if not pl.Path(self.manifest_file).exists():
            raise ValueError(f"manifest_file not exists: {self.manifest_file}")
        try:
            manifest_df: pd.DataFrame = pd.read_excel(self.manifest_file, dtype=self.column_dtype_map)
        except Exception as e:
            raise ValueError(f"can not load manifest: {str(e)}")

        # Validate required columns
        retrieved_columns: List[str] = self.id_keys + self.volume_keys
        for col in retrieved_columns:
            if col not in manifest_df.columns:
                raise ValueError(f"required columns missing: {col}")
        retrieved_manifest_df: pd.DataFrame = manifest_df[retrieved_columns]

        # Convert DataFrame to list of dictionaries
        manifest: List[Dict[str, Any]] = retrieved_manifest_df.to_dict(orient="records")

        all_path_keys: List[str] = self.volume_keys
        # Process relative paths to absolute paths
        self.modified_manifest: List[Dict[str, Any]] = []
        for sample in manifest:
            modified_sample: Dict[str, Any] = {}
            for key, value in sample.items():
                if key in all_path_keys:
                    modified_sample[key] = (pl.Path(self.dataset_root_dir) / value).as_posix()
                else:
                    modified_sample[key] = value
            self.modified_manifest.append(modified_sample)

        # Define transforms
        self.tf_load_image: mT.LoadImaged = mT.LoadImaged(
            keys=all_path_keys,
            ensure_channel_first=True,
            allow_missing_keys=True
        )

        self.tf_as_discrete: mT.AsDiscreted = mT.AsDiscreted(
            keys=self.pred_key,
            argmax=True,
            dim=1,
            dtype=torch.int,
            keepdim=False,
            allow_missing_keys=True
        )

        self.tf_save_image_volume: mT.SaveImaged = mT.SaveImaged(
            keys=self.volume_keys,
            output_dir=self.export_root_dir,
            data_root_dir=self.dataset_root_dir,
            output_postfix="",
            output_dtype=torch.float32,
            separate_folder=False
        )

        return self

    @override
    def __call__(self, ret_dict: Dict[str, Any]) -> None:
        self._assert_init_essentials()
        # batch_idx is the only index we can use to identify samples, which shall certainly exist
        assert "batch_idx" in ret_dict, f"Key 'barch_idx' must exist, but not found in {ret_dict}"

        idx: int = int(ret_dict["batch_idx"].item())

        # Keys: volume, mask
        all_path_keys: List[str] = self.volume_keys

        record_dict: Dict[str, Any] = {
            k: v for k, v in self.modified_manifest[idx].items()
            if k in all_path_keys
        }

        # It contains all volumes, all binary masks and the combined mask
        # They shall share the same size
        loaded_dict: Dict[str, Any] = self.tf_load_image(record_dict)  # Tensor shall be (1, X, Y, Z)
        shared_size: Optional[Tuple[int, int, int]] = None
        for k in all_path_keys:
            assert k in loaded_dict, f"Key '{k}' not found in {loaded_dict}"
            ts_size = (loaded_dict[k].size(1), loaded_dict[k].size(2), loaded_dict[k].size(3))
            if shared_size is None:
                shared_size = ts_size
            else:
                assert ts_size == shared_size, f"Size of {k} is not consistent, {k}: {ts_size} != {shared_size}"

        pred_logits = ret_dict[self.pred_key]  # (1, C, X, Y, Z)
        num_classes: int = pred_logits.size(1)
        assert num_classes == len(self.mask_keys), f"num_classes shall be same length with {self.mask_keys}"

        loaded_dict[self.pred_key] = pred_logits
        dis_dict: Dict[Hashable, Any] = self.tf_as_discrete(loaded_dict)
        pred_mask: Tensor = dis_dict[self.pred_key]  # (1, X, Y, Z)
        pred_size: Tuple[int, int, int] = (pred_mask.size(1), pred_mask.size(2), pred_mask.size(3))
        assert pred_size == shared_size, f"Size of {self.pred_key} is not consistent, {self.pred_key}: {pred_size} != {shared_size}"

        # Save all sample files
        if "volume" in self.save_option:
            self.tf_save_image_volume(dis_dict)

        # Save pred logits
        tf_save_pred_logits: mT.SaveImage = mT.SaveImage(
            output_dir=self.export_root_dir,
            output_ext="",
            output_postfix="",
            output_dtype=torch.float32,
            separate_folder=False
        )
        pred_softmax: Tensor = torch.softmax(pred_logits, dim=1)  # (1, C, X, Y, Z)

        tf_save_pred_mask: mT.SaveImage = mT.SaveImage(
            output_dir=self.export_root_dir,
            output_ext="",
            output_postfix="",
            output_dtype=torch.uint8,
            separate_folder=False
        )

        # Use record of volume_keys[0] to determine parent dir and meta
        pred_save_rel_dir: pl.Path = pl.Path(record_dict[self.volume_keys[0]]).relative_to(self.dataset_root_dir).parent
        pred_save_abs_dir: pl.Path = pl.Path(self.export_root_dir) / pred_save_rel_dir
        pred_file_id: str = "_".join([self.modified_manifest[idx][k] for k in self.id_keys])
        meta: Dict[str, Any] = dis_dict[self.volume_keys[0]].meta

        # Softmax logits per class
        for i in range(num_classes):
            class_logits: Tensor = pred_softmax[:, i]  # (1, X, Y, Z)
            class_mask: Tensor = pred_mask.cpu() == i
            postfix: str = self.mask_keys[i].split("_", maxsplit=1)[1]

            if "pred" in self.save_option:
                # Softmax logits per class
                logits_softmax_class_save_file: pl.Path = pred_save_abs_dir / f"{pred_file_id}_pred_logits_{postfix}.nii.gz"
                tf_save_pred_logits(
                    class_logits,
                    meta_data=meta,
                    filename=logits_softmax_class_save_file
                )

                # pred mask per class
                pred_mask_class_save_file: pl.Path = pred_save_abs_dir / f"{pred_file_id}_pred_{self.mask_keys[i]}.nii.gz"
                tf_save_pred_mask(
                    class_mask,
                    meta_data=meta,
                    filename=pred_mask_class_save_file
                )

        if "pred" in self.save_option:
            # Combined pred
            pred_save_file: pl.Path = pred_save_abs_dir / f"{pred_file_id}_pred_{self.combined_mask_key}.nii.gz"
            tf_save_pred_mask(
                dis_dict[self.pred_key],
                meta_data=dis_dict[self.combined_mask_key].meta,
                filename=pred_save_file
            )
