# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import monai.transforms as mT

from monai.data import MetaTensor
from typing import Dict, Any, Optional, Union, List, Sequence
from monai.utils import GridSampleMode, GridSamplePadMode, PytorchPadMode, NumpyPadMode

from Transform.tf_custom import DuplicateItemsd, RandCropByLabelClassesd

PathLike = Union[str, os.PathLike]
DtypeLike = Union[np.dtype, type, str, None]


class TransformSegmentationDefaultBase(mT.Transform):
    def __init__(
            self,
            volume_key: Optional[str],
            mask_key: Optional[str]
    ) -> None:
        self.volume_key: Optional[str] = volume_key
        self.mask_key: Optional[str] = mask_key
        self.transform_dict: Dict[str, mT.Transform] = {}
        self._comp_transform: mT.Compose = mT.Compose()

    def get_composed_transform(self) -> mT.Compose:
        return self._comp_transform

    def __call__(
            self,
            data: Dict[str, Union[Sequence[PathLike], PathLike]]
    ) -> Union[List[Dict[str, Any]], Dict[str, MetaTensor]]:
        result_data = self._comp_transform(data)
        return result_data

    def execute(
            self,
            data: Dict[str, Union[Sequence[PathLike], PathLike]]
    ) -> Union[List[Dict[str, Any]], Dict[str, MetaTensor]]:
        result_data: Dict[str, Any] = data
        for name, transform in self.transform_dict.items():
            if isinstance(result_data, List):
                tmp_data: List[Dict[str, Any]] = []
                for tup in result_data:
                    tmp_data.append(transform(tup))
                result_data: List[Dict[str, Any]] = tmp_data
            else:
                result_data: Dict[str, Any] = transform(result_data)
        return result_data


class TransformSegmentationDefaultTrain(TransformSegmentationDefaultBase):
    def __init__(
            self,
            volume_key: str = 'volume',
            mask_key: str = 'mask',
            param_volume_tf_duplicate_items_dup_keys_volume: str = 'volume_raw',
            param_mask_tf_duplicate_items_dup_keys_mask: str = 'mask_raw',
            param_tf_spacing_pixdim: Union[Sequence[float], float] = (1.0, 1.0, 1.0),
            param_tf_spacing_mode_volume: GridSampleMode = GridSampleMode.BILINEAR,
            param_tf_spacing_mode_mask: GridSampleMode = GridSampleMode.NEAREST,
            param_tf_padding_mode_volume: GridSamplePadMode = GridSamplePadMode.BORDER,
            param_tf_padding_mode_mask: GridSamplePadMode = GridSamplePadMode.BORDER,
            param_tf_spatial_pad_spatial_size: Union[Sequence[int], int] = (128, 128, 128),
            param_tf_spatial_pad_mode: Union[PytorchPadMode, NumpyPadMode] = PytorchPadMode.REPLICATE,
            param_tf_rand_crop_by_label_classes_spatial_size: Union[Sequence[int], int] = (128, 128, 128),
            param_tf_rand_crop_by_label_classes_ratios: Optional[List[Union[float, int]]] = None,
            param_tf_rand_crop_by_label_classes_num_classes: Optional[int] = None,
            param_tf_rand_crop_by_label_classes_num_samples: int = 1,
            param_tf_scale_intensity_range_a_min: float = -1000.0,
            param_tf_scale_intensity_range_a_max: float = 1000.0,
            param_tf_scale_intensity_range_b_min: Optional[float] = 0.0,
            param_tf_scale_intensity_range_b_max: Optional[float] = 1.0,
            param_tf_scale_intensity_range_clip: bool = True,
            param_tf_allow_missing_keys: bool = False,
            random_seed: Optional[int] = None
    ) -> None:
        super().__init__(volume_key, mask_key)
        self.random_seed: Optional[int] = random_seed

        self._tf_load_image: mT.LoadImaged = mT.LoadImaged(
            keys=[volume_key, mask_key],
            ensure_channel_first=True,
            allow_missing_keys=param_tf_allow_missing_keys
        )
        self._tf_duplicate_items: DuplicateItemsd = DuplicateItemsd(
            keys=[volume_key, mask_key],
            dup_keys=[param_volume_tf_duplicate_items_dup_keys_volume, param_mask_tf_duplicate_items_dup_keys_mask]
        )
        self._tf_spacing: mT.Spacingd = mT.Spacingd(
            keys=[volume_key, mask_key],
            pixdim=param_tf_spacing_pixdim,
            mode=[param_tf_spacing_mode_volume, param_tf_spacing_mode_mask],
            padding_mode=[param_tf_padding_mode_volume, param_tf_padding_mode_mask],
            allow_missing_keys=param_tf_allow_missing_keys
        )
        self._tf_spatial_pad: mT.SpatialPadd = mT.SpatialPadd(
            keys=[volume_key, mask_key],
            spatial_size=param_tf_spatial_pad_spatial_size,
            mode=param_tf_spatial_pad_mode,
            allow_missing_keys=param_tf_allow_missing_keys
        )
        self._tf_scale_intensity_range: mT.ScaleIntensityRanged = mT.ScaleIntensityRanged(
            keys=volume_key,
            a_min=param_tf_scale_intensity_range_a_min,
            a_max=param_tf_scale_intensity_range_a_max,
            b_min=param_tf_scale_intensity_range_b_min,
            b_max=param_tf_scale_intensity_range_b_max,
            clip=param_tf_scale_intensity_range_clip,
            allow_missing_keys=param_tf_allow_missing_keys
        )
        self._tf_rand_crop_by_label_classes: RandCropByLabelClassesd = RandCropByLabelClassesd(
            keys=[volume_key, mask_key],
            label_key=mask_key,
            spatial_size=param_tf_rand_crop_by_label_classes_spatial_size,
            ratios=param_tf_rand_crop_by_label_classes_ratios,
            num_classes=param_tf_rand_crop_by_label_classes_num_classes,
            num_samples=param_tf_rand_crop_by_label_classes_num_samples,
            allow_missing_keys=param_tf_allow_missing_keys
        )
        self._tf_cast_to_type: mT.CastToTyped = mT.CastToTyped(
            keys=[volume_key, mask_key],
            dtype=torch.float,
            allow_missing_keys=param_tf_allow_missing_keys
        )
        self.transform_dict: Dict[str, mT.Transform] = {
            'LoadImaged': self._tf_load_image,
            'DuplicateItemsd': self._tf_duplicate_items,
            'Spacingd': self._tf_spacing,
            'SpatialPadd': self._tf_spatial_pad,
            'ScaleIntensityRanged': self._tf_scale_intensity_range,
            'RandCropByLabelClassesd': self._tf_rand_crop_by_label_classes,
            'CastToTyped': self._tf_cast_to_type
        }
        self._initialize_transforms()
        self._comp_transform: mT.Compose = mT.Compose(list(self.transform_dict.values()))

    def _initialize_transforms(self) -> None:
        if self.random_seed is None: return
        for name, transform in self.transform_dict.items():
            if hasattr(transform, 'set_random_state'):
                transform_seed: int = self.random_seed + hash(name) % 10000
                random_state: np.random.RandomState = np.random.RandomState(transform_seed)
                transform.set_random_state(state=random_state)

    def get_state(self) -> Dict[str, Dict[str, Any]]:
        state_dict: Dict[str, Dict[str, Any]] = {}
        for name, transform in self.transform_dict.items():
            if hasattr(transform, 'get_random_state'):
                try:
                    state_dict[name] = {
                        'random_state': transform.get_random_state()
                    }
                except Exception as e:
                    raise ValueError(f"Fail to acquire random_state for {name}: {str(e)}")

        return state_dict

    def set_state(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        for name, state in state_dict.items():
            if name not in self.transform_dict:
                raise ValueError(f"{name} do not exists")

            if hasattr(self.transform_dict[name], 'set_random_state') and 'random_state' in state:
                random_state: np.random.RandomState = np.random.RandomState()
                random_state.set_state(state_dict[name]['random_state'])
                try:
                    self.transform_dict[name].set_random_state(state=random_state)
                except Exception as e:
                    raise ValueError(f"Fail to set random_state for {name}: {str(e)}")


class TransformSegmentationDefaultInferencePre(TransformSegmentationDefaultBase):
    def __init__(
            self,
            volume_key: str = 'volume',
            mask_key: Optional[str] = None,
            param_volume_tf_duplicate_items_dup_keys_volume: str = 'volume_raw',
            param_mask_tf_duplicate_items_dup_keys_mask: Optional[str] = None,
            param_tf_spacing_pixdim: Union[Sequence[float], float] = (1.0, 1.0, 1.0),
            param_tf_spacing_mode_volume: GridSampleMode = GridSampleMode.BILINEAR,
            param_tf_spacing_mode_mask: GridSampleMode = GridSampleMode.NEAREST,
            param_tf_padding_mode_volume: GridSamplePadMode = GridSamplePadMode.BORDER,
            param_tf_padding_mode_mask: GridSamplePadMode = GridSamplePadMode.BORDER,
            param_tf_scale_intensity_range_a_min: float = -1000.0,
            param_tf_scale_intensity_range_a_max: float = 1000.0,
            param_tf_scale_intensity_range_b_min: Optional[float] = 0.0,
            param_tf_scale_intensity_range_b_max: Optional[float] = 1.0,
            param_tf_scale_intensity_range_clip: bool = True
    ) -> None:
        super().__init__(volume_key, mask_key)
        self._tf_load_image: mT.LoadImaged = mT.LoadImaged(
            keys=[volume_key, mask_key],
            ensure_channel_first=True,
            allow_missing_keys=True
        )
        self._tf_duplicate_items: DuplicateItemsd = DuplicateItemsd(
            keys=[volume_key, mask_key],
            dup_keys=[param_volume_tf_duplicate_items_dup_keys_volume, param_mask_tf_duplicate_items_dup_keys_mask]
        )
        self._tf_spacing: mT.Spacingd = mT.Spacingd(
            keys=[volume_key, mask_key],
            pixdim=param_tf_spacing_pixdim,
            mode=[param_tf_spacing_mode_volume, param_tf_spacing_mode_mask],
            padding_mode=[param_tf_padding_mode_volume, param_tf_padding_mode_mask],
            allow_missing_keys=True
        )
        self._tf_scale_intensity_range: mT.ScaleIntensityRanged = mT.ScaleIntensityRanged(
            keys=volume_key,
            a_min=param_tf_scale_intensity_range_a_min,
            a_max=param_tf_scale_intensity_range_a_max,
            b_min=param_tf_scale_intensity_range_b_min,
            b_max=param_tf_scale_intensity_range_b_max,
            clip=param_tf_scale_intensity_range_clip,
            allow_missing_keys=True
        )
        self._tf_cast_to_type: mT.CastToTyped = mT.CastToTyped(
            keys=[volume_key, mask_key],
            dtype=torch.float,
            allow_missing_keys=True
        )
        self.transform_dict: Dict[str, mT.Transform] = {
            'LoadImaged': self._tf_load_image,
            'DuplicateItemsd': self._tf_duplicate_items,
            'Spacingd': self._tf_spacing,
            'ScaleIntensityRanged': self._tf_scale_intensity_range,
            'CastToTyped': self._tf_cast_to_type
        }
        self._comp_transform: mT.Compose = mT.Compose(list(self.transform_dict.values()))


class TransformSegmentationDefaultInferencePost(TransformSegmentationDefaultBase):
    def __init__(
            self,
            volume_key: Optional[str] = None,
            mask_key: str = 'mask',
            ref_key: str = 'ref',
            param_tf_resample_to_match_mode_volume: GridSampleMode = GridSampleMode.BILINEAR,
            param_tf_resample_to_match_mode_mask: GridSampleMode = GridSampleMode.NEAREST,
            param_tf_resample_to_match_padding_mode_volume: GridSamplePadMode = GridSamplePadMode.BORDER,
            param_tf_resample_to_match_padding_mode_mask: GridSamplePadMode = GridSamplePadMode.BORDER,
            param_tf_scale_intensity_range_a_min: float = 0.0,
            param_tf_scale_intensity_range_a_max: float = 1.0,
            param_tf_scale_intensity_range_b_min: Optional[float] = -1000.0,
            param_tf_scale_intensity_range_b_max: Optional[float] = 1000.0,
            param_tf_scale_intensity_range_clip: bool = True,
            param_tf_cast_to_type_dtype_volume: Union[DtypeLike, torch.dtype] = torch.float32,
            param_tf_cast_to_type_dtype_mask: Union[DtypeLike, torch.dtype] = torch.uint8
    ) -> None:
        super().__init__(volume_key, mask_key)
        self.ref_key: str = ref_key

        self.tf_resample_to_match: mT.ResampleToMatchd = mT.ResampleToMatchd(
            keys=[volume_key, mask_key],
            key_dst=ref_key,
            mode=[param_tf_resample_to_match_mode_volume, param_tf_resample_to_match_mode_mask],
            padding_mode=[param_tf_resample_to_match_padding_mode_volume, param_tf_resample_to_match_padding_mode_mask],
            allow_missing_keys=True
        )

        self._tf_scale_intensity_range: mT.ScaleIntensityRanged = mT.ScaleIntensityRanged(
            keys=volume_key,
            a_min=param_tf_scale_intensity_range_a_min,
            a_max=param_tf_scale_intensity_range_a_max,
            b_min=param_tf_scale_intensity_range_b_min,
            b_max=param_tf_scale_intensity_range_b_max,
            clip=param_tf_scale_intensity_range_clip,
            allow_missing_keys=True
        )

        self._tf_cast_to_type: mT.CastToTyped = mT.CastToTyped(
            keys=[volume_key, mask_key],
            dtype=[param_tf_cast_to_type_dtype_volume, param_tf_cast_to_type_dtype_mask],
            allow_missing_keys=True
        )

        self.transform_dict: Dict[str, mT.Transform] = {
            'ResampleToMatchd': self.tf_resample_to_match,
            'ScaleIntensityRanged': self._tf_scale_intensity_range,
            'CastToTyped': self._tf_cast_to_type
        }
        self._comp_transform: mT.Compose = mT.Compose(list(self.transform_dict.values()))


if __name__ == "__main__":
    import pathlib as pl

    # Prepare samples
    # Simulate 3 modalities
    volume_paths: List[PathLike] = \
        [
            pl.Path(r".\Samples\TJ\pre\TJ_1004520369_pre\TJ_1004520369_pre_volume.nii.gz")
        ] * 3

    # Mask for background and ROI
    mask_paths: List[PathLike] = \
        [
            pl.Path(r".\Samples\TJ\pre\TJ_1004520369_pre\TJ_1004520369_pre_mask_00_Bg.nii.gz"),
            pl.Path(r".\Samples\TJ\pre\TJ_1004520369_pre\TJ_1004520369_pre_mask_01_EsoROI.nii.gz")
        ]

    # Sew in input dict
    data: Dict[str, List[PathLike]] = {
        'volume': volume_paths,
        'mask': mask_paths,
    }

    print(data)

    # Transform for train
    tf_train: TransformSegmentationDefaultTrain = TransformSegmentationDefaultTrain(
        volume_key='volume',
        mask_key='mask',
        param_tf_spatial_pad_spatial_size=(200, 200, 200),
        param_tf_rand_crop_by_label_classes_num_samples=4,
        param_tf_allow_missing_keys=True
    )

    # Explicitly execute
    transformed_data: List[Dict[str, MetaTensor]] = tf_train.execute(data)
    print(f'[{type(tf_train).__name__}.execute]')
    # print(transformed_data)
    print(f'len of transformed_data: {len(transformed_data)}')
    print(f'key of transformed_data: {transformed_data[0].keys()}')
    for idx, tup in enumerate(transformed_data):
        print(f'[{idx}]')
        print(f'volume: {tup["volume"].shape}, {tup["volume"].dtype}')
        print(f'mask: {tup["mask"].shape}, {tup["mask"].dtype}')
        print(f'volume_raw: {tup["volume_raw"].shape}, {tup["volume_raw"].dtype}')
        print(f'mask_raw: {tup["mask_raw"].shape}, {tup["mask_raw"].dtype}')

    # Debug state_dict
    print(tf_train.get_state())

    # Composed execute
    transformed_data: List[Dict[str, MetaTensor]] = tf_train(data)
    print(f'[{type(tf_train).__name__}.__call__]')
    # print(transformed_data)
    print(f'len of transformed_data: {len(transformed_data)}')
    print(f'key of transformed_data: {transformed_data[0].keys()}')
    for idx, tup in enumerate(transformed_data):
        print(f'[{idx}]')
        print(f'volume: {tup["volume"].shape}, {tup["volume"].dtype}')
        print(f'mask: {tup["mask"].shape}, {tup["mask"].dtype}')
        print(f'volume_raw: {tup["volume_raw"].shape}, {tup["volume_raw"].dtype}')
        print(f'mask_raw: {tup["mask_raw"].shape}, {tup["mask_raw"].dtype}')

    # Transform for inference preprocess (with mask)
    tf_infer_pre_w_mask: TransformSegmentationDefaultInferencePre = TransformSegmentationDefaultInferencePre(
        volume_key='volume',
        mask_key='mask',
        param_volume_tf_duplicate_items_dup_keys_volume='volume_raw',
        param_mask_tf_duplicate_items_dup_keys_mask='mask_raw'
    )

    # Explicitly execute
    transformed_data_w_mask: Dict[str, MetaTensor] = tf_infer_pre_w_mask.execute(data)
    print(f'[{type(tf_infer_pre_w_mask).__name__}.execute] with mask')
    # print(transformed_data_w_mask)
    print(f'key of transformed_data_w_mask: {transformed_data_w_mask.keys()}')
    print(f'volume: {transformed_data_w_mask["volume"].shape}, {transformed_data_w_mask["volume"].dtype}')
    print(f'mask: {transformed_data_w_mask["mask"].shape}, {transformed_data_w_mask["mask"].dtype}')
    print(f'volume_raw: {transformed_data_w_mask["volume_raw"].shape}, {transformed_data_w_mask["volume_raw"].dtype}')
    print(f'mask_raw: {transformed_data_w_mask["mask_raw"].shape}, {transformed_data_w_mask["mask_raw"].dtype}')

    # Composed execute
    transformed_data_w_mask: Dict[str, MetaTensor] = tf_infer_pre_w_mask(data)
    print(f'[{type(tf_infer_pre_w_mask).__name__}.__call__] with mask')
    # print(transformed_data_w_mask)
    print(f'key of transformed_data_w_mask: {transformed_data_w_mask.keys()}')
    print(f'volume: {transformed_data_w_mask["volume"].shape}, {transformed_data_w_mask["volume"].dtype}')
    print(f'mask: {transformed_data_w_mask["mask"].shape}, {transformed_data_w_mask["mask"].dtype}')
    print(f'volume_raw: {transformed_data_w_mask["volume_raw"].shape}, {transformed_data_w_mask["volume_raw"].dtype}')
    print(f'mask_raw: {transformed_data_w_mask["mask_raw"].shape}, {transformed_data_w_mask["mask_raw"].dtype}')

    # Transform for inference preprocess (without mask)
    tf_infer_pre_wo_mask: TransformSegmentationDefaultInferencePre = TransformSegmentationDefaultInferencePre(
        volume_key='volume',
        param_volume_tf_duplicate_items_dup_keys_volume='volume_raw'
    )

    # Explicitly execute
    transformed_data_wo_mask: Dict[str, MetaTensor] = tf_infer_pre_wo_mask.execute(data)
    print(f'[{type(tf_infer_pre_wo_mask).__name__}.execute] without mask')
    # print(transformed_data_wo_mask)
    print(f'key of transformed_data_wo_mask: {transformed_data_wo_mask.keys()}')
    print(f'volume: {transformed_data_wo_mask["volume"].shape}, {transformed_data_wo_mask["volume"].dtype}')
    print(f'volume_raw: {transformed_data_wo_mask["volume_raw"].shape}, {transformed_data_wo_mask["volume_raw"].dtype}')

    # Composed execute
    transformed_data_wo_mask: Dict[str, MetaTensor] = tf_infer_pre_wo_mask(data)
    print(f'[{type(tf_infer_pre_wo_mask).__name__}.__call__] without mask')
    # print(transformed_data_wo_mask)
    print(f'key of transformed_data_wo_mask: {transformed_data_wo_mask.keys()}')
    print(f'volume: {transformed_data_wo_mask["volume"].shape}, {transformed_data_wo_mask["volume"].dtype}')
    print(f'volume_raw: {transformed_data_wo_mask["volume_raw"].shape}, {transformed_data_wo_mask["volume_raw"].dtype}')

    # Transform for inference postprocess (with volume)
    tf_infer_post_w_volume: TransformSegmentationDefaultInferencePost = TransformSegmentationDefaultInferencePost(
        volume_key='volume',
        mask_key='mask',
        ref_key='volume_raw'
    )

    # Explicitly execute
    transformed_data_w_vol: Union[List[Dict[str, MetaTensor]], Dict[str, MetaTensor]] = \
        tf_infer_post_w_volume.execute(transformed_data_w_mask)
    print(f'[{type(tf_infer_post_w_volume).__name__}.execute] with volume')
    # print(transformed_data_w_vol)
    print(f'key of transformed_data_w_vol: {transformed_data_w_vol.keys()}')
    print(f'volume: {transformed_data_w_vol["volume"].shape}, {transformed_data_w_vol["volume"].dtype}')
    print(f'mask: {transformed_data_w_vol["mask"].shape}, {transformed_data_w_vol["mask"].dtype}')
    print(f'volume_raw: {transformed_data_w_vol["volume_raw"].shape}, {transformed_data_w_vol["volume_raw"].dtype}')
    print(f'mask_raw: {transformed_data_w_vol["mask_raw"].shape}, {transformed_data_w_vol["mask_raw"].dtype}')

    # Composed execute
    transformed_data_w_vol: Union[List[Dict[str, MetaTensor]], Dict[str, MetaTensor]] = \
        tf_infer_post_w_volume(transformed_data_w_mask)
    print(f'[{type(tf_infer_post_w_volume).__name__}.__call__] with volume')
    # print(transformed_data_w_vol)
    print(f'key of transformed_data_w_vol: {transformed_data_w_vol.keys()}')
    print(f'volume: {transformed_data_w_vol["volume"].shape}, {transformed_data_w_vol["volume"].dtype}')
    print(f'mask: {transformed_data_w_vol["mask"].shape}, {transformed_data_w_vol["mask"].dtype}')
    print(f'volume_raw: {transformed_data_w_vol["volume_raw"].shape}, {transformed_data_w_vol["volume_raw"].dtype}')
    print(f'mask_raw: {transformed_data_w_vol["mask_raw"].shape}, {transformed_data_w_vol["mask_raw"].dtype}')

    # transform for inference postprocess (without volume)
    tf_infer_post_wo_volume: TransformSegmentationDefaultInferencePost = TransformSegmentationDefaultInferencePost(
        mask_key='mask',
        ref_key='volume_raw'
    )

    # Explicitly execute
    transformed_data_wo_vol: Union[List[Dict[str, MetaTensor]], Dict[str, MetaTensor]] = \
        tf_infer_post_wo_volume.execute(transformed_data_w_mask)
    print(f'[{type(tf_infer_post_wo_volume).__name__}.execute] without volume')
    # print(transformed_data_wo_vol)
    print(f'key of transformed_data_wo_vol: {transformed_data_wo_vol.keys()}')
    print(f'volume: {transformed_data_wo_vol["volume"].shape}, {transformed_data_wo_vol["volume"].dtype}')
    print(f'mask: {transformed_data_wo_vol["mask"].shape}, {transformed_data_wo_vol["mask"].dtype}')
    print(f'volume_raw: {transformed_data_wo_vol["volume_raw"].shape}, {transformed_data_wo_vol["volume_raw"].dtype}')
    print(f'mask_raw: {transformed_data_wo_vol["mask_raw"].shape}, {transformed_data_wo_vol["mask_raw"].dtype}')

    # Composed execute
    transformed_data_wo_vol: Union[List[Dict[str, MetaTensor]], Dict[str, MetaTensor]] = \
        tf_infer_post_wo_volume(transformed_data_w_mask)
    print(f'[{type(tf_infer_post_wo_volume).__name__}.__call__] without volume')
    # print(transformed_data_wo_vol)
    print(f'key of transformed_data_wo_vol: {transformed_data_wo_vol.keys()}')
    print(f'volume: {transformed_data_wo_vol["volume"].shape}, {transformed_data_wo_vol["volume"].dtype}')
    print(f'mask: {transformed_data_wo_vol["mask"].shape}, {transformed_data_wo_vol["mask"].dtype}')
    print(f'volume_raw: {transformed_data_wo_vol["volume_raw"].shape}, {transformed_data_wo_vol["volume_raw"].dtype}')
    print(f'mask_raw: {transformed_data_wo_vol["mask_raw"].shape}, {transformed_data_wo_vol["mask_raw"].dtype}')

    pass
