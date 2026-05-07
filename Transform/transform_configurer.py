# -*- coding: utf-8 -*-
"""
Default Segmentation Transform Module

This module provides default transform pipelines for segmentation tasks using MONAI.

Classes:
    TransformSegmentationDefaultBase: Base class for segmentation transforms
    TransformSegmentationDefaultTrain: Transform pipeline for training
    TransformSegmentationDefaultInferencePre: Transform pipeline for inference preprocessing
    TransformSegmentationDefaultInferencePost: Transform pipeline for inference postprocessing

Key Features:
    - Supports volume and mask preprocessing
    - Provides train-specific augmentations (e.g., random cropping)
    - Includes inference-specific pre and post-processing
    - Supports random state management for reproducibility
"""
import os
import torch
import numpy as np
import monai.transforms as mT
from monai.data import MetaTensor
from monai.utils import GridSampleMode, GridSamplePadMode, PytorchPadMode, NumpyPadMode
from typing import Dict, Any, Optional, Union, List, Sequence, Tuple
from typing_extensions import override
from Transform.monai_transform_custom import DuplicateItemsd, RandCropByLabelClassesd
from dataclasses import dataclass
from abc import ABC

PathLike = Union[str, os.PathLike]
DtypeLike = Union[np.dtype, type, str, None]


@dataclass
class ConfigTransformBase(ABC):
    """
    Base class for transforms
    
    Provides common functionality for all transform pipelines
    """
    volume_key: Optional[str] = None
    mask_key: Optional[str] = None

    def is_ready(self) -> bool:
        return hasattr(self, "_composed_transform") and \
            hasattr(self, "transform_dict")

    def init_essentials(self) -> 'ConfigTransformBase':
        self._composed_transform: Optional[mT.Compose] = mT.Compose()
        self.transform_dict: Dict[str, mT.Compose] = {}
        return self

    def _assert_init_essentials(self) -> None:
        if self.is_ready(): return
        self.init_essentials()

    def get_composed_transform(self) -> mT.Compose:
        """
        Get the composed transform pipeline
        
        Returns:
            Composed transform pipeline
        """
        self._assert_init_essentials()
        return self._composed_transform

    def __call__(
            self,
            data: Dict[str, Union[Sequence[PathLike], PathLike]]
    ) -> Union[List[Dict[str, Any]], Dict[str, MetaTensor]]:
        """
        Apply the composed transform to the data
        
        Args:
            data: Input data dictionary
            
        Returns:
            Transformed data
        """
        self._assert_init_essentials()
        if self._composed_transform is not None:
            return self._composed_transform(data)

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

    def get_state(self) -> Dict[str, Dict[str, Any]]:
        return dict()

    def set_state(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        return


@dataclass
class ConfigTransformSegmentationDefaultTrain(ConfigTransformBase):
    """
    Transform pipeline for training segmentation models
    
    Includes a sequence of transforms for data loading, preprocessing, augmentation,
    and preparation for model training

    Initialize the training transform pipeline

    Attributes:
        volume_key: Key for volume data in the input dictionary
        mask_key: Key for mask data in the input dictionary
        param_volume_tf_duplicate_items_dup_keys_volume: Key for duplicated volume data
        param_mask_tf_duplicate_items_dup_keys_mask: Key for duplicated mask data
        param_tf_spacing_pixdim: Voxel spacing for resampling
        param_tf_spacing_mode_volume: Interpolation mode for volume resampling
        param_tf_spacing_mode_mask: Interpolation mode for mask resampling
        param_tf_padding_mode_volume: Padding mode for volume resampling
        param_tf_padding_mode_mask: Padding mode for mask resampling
        param_tf_spatial_pad_spatial_size: Spatial size for padding
        param_tf_spatial_pad_mode: Padding mode for spatial padding
        param_tf_rand_crop_by_label_classes_spatial_size: Spatial size for random cropping
        param_tf_rand_crop_by_label_classes_ratios: Ratios for class-based cropping
        param_tf_rand_crop_by_label_classes_num_classes: Number of classes for cropping
        param_tf_rand_crop_by_label_classes_num_samples: Number of samples to generate
        param_tf_scale_intensity_range_a_min: Minimum intensity value for scaling
        param_tf_scale_intensity_range_a_max: Maximum intensity value for scaling
        param_tf_scale_intensity_range_b_min: Minimum scaled intensity value
        param_tf_scale_intensity_range_b_max: Maximum scaled intensity value
        param_tf_scale_intensity_range_clip: Whether to clip intensity values
        param_tf_allow_missing_keys: Whether to allow missing keys
        random_seed: Random seed for reproducibility
    """
    volume_key: str = 'volume'
    mask_key: str = 'mask'
    param_volume_tf_duplicate_items_dup_keys_volume: Optional[str] = 'volume_raw'
    param_mask_tf_duplicate_items_dup_keys_mask: Optional[str] = 'mask_raw'
    param_tf_spacing_pixdim: Union[Tuple[float, ...], List[float], float] = (1.0, 1.0, 1.0)
    param_tf_spacing_mode_volume: GridSampleMode = GridSampleMode.BILINEAR
    param_tf_spacing_mode_mask: GridSampleMode = GridSampleMode.NEAREST
    param_tf_padding_mode_volume: GridSamplePadMode = GridSamplePadMode.BORDER
    param_tf_padding_mode_mask: GridSamplePadMode = GridSamplePadMode.BORDER
    param_tf_spatial_pad_spatial_size: Union[Tuple[int, ...], List[float], int] = (128, 128, 128)
    param_tf_spatial_pad_mode: Union[PytorchPadMode, NumpyPadMode] = PytorchPadMode.REPLICATE
    param_tf_rand_crop_by_label_classes_spatial_size: Union[Tuple[int, ...], List[float], int] = (128, 128, 128)
    param_tf_rand_crop_by_label_classes_ratios: Optional[List[Union[float, int]]] = None
    param_tf_rand_crop_by_label_classes_num_classes: Optional[int] = None
    param_tf_rand_crop_by_label_classes_num_samples: int = 1
    param_tf_scale_intensity_range_a_min: float = -1000.0
    param_tf_scale_intensity_range_a_max: float = 1000.0
    param_tf_scale_intensity_range_b_min: Optional[float] = 0.0
    param_tf_scale_intensity_range_b_max: Optional[float] = 1.0
    param_tf_scale_intensity_range_clip: bool = True
    param_tf_allow_missing_keys: bool = False
    random_seed: Optional[int] = None

    @override
    def init_essentials(self) -> 'ConfigTransformSegmentationDefaultTrain':
        # Initialize individual transforms
        self._tf_load_image: mT.LoadImaged = mT.LoadImaged(
            keys=[self.volume_key, self.mask_key],
            ensure_channel_first=True,
            allow_missing_keys=self.param_tf_allow_missing_keys
        )
        self._tf_duplicate_items: DuplicateItemsd = DuplicateItemsd(
            keys=[self.volume_key, self.mask_key],
            dup_keys=[
                self.param_volume_tf_duplicate_items_dup_keys_volume,
                self.param_mask_tf_duplicate_items_dup_keys_mask
            ]
        )
        self._tf_spacing: mT.Spacingd = mT.Spacingd(
            keys=[self.volume_key, self.mask_key],
            pixdim=self.param_tf_spacing_pixdim,
            mode=[self.param_tf_spacing_mode_volume, self.param_tf_spacing_mode_mask],
            padding_mode=[self.param_tf_padding_mode_volume, self.param_tf_padding_mode_mask],
            allow_missing_keys=self.param_tf_allow_missing_keys
        )
        self._tf_spatial_pad: mT.SpatialPadd = mT.SpatialPadd(
            keys=[self.volume_key, self.mask_key],
            spatial_size=self.param_tf_spatial_pad_spatial_size,
            mode=self.param_tf_spatial_pad_mode,
            allow_missing_keys=self.param_tf_allow_missing_keys
        )
        self._tf_scale_intensity_range: mT.ScaleIntensityRanged = mT.ScaleIntensityRanged(
            keys=self.volume_key,
            a_min=self.param_tf_scale_intensity_range_a_min,
            a_max=self.param_tf_scale_intensity_range_a_max,
            b_min=self.param_tf_scale_intensity_range_b_min,
            b_max=self.param_tf_scale_intensity_range_b_max,
            clip=self.param_tf_scale_intensity_range_clip,
            allow_missing_keys=self.param_tf_allow_missing_keys
        )
        self._tf_rand_crop_by_label_classes: RandCropByLabelClassesd = RandCropByLabelClassesd(
            keys=[self.volume_key, self.mask_key],
            label_key=self.mask_key,
            spatial_size=self.param_tf_rand_crop_by_label_classes_spatial_size,
            ratios=self.param_tf_rand_crop_by_label_classes_ratios,
            num_classes=self.param_tf_rand_crop_by_label_classes_num_classes,
            num_samples=self.param_tf_rand_crop_by_label_classes_num_samples,
            allow_missing_keys=self.param_tf_allow_missing_keys
        )
        self._tf_cast_to_type: mT.CastToTyped = mT.CastToTyped(
            keys=[self.volume_key, self.mask_key],
            dtype=torch.float,
            allow_missing_keys=self.param_tf_allow_missing_keys
        )

        # Build transform dictionary
        self.transform_dict: Dict[str, mT.Transform] = {
            'LoadImaged': self._tf_load_image,
            'DuplicateItemsd': self._tf_duplicate_items,
            'Spacingd': self._tf_spacing,
            'SpatialPadd': self._tf_spatial_pad,
            'ScaleIntensityRanged': self._tf_scale_intensity_range,
            'RandCropByLabelClassesd': self._tf_rand_crop_by_label_classes,
            'CastToTyped': self._tf_cast_to_type
        }

        # Initialize transforms with random seed
        self._initialize_transforms()

        # Compose transforms into a pipeline
        self._composed_transform: mT.Compose = mT.Compose(list(self.transform_dict.values()))

        return self

    def _initialize_transforms(self) -> None:
        """
        Initialize transforms with random seed for reproducibility
        """
        if self.random_seed is None: return
        for name, transform in self.transform_dict.items():
            if hasattr(transform, 'set_random_state'):
                transform_seed: int = self.random_seed + hash(name) % 10000
                random_state: np.random.RandomState = np.random.RandomState(transform_seed)
                transform.set_random_state(state=random_state)

    @override
    def get_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current state of all transforms for reproducibility

        Returns:
            Dictionary containing the random states of all applicable transforms
        """
        self._assert_init_essentials()
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

    @override
    def set_state(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Set the state of transforms from a saved state dictionary

        Args:
            state_dict: Dictionary containing the random states of transforms
        """
        self._assert_init_essentials()
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


@dataclass
class ConfigTransformSegmentationDefaultInferencePre(ConfigTransformBase):
    """
    Transform pipeline for inference preprocessing
    
    Includes a sequence of transforms for data loading, preprocessing,
    and preparation for model inference

    Initialize the inference preprocessing transform pipeline

    Attributes:
        volume_key: Key for volume data in the input dictionary
        mask_key: Key for mask data in the input dictionary (optional)
        param_volume_tf_duplicate_items_dup_keys_volume: Key for duplicated volume data
        param_mask_tf_duplicate_items_dup_keys_mask: Key for duplicated mask data (optional)
        param_tf_spacing_pixdim: Voxel spacing for resampling
        param_tf_spacing_mode_volume: Interpolation mode for volume resampling
        param_tf_spacing_mode_mask: Interpolation mode for mask resampling
        param_tf_padding_mode_volume: Padding mode for volume resampling
        param_tf_padding_mode_mask: Padding mode for mask resampling
        param_tf_scale_intensity_range_a_min: Minimum intensity value for scaling
        param_tf_scale_intensity_range_a_max: Maximum intensity value for scaling
        param_tf_scale_intensity_range_b_min: Minimum scaled intensity value
        param_tf_scale_intensity_range_b_max: Maximum scaled intensity value
        param_tf_scale_intensity_range_clip: Whether to clip intensity values
    """

    volume_key: str = 'volume'
    mask_key: Optional[str] = None
    param_volume_tf_duplicate_items_dup_keys_volume: str = 'volume_raw'
    param_mask_tf_duplicate_items_dup_keys_mask: Optional[str] = None
    param_tf_spacing_pixdim: Union[Tuple[float, ...], List[float], float] = (1.0, 1.0, 1.0)
    param_tf_spacing_mode_volume: GridSampleMode = GridSampleMode.BILINEAR
    param_tf_spacing_mode_mask: GridSampleMode = GridSampleMode.NEAREST
    param_tf_padding_mode_volume: GridSamplePadMode = GridSamplePadMode.BORDER
    param_tf_padding_mode_mask: GridSamplePadMode = GridSamplePadMode.BORDER
    param_tf_scale_intensity_range_a_min: float = -1000.0
    param_tf_scale_intensity_range_a_max: float = 1000.0
    param_tf_scale_intensity_range_b_min: Optional[float] = 0.0
    param_tf_scale_intensity_range_b_max: Optional[float] = 1.0
    param_tf_scale_intensity_range_clip: bool = True

    @override
    def init_essentials(self) -> 'ConfigTransformSegmentationDefaultInferencePre':
        # Initialize individual transforms
        self._tf_load_image: mT.LoadImaged = mT.LoadImaged(
            keys=[self.volume_key, self.mask_key],
            ensure_channel_first=True,
            allow_missing_keys=True
        )
        self._tf_duplicate_items: DuplicateItemsd = DuplicateItemsd(
            keys=[self.volume_key, self.mask_key],
            dup_keys=[
                self.param_volume_tf_duplicate_items_dup_keys_volume,
                self.param_mask_tf_duplicate_items_dup_keys_mask
            ]
        )
        self._tf_spacing: mT.Spacingd = mT.Spacingd(
            keys=[self.volume_key, self.mask_key],
            pixdim=self.param_tf_spacing_pixdim,
            mode=[self.param_tf_spacing_mode_volume, self.param_tf_spacing_mode_mask],
            padding_mode=[self.param_tf_padding_mode_volume, self.param_tf_padding_mode_mask],
            allow_missing_keys=True
        )
        self._tf_scale_intensity_range: mT.ScaleIntensityRanged = mT.ScaleIntensityRanged(
            keys=self.volume_key,
            a_min=self.param_tf_scale_intensity_range_a_min,
            a_max=self.param_tf_scale_intensity_range_a_max,
            b_min=self.param_tf_scale_intensity_range_b_min,
            b_max=self.param_tf_scale_intensity_range_b_max,
            clip=self.param_tf_scale_intensity_range_clip,
            allow_missing_keys=True
        )
        self._tf_cast_to_type: mT.CastToTyped = mT.CastToTyped(
            keys=[self.volume_key, self.mask_key],
            dtype=torch.float,
            allow_missing_keys=True
        )

        # Build transform dictionary
        self.transform_dict: Dict[str, mT.Transform] = {
            'LoadImaged': self._tf_load_image,
            'DuplicateItemsd': self._tf_duplicate_items,
            'Spacingd': self._tf_spacing,
            'ScaleIntensityRanged': self._tf_scale_intensity_range,
            'CastToTyped': self._tf_cast_to_type
        }

        # Compose transforms into a pipeline
        self._composed_transform: mT.Compose = mT.Compose(list(self.transform_dict.values()))

        return self


@dataclass
class ConfigTransformSegmentationDefaultInferencePost(ConfigTransformBase):
    """
    Transform pipeline for inference postprocessing
    
    Includes a sequence of transforms for post-processing model outputs,
    such as resampling to match original dimensions and intensity scaling

    Initialize the inference postprocessing transform pipeline

    Attributes:
        volume_key: Key for volume data in the input dictionary (optional)
        mask_key: Key for mask data in the input dictionary
        ref_key: Key for reference data used for resampling
        param_tf_resample_to_match_mode_volume: Interpolation mode for volume resampling
        param_tf_resample_to_match_mode_mask: Interpolation mode for mask resampling
        param_tf_resample_to_match_padding_mode_volume: Padding mode for volume resampling
        param_tf_resample_to_match_padding_mode_mask: Padding mode for mask resampling
        param_tf_scale_intensity_range_a_min: Minimum intensity value for scaling
        param_tf_scale_intensity_range_a_max: Maximum intensity value for scaling
        param_tf_scale_intensity_range_b_min: Minimum scaled intensity value
        param_tf_scale_intensity_range_b_max: Maximum scaled intensity value
        param_tf_scale_intensity_range_clip: Whether to clip intensity values
        param_tf_cast_to_type_dtype_volume: Data type for volume casting
        param_tf_cast_to_type_dtype_mask: Data type for mask casting
    """
    volume_key: Optional[str] = None
    mask_key: str = 'mask'
    ref_key: str = 'ref'
    param_tf_resample_to_match_mode_volume: GridSampleMode = GridSampleMode.BILINEAR
    param_tf_resample_to_match_mode_mask: GridSampleMode = GridSampleMode.NEAREST
    param_tf_resample_to_match_padding_mode_volume: GridSamplePadMode = GridSamplePadMode.BORDER
    param_tf_resample_to_match_padding_mode_mask: GridSamplePadMode = GridSamplePadMode.BORDER
    param_tf_scale_intensity_range_a_min: float = 0.0
    param_tf_scale_intensity_range_a_max: float = 1.0
    param_tf_scale_intensity_range_b_min: Optional[float] = -1000.0
    param_tf_scale_intensity_range_b_max: Optional[float] = 1000.0
    param_tf_scale_intensity_range_clip: bool = True
    param_tf_cast_to_type_dtype_volume: Union[DtypeLike, torch.dtype] = torch.float32
    param_tf_cast_to_type_dtype_mask: Union[DtypeLike, torch.dtype] = torch.uint8

    @override
    def init_essentials(self) -> 'ConfigTransformSegmentationDefaultInferencePost':
        # Initialize individual transforms
        self.tf_resample_to_match: mT.ResampleToMatchd = mT.ResampleToMatchd(
            keys=[self.volume_key, self.mask_key],
            key_dst=self.ref_key,
            mode=[self.param_tf_resample_to_match_mode_volume, self.param_tf_resample_to_match_mode_mask],
            padding_mode=[
                self.param_tf_resample_to_match_padding_mode_volume,
                self.param_tf_resample_to_match_padding_mode_mask
            ],
            allow_missing_keys=True
        )

        self._tf_scale_intensity_range: mT.ScaleIntensityRanged = mT.ScaleIntensityRanged(
            keys=self.volume_key,
            a_min=self.param_tf_scale_intensity_range_a_min,
            a_max=self.param_tf_scale_intensity_range_a_max,
            b_min=self.param_tf_scale_intensity_range_b_min,
            b_max=self.param_tf_scale_intensity_range_b_max,
            clip=self.param_tf_scale_intensity_range_clip,
            allow_missing_keys=True
        )

        self._tf_cast_to_type: mT.CastToTyped = mT.CastToTyped(
            keys=[self.volume_key, self.mask_key],
            dtype=[self.param_tf_cast_to_type_dtype_volume, self.param_tf_cast_to_type_dtype_mask],
            allow_missing_keys=True
        )

        # Build transform dictionary
        self.transform_dict: Dict[str, mT.Transform] = {
            'ResampleToMatchd': self.tf_resample_to_match,
            'ScaleIntensityRanged': self._tf_scale_intensity_range,
            'CastToTyped': self._tf_cast_to_type
        }

        # Compose transforms into a pipeline
        self._composed_transform: mT.Compose = mT.Compose(list(self.transform_dict.values()))

        return self

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
    tf_train: ConfigTransformSegmentationDefaultTrain = ConfigTransformSegmentationDefaultTrain(
        volume_key='volume',
        mask_key='mask',
        param_tf_spatial_pad_spatial_size=(200, 200, 200),
        param_tf_rand_crop_by_label_classes_num_samples=4,
        param_tf_allow_missing_keys=True
    )

    # Debug state_dict
    print(tf_train.get_state())

    # Call transform
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
    tf_infer_pre_w_mask: ConfigTransformSegmentationDefaultInferencePre = ConfigTransformSegmentationDefaultInferencePre(
        volume_key='volume',
        mask_key='mask',
        param_volume_tf_duplicate_items_dup_keys_volume='volume_raw',
        param_mask_tf_duplicate_items_dup_keys_mask='mask_raw'
    )

    # Call transform
    transformed_data_w_mask: Dict[str, MetaTensor] = tf_infer_pre_w_mask(data)
    print(f'[{type(tf_infer_pre_w_mask).__name__}.__call__] with mask')
    # print(transformed_data_w_mask)
    print(f'key of transformed_data_w_mask: {transformed_data_w_mask.keys()}')
    print(f'volume: {transformed_data_w_mask["volume"].shape}, {transformed_data_w_mask["volume"].dtype}')
    print(f'mask: {transformed_data_w_mask["mask"].shape}, {transformed_data_w_mask["mask"].dtype}')
    print(f'volume_raw: {transformed_data_w_mask["volume_raw"].shape}, {transformed_data_w_mask["volume_raw"].dtype}')
    print(f'mask_raw: {transformed_data_w_mask["mask_raw"].shape}, {transformed_data_w_mask["mask_raw"].dtype}')

    # Transform for inference preprocess (without mask)
    tf_infer_pre_wo_mask: ConfigTransformSegmentationDefaultInferencePre = ConfigTransformSegmentationDefaultInferencePre(
        volume_key='volume',
        param_volume_tf_duplicate_items_dup_keys_volume='volume_raw'
    )

    # Call transform
    transformed_data_wo_mask: Dict[str, MetaTensor] = tf_infer_pre_wo_mask(data)
    print(f'[{type(tf_infer_pre_wo_mask).__name__}.__call__] without mask')
    # print(transformed_data_wo_mask)
    print(f'key of transformed_data_wo_mask: {transformed_data_wo_mask.keys()}')
    print(f'volume: {transformed_data_wo_mask["volume"].shape}, {transformed_data_wo_mask["volume"].dtype}')
    print(f'volume_raw: {transformed_data_wo_mask["volume_raw"].shape}, {transformed_data_wo_mask["volume_raw"].dtype}')

    # Transform for inference postprocess (with volume)
    tf_infer_post_w_volume: ConfigTransformSegmentationDefaultInferencePost = ConfigTransformSegmentationDefaultInferencePost(
        volume_key='volume',
        mask_key='mask',
        ref_key='volume_raw'
    )

    # Call transform
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
    tf_infer_post_wo_volume: ConfigTransformSegmentationDefaultInferencePost = ConfigTransformSegmentationDefaultInferencePost(
        mask_key='mask',
        ref_key='volume_raw'
    )

    # Call transform
    transformed_data_wo_vol: Union[List[Dict[str, MetaTensor]], Dict[str, MetaTensor]] = \
        tf_infer_post_wo_volume(transformed_data_w_mask)
    print(f'[{type(tf_infer_post_wo_volume).__name__}.__call__] without volume')
    # print(transformed_data_wo_vol)
    print(f'key of transformed_data_wo_vol: {transformed_data_wo_vol.keys()}')
    print(f'volume: {transformed_data_wo_vol["volume"].shape}, {transformed_data_wo_vol["volume"].dtype}')
    print(f'mask: {transformed_data_wo_vol["mask"].shape}, {transformed_data_wo_vol["mask"].dtype}')
    print(f'volume_raw: {transformed_data_wo_vol["volume_raw"].shape}, {transformed_data_wo_vol["volume_raw"].dtype}')
    print(f'mask_raw: {transformed_data_wo_vol["mask_raw"].shape}, {transformed_data_wo_vol["mask_raw"].dtype}')
