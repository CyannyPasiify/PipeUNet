# -*- coding: utf-8 -*-
import os
import pandas as pd
import pathlib as pl
import monai.data as mD

from typing import Dict, Any, Optional, List, Union, Sequence, Callable, Iterable, Set, Type
from pandas._typing import DtypeArg

PathLike = Union[str, os.PathLike]


class DatasetManifestRetrieverSegmentationDefault:
    def __init__(
            self,
            root_dir: PathLike,
            manifest_file: PathLike,
            column_key_map: Dict[str, str],
            column_key_relative_path: Iterable[str],
            column_group_map: Dict[str, Iterable[str]],
            column_dtype_map: Optional[DtypeArg] = None
    ) -> None:
        # validate root_dir and manifest_file
        if not pl.Path(root_dir).exists():
            raise ValueError(f"root_dir not exists: {root_dir}")

        if not pl.Path(manifest_file).exists():
            raise ValueError(f"manifest_file not exists: {manifest_file}")

        self.root_dir: pl.Path = pl.Path(root_dir)
        self.manifest_file: pl.Path = pl.Path(manifest_file)
        self.column_dtype_map: Optional[DtypeArg] = column_dtype_map
        self.column_key_map: Dict[str, str] = column_key_map
        self.column_key_relative_path: Iterable[str] = column_key_relative_path
        self.column_group_map: Dict[str, Set[str]] = {group: set(keys) for group, keys in column_group_map.items()}
        self._column_group_inv_map: Dict[str, Set[str]] = {}
        for group, keys in self.column_group_map.items():
            for key in keys:
                if key not in self._column_group_inv_map:
                    self._column_group_inv_map[key] = {group}
                else:
                    self._column_group_inv_map[key].add(group)

        # manifest: {
        #   'volume': ['modality_0.nii.gz', 'modality_1.nii.gz', 'modality_2.nii.gz', ...],
        #   'mask': ['label_0.nii.gz', 'label_1.nii.gz', 'label_2.nii.gz', ...]
        # }
        self.manifest: List[Dict[str, Any]] = self._load_and_validate_manifest()

    def _load_and_validate_manifest(self) -> List[Dict[str, Any]]:
        try:
            manifest_df: pd.DataFrame = pd.read_excel(self.manifest_file, dtype=self.column_dtype_map)
        except Exception as e:
            raise ValueError(f"can not load manifest: {str(e)}")

        retrieved_columns: Iterable[str] = self.column_key_map.keys()
        for col in retrieved_columns:
            if col not in manifest_df.columns:
                raise ValueError(f"required columns missing: {col}")

        retrieved_manifest_df: pd.DataFrame = manifest_df[retrieved_columns]
        retrieved_manifest_df = retrieved_manifest_df.rename(columns=self.column_key_map)

        manifest: List[Dict[str, Any]] = retrieved_manifest_df.to_dict(orient="records")

        # modify path attributes, cat root_dir
        modified_manifest: List[Dict[str, Any]] = []
        for sample in manifest:
            modified_sample: Dict[str, Any] = {}
            for key, value in sample.items():
                if key in self.column_key_relative_path:
                    modified_sample[key] = (self.root_dir / pl.Path(value)).as_posix()
                else:
                    modified_sample[key] = value
            modified_manifest.append(modified_sample)

        # bind groups
        grouped_manifest: List[Dict[str, Any]] = []
        for sample in modified_manifest:
            grouped_sample: Dict[str, Any] = {group: [] for group in self.column_group_map.keys()}
            for key, value in sample.items():
                if key in self._column_group_inv_map:
                    for group in self._column_group_inv_map[key]:
                        grouped_sample[group].append(value)
                else:
                    grouped_sample[key] = value
            grouped_manifest.append(grouped_sample)

        return grouped_manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def get_monai_dataset(
            self,
            dataset_class: Type[Union[mD.PersistentDataset, mD.CacheDataset, mD.LMDBDataset]],
            transform: Union[Sequence[Callable], Callable],
            **kwargs
    ) -> mD.Dataset:
        return dataset_class(self.manifest, transform, **kwargs)


if __name__ == "__main__":
    from Transform.tf_default_seg import TransformSegmentationDefaultTrain
    from typing import cast
    from monai.utils import GridSampleMode, GridSamplePadMode, PytorchPadMode, NumpyPadMode

    manifest_file: pl.Path = pl.Path(r'./Samples/split01_TJ/split01_TJ_train.xlsx')
    ds: DatasetManifestRetrieverSegmentationDefault = DatasetManifestRetrieverSegmentationDefault(
        root_dir='./Samples',
        manifest_file=manifest_file,
        column_key_map={
            'volume': 'volume',
            'mask_00_Bg': 'mask_0',
            'mask_01_EsoROI': 'mask_1',
        },
        column_key_relative_path=['volume', 'mask_0', 'mask_1'],
        column_group_map={'volume': ['volume'], 'mask': ['mask_0', 'mask_1']},
        column_dtype_map=None
    )

    transform_params = {
        'volume_key': 'volume',
        'mask_key': 'mask',
        'param_volume_tf_duplicate_items_dup_keys_volume': None,
        'param_mask_tf_duplicate_items_dup_keys_mask': None,
        'param_tf_spacing_pixdim': (1.0, 1.0, 1.0),
        'param_tf_spacing_mode_volume': GridSampleMode.BILINEAR,
        'param_tf_spacing_mode_mask': GridSampleMode.NEAREST,
        'param_tf_padding_mode_volume': GridSamplePadMode.BORDER,
        'param_tf_padding_mode_mask': GridSamplePadMode.BORDER,
        'param_tf_spatial_pad_spatial_size': (128, 128, 128),
        'param_tf_spatial_pad_mode': PytorchPadMode.REPLICATE,
        'param_tf_rand_crop_by_label_classes_spatial_size': (128, 128, 128),
        'param_tf_rand_crop_by_label_classes_ratios': (0.0, 1.0),
        'param_tf_rand_crop_by_label_classes_num_classes': 2,
        'param_tf_rand_crop_by_label_classes_num_samples': 2,
        'param_tf_scale_intensity_range_a_min': -1000,
        'param_tf_scale_intensity_range_a_max': 1000,
        'param_tf_scale_intensity_range_b_min': 0.0,
        'param_tf_scale_intensity_range_b_max': 1.0,
        'param_tf_scale_intensity_range_clip': True,
        'param_tf_allow_missing_keys': False,
        'random_seed': 0
    }
    transform: TransformSegmentationDefaultTrain = TransformSegmentationDefaultTrain(**transform_params)

    pds: mD.PersistentDataset = cast(
        mD.PersistentDataset,
        ds.get_monai_dataset(
            dataset_class=mD.PersistentDataset,
            transform=transform.get_composed_transform(),
            cache_dir='./Samples/cache'
        )
    )

    for idx, batch in enumerate(pds):
        print(f'[{idx}]')
        volume = batch[0]['volume']
        shape = volume.shape
        inspect_slice = volume[0, :, :, shape[3] // 2].numpy()
        print(inspect_slice)

    print()

    # Check reproducibility
    import random
    import torch

    all_pass: bool = True
    max_len = len(ds)
    for i in range(10):
        print(f'Reproducibility [{i}]')
        transform_1: TransformSegmentationDefaultTrain = TransformSegmentationDefaultTrain(random_seed=0)
        pds_1: mD.PersistentDataset = cast(
            mD.PersistentDataset,
            ds.get_monai_dataset(
                dataset_class=mD.PersistentDataset,
                transform=transform_1.get_composed_transform(),
                cache_dir='./Samples/cache'
            )
        )
        start_round = random.randint(0, max_len - 2)
        print(f'  Warmup Round: {start_round}', end='\n ')
        for idx in range(start_round):
            print(f' [{idx}]', end='')
            volume = pds_1[idx][0]['volume']

        tf_state = transform_1.get_state()

        transform_2: TransformSegmentationDefaultTrain = TransformSegmentationDefaultTrain(random_seed=0)
        transform_2.set_state(tf_state)
        pds_2: mD.PersistentDataset = cast(
            mD.PersistentDataset,
            ds.get_monai_dataset(
                dataset_class=mD.PersistentDataset,
                transform=transform_2.get_composed_transform(),
                cache_dir='./Samples/cache'
            )
        )
        print()

        for idx in range(start_round + 1, max_len):
            print(f'  Compare Round [{idx}]:', end=' ')
            volume_1 = pds_1[idx][0]['volume']
            volume_2 = pds_2[idx][0]['volume']
            identity: bool = torch.all(volume_1 == volume_2).item()
            print('PASS' if identity else 'FAIL')
            all_pass &= identity
        print()

    print(f'Overall Reproducibility: {all_pass}')
