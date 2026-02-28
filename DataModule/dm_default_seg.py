# -*- coding: utf-8 -*-
import os
import torch
import pathlib as pl
import monai.data.dataset as mD
from monai.data.dataloader import DataLoader
import lightning as L
from typing import Dict, Any, Optional, Iterable, Union, Literal, Type
from pandas._typing import DtypeArg
from dataclasses import dataclass

from Transform.tf_default_seg import (
    TransformSegmentationDefaultBase,
    TransformSegmentationDefaultTrain, TransformSegmentationDefaultInferencePre
)
from Dataset.dmr_default_seg import DatasetManifestRetrieverSegmentationDefault

PathLike = Union[str, os.PathLike]
PhaseLike = Literal['train', 'val', 'test', 'predict']
PipeStage = Literal['fit', 'val', 'test', 'predict']


@dataclass(frozen=True)
class DataModuleSegmentationDefaultInitArgs:
    # Dataset Retriever
    root_dir: PathLike
    manifest_file: PathLike
    column_key_map: Dict[str, str]
    column_key_relative_path: Iterable[str]
    column_group_map: Dict[str, Iterable[str]]
    column_dtype_map: Optional[DtypeArg]
    # Runtime Dataset
    dataset_cls: Type[Union[mD.PersistentDataset, mD.CacheDataset, mD.LMDBDataset]]
    dataset_params: Dict[str, Any]
    # Transform
    transform_cls: Type[TransformSegmentationDefaultBase]
    transform_params: Dict[str, Any]
    # Dataloader
    batch_size: Optional[int] = 1
    shuffle: Optional[bool] = None
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool = False
    generator_random_seed: Optional[int] = None

    @staticmethod
    def convert_to_dict(**kwargs) -> Dict[str, Any]:
        return kwargs

    def retriever_args(self) -> Dict[str, Any]:
        return DataModuleSegmentationDefaultInitArgs.convert_to_dict(
            root_dir=self.root_dir,
            manifest_file=self.manifest_file,
            column_key_map=self.column_key_map,
            column_key_relative_path=self.column_key_relative_path,
            column_group_map=self.column_group_map,
            column_dtype_map=self.column_dtype_map,
        )

    def dataset_args(self) -> Dict[str, Any]:
        return DataModuleSegmentationDefaultInitArgs.convert_to_dict(
            dataset_cls=self.dataset_cls,
            dataset_params=self.dataset_params
        )

    def transform_args(self) -> Dict[str, Any]:
        return DataModuleSegmentationDefaultInitArgs.convert_to_dict(
            transform_cls=self.transform_cls,
            transform_params=self.transform_params
        )

    def dataloader_args(self) -> Dict[str, Any]:
        return DataModuleSegmentationDefaultInitArgs.convert_to_dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            generator_random_seed=self.generator_random_seed
        )


class DataModuleSegmentationDefault(L.LightningDataModule):
    def __init__(
            self,
            train_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None,
            val_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None,
            test_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None,
            predict_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    ) -> None:
        super().__init__()
        self.init_args: Dict[str, DataModuleSegmentationDefaultInitArgs] = {}
        for stage, init_args in zip(
                ['train', 'val', 'test', 'predict'],
                [train_init_args, val_init_args, test_init_args, predict_init_args]
        ):
            if isinstance(init_args, DataModuleSegmentationDefaultInitArgs):
                self.init_args[stage] = init_args

        # records after initialization
        self.transforms: Dict[PhaseLike, TransformSegmentationDefaultBase] = {}
        self.retrievers: Dict[PhaseLike, DatasetManifestRetrieverSegmentationDefault] = {}
        self.datasets: Dict[PhaseLike, mD.Dataset] = {}
        self.dataloaders: Dict[PhaseLike, DataLoader] = {}

    def prepare_data(self) -> None:
        stage: str
        init: DataModuleSegmentationDefaultInitArgs
        for stage, init in self.init_args.items():
            root_dir: PathLike = init.root_dir
            manifest_file: PathLike = init.manifest_file
            if not pl.Path(root_dir).exists():
                raise ValueError(f"Root for '{stage}' dataset not exists: {root_dir}")
            if not pl.Path(manifest_file).exists():
                raise ValueError(f"Manifest for '{stage}' dataset not exists: {manifest_file}")

    def setup(self, stage: Optional[PipeStage] = None) -> None:
        if stage == 'fit' or stage is None:
            if 'train' in self.init_args:
                init: DataModuleSegmentationDefaultInitArgs = self.init_args['train']
                # Transform
                transform_cls: Type[TransformSegmentationDefaultBase] = init.transform_cls
                transform_params: Dict[str, Any] = init.transform_params

                # Create Transform pipe
                self.transforms['train'] = transform_cls(**transform_params)  # noqa

                # Create Dataset
                self.retrievers['train'] = DatasetManifestRetrieverSegmentationDefault(**init.retriever_args())
                dataset: mD.Dataset = self.retrievers['train'].get_monai_dataset(
                    dataset_class=init.dataset_cls,
                    transform=self.transforms['train'].get_composed_transform(),  # use composed transform to accelerate
                    **init.dataset_params
                )
                self.datasets['train'] = dataset

            elif stage is not None:
                raise ValueError(f"Stage 'train' has invalid init arguments")

            if 'val' in self.init_args:
                init: DataModuleSegmentationDefaultInitArgs = self.init_args['val']
                # Transform
                transform_cls: Type[TransformSegmentationDefaultBase] = init.transform_cls
                transform_params: Dict[str, Any] = init.transform_params

                # Create Transform pipe
                self.transforms['val'] = transform_cls(**transform_params)  # noqa

                # Create Dataset
                self.retrievers['val'] = DatasetManifestRetrieverSegmentationDefault(**init.retriever_args())
                # Create Dataset
                self.retrievers['val'] = DatasetManifestRetrieverSegmentationDefault(**init.retriever_args())
                dataset: mD.Dataset = self.retrievers['val'].get_monai_dataset(
                    dataset_class=init.dataset_cls,
                    transform=self.transforms['val'].get_composed_transform(),  # use composed transform to accelerate
                    **init.dataset_params
                )
                self.datasets['val'] = dataset

            elif stage is not None:
                raise ValueError(f"Stage 'val' has invalid init arguments")

        if stage == 'val':
            if 'val' in self.init_args:
                init: DataModuleSegmentationDefaultInitArgs = self.init_args['val']
                # Transform
                transform_cls: Type[TransformSegmentationDefaultBase] = init.transform_cls
                transform_params: Dict[str, Any] = init.transform_params

                # Create Transform pipe
                self.transforms['val'] = transform_cls(**transform_params)  # noqa

                # Create Dataset
                self.retrievers['val'] = DatasetManifestRetrieverSegmentationDefault(**init.retriever_args())
                dataset: mD.Dataset = self.retrievers['val'].get_monai_dataset(
                    dataset_class=init.dataset_cls,
                    transform=self.transforms['val'].get_composed_transform(),  # use composed transform to accelerate
                    **init.dataset_params
                )
                self.datasets['val'] = dataset

            elif stage is not None:
                raise ValueError(f"Stage 'val' has invalid init arguments")

        if stage == 'test' or stage is None:
            if 'test' in self.init_args:
                init: DataModuleSegmentationDefaultInitArgs = self.init_args['test']
                # Transform
                transform_cls: Type[TransformSegmentationDefaultBase] = init.transform_cls
                transform_params: Dict[str, Any] = init.transform_params

                # Create Transform pipe
                self.transforms['test'] = transform_cls(**transform_params)  # noqa

                # Create Dataset
                self.retrievers['test'] = DatasetManifestRetrieverSegmentationDefault(**init.retriever_args())
                dataset: mD.Dataset = self.retrievers['test'].get_monai_dataset(
                    dataset_class=init.dataset_cls,
                    transform=self.transforms['test'].get_composed_transform(),  # use composed transform to accelerate
                    **init.dataset_params
                )
                self.datasets['test'] = dataset

            elif stage is not None:
                raise ValueError(f"Stage 'test' has invalid init arguments")

        if stage == 'predict' or stage is None:
            if 'predict' in self.init_args:
                init: DataModuleSegmentationDefaultInitArgs = self.init_args['test']
                # Transform
                transform_cls: Type[TransformSegmentationDefaultBase] = init.transform_cls
                transform_params: Dict[str, Any] = init.transform_params

                # Create Transform pipe
                self.transfrom['predict'] = transform_cls(**transform_params)  # noqa

                # Create Dataset
                self.retrievers['predict'] = DatasetManifestRetrieverSegmentationDefault(**init.retriever_args())
                dataset: mD.Dataset = self.retrievers['predict'].get_monai_dataset(
                    dataset_class=init.dataset_cls,
                    transform=self.transforms['predict'].get_composed_transform(),
                    # use composed transform to accelerate
                    **init.dataset_params
                )
                self.datasets['predict'] = dataset

            elif stage is not None:
                raise ValueError(f"Stage 'predict' has invalid init arguments")

    def train_dataloader(self) -> DataLoader:
        if 'train' not in self.datasets:
            raise AttributeError("Dataset for stage 'train' has not been initialized, please initialize first")

        init: DataModuleSegmentationDefaultInitArgs = self.init_args['train']
        dataloader_args: Dict[str, Any] = init.dataloader_args()
        if init.generator_random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(init.generator_random_seed)
            dataloader_args['generator'] = generator
        del dataloader_args['generator_random_seed']
        loader: DataLoader = DataLoader(
            dataset=self.datasets['train'],
            **dataloader_args
        )
        self.dataloaders['train'] = loader
        return loader

    def val_dataloader(self) -> DataLoader:
        if 'val' not in self.datasets:
            raise KeyError("Dataset for stage 'val' has not been initialized, please initialize first")

        init: DataModuleSegmentationDefaultInitArgs = self.init_args['val']
        dataloader_args: Dict[str, Any] = init.dataloader_args()
        if init.generator_random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(init.generator_random_seed)
            dataloader_args['generator'] = generator
        del dataloader_args['generator_random_seed']
        loader: DataLoader = DataLoader(
            dataset=self.datasets['val'],
            **dataloader_args
        )
        self.dataloaders['val'] = loader
        return loader

    def test_dataloader(self) -> DataLoader:
        if 'test' not in self.datasets:
            raise KeyError("Dataset for stage 'test' has not been initialized, please initialize first")

        init: DataModuleSegmentationDefaultInitArgs = self.init_args['test']
        dataloader_args: Dict[str, Any] = init.dataloader_args()
        if init.generator_random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(init.generator_random_seed)
            dataloader_args['generator'] = generator
        del dataloader_args['generator_random_seed']
        loader: DataLoader = DataLoader(
            dataset=self.datasets['test'],
            **dataloader_args
        )
        self.dataloaders['test'] = loader
        return loader

    def predict_dataloader(self) -> DataLoader:
        if 'predict' not in self.datasets:
            raise KeyError("Dataset for stage 'predict' has not been initialized, please initialize first")

        init: DataModuleSegmentationDefaultInitArgs = self.init_args['predict']
        dataloader_args: Dict[str, Any] = init.dataloader_args()
        if init.generator_random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(init.generator_random_seed)
            dataloader_args['generator'] = generator
        del dataloader_args['generator_random_seed']
        loader: DataLoader = DataLoader(
            dataset=self.datasets['predict'],
            **dataloader_args
        )
        self.dataloaders['predict'] = loader
        return loader

    def state_dict(self) -> Dict[str, Any]:
        state_dict: Dict[PhaseLike, Any] = {}

        # save Transform states
        for phase, transform in self.transforms.items():
            if hasattr(transform, 'get_state'):
                try:
                    if phase not in state_dict:
                        state_dict[phase] = {}
                    state_dict[phase][f'transform_{type(transform).__name__}'] = transform.get_state()
                except Exception as e:
                    raise ValueError(f"Fail to acquire {type(transform).__name__} state for phase {phase}: {str(e)}")

        # save Dataloader states
        for phase, loader in self.dataloaders.items():
            if hasattr(loader, 'generator') and loader.generator is not None and hasattr(loader.generator, 'get_state'):
                try:
                    if phase not in state_dict:
                        state_dict[phase] = {}
                    state_dict[phase][f'dataloader_generator'] = loader.generator.get_state()
                except Exception as e:
                    raise ValueError(f"Fail to acquire generator state for phase {phase}: {str(e)}")

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # load transform states
        for phase, transform in self.transforms.items():
            transform_name: str = f'transform_{type(transform).__name__}'
            if phase in state_dict and transform_name in state_dict[phase] and hasattr(transform, 'set_state'):
                try:
                    transform.set_state(state_dict[phase][transform_name])
                except Exception as e:
                    raise ValueError(f"Fail to load {type(transform).__name__} state for phase {phase}: {str(e)}")

        # load Dataloader states
        for phase, loader in self.dataloaders.items():
            if ('dataloader_generator' in state_dict[phase] and hasattr(loader, 'generator')
                    and hasattr(loader.generator, 'set_state')):
                try:
                    loader.generator.set_state(state_dict[phase][f'dataloader_generator'])
                except Exception as e:
                    raise ValueError(f"Fail to acquire generator state for phase {phase}: {str(e)}")


if __name__ == "__main__":
    from monai.utils import GridSampleMode, GridSamplePadMode, PytorchPadMode, NumpyPadMode
    import torch
    import numpy as np

    init_args_train: DataModuleSegmentationDefaultInitArgs = DataModuleSegmentationDefaultInitArgs(
        root_dir='./Samples',
        manifest_file='./Samples/split01_TJ/split01_TJ_train.xlsx',
        column_key_map={
            'volume': 'volume',
            'mask_00_Bg': 'mask_0',
            'mask_01_EsoROI': 'mask_1',
        },
        column_key_relative_path=['volume', 'mask_0', 'mask_1'],
        column_group_map={'volume': ['volume'], 'mask': ['mask_0', 'mask_1']},
        column_dtype_map=None,
        dataset_cls=mD.PersistentDataset,
        dataset_params={
            'cache_dir': './Samples/cache'
        },
        transform_cls=TransformSegmentationDefaultTrain,
        transform_params={
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
            'random_seed': 0,
        },
        batch_size=3,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        generator_random_seed=0
    )

    init_args_test: DataModuleSegmentationDefaultInitArgs = DataModuleSegmentationDefaultInitArgs(
        root_dir='./Samples',
        manifest_file='./Samples/split01_TJ/split01_TJ_test.xlsx',
        column_key_map={
            'volume': 'volume',
            'mask_00_Bg': 'mask_0',
            'mask_01_EsoROI': 'mask_1'
        },
        column_key_relative_path=['volume', 'mask_0', 'mask_1'],
        column_group_map={'volume': ['volume'], 'mask': ['mask_0', 'mask_1']},
        column_dtype_map=None,
        dataset_cls=mD.PersistentDataset,
        dataset_params={
            'cache_dir': './Samples/cache'
        },
        transform_cls=TransformSegmentationDefaultInferencePre,
        transform_params={
            'volume_key': 'volume',
            'mask_key': 'mask',
            'param_volume_tf_duplicate_items_dup_keys_volume': 'volume_raw',
            'param_mask_tf_duplicate_items_dup_keys_mask': 'mask_raw',
            'param_tf_spacing_pixdim': (1.0, 1.0, 1.0),
            'param_tf_spacing_mode_volume': GridSampleMode.BILINEAR,
            'param_tf_spacing_mode_mask': GridSampleMode.NEAREST,
            'param_tf_padding_mode_volume': GridSamplePadMode.BORDER,
            'param_tf_padding_mode_mask': GridSamplePadMode.BORDER,
            'param_tf_scale_intensity_range_a_min': -1000,
            'param_tf_scale_intensity_range_a_max': 1000,
            'param_tf_scale_intensity_range_b_min': 0.0,
            'param_tf_scale_intensity_range_b_max': 1.0,
            'param_tf_scale_intensity_range_clip': True
        },
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    print(f"[1] Test {DataModuleSegmentationDefault.__name__} in phase 'train'")
    print(f"------------------------------------------------------")

    # Initialize DataModule
    dm = DataModuleSegmentationDefault(
        train_init_args=init_args_train,
        test_init_args=init_args_test
    )

    # Prepare data and setup
    dm.prepare_data()
    dm.setup()

    # # Get train loader
    train_loader = dm.train_dataloader()

    # Try loading data
    print(f"Try loading (train) ...")
    train_len = len(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        print(f"  Batch {batch_idx + 1}/{train_len}")

        # print batch info
        print(f"  Batch keys: {list(batch.keys())}")
        required_keys = ['volume', 'mask']
        for key in required_keys:
            assert key in batch, f"Key '{key}' is missing"
            print(f"  '{key}' shape: {batch[key].shape}")

        print()

    print(f"[2] Test {DataModuleSegmentationDefault.__name__} in phase 'test'")
    print(f"------------------------------------------------------")

    # Get test loader
    test_loader = dm.test_dataloader()

    # Try loading data
    print(f"Try loading (test)...")
    test_len = len(test_loader)
    for batch_idx, batch in enumerate(test_loader):
        print(f"  Batch {batch_idx + 1}/{test_len}")

        # print batch info
        print(f"  Batch keys: {list(batch.keys())}")
        required_keys = ['volume', 'mask', 'volume_raw', 'mask_raw']
        for key in required_keys:
            assert key in batch, f"Key '{key}' is missing"
            print(f"  '{key}' shape: {batch[key].shape}")

        print()

    print(f"[3] Test {DataModuleSegmentationDefault.__name__} reproducibility")
    print(f"------------------------------------------------------")

    # Check reproducibility
    epochs: int = 10
    warmup_epochs: int = np.random.randint(1, epochs - 1)
    train_loader = dm.train_dataloader()
    len_batches: int = len(train_loader)

    # Warmup
    for epoch in range(warmup_epochs):
        print(f'  Warmup Epoch [{epoch + 1}/{warmup_epochs}]:\n'
              f'    Batch:', end='')
        for batch_idx, batch in enumerate(train_loader):
            print(f" [{batch_idx}]", end='')
        print()
    print()

    # Get-Set DataModule state
    dm_state = dm.state_dict()
    state_path: pl.Path = pl.Path('./Samples/ckpt/dm_state.pth')
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dm_state, state_path)
    loaded_state = torch.load(state_path)

    # Initialize DataModule
    dm_2 = DataModuleSegmentationDefault(
        train_init_args=init_args_train,
        test_init_args=init_args_test
    )

    # Prepare data and setup
    dm_2.prepare_data()
    dm_2.setup()
    train_loader_2 = dm_2.train_dataloader()
    dm_2.load_state_dict(loaded_state)  # cation: shall always load state after DataLoader initialization
    assert len(train_loader_2) == len_batches

    all_pass: bool = True
    for epoch in range(warmup_epochs, epochs):
        print(f'  Compare Epoch [{epoch + 1}/{epochs}]:\n',
              f'    Compare Batch:')
        for batch_idx, (batch_1, batch_2) in enumerate(zip(train_loader, train_loader_2)):
            print(f"      [{batch_idx}]", end=' ')
            volume_1 = batch_1['volume']
            volume_2 = batch_2['volume']
            mask_1 = batch_1['mask']
            mask_2 = batch_2['mask']
            identity: bool = torch.all(volume_1 == volume_2).item() and torch.all(mask_1 == mask_2).item()
            print('PASS' if identity else 'FAIL')
            all_pass &= identity

        print()

    print(f'Overall Reproducibility: {all_pass}')

    pass
