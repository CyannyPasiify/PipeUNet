# -*- coding: utf-8 -*-
"""
Default Segmentation DataModule

This module provides a PyTorch Lightning DataModule for segmentation tasks,
managing data loading, preprocessing, and batching for training, validation,
testing, and prediction phases.

Classes:
    DataModuleSegmentationDefaultInitArgs: Dataclass for DataModule initialization arguments
    DataModuleSegmentationDefault: PyTorch Lightning DataModule for segmentation tasks

Key Features:
    - Supports multiple phases (train, val, test, predict)
    - Configurable dataset types (PersistentDataset, CacheDataset, LMDBDataset)
    - Customizable transforms for each phase
    - Reproducible data loading with random seed management
    - State management for reproducibility
"""
import os
import pathlib as pl
from monai.data.dataloader import DataLoader
import lightning as L
from typing import Dict, Any, Optional, Union, Literal
from dataclasses import dataclass
from Dataset.dataset_configurer import ConfigDatasetBase, ConfigDatasetPersistent
from Transform.transform_configurer import (
    ConfigTransformBase,
    ConfigTransformSegmentationDefaultTrain,
    ConfigTransformSegmentationDefaultInferencePre,
    ConfigTransformSegmentationDefaultInferencePost
)
from Dataset.dataset_manifest_retriever_configurer import ConfigDatasetManifestRetrieverSegmentationDefault

PathLike = Union[str, os.PathLike]
PhaseLike = Literal['train', 'val', 'test', 'predict']
PipeStage = Literal['fit', 'val', 'test', 'predict']


@dataclass
class DataModuleSegmentationDefaultInitArgs:
    """
    Dataclass for DataModule initialization arguments
    
    Contains all parameters needed to initialize a DataModule for segmentation tasks
    """
    # Dataset Retriever
    retriever: ConfigDatasetManifestRetrieverSegmentationDefault = ConfigDatasetManifestRetrieverSegmentationDefault()
    # Runtime Dataset
    dataset: ConfigDatasetBase = ConfigDatasetPersistent()  # Wrapped MONAI dataset class to use
    # Transform
    transform: ConfigTransformBase = ConfigTransformSegmentationDefaultInferencePre  # Transform class to use
    # Dataloader (We use original torch Dataloader)
    batch_size: Optional[int] = 1  # Batch size for dataloader
    shuffle: Optional[bool] = None  # Whether to shuffle data
    num_workers: int = 0  # Number of worker processes
    pin_memory: bool = False  # Whether to pin memory
    drop_last: bool = False  # Whether to drop the last incomplete batch
    persistent_workers: bool = False  # Whether to use persistent workers
    generator_random_seed: Optional[int] = None  # Random seed for dataloader generator

    @staticmethod
    def convert_to_dict(**kwargs) -> Dict[str, Any]:
        """
        Convert keyword arguments to a dictionary
        
        Args:
            **kwargs: Keyword arguments to convert
            
        Returns:
            Dictionary of keyword arguments
        """
        return kwargs

    def dataloader_args(self) -> Dict[str, Any]:
        """
        Get arguments for the dataloader
        
        Returns:
            Dictionary of dataloader arguments
        """
        return DataModuleSegmentationDefaultInitArgs.convert_to_dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            generator_random_seed=self.generator_random_seed
        )


@dataclass
class DataModuleSegmentationDefault(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for segmentation tasks
    
    Manages data loading, preprocessing, and batching for training, validation,
    testing, and prediction phases

    Initialize the DataModule

    Args:
        train_init_args: Initialization arguments for the training phase
        val_init_args: Initialization arguments for the validation phase
        test_init_args: Initialization arguments for the testing phase
        predict_init_args: Initialization arguments for the prediction phase
    """

    def __init__(
            self,
            train_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None,
            val_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None,
            test_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None,
            predict_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = None
    ):
        super().__init__()
        self.train_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = train_init_args
        self.val_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = val_init_args
        self.test_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = test_init_args
        self.predict_init_args: Optional[DataModuleSegmentationDefaultInitArgs] = predict_init_args

        # Store initialization arguments for each phase
        self.init_args: Dict[str, DataModuleSegmentationDefaultInitArgs] = {}
        for stage, init_args in zip(
                ['train', 'val', 'test', 'predict'],
                [self.train_init_args, self.val_init_args, self.test_init_args, self.predict_init_args]
        ):
            if isinstance(init_args, DataModuleSegmentationDefaultInitArgs):
                self.init_args[stage] = init_args

        # Records after initialization
        self.transforms: Dict[PhaseLike, ConfigTransformBase] = {}  # Transforms for each phase
        # Dataset retrievers for each phase
        self.retrievers: Dict[PhaseLike, ConfigDatasetManifestRetrieverSegmentationDefault] = {}
        self.datasets: Dict[PhaseLike, ConfigDatasetBase] = {}  # Datasets for each phase
        self.dataloaders: Dict[PhaseLike, DataLoader] = {}  # Dataloaders for each phase

    def prepare_data(self) -> None:
        """
        Prepare data for use
        
        This method is called once on the main process before training
        Used to validate that all required data directories and files exist
        
        Raises:
            ValueError: If root directory or manifest file does not exist for any phase
        """
        stage: str
        init: DataModuleSegmentationDefaultInitArgs
        for stage, init in self.init_args.items():
            root_dir: PathLike = init.retriever.root_dir
            manifest_file: PathLike = init.retriever.manifest_file
            if not pl.Path(root_dir).exists():
                raise ValueError(f"Root for '{stage}' dataset not exists: {root_dir}")
            if not pl.Path(manifest_file).exists():
                raise ValueError(f"Manifest for '{stage}' dataset not exists: {manifest_file}")

    def setup(self, stage: Optional[PipeStage] = None) -> None:
        """
        Setup data for each phase
        
        This method is called on every process before training/validation/testing/prediction
        Initializes transforms, retrievers, and datasets for the specified stage
        
        Args:
            stage: Current stage (fit, val, test, predict)
            
        Raises:
            ValueError: If any stage has invalid init arguments
        """
        # Setup for training and validation phases
        if stage == 'fit' or stage is None:
            if 'train' in self.init_args:
                init: DataModuleSegmentationDefaultInitArgs = self.init_args['train']
                # Create transform pipeline
                init.transform.init_essentials()
                self.transforms['train'] = init.transform
                # Create dataset retriever
                init.retriever.init_essentials()
                self.retrievers['train'] = init.retriever
                # Create dataset with composed transform for acceleration
                self.datasets['train'] = self.retrievers['train'].get_assembled_dataset(
                    init.dataset, init.transform.get_composed_transform()
                )

            elif stage is not None:
                raise ValueError(f"Stage 'train' has invalid init arguments")

            if 'val' in self.init_args:
                init: DataModuleSegmentationDefaultInitArgs = self.init_args['val']
                # Create transform pipeline
                init.transform.init_essentials()
                self.transforms['val'] = init.transform
                # Create dataset retriever
                init.retriever.init_essentials()
                self.retrievers['val'] = init.retriever
                # Create dataset with composed transform for acceleration
                self.datasets['val'] = self.retrievers['val'].get_assembled_dataset(
                    init.dataset, init.transform.get_composed_transform()
                )

            elif stage is not None:
                raise ValueError(f"Stage 'val' has invalid init arguments")

        # Setup for validation phase only
        if stage == 'val':
            if 'val' in self.init_args:
                init: DataModuleSegmentationDefaultInitArgs = self.init_args['val']
                # Create transform pipeline
                init.transform.init_essentials()
                self.transforms['val'] = init.transform
                # Create dataset retriever
                init.retriever.init_essentials()
                self.retrievers['val'] = init.retriever
                # Create dataset with composed transform for acceleration
                self.datasets['val'] = self.retrievers['val'].get_assembled_dataset(
                    init.dataset, init.transform.get_composed_transform()
                )

            elif stage is not None:
                raise ValueError(f"Stage 'val' has invalid init arguments")

        # Setup for testing phase
        if stage == 'test' or stage is None:
            if 'test' in self.init_args:
                init: DataModuleSegmentationDefaultInitArgs = self.init_args['test']
                # Create transform pipeline
                init.transform.init_essentials()
                self.transforms['test'] = init.transform
                # Create dataset retriever
                init.retriever.init_essentials()
                self.retrievers['test'] = init.retriever
                # Create dataset with composed transform for acceleration
                self.datasets['test'] = self.retrievers['test'].get_assembled_dataset(
                    init.dataset, init.transform.get_composed_transform()
                )

            elif stage is not None:
                raise ValueError(f"Stage 'test' has invalid init arguments")

        # Setup for prediction phase
        if stage == 'predict' or stage is None:
            if 'predict' in self.init_args:
                init: DataModuleSegmentationDefaultInitArgs = self.init_args['predict']
                # Create transform pipeline
                init.transform.init_essentials()
                self.transforms['predict'] = init.transform
                # Create dataset retriever
                init.retriever.init_essentials()
                self.retrievers['predict'] = init.retriever
                # Create dataset with composed transform for acceleration
                self.datasets['predict'] = self.retrievers['predict'].get_assembled_dataset(
                    init.dataset, init.transform.get_composed_transform()
                )

            elif stage is not None:
                raise ValueError(f"Stage 'predict' has invalid init arguments")

    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader
        
        Returns:
            DataLoader for training data
            
        Raises:
            AttributeError: If training dataset has not been initialized
        """
        if 'train' not in self.datasets:
            raise AttributeError("Dataset for stage 'train' has not been initialized, please setup first")

        init: DataModuleSegmentationDefaultInitArgs = self.init_args['train']
        dataloader_args: Dict[str, Any] = init.dataloader_args()
        # Set random seed for reproducibility
        if init.generator_random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(init.generator_random_seed)
            dataloader_args['generator'] = generator
        del dataloader_args['generator_random_seed']

        loader: DataLoader = DataLoader(
            dataset=self.datasets['train'].get_dataset(),
            **dataloader_args
        )
        self.dataloaders['train'] = loader
        return loader

    def val_dataloader(self) -> DataLoader:
        """
        Create validation dataloader
        
        Returns:
            DataLoader for validation data
            
        Raises:
            KeyError: If validation dataset has not been initialized
        """
        if 'val' not in self.datasets:
            raise KeyError("Dataset for stage 'val' has not been initialized, please setup first")

        init: DataModuleSegmentationDefaultInitArgs = self.init_args['val']
        dataloader_args: Dict[str, Any] = init.dataloader_args()
        # Set random seed for reproducibility
        if init.generator_random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(init.generator_random_seed)
            dataloader_args['generator'] = generator
        del dataloader_args['generator_random_seed']
        loader: DataLoader = DataLoader(
            dataset=self.datasets['val'].get_dataset(),
            **dataloader_args
        )
        self.dataloaders['val'] = loader
        return loader

    def test_dataloader(self) -> DataLoader:
        """
        Create testing dataloader
        
        Returns:
            DataLoader for testing data
            
        Raises:
            KeyError: If testing dataset has not been initialized
        """
        if 'test' not in self.datasets:
            raise KeyError("Dataset for stage 'test' has not been initialized, please setup first")

        init: DataModuleSegmentationDefaultInitArgs = self.init_args['test']
        dataloader_args: Dict[str, Any] = init.dataloader_args()
        # Set random seed for reproducibility
        if init.generator_random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(init.generator_random_seed)
            dataloader_args['generator'] = generator
        del dataloader_args['generator_random_seed']
        loader: DataLoader = DataLoader(
            dataset=self.datasets['test'].get_dataset(),
            **dataloader_args
        )
        self.dataloaders['test'] = loader
        return loader

    def predict_dataloader(self) -> DataLoader:
        """
        Create prediction dataloader
        
        Returns:
            DataLoader for prediction data
            
        Raises:
            KeyError: If prediction dataset has not been initialized
        """
        if 'predict' not in self.datasets:
            raise KeyError("Dataset for stage 'predict' has not been initialized, please setup first")

        init: DataModuleSegmentationDefaultInitArgs = self.init_args['predict']
        dataloader_args: Dict[str, Any] = init.dataloader_args()
        # Set random seed for reproducibility
        if init.generator_random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(init.generator_random_seed)
            dataloader_args['generator'] = generator
        del dataloader_args['generator_random_seed']
        loader: DataLoader = DataLoader(
            dataset=self.datasets['predict'].get_dataset(),
            **dataloader_args
        )
        self.dataloaders['predict'] = loader
        return loader

    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary for the DataModule
        
        Returns:
            Dictionary containing the state of transforms and dataloaders
            
        Raises:
            ValueError: If unable to acquire state for any transform or dataloader
        """
        state_dict: Dict[PhaseLike, Any] = {}

        # Save transform states
        for phase, transform in self.transforms.items():
            if hasattr(transform, 'get_state'):
                try:
                    if phase not in state_dict:
                        state_dict[phase] = {}
                    state_dict[phase][f'transform_{type(transform).__name__}'] = transform.get_state()
                except Exception as e:
                    raise ValueError(f"Fail to acquire {type(transform).__name__} state for phase {phase}: {str(e)}")

        # Save dataloader states
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
        """
        Load the state dictionary for the DataModule
        
        Args:
            state_dict: Dictionary containing the state of transforms and dataloaders
            
        Raises:
            ValueError: If unable to load state for any transform or dataloader
        """
        # Load transform states
        for phase, transform in self.transforms.items():
            transform_name: str = f'transform_{type(transform).__name__}'
            if phase in state_dict and transform_name in state_dict[phase] and hasattr(transform, 'set_state'):
                try:
                    transform.set_state(state_dict[phase][transform_name])
                except Exception as e:
                    raise ValueError(f"Fail to load {type(transform).__name__} state for phase {phase}: {str(e)}")

        # Load dataloader states
        for phase, loader in self.dataloaders.items():
            if ('dataloader_generator' in state_dict[phase] and hasattr(loader, 'generator')
                    and hasattr(loader.generator, 'set_state')):
                try:
                    loader.generator.set_state(state_dict[phase][f'dataloader_generator'])
                except Exception as e:
                    raise ValueError(f"Fail to acquire generator state for phase {phase}: {str(e)}")


if __name__ == "__main__":
    from monai.utils import GridSampleMode, GridSamplePadMode, PytorchPadMode
    import torch
    import numpy as np

    init_args_train: DataModuleSegmentationDefaultInitArgs = DataModuleSegmentationDefaultInitArgs(
        ConfigDatasetManifestRetrieverSegmentationDefault(
            root_dir='./Samples',
            manifest_file='./Samples/split01_TJ/split01_TJ_train.xlsx',
            column_key_map={
                'volume': 'volume',
                'mask_00_Bg': 'mask_0',
                'mask_01_EsoROI': 'mask_1',
            },
            column_key_relative_path=['volume', 'mask_0', 'mask_1'],
            column_group_map={'volume': ['volume'], 'mask': ['mask_0', 'mask_1']},
            column_dtype_map=None
        ),
        dataset=ConfigDatasetPersistent(cache_dir='./Samples/datamodule_test/cache'),
        transform=ConfigTransformSegmentationDefaultTrain(
            volume_key='volume',
            mask_key='mask',
            param_volume_tf_duplicate_items_dup_keys_volume=None,
            param_mask_tf_duplicate_items_dup_keys_mask=None,
            param_tf_spacing_pixdim=(1.0, 1.0, 1.0),
            param_tf_spacing_mode_volume=GridSampleMode.BILINEAR,
            param_tf_spacing_mode_mask=GridSampleMode.NEAREST,
            param_tf_padding_mode_volume=GridSamplePadMode.BORDER,
            param_tf_padding_mode_mask=GridSamplePadMode.BORDER,
            param_tf_spatial_pad_spatial_size=(128, 128, 128),
            param_tf_spatial_pad_mode=PytorchPadMode.REPLICATE,
            param_tf_rand_crop_by_label_classes_spatial_size=(128, 128, 128),
            param_tf_rand_crop_by_label_classes_ratios=[0.0, 1.0],
            param_tf_rand_crop_by_label_classes_num_classes=2,
            param_tf_rand_crop_by_label_classes_num_samples=2,
            param_tf_scale_intensity_range_a_min=-1000,
            param_tf_scale_intensity_range_a_max=1000,
            param_tf_scale_intensity_range_b_min=0.0,
            param_tf_scale_intensity_range_b_max=1.0,
            param_tf_scale_intensity_range_clip=True,
            param_tf_allow_missing_keys=False,
            random_seed=0,
        ),
        batch_size=3,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        # Set persistent_workers=false to enable resumable reproducibility.
        # When false, Dataloader will use the specified RNG to init each worker every epoch,
        # so by recording the RNG state, worker states are recorded properly.
        # But if persistent_workers=true, workers will be initialized only one time at the beginning, after which
        # their states can not be reached anymore, so can not be identified by recording the RNG state.
        # After resuming, workers will be reinitialized by the recoded RNG, which differs from the former state.
        persistent_workers=False,
        generator_random_seed=0
    )

    init_args_test: DataModuleSegmentationDefaultInitArgs = DataModuleSegmentationDefaultInitArgs(
        ConfigDatasetManifestRetrieverSegmentationDefault(
            root_dir='./Samples',
            manifest_file='./Samples/split01_TJ/split01_TJ_test.xlsx',
            column_key_map={
                'volume': 'volume',
                'mask_00_Bg': 'mask_0',
                'mask_01_EsoROI': 'mask_1'
            },
            column_key_relative_path=['volume', 'mask_0', 'mask_1'],
            column_group_map={'volume': ['volume'], 'mask': ['mask_0', 'mask_1']},
            column_dtype_map=None
        ),
        dataset=ConfigDatasetPersistent(cache_dir='./Samples/datamodule_test/cache'),
        transform=ConfigTransformSegmentationDefaultInferencePre(
            volume_key='volume',
            mask_key='mask',
            param_volume_tf_duplicate_items_dup_keys_volume='volume_raw',
            param_mask_tf_duplicate_items_dup_keys_mask='mask_raw',
            param_tf_spacing_pixdim=(1.0, 1.0, 1.0),
            param_tf_spacing_mode_volume=GridSampleMode.BILINEAR,
            param_tf_spacing_mode_mask=GridSampleMode.NEAREST,
            param_tf_padding_mode_volume=GridSamplePadMode.BORDER,
            param_tf_padding_mode_mask=GridSamplePadMode.BORDER,
            param_tf_scale_intensity_range_a_min=-1000,
            param_tf_scale_intensity_range_a_max=1000,
            param_tf_scale_intensity_range_b_min=0.0,
            param_tf_scale_intensity_range_b_max=1.0,
            param_tf_scale_intensity_range_clip=True
        ),
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
    state_path: pl.Path = pl.Path('./Samples/datamodule_test/ckpt/dm_state.pth')
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dm_state, state_path)
    loaded_state = torch.load(state_path, weights_only=False)

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
