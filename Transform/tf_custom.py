# -*- coding: utf-8 -*-
"""
Custom Transform Module

This module provides custom extended MONAI transforms for data preprocessing and augmentation.

Classes:
    RenameItemsd: Renames specified keys in the data dictionary
    DuplicateItemsd: Creates copies of specified keys in the data dictionary
    RandCropByLabelClassesd: Extends MONAI's RandCropByLabelClassesd to support random state retrieval
"""
import monai.transforms as mT
from typing import Collection, Union, Hashable, Dict, Any
from monai.utils import TransformBackends

KeysCollection = Union[Collection[Hashable], Hashable]


class RenameItemsd(mT.MapTransform):
    """
    Transform that renames specified keys in the data dictionary
    
    Args:
        keys: Collection of keys to rename
        renamed_keys: Collection of new names for the keys
        
    Note:
        The length of keys and renamed_keys must be the same
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, renamed_keys: KeysCollection) -> None:
        """
        Initialize the transform
        
        Args:
            keys: Collection of keys to rename
            renamed_keys: Collection of new names for the keys
        """
        super().__init__(keys)
        self.renamed_keys = renamed_keys
        assert len(keys) == len(renamed_keys), 'keys and renamed_keys must have the same length'

    def __call__(self, data):
        """
        Apply the transform to the data
        
        Args:
            data: Input data dictionary
            
        Returns:
            Data dictionary with renamed keys
        """
        d = dict(data)
        for k, rk in zip(self.keys, self.renamed_keys):
            v = d.pop(k)
            d[rk] = v
        return d


class DuplicateItemsd(mT.MapTransform):
    """
    Transform that creates copies of specified keys in the data dictionary
    
    Args:
        keys: Collection of keys to duplicate
        dup_keys: Collection of new keys for the duplicates
        
    Note:
        The length of keys and dup_keys must be the same
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, dup_keys: KeysCollection) -> None:
        """
        Initialize the transform
        
        Args:
            keys: Collection of keys to duplicate
            dup_keys: Collection of new keys for the duplicates
        """
        super().__init__(keys)
        self.dup_keys = dup_keys
        assert len(keys) == len(dup_keys), 'keys and dup_keys must have the same length'

    def __call__(self, data):
        """
        Apply the transform to the data
        
        Args:
            data: Input data dictionary
            
        Returns:
            Data dictionary with duplicated keys
        """
        d = dict(data)
        for k, dk in zip(self.keys, self.dup_keys):
            if isinstance(dk, str) and k in d:
                d[dk] = d[k]
        return d

class RandCropByLabelClassesd(mT.RandCropByLabelClassesd):
    """
    Extended version of MONAI's RandCropByLabelClassesd that supports random state retrieval
    
    This transform randomly crops the input based on label classes distribution
    and adds the ability to get the random state for reproducibility
    """
    def get_random_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current random state of the cropper
        
        Returns:
            Dictionary containing the random state
        """
        return self.cropper.R.get_state()