# -*- coding: utf-8 -*-
import monai.transforms as mT
from typing import Collection, Union, Hashable, Dict, Any
from monai.utils import TransformBackends

KeysCollection = Union[Collection[Hashable], Hashable]


class RenameItemsd(mT.MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, renamed_keys: KeysCollection) -> None:
        super().__init__(keys)
        self.renamed_keys = renamed_keys
        assert len(keys) == len(renamed_keys), 'keys and renamed_keys must have the same length'

    def __call__(self, data):
        d = dict(data)
        for k, rk in zip(self.keys, self.renamed_keys):
            v = d.pop(k)
            d[rk] = v
        return d


class DuplicateItemsd(mT.MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, dup_keys: KeysCollection) -> None:
        super().__init__(keys)
        self.dup_keys = dup_keys
        assert len(keys) == len(dup_keys), 'keys and dup_keys must have the same length'

    def __call__(self, data):
        d = dict(data)
        for k, dk in zip(self.keys, self.dup_keys):
            if isinstance(dk, str) and k in d:
                d[dk] = d[k]
        return d

class RandCropByLabelClassesd(mT.RandCropByLabelClassesd):
    def get_random_state(self) -> Dict[str, Dict[str, Any]]:
        return self.cropper.R.get_state()