# -*- coding: utf-8 -*-
from torch.nn import Module
from typing import TypeVar, Dict, Any, Optional, Union, Tuple, List, Literal, cast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import override

T = TypeVar("T")
TLSeq = Union[List[T], Tuple[T, ...]]

from Network.module_unet import UNet


@dataclass
class ConfigNetworkBase(ABC):
    def is_ready(self) -> bool:
        return hasattr(self, "network")

    def _assert_init_essentials(
            self,
            *args,
            **kwargs
    ) -> None:
        if self.is_ready(): return
        self.init_essentials(*args, **kwargs)

    @abstractmethod
    def init_essentials(
            self,
            *args,
            **kwargs
    ) -> 'ConfigNetworkBase':
        self.network = None  # Just placeholder
        return self

    def get_network_module(self, *args, **kwargs) -> Module:
        self._assert_init_essentials(*args, **kwargs)
        return self.network


@dataclass
class ConfigNetworkUNet(ConfigNetworkBase):
    focuser_in_channels: int = 1
    focuser_out_channels: int = 16
    encoder_primary_in_channels: TLSeq[int] = (16, 32)
    encoder_primary_out_channels: TLSeq[int] = (32, 64)
    encoder_primary_depth: Union[int, TLSeq[int]] = 2
    encoder_advanced_in_channels: TLSeq[int] = (64, 128)
    encoder_advanced_out_channels: TLSeq[int] = (128, 256)
    encoder_advanced_depth: Union[int, TLSeq[int]] = 2
    bottleneck_in_channels: int = 256
    bottleneck_out_channels: int = 512
    bottleneck_depth: int = 2
    decoder_advanced_in_channels: TLSeq[int] = (512, 256)
    decoder_advanced_upsample_channels: TLSeq[int] = (256, 128)
    decoder_advanced_bridge_channels: TLSeq[int] = (256, 128)
    decoder_advanced_out_channels: TLSeq[int] = (256, 128)
    decoder_advanced_depth: Union[int, TLSeq[int]] = 2
    decoder_primary_in_channels: TLSeq[int] = (128, 64)
    decoder_primary_upsample_channels: TLSeq[int] = (64, 32)
    decoder_primary_bridge_channels: TLSeq[int] = (64, 32)
    decoder_primary_out_channels: TLSeq[int] = (64, 32)
    decoder_primary_depth: Union[int, TLSeq[int]] = 2
    auxiliary_classifier_in_channels: TLSeq[int] = (256, 128, 64, 32)
    auxiliary_classifier_out_channels: TLSeq[int] = (2, 2, 2, 2)
    distributor_in_channels: int = 32
    distributor_out_channels: int = 16
    classifier_in_channels: int = 16
    classifier_out_channels: int = 2
    reserve_io: bool = False

    @override
    def init_essentials(self) -> 'ConfigNetworkUNet':
        # Create original UNet
        self.network: UNet = UNet(
            focuser_in_channels=self.focuser_in_channels,
            focuser_out_channels=self.focuser_out_channels,
            encoder_primary_in_channels=self.encoder_primary_in_channels,
            encoder_primary_out_channels=self.encoder_primary_out_channels,
            encoder_primary_depth=self.encoder_primary_depth,
            encoder_advanced_in_channels=self.encoder_advanced_in_channels,
            encoder_advanced_out_channels=self.encoder_advanced_out_channels,
            encoder_advanced_depth=self.encoder_advanced_depth,
            bottleneck_in_channels=self.bottleneck_in_channels,
            bottleneck_out_channels=self.bottleneck_out_channels,
            bottleneck_depth=self.bottleneck_depth,
            decoder_advanced_in_channels=self.decoder_advanced_in_channels,
            decoder_advanced_upsample_channels=self.decoder_advanced_upsample_channels,
            decoder_advanced_bridge_channels=self.decoder_advanced_bridge_channels,
            decoder_advanced_out_channels=self.decoder_advanced_out_channels,
            decoder_advanced_depth=self.decoder_advanced_depth,
            decoder_primary_in_channels=self.decoder_primary_in_channels,
            decoder_primary_upsample_channels=self.decoder_primary_upsample_channels,
            decoder_primary_bridge_channels=self.decoder_primary_bridge_channels,
            decoder_primary_out_channels=self.decoder_primary_out_channels,
            decoder_primary_depth=self.decoder_primary_depth,
            auxiliary_classifier_in_channels=self.auxiliary_classifier_in_channels,
            auxiliary_classifier_out_channels=self.auxiliary_classifier_out_channels,
            distributor_in_channels=self.distributor_in_channels,
            distributor_out_channels=self.distributor_out_channels,
            classifier_in_channels=self.classifier_in_channels,
            classifier_out_channels=self.classifier_out_channels,
            reserve_io=self.reserve_io
        )
        return self
