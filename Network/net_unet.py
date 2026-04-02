# -*- coding: utf-8 -*-
"""
UNet 3D Neural Network Architecture

This module implements a 3D U-Net architecture for volumetric image segmentation.
The U-Net is an encoder-decoder framework with skip connections that preserves
spatial information at multiple scales.

Architecture Overview:
    The network consists of the following components:
    - Focuser: Initial feature extraction stem
    - Encoder: Multi-scale feature extraction with primary and advanced stages
    - Repeater (Bottleneck): Bridge and deepest feature processing layer
    - Decoder: Upsampling and feature reconstruction with skip connections
    - Auxiliary Classifier: Deep supervision at multiple decoder stages
    - Distributor: Feature refinement before final classification
    - Classifier: Final segmentation output layer

Key Features:
    - Multi-scale representation learning
    - Skip connections for preserving spatial details
    - Deep supervision through auxiliary classifiers
    - Configurable depth and channel dimensions
    - I/O shape tracking for debugging

Classes:
    UNet: Main U-Net architecture
    UNetFocuser: Initial feature extraction module
    UNetEncoderPriorBank: Optional prior injection module
    UNetEncoder: Multi-stage encoder
    UNetRepeater: Stage feature repeater and bottleneck processing module
    UNetDecoder: Multi-stage decoder with upsampling
    UNetAuxiliaryClassifier: Deep supervision classifier
    UNetDistributor: Feature refinement module
    UNetClassifier: Final output classifier
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List, Collection, Sequence, Union, Dict, Any, Type, Iterable, cast
from Network.net_block import IODescriptive, ConvNormAct, ConvBNReLU, Concat


# 00 UNet
# UNet is an Encoder-Decoder framework for neural network building, which features multiscale representation ability and is suitable for segmentation task
# With stages auxiliary logits for deep supervision
# Data Shape Flow:
# Channel (default 1 modality, 2 classes):               Spatial (default 128³):
# volume                     mask                           volume                    mask
#   1                        2                                128³                    128³
#   Focuser                  Distributor                      Focuser                 Distributor
#     ↓                      ↑                                  ↓                     ↑
#     16                     16                                 128³                128³
#     Encoder    Repeater     ↑ Decoder  Auxiliary               Encoder  Repeater   ↑ Decoder
#     primary                 │          Classifier              primary             │
#       ↓                     └──────────┐                       Dn ↓             Up │
#       32       ── + ─→     32  = 64  → 32  →  2                   64³      c       64³
#       ↓                     ↖──────────┐                       Dn ↓             Up ↑
#       64       ── + ─→     64  = 128 → 64  →  2                   32³      c       32³
#     advanced                ↑                                  advanced            ↑
#       ↓                     └──────────┐                       Dn ↓             Up │
#       128      ── + ─→     128 = 256 → 128 →  2                   16³      c       16³
#       ↓                     ↖──────────┐                       Dn ↓             Up ↑
#       256      ── + ─→     256 = 512 → 256 →  2                   8³       c       8³
#         │                  ↑                                      │                ↑
#         └────→   512   ────┘                                      └────→   8³  ────┘
#              Bottleneck                                                Bottleneck
# Pinout Diagram: [Valid]
#                 ┌──────┐
#   input_source ─│ 00   │─ cls_logits
# (B,FCin,X,Y,Z)  │ UNet │  (B,CCout,X,Y,Z)
#                 │      │
#                 │      │─ aux_cls_logits[S]
#                 │      │  S:=S1+S2
#                 │      │  [0]:  (B,ACCout[0],  X/(2^(S-1)),Y/(2^(S-1)),Z/(2^(S-1))))
#                 │      │  [1]:  (B,ACCout[1],  X/(2^(S-2)),Y/(2^(S-2)),Z/(2^(S-2))))
#                 │      │  ⋮
#                 │      │  [S-1]:(B,ACCout[S-1],X  ,        Y  ,        Z           )
#                 └──────┘
# Expanded Diagram:
#                      ┌──────────────┐                                                                                                                ┌──────────────────┐                              ┌─────────────────┐
#        input_source ─│ 00-00        │─┬ {output_feat}                                                                                 {input_feat} ┌─│ 00-07            │──────────────────────────────│ 00-08           │─ logits
#      (B,F Cin,X,Y,Z) │ UNet-Focuser │ │ (B,FCout,X,Y,Z)                                                                             (B,DCin,X,Y,Z) │ │ UNet-Distributor │ {output_feat}   {input_feat} │ UNet-Classifier │  (B,CCout,X,Y,Z)
#                      └──────────────┘ │                                                                                                            │ └──────────────────┘ (B,DCout/CCin,X,Y,Z)         └─────────────────┘
#                                       │                                                                                      {output_feats[S-1:S]} │
#                                       │ ┌──────────────┐                                                                          ┌──────────────┐ │                                    ┌────────────────┐  aux_cls_logits[S]
#                          {input_feat} └─│ 00-02        │─ {output_feats[S]} ─────┐                                                │ 00-05        │─┴────────────────────────────────────│ 00-06          │─ {output_logits[S]}
#                    (B,EPCin[0],X,Y,Z)   │ UNet-Encoder │                         │                                                │ UNet-Decoder │ {output_feats[S]}   {input_feats[S]} │ UNet-Auxiliary │
#                                         │              │                         │                            ×{inject_feats[S]} ─│              │                                      │ Classifier     │
#                     ×{inject_feats[S]} ─│              │─ {dn_feats[S]} ─────────┤                                      (Unused)  │              │                                      └────────────────┘
#                               (Unused)  │              │                         │                                                │              │
#                                         │              │  {input_feats[S+1]} ←─ {{output_feats[0:S]},                             │              │
#                                         └──────────────┘     │                   {dn_feats[S-1:S]}  }       ┌─────────────────────│              │
#                                                              │                                              │   {bridge_feats[S]} │              │
#                                                              │ ┌───────────────┐                            │                     │              │
#                                           {input_feats[S+1]} └─│ 00-03-00      │─ {output_feats[S+1]}       │                   ┌─│              │
#                                                                │ UNet-Repeater │       ↓                    │      {input_feat} │ │              │
#                                                                │               │  {{input_feat}, ───────────┼───────────────────┘ └──────────────┘
#                                                                └───────────────┘     {bridge_feats[0:S]}} ──┘      (B,DACin[0],
#                                                                                                                     X/(2^(S)),
#                                                                                                                     Y/(2^(S)),
#                                                                                                                     Z/(2^(S)))

class UNet(nn.Module, IODescriptive):
    """
    3D U-Net Architecture for Volumetric Image Segmentation
    
    A deep encoder-decoder network with skip connections for medical image
    segmentation. Features multi-scale feature extraction and deep supervision.
    
    Architecture Flow:
                                    (skip paths)
                               ┌────────────────────┐
                               ↑      Repeater      ↓
        Input → Focuser → Encoder → (bottleneck) → Decoder → Distributor → Classifier
                                                        ↓
                                                (auxiliary outputs)
    
    Attributes:
        stages: Total number of resolution stages
        focuser: Initial feature extraction module
        encoder: Multi-scale encoder (primary + advanced stages)
        repeater: Bridge and bottleneck feature processing
        decoder: Multi-scale decoder with skip connections
        auxiliary_classifier: Deep supervision outputs
        distributor: Feature refinement module
        classifier: Final segmentation output
        reserve_io: Whether to store intermediate I/O tensors
    """
    def __init__(
            self,
            focuser_in_channels: int = 1,  # FCin: Input channels (in accordance with modalities)
            focuser_out_channels: int = 16,  # FCout: Initial feature channels
            encoder_primary_in_channels: Sequence[int] = (16, 32),  # EPCin[S1]: Primary encoder input channels
            encoder_primary_out_channels: Sequence[int] = (32, 64),  # EPCout[S1]: Primary encoder output channels
            encoder_primary_depth: Union[int, Sequence[int]] = 2,  # EPD[S1]: Primary encoder layer depth
            encoder_advanced_in_channels: Sequence[int] = (64, 128),  # EACin[S2]: Advanced encoder input channels
            encoder_advanced_out_channels: Sequence[int] = (128, 256),  # EACout[S2]: Advanced encoder output channels
            encoder_advanced_depth: Union[int, Sequence[int]] = 2,  # EAD[S2]: Advanced encoder layer depth
            bottleneck_in_channels: int = 256,  # RBCin: Bottleneck input channels
            bottleneck_out_channels: int = 512,  # RBCout: Bottleneck output channels
            bottleneck_depth: int = 2,  # RBD: Bottleneck layer depth
            decoder_advanced_in_channels: Sequence[int] = (512, 256),  # DACin[S2]: Advanced decoder input channels
            decoder_advanced_upsample_channels: Sequence[int] = (256, 128),  # DAUC[S2]: Advanced decoder upsample channels
            decoder_advanced_bridge_channels: Sequence[int] = (256, 128),  # DASC[S2]: Advanced decoder bridge channels
            decoder_advanced_out_channels: Sequence[int] = (256, 128),  # DACout[S2]: Advanced decoder output channels
            decoder_advanced_depth: Union[int, Sequence[int]] = 2,  # DAD[S2]: Advanced decoder layer depth
            decoder_primary_in_channels: Sequence[int] = (128, 64),  # DPCin[S1]: Primary decoder input channels
            decoder_primary_upsample_channels: Sequence[int] = (64, 32),  # DPUC[S1]: Primary decoder upsample channels
            decoder_primary_bridge_channels: Sequence[int] = (64, 32),  # DPSC[S1]: Primary decoder bridge channels
            decoder_primary_out_channels: Sequence[int] = (64, 32),  # DPCout[S1]: Primary decoder output channels
            decoder_primary_depth: Union[int, Sequence[int]] = 2,  # DPD[S1]: Primary decoder layer depth
            auxiliary_classifier_in_channels: Sequence[int] = (256, 128, 64, 32),  # ACCin[S1+S2]: Auxiliary classifier input channels
            auxiliary_classifier_out_channels: Sequence[int] = (2, 2, 2, 2),  # ACCout[S1+S2]: Auxiliary classifier output channels (classes)
            distributor_in_channels: int = 32,  # DCin: Distributor input channels
            distributor_out_channels: int = 16,  # DCout: Distributor output channels
            classifier_in_channels: int = 16,  # CCin: Classifier input channels
            classifier_out_channels: int = 2,  # CCout: Number of output classes
            reserve_io: bool = False  # Whether to store intermediate I/O for debugging
    ):
        """
        Initialize UNet architecture
        
        Args:
            focuser_in_channels: Number of input channels (in accordance with modalities)
            focuser_out_channels: Number of output channels from focuser module
            encoder_primary_in_channels: Input channels for each primary encoder stage
            encoder_primary_out_channels: Output channels for each primary encoder stage
            encoder_primary_depth: Number of layers in each primary encoder stage
            encoder_advanced_in_channels: Input channels for each advanced encoder stage
            encoder_advanced_out_channels: Output channels for each advanced encoder stage
            encoder_advanced_depth: Number of layers in each advanced encoder stage
            bottleneck_in_channels: Input channels for bottleneck
            bottleneck_out_channels: Output channels from bottleneck
            bottleneck_depth: Number of layers in bottleneck
            decoder_advanced_in_channels: Input channels for each advanced decoder stage
            decoder_advanced_upsample_channels: Upsample channels for each advanced decoder stage
            decoder_advanced_bridge_channels: Bridge channels for each advanced decoder stage
            decoder_advanced_out_channels: Output channels for each advanced decoder stage
            decoder_advanced_depth: Number of layers in each advanced decoder stage
            decoder_primary_in_channels: Input channels for each primary decoder stage
            decoder_primary_upsample_channels: Upsample channels for each primary decoder stage
            decoder_primary_bridge_channels: Bridge channels for each primary decoder stage
            decoder_primary_out_channels: Output channels for each primary decoder stage
            decoder_primary_depth: Number of layers in each primary decoder stage
            auxiliary_classifier_in_channels: Input channels for auxiliary classifiers
            auxiliary_classifier_out_channels: Output channels (classes) for auxiliary classifiers
            distributor_in_channels: Input channels for distributor
            distributor_out_channels: Output channels from distributor
            classifier_in_channels: Input channels for final classifier
            classifier_out_channels: Number of output classes
            reserve_io: If True, store intermediate I/O tensors for debugging
            
        Raises:
            AssertionError: If channel dimensions don't match between connected modules
        """
        super(UNet, self).__init__()
        # region Assertions
        # Encoder layer count match
        assert len(encoder_primary_in_channels) == len(encoder_primary_out_channels)
        if isinstance(encoder_primary_depth, Sequence):
            assert len(encoder_primary_depth) == len(encoder_primary_in_channels)
        assert len(encoder_advanced_in_channels) == len(encoder_advanced_out_channels)
        if isinstance(encoder_advanced_depth, Sequence):
            assert len(encoder_advanced_depth) == len(encoder_advanced_in_channels)
        # Encoder channel match
        assert focuser_out_channels == encoder_primary_in_channels[0]
        for ic, oc in zip(encoder_primary_in_channels[1:], encoder_primary_out_channels[:-1]):
            assert ic == oc
        assert encoder_primary_out_channels[-1] == encoder_advanced_in_channels[0]
        for ic, oc in zip(encoder_advanced_in_channels[1:], encoder_advanced_out_channels[:-1]):
            assert ic == oc
        # Bottleneck channel match
        assert encoder_advanced_out_channels[-1] == bottleneck_in_channels
        assert bottleneck_out_channels == decoder_advanced_in_channels[0]
        # Encoder-Decoder layer count match
        assert len(encoder_primary_out_channels) == len(decoder_primary_bridge_channels)
        assert len(encoder_advanced_out_channels) == len(decoder_advanced_bridge_channels)
        # Encoder-Decoder channel match
        for oc, sc in zip(encoder_primary_out_channels, reversed(decoder_primary_bridge_channels)):
            assert oc == sc
        for oc, sc in zip(encoder_advanced_out_channels, reversed(decoder_advanced_bridge_channels)):
            assert oc == sc
        # Decoder layer count match
        assert len(decoder_advanced_in_channels) == len(decoder_advanced_upsample_channels)
        assert len(decoder_advanced_bridge_channels) == len(decoder_advanced_out_channels)
        assert len(decoder_advanced_in_channels) == len(decoder_advanced_out_channels)
        if isinstance(decoder_advanced_depth, Sequence):
            assert len(decoder_advanced_depth) == len(decoder_advanced_in_channels)
        assert len(decoder_primary_in_channels) == len(decoder_primary_upsample_channels)
        assert len(decoder_primary_bridge_channels) == len(decoder_primary_out_channels)
        assert len(decoder_primary_in_channels) == len(decoder_primary_out_channels)
        if isinstance(decoder_primary_depth, Sequence):
            assert len(decoder_primary_depth) == len(decoder_primary_in_channels)
        # Decoder channel match
        for ic, oc in zip(decoder_advanced_in_channels[1:], decoder_advanced_out_channels[:-1]):
            assert ic == oc
        for ic, oc in zip(decoder_primary_in_channels[1:], decoder_primary_out_channels[:-1]):
            assert ic == oc
        assert decoder_primary_out_channels[-1] == distributor_in_channels
        # Auxiliary Classifier layer count match
        assert len(auxiliary_classifier_in_channels) == len(auxiliary_classifier_out_channels)
        assert len(auxiliary_classifier_in_channels) == \
               len(decoder_advanced_out_channels) + len(decoder_primary_out_channels)
        # Auxiliary Classifier channel match
        for oc, ic in zip(decoder_advanced_out_channels,
                          auxiliary_classifier_in_channels[:len(decoder_advanced_out_channels)]):
            assert ic == oc
        for oc, ic in zip(decoder_primary_out_channels,
                          auxiliary_classifier_in_channels[len(decoder_advanced_out_channels):]):
            assert ic == oc
        # Main Classifier channel match
        assert distributor_out_channels == classifier_in_channels
        # endregion

        # Count the feature spatial size variants
        self.stages: int = len(encoder_primary_in_channels) + len(encoder_advanced_in_channels) + 1
        
        # Initialize network modules
        self.focuser: UNetFocuser = UNetFocuser(focuser_in_channels, focuser_out_channels, reserve_io)
        self.encoder: UNetEncoder = UNetEncoder(
            encoder_primary_in_channels,
            encoder_primary_out_channels,
            encoder_primary_depth,
            encoder_advanced_in_channels,
            encoder_advanced_out_channels,
            encoder_advanced_depth,
            reserve_io
        )
        self.repeater: UNetRepeater = UNetRepeater(
            self.stages,
            bottleneck_in_channels,
            bottleneck_out_channels,
            bottleneck_depth,
            reserve_io
        )
        self.decoder: UNetDecoder = UNetDecoder(
            decoder_advanced_in_channels,
            decoder_advanced_upsample_channels,
            decoder_advanced_bridge_channels,
            decoder_advanced_out_channels,
            decoder_advanced_depth,
            decoder_primary_in_channels,
            decoder_primary_upsample_channels,
            decoder_primary_bridge_channels,
            decoder_primary_out_channels,
            decoder_primary_depth,
            reserve_io
        )
        self.auxiliary_classifier: UNetAuxiliaryClassifier = UNetAuxiliaryClassifier(
            auxiliary_classifier_in_channels,
            auxiliary_classifier_out_channels,
            reserve_io
        )
        self.distributor: UNetDistributor = UNetDistributor(
            distributor_in_channels,
            distributor_out_channels,
            reserve_io
        )
        self.classifier: UNetClassifier = UNetClassifier(classifier_in_channels, classifier_out_channels, reserve_io)

        self.reserve_io: bool = reserve_io

    def forward(self, input_source: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass through UNet
        
        Args:
            input_source: Input tensor of shape (B, C, X, Y, Z)
            
        Returns:
            Tuple of (main_logits, auxiliary_logits_list)
            - main_logits: Final segmentation output (B, num_classes, X, Y, Z)
            - auxiliary_logits_list: List of auxiliary outputs for deep supervision
            
        Raises:
            AssertionError: If input is not 5D or spatial dimensions not divisible by 2^(stages-1)
        """
        # Validate input dimensions
        assert input_source.ndim == 5, f"Expected 5D input (B,C,D,H,W), got {input_source.ndim}D"
        assert divmod(input_source.size(2), 2 ** (self.stages - 1))[1] == 0, \
            f"Spatial dimension must be divisible by {2 ** (self.stages - 1)}"
        
        # Forward pass through network components
        f_feat: Tensor = self.focuser(input_source)
        enc_feats: List[Tensor]
        dn_feats: List[Tensor]
        enc_feats, dn_feats = self.encoder(f_feat)
        # enc_feats to skip, dn_feats[-1] to bottleneck
        rep_feats: List[Tensor] = self.repeater(enc_feats + [dn_feats[-1]])
        # rep_feats[0] from bottleneck, rep_feats[1:] are bridge features
        dec_feats: List[Tensor] = self.decoder(rep_feats[0], rep_feats[1:])
        d_feat: Tensor = self.distributor(dec_feats[-1])
        aux_cls_logits: List[Tensor] = self.auxiliary_classifier(dec_feats)  # Map from small to large
        cls_logits: Tensor = self.classifier(d_feat)

        # Store I/O tensors for debugging if enabled
        if self.reserve_io:
            setattr(self, 'input_source', input_source.cpu())
            setattr(self, 'cls_logits', cls_logits.cpu())
            setattr(self, 'aux_cls_logits', [lt.cpu() for lt in aux_cls_logits])
        return cls_logits, aux_cls_logits

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        """
        Generate hierarchical I/O shape description
        
        Args:
            max_level: Maximum recursion level for nested modules
            indent: Current indentation level
            indent_placeholder: String used for indentation
            target_level: Target level for this description
            
        Returns:
            Formatted I/O description string with nested module information
        """
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_source')
                        and hasattr(self, 'cls_logits')
                        and hasattr(self, 'aux_cls_logits')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_source: {tuple(self.input_source.size())}\n"
                     f"{prefix}    cls_logits: {tuple(self.cls_logits.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    aux_cls_logits: {[tuple(lt.size()) for lt in self.aux_cls_logits]}\n")
        # Recursively get I/O descriptions from submodules
        for module in [self.focuser, self.encoder, self.repeater, self.decoder, self.auxiliary_classifier,
                       self.distributor, self.classifier]:
            if hasattr(module, 'io_description'):
                desc += module.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-00 UNet-Focuser
# [Optional]
# Focuser is a Stem Feature Extractor before Encoder, which is intended to generate or enhance key region
# [Forbidden] Stem Feature shall not be used to formulate Feature Pyramid, if you wish to do so, move the pipe to Encoder
# Pinout Diagram: [Valid]
#                ┌──────────────┐
#  input_source ─│ 00-00        │─ output_feat
# (B,Cin,X,Y,Z)  │ UNet-Focuser │  (B,Cout,X,Y,Z)
#                └──────────────┘
class UNetFocuser(nn.Module, IODescriptive):
    """
    UNet Focuser Module - Initial Feature Extractor
    
    A stem feature extractor that processes the input before the encoder.
    Designed to generate or enhance key regions in the input data.
    
    Architecture:
        Input → Conv3d(3x3x3) → BatchNorm → ReLU → Output
    
    Note:
        This module should NOT be used to formulate Feature Pyramid.
        If feature pyramid is needed, move the processing to the Encoder.
    
    Attributes:
        pipe: Sequential container with ConvBNReLU layer
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: int,  # Cin: Number of input channels
            out_channels: int,  # Cout: Number of output channels
            reserve_io: bool = False
    ):
        """
        Initialize UNetFocuser
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetFocuser, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_source: Tensor) -> Tensor:
        """
        Forward pass through focuser
        
        Args:
            input_source: Input tensor of shape (B, Cin, X, Y, Z)
            
        Returns:
            Output tensor of shape (B, Cout, X, Y, Z)
        """
        output_feat: Tensor = self.pipe(input_source)

        if self.reserve_io:
            setattr(self, 'input_source', input_source.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_source')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_source: {tuple(self.input_source.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        return desc


# 00-01 UNet-EncoderPriorBank
# [Optional]
# This is for prior injection, you may inject anything to each stage of the Encoder
# Pinout Diagram: [Placeholder]
#                  ┌──────────────┐
# prior_source[S] ─│ 00-01        │─ prior_feats[S]
#   (B,Cin,X,Y,Z)  │ UNet-Encoder │  (B,Cout,X,Y,Z)
#                  │ PriorBank    │
#                  └──────────────┘
class UNetEncoderPriorBank(nn.Module, IODescriptive):
    """
    UNet Encoder Prior Bank Module - Optional Prior Feature Injection
    
    A placeholder module for injecting prior features into each encoder stage.
    This allows external information to be incorporated into the encoding process.
    
    Note:
        This is a placeholder implementation that returns None.
        Subclass and override to implement custom prior injection logic.
    
    Use Cases:
        - Inject anatomical priors
        - Add multi-modal information
        - Incorporate pre-computed features
    
    Attributes:
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            reserve_io: bool = False
    ):
        """
        Initialize UNetEncoderPriorBank
        
        Args:
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetEncoderPriorBank, self).__init__()
        self.reserve_io: bool = reserve_io

    def forward(self, prior_source: Sequence[Tensor]) -> List[Tensor]:
        """
        Forward pass through prior bank
        
        Args:
            prior_source: Sequence of prior tensors to inject
            
        Returns:
            List of processed prior features (None in base implementation)
        """
        return None

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        return ''


# 00-01-00 UNet-EncoderPriorBank-Injector
# Injection process pipe
# Pinout Diagram: [Placeholder]
#                ┌───────────────────┐
# inject_source ─│ 00-01-00          │─ inject_feat
# (B,Cin,X,Y,Z)  │ UNet-EncoderPrior │  (B,Cout,X,Y,Z)
#                │ Bank-Injector     │
#                └───────────────────┘
class UNetEncoderPriorBankInjector(nn.Module, IODescriptive):
    """
    UNet Encoder Prior Bank Injector Module - Single Stage Prior Injection
    
    A placeholder module for injecting prior features at a single encoder stage.
    Used within the PriorBank to process individual stage injections.
    
    Note:
        This is a placeholder implementation that returns None.
        Subclass and override to implement custom injection processing.
    
    Attributes:
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            reserve_io: bool = False
    ):
        """
        Initialize UNetEncoderPriorBankInjector
        
        Args:
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetEncoderPriorBankInjector, self).__init__()
        self.reserve_io: bool = reserve_io

    def forward(self, inject_source: Tensor) -> Tensor:
        """
        Forward pass through injector
        
        Args:
            inject_source: Prior tensor to inject
            
        Returns:
            Processed injection feature (None in base implementation)
        """
        return None

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        return ''


# 00-02 UNet-Encoder
# Main Feature Pyramid Extractor, which is intended to generate multi-granularity features
# Pinout Diagram: [Valid]
#                      ┌──────────────┐
#          input_feat ─│ 00-02        │─ output_feats[S1+S2]
#   (B,PCin[0],X,Y,Z)  │ UNet-Encoder │  # Primary part
#                      │              │  [0]:   (B,PCout[0],   X,           Y,           Z           )
# inject_feats[S1+S2] ─│              │  [1]:   (B,PCout[1],   X/2,         Y/2,         Z/2         )
#            (Unused)  │              │  ⋮
#                      │              │  [S1-1]:(B,PCout[S1-1],X/(2^(S1-1)),Y/(2^(S1-1)),Z/(2^(S1-1)))
#                      │              │  # Advanced part
#                      │              │  [0]:   (B,ACout[0],   X/(2^(S1)),     Y/(2^(S1)),     Z/(2^(S1))     )
#                      │              │  [1]:   (B,ACout[1],   X/(2^(S1+1)),   Y/(2^(S1+1)),   Z/(2^(S1+1))   )
#                      │              │  ⋮
#                      │              │  [S2-1]:(B,ACout[S2-1],X/(2^(S1+S2-1)),Y/(2^(S1+S2-1)),Z/(2^(S1+S2-1)))
#                      │              │
#                      │              │─ dn_feats[S1+S2]
#                      │              │  # Primary part
#                      │              │  [0]:   (B,PCout[0],   X/2,     Y/2,     Z/2     )
#                      │              │  [1]:   (B,PCout[1],   X/4,     Y/4,     Z/4     )
#                      │              │  ⋮
#                      │              │  [S1-1]:(B,PCout[S1-1],X/(2^S1),Y/(2^S1),Z/(2^S1))
#                      │              │  # Advanced part
#                      │              │  [0]:   (B,ACout[0],   X/(2^(S1+1)), Y/(2^(S1+1)), Z/(2^(S1+1)) )
#                      │              │  [1]:   (B,ACout[1],   X/(2^(S1+2)), Y/(2^(S1+2)), Z/(2^(S1+2)) )
#                      │              │  ⋮
#                      │              │  [S2-1]:(B,ACout[S2-1],X/(2^(S1+S2)),Y/(2^(S1+S2)),Z/(2^(S1+S2)))
#                      └──────────────┘
# Expanded Diagram:
#                     ┌──────────────────┐
#         input_feat ─│ 00-02-00         │─ output_feats[0:S1]
#  (B,PCin[0],X,Y,Z)  │ UNet-Encoder-    │  # Primary part
#                     │ PrimaryExtractor │  [0]:   (B,PCout[0],   X,           Y,           Z           )
# inject_feats[0:S1] ─│                  │  [1]:   (B,PCout[1],   X/2,         Y/2,         Z/2         )
#           (Unused)  │                  │  ⋮
#                     │                  │  [S1-1]:(B,PCout[S1-1],X/(2^(S1-1)),Y/(2^(S1-1)),Z/(2^(S1-1)))
#                     │                  │
#                     │                  │─ dn_feats[0:S1]
#                     │                  │  # Primary part
#                     │                  │  [0]:   (B,PCout[0],   X/2,     Y/2,     Z/2     )
#                     │                  │  [1]:   (B,PCout[1],   X/4,     Y/4,     Z/4     )
#                     │                  │  ⋮
#                     │                  │  [S1-1]:(B,PCout[S1-1],X/(2^S1),Y/(2^S1),Z/(2^S1))
#                     └──────────────────┘     │
#                                              │       ┌───────────────────┐
#                               dn_feats[S1-1] └───────│ 00-02-01          │─ output_feats[S1:S1+S2]
#                                                      │ UNet-Encoder-     │  # Advanced part
#                              inject_feats[S1:S1+S2] ─│ AdvancedExtractor │  [0]:   (B,ACout[0],   X/(2^(S1)),     Y/(2^(S1)),     Z/(2^(S1))     )
#                                            (Unused)  │                   │  [1]:   (B,ACout[1],   X/(2^(S1+1)),   Y/(2^(S1+1)),   Z/(2^(S1+1))   )
#                                                      │                   │  ⋮
#                                                      │                   │  [S2-1]:(B,ACout[S2-1],X/(2^(S1+S2-1)),Y/(2^(S1+S2-1)),Z/(2^(S1+S2-1)))
#                                                      │                   │
#                                                      │                   │─ dn_feats[S1:S1+S2]
#                                                      │                   │  # Advanced part
#                                                      │                   │  [0]:   (B,ACout[0],   X/(2^(S1+1)), Y/(2^(S1+1)), Z/(2^(S1+1)) )
#                                                      │                   │  [1]:   (B,ACout[1],   X/(2^(S1+2)), Y/(2^(S1+2)), Z/(2^(S1+2)) )
#                                                      │                   │  ⋮
#                                                      │                   │  [S2-1]:(B,ACout[S2-1],X/(2^(S1+S2)),Y/(2^(S1+S2)),Z/(2^(S1+S2)))
#                                                      └───────────────────┘
class UNetEncoder(nn.Module, IODescriptive):
    """
    UNet Encoder Module - Multi-Scale Feature Extraction
    
    The encoder consists of two parts:
    - Primary Extractor: Initial feature extraction stages
    - Advanced Extractor: Deeper feature extraction stages
    
    Each stage reduces spatial dimensions by 2x while increasing channel depth.
    Features from each stage are saved for skip connections in the decoder.
    
    Architecture:
        Input → Primary Extractor (S1 stages) → Advanced Extractor (S2 stages)
                  ↓                                    ↓
            (output_feats,                    (output_feats,
             downsample_feats)                 downsample_feats)
    
    Attributes:
        primary_extractor: Primary feature extraction stages
        advanced_extractor: Advanced feature extraction stages
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            primary_in_channels: Sequence[int],  # PCin[S1]: Primary stage input channels
            primary_out_channels: Sequence[int],  # PCout[S1]: Primary stage output channels
            primary_depth: Union[int, Sequence[int]],  # PD[S1]: Primary stage layer depth
            advanced_in_channels: Sequence[int],  # ACin[S2]: Advanced stage input channels
            advanced_out_channels: Sequence[int],  # ACout[S2]: Advanced stage output channels
            advanced_depth: Union[int, Sequence[int]],  # AD[S2]: Advanced stage layer depth
            reserve_io: bool = False
    ):
        """
        Initialize UNetEncoder
        
        Args:
            primary_in_channels: Input channels for each primary stage
            primary_out_channels: Output channels for each primary stage
            primary_depth: Layer depth for each primary stage (int or sequence)
            advanced_in_channels: Input channels for each advanced stage
            advanced_out_channels: Output channels for each advanced stage
            advanced_depth: Layer depth for each advanced stage (int or sequence)
            reserve_io: If True, store I/O tensors for debugging
            
        Raises:
            AssertionError: If channel/depth sequence lengths don't match
        """
        super(UNetEncoder, self).__init__()
        assert len(primary_in_channels) == len(primary_out_channels)
        assert len(advanced_in_channels) == len(advanced_out_channels)
        if isinstance(primary_depth, Sequence):
            assert len(primary_in_channels) == len(primary_depth)
        if isinstance(advanced_depth, Sequence):
            assert len(advanced_in_channels) == len(advanced_depth)
        
        # Initialize primary and advanced extractors
        self.primary_extractor: UNetEncoderPrimaryExtractor = UNetEncoderPrimaryExtractor(
            primary_in_channels,
            primary_out_channels,
            primary_depth,
            reserve_io
        )
        self.advanced_extractor: UNetEncoderAdvancedExtractor = UNetEncoderAdvancedExtractor(
            advanced_in_channels,
            advanced_out_channels,
            advanced_depth,
            reserve_io
        )

        self.reserve_io: bool = reserve_io

    def forward(
            self,
            input_feat: Tensor,
            inject_feats: Optional[Sequence[Tensor]] = None
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Forward pass through encoder
        
        Args:
            input_feat: Input tensor of shape (B, C, X, Y, Z)
            inject_feats: Optional prior features to inject at each stage
            
        Returns:
            Tuple of (output_features, downsample_features)
            - output_features: List of features from each stage for skip connections
            - downsample_features: List of downsampled features passed between stages
            
        Raises:
            AssertionError: If inject_feats length doesn't match total stages
        """
        assert (inject_feats is None or
                len(inject_feats) == self.primary_extractor.stages + self.advanced_extractor.stages)

        primary_feats: List[Tensor]
        primary_dn_feats: List[Tensor]
        advanced_feats: List[Tensor]
        advanced_dn_feats: List[Tensor]
        if inject_feats is None:
            primary_feats, primary_dn_feats = self.primary_extractor(
                input_feat)  # (S1)*(B,PCout[s1],*),
            advanced_feats, advanced_dn_feats = self.advanced_extractor(
                primary_dn_feats[self.primary_extractor.stages - 1])  # (S2)*(B,ACout[s2],*)
        else:
            primary_feats, primary_dn_feats = self.primary_extractor(
                input_feat,
                inject_feats[:self.primary_extractor.stages]
            )  # (S1)*(B,PCout[s1],*)
            advanced_feats, advanced_dn_feats = self.advanced_extractor(
                primary_dn_feats[self.primary_extractor.stages - 1],
                inject_feats[self.primary_extractor.stages:]
            )  # (S2)*(B,ACout[s2],*)

        output_feats: List[Tensor] = primary_feats + advanced_feats
        dn_feats: List[Tensor] = primary_dn_feats + advanced_dn_feats
        # (S1+S2)*(B,P/ACout[s],*)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            if inject_feats is not None:
                setattr(self, 'inject_feats', [ft.cpu() for ft in inject_feats])
            setattr(self, 'output_feats', [ft.cpu() for ft in output_feats])
            setattr(self, 'dn_feats', [ft.cpu() for ft in dn_feats])
        return output_feats, dn_feats

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feats')
                        and hasattr(self, 'dn_feats')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n")
        if hasattr(self, 'inject_feats'):
            desc += f"{prefix}  inject_feats: {[tuple(ft.size()) for ft in self.inject_feats]}\n"
        desc += (f"{prefix}  O: \n"
                 f"{prefix}    output_feats: {[tuple(ft.size()) for ft in self.output_feats]}\n"
                 f"{prefix}    dn_feats: {[tuple(ft.size()) for ft in self.dn_feats]}\n")
        for module in [self.primary_extractor, self.advanced_extractor]:
            if hasattr(module, 'io_description'):
                desc += module.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-02-00 UNet-Encoder-PrimaryExtractor
# Shallow part of the Encoder, representing primary semantics
# Pinout Diagram: [Valid]
#                   ┌──────────────────┐
#       input_feat ─│ 00-02-00         │─ output_feats[S]
# (B,Cin[0],X,Y,Z)  │ UNet-Encoder-    │  [0]:  (B,Cout[0],  X,          Y,          Z          )
#                   │ PrimaryExtractor │  [1]:  (B,Cout[1],  X/2,        Y/2,        Z/2        )
#  inject_feats[S] ─│                  │  ⋮
#         (Unused)  │                  │  [S-1]:(B,Cout[S-1],X/(2^(S-1)),Y/(2^(S-1)),Z/(2^(S-1)))
#                   │                  │
#                   │                  │─ dn_feats[S]
#                   │                  │  [0]:  (B,Cout[0],  X/2,    Y/2,    Z/2    )
#                   │                  │  [1]:  (B,Cout[1],  X/4,    Y/4,    Z/4    )
#                   │                  │  ⋮
#                   │                  │  [S-1]:(B,Cout[S-1],X/(2^S),Y/(2^S),Z/(2^S))
#                   └──────────────────┘
# Expanded Diagram:
#                                     ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
#                                     │  ┌──────────────────────┐ output_feats[0]                              ┌──────────────────────┐  │  dn_feats[0]
#                         input_feat ─┼──│ 00-02-00-00          │──────────────────────────────────────────────│ 00-02-00-01          │──┼─ {output_feat}→
#                   (B,Cin[0],X,Y,Z)  │  │ UNet-Encoder-Primary │ {output_feat}                   {input_feat} │ UNet-Encoder-Primary │  │  (B,Cout[0]/Cin[1],X/2,Y/2,Z/2)
#                    inject_feats[0] ─┼──│ Extractor-Stage      │ (B,Cout[0]/Cin[1],X,Y,Z)                     │ Extractor-Downsample │  │
#                           (Unused)  │  │ :depth=D/D[0]        │                                              └──────────────────────┘  │
#                                     │  └──────────────────────┘                                                                        │
#                                     └──────────────────────────────────────────────────────────────────────────────────────────────────┘
#                                     ⋮                                                ⋮                                                 ⋮
#                                     ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
#                      dn_feats[s-1]  │  ┌──────────────────────┐ output_feats[s]                              ┌──────────────────────┐  │  dn_feats[s]
#                     →{output_feat} ─┼──│ 00-02-00-00          │──────────────────────────────────────────────│ 00-02-00-01          │──┼─ {output_feat}→
# (B,Cin[s],X/(2^s),Y/(2^s),Z/(2^s))  │  │ UNet-Encoder-Primary │ {output_feat}                   {input_feat} │ UNet-Encoder-Primary │  │  (B,Cout[s]/Cin[s+1],X/(2^(s+1)),Y/(2^(s+1)),Z/(2^(s+1)))
#                    inject_feats[s] ─┼──│ Extractor-Stage      │ (B,Cout[s]/Cin[s+1],X/(2^s),Y/(2^s),Z/(2^s)) │ Extractor-Downsample │  │
#                           (Unused)  │  │ :depth=D/D[s]        │                                              └──────────────────────┘  │
#                                     │  └──────────────────────┘                                                                        │
#                                     └──────────────────────────────────────────────────────────────────────────────────────────────────┘
class UNetEncoderPrimaryExtractor(nn.Module, IODescriptive):
    """
    UNet Encoder Primary Extractor Module - Shallow Feature Extraction
    
    Extracts features from the shallow (primary) part of the encoder.
    Represents primary semantics with relatively high spatial resolution.
    
    Architecture:
        For each stage s:
            stage_input → Stage → output_feats[s] → Downsample → dn_feats[s]
    
    Each stage consists of:
    - Stage: Conv-BN-ReLU layers for feature extraction
    - Downsample: 2x downsampling for next stage
    
    Attributes:
        stages: Number of extraction stages
        pipe: ModuleList of (Stage, Downsample) sequences
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: Sequence[int],  # Cin[S]: Input channels for each stage
            out_channels: Sequence[int],  # Cout[S]: Output channels for each stage
            depth: Union[int, Sequence[int]],  # D[S]: Layer depth for each stage
            reserve_io: bool = False
    ):
        """
        Initialize UNetEncoderPrimaryExtractor
        
        Args:
            in_channels: Input channels for each stage
            out_channels: Output channels for each stage
            depth: Number of ConvBNReLU layers per stage (int or sequence)
            reserve_io: If True, store I/O tensors for debugging
            
        Raises:
            AssertionError: If channel/depth sequence lengths don't match
        """
        super(UNetEncoderPrimaryExtractor, self).__init__()
        assert len(in_channels) == len(out_channels)
        if isinstance(depth, Sequence):
            assert len(in_channels) == len(depth)
        else:
            depth: List[int] = [depth] * len(in_channels)
        self.stages: int = len(in_channels)
        self.pipe: nn.ModuleList = nn.ModuleList()

        # Create stage-downsample sequence for each resolution level
        for ic, oc, dp in zip(in_channels, out_channels, depth):
            self.pipe.append(nn.Sequential(
                UNetEncoderPrimaryExtractorStage(ic, oc, dp, reserve_io),
                UNetEncoderPrimaryExtractorDownsample(reserve_io)
            ))

        self.reserve_io: bool = reserve_io

    def forward(
            self,
            input_feat: Tensor,
            inject_feats: Optional[Sequence[Tensor]] = None
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Forward pass through primary extractor
        
        Args:
            input_feat: Input tensor of shape (B, Cin[0], X, Y, Z)
            inject_feats: Optional prior features to inject at each stage
            
        Returns:
            Tuple of (output_features, downsample_features)
            - output_features: Features from each stage for skip connections
            - downsample_features: Downsampled features for next stage
            
        Raises:
            AssertionError: If inject_feats length doesn't match stages
        """
        output_feats: List[Tensor] = []
        dn_feats: List[Tensor] = []
        if inject_feats is None:
            stage_feat: Tensor = input_feat
            for module in self.pipe:
                stage_feat = module[0](stage_feat)  # Stage
                output_feats.append(stage_feat)  # Record features for all stages
                stage_feat = module[1](stage_feat)  # Downsample
                dn_feats.append(stage_feat)  # Record features after downsample
        else:
            assert len(inject_feats) == self.stages
            stage_feat: Tensor = input_feat
            for module, inject_feat in zip(self.pipe, inject_feats):
                stage_feat = module[0](stage_feat, inject_feat)  # Stage
                output_feats.append(stage_feat)  # Record features for all stages
                stage_feat = module[1](stage_feat)  # Downsample
                dn_feats.append(stage_feat)  # Record features after downsample

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            if inject_feats is not None:
                setattr(self, 'inject_feats', [ft.cpu() for ft in inject_feats])
            setattr(self, 'output_feats', [ft.cpu() for ft in output_feats])
            setattr(self, 'dn_feats', [ft.cpu() for ft in dn_feats])
        return output_feats, dn_feats

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feats')
                        and hasattr(self, 'dn_feats')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n")
        if hasattr(self, 'inject_feats'):
            desc += f"{prefix}  inject_feats: {[tuple(ft.size()) for ft in self.inject_feats]}\n"
        desc += (f"{prefix}  O: \n"
                 f"{prefix}    output_feats: {[tuple(ft.size()) for ft in self.output_feats]}\n"
                 f"{prefix}    dn_feats: {[tuple(ft.size()) for ft in self.dn_feats]}\n")
        for module in self.pipe:
            for submodule in cast(nn.Sequential, module):
                if hasattr(submodule, 'io_description'):
                    desc += submodule.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-02-00-00 UNet-Encoder-PrimaryExtractor-Stage
# One stage within the Encoder primary extractor, representing one specified granularity
# Pinout Diagram: [Valid]
#                ┌──────────────────────┐
#    input_feat ─│ 00-02-00-00          │─ output_feat
# (B,Cin,X,Y,Z)  │ UNet-Encoder-Primary │  (B,Cout,X,Y,Z)
#   inject_feat ─│ Extractor-Stage      │
#      (Unused)  │                      │
#                └──────────────────────┘
# Expanded Diagram:
#                ┌──────────────┐                              ┌──────────────┐                              ┌──────────────┐
#    input_feat ─│ [0]          │───────────── ··· ────────────│ [d]          │──────────────────────────────│ [D-1]        │─ output_feat
# (B,Cin,X,Y,Z)  │ Conv-BN-ReLU │ {output_feat}   {input_feat} │ Conv-BN-ReLU │ {output_feat}   {input_feat} │ Conv-BN-ReLU │  (B,Cout,X,Y,Z)
#                └──────────────┘ (B,Cin,X,Y,Z)                └──────────────┘ (B,Cin,X,Y,Z)                └──────────────┘
class UNetEncoderPrimaryExtractorStage(nn.Module, IODescriptive):
    """
    UNet Encoder Primary Extractor Stage Module - Single Resolution Feature Processing
    
    Processes features at a single resolution level within the primary extractor.
    Consists of D Conv-BN-ReLU layers for feature extraction and transformation.
    
    Architecture:
        Input → [Conv-BN-ReLU] × (D-1) → Conv-BN-ReLU (Cin→Cout) → Output
    
    The first D-1 layers maintain channel dimensions, the last layer changes
    from Cin to Cout.
    
    Attributes:
        pipe: Sequential container with D ConvBNReLU layers
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: int,  # Cin: Input channels
            out_channels: int,  # Cout: Output channels
            depth: int,  # D: Number of ConvBNReLU layers
            reserve_io: bool = False
    ):
        """
        Initialize UNetEncoderPrimaryExtractorStage
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            depth: Number of ConvBNReLU layers
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetEncoderPrimaryExtractorStage, self).__init__()
        # Create D-1 layers that maintain channel dimensions
        # followed by 1 layer that changes from Cin to Cout
        self.pipe: nn.Sequential = nn.Sequential(
            *[ConvBNReLU(in_channels, in_channels, 3, padding='same', reserve_io=reserve_io)
              for _ in range(depth - 1)],
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor, inject_feat: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through encoder stage
        
        Args:
            input_feat: Input tensor of shape (B, Cin, X, Y, Z)
            inject_feat: Optional injection tensor (currently unused)
            
        Returns:
            Output tensor of shape (B, Cout, X, Y, Z)
        """
        # inject_feat is always ignored now
        output_feat: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            if inject_feat is not None:
                setattr(self, 'inject_feat', inject_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n")
        if hasattr(self, 'inject_feat'):
            desc += f"{prefix}  inject_feat: {tuple(self.inject_feat.size())}\n"
        desc += (f"{prefix}  O: \n"
                 f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        for module in self.pipe:
            if hasattr(module, 'io_description'):
                desc += module.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-02-00-01 UNet-Encoder-PrimaryExtractor-Downsample
# Downsampler after each primary extractor stage
# Pinout Diagram: [Valid]
#                ┌──────────────────────┐
#    input_feat ─│ 00-02-00-01          │─ output_feat
#   (B,C,X,Y,Z)  │ UNet-Encoder-Primary │  (B,C,X/2,Y/2,Z/2)
#                │ Extractor-Downsample │
#                └──────────────────────┘
class UNetEncoderPrimaryExtractorDownsample(nn.Module, IODescriptive):
    """
    UNet Encoder Primary Extractor Downsample Module
    
    Downsamples features by 2x in each spatial dimension using max pooling.
    This reduces spatial resolution while preserving channel dimensions.
    
    Architecture:
        Input → MaxPool3d(2x2x2, stride=2) → Output
    
    Spatial dimensions are halved: (X, Y, Z) → (X/2, Y/2, Z/2)
    
    Attributes:
        pipe: Sequential container with max pooling layer
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            reserve_io: bool = False
    ):
        """
        Initialize UNetEncoderPrimaryExtractorDownsample
        
        Args:
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetEncoderPrimaryExtractorDownsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.MaxPool3d(2, 2)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor) -> Tensor:
        """
        Forward pass through downsampling layer
        
        Args:
            input_feat: Input tensor of shape (B, C, X, Y, Z)
            
        Returns:
            Output tensor of shape (B, C, X/2, Y/2, Z/2)
        """
        output_feat: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        return desc


# 00-02-01 UNet-Encoder-AdvancedExtractor
# Deep part of the Encoder, representing advanced semantics
# Pinout Diagram: [Valid]
#                   ┌──────────────────┐
#       input_feat ─│ 00-02-01         │─ output_feats[S]
# (B,Cin[0],X,Y,Z)  │ UNet-Encoder-    │  [0]:  (B,Cout[0],  X,          Y,          Z          )
#                   │ Advanced         │  [1]:  (B,Cout[1],  X/2,        Y/2,        Z/2        )
#  inject_feats[S] ─│ Extractor        │  ⋮
#         (Unused)  │                  │  [S-1]:(B,Cout[S-1],X/(2^(S-1)),Y/(2^(S-1)),Z/(2^(S-1)))
#                   │                  │
#                   │                  │─ dn_feats[S]
#                   │                  │  [0]:  (B,Cout[0],  X/2,    Y/2,    Z/2    )
#                   │                  │  [1]:  (B,Cout[1],  X/4,    Y/4,    Z/4    )
#                   │                  │  ⋮
#                   │                  │  [S-1]:(B,Cout[S-1],X/(2^S),Y/(2^S),Z/(2^S))
#                   └──────────────────┘
# Expanded Diagram:
#                                     ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
#                                     │  ┌───────────────────────┐ output_feats[0]                              ┌───────────────────────┐  │  dn_feats[0]
#                         input_feat ─┼──│ 00-02-01-00           │──────────────────────────────────────────────│ 00-02-01-01           │──┼─ {output_feat}→
#                   (B,Cin[0],X,Y,Z)  │  │ UNet-Encoder-Advanced │ {output_feat}                   {input_feat} │ UNet-Encoder-Advanced │  │  (B,Cout[0]/Cin[1],X/2,Y/2,Z/2)
#                    inject_feats[0] ─┼──│ Extractor-Stage       │ (B,Cout[0]/Cin[1],X,Y,Z)                     │ Extractor-Downsample  │  │
#                           (Unused)  │  │ :depth=D/D[0]         │                                              └───────────────────────┘  │
#                                     │  └───────────────────────┘                                                                         │
#                                     └────────────────────────────────────────────────────────────────────────────────────────────────────┘
#                                     ⋮                                                 ⋮                                                  ⋮
#                                     ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
#                      dn_feats[s-1]  │  ┌───────────────────────┐ output_feats[s]                              ┌───────────────────────┐  │  dn_feats[s]
#                     →{output_feat} ─┼──│ 00-02-01-00           │──────────────────────────────────────────────│ 00-02-01-01           │──┼─ {output_feat}→
# (B,Cin[s],X/(2^s),Y/(2^s),Z/(2^s))  │  │ UNet-Encoder-Advanced │ {output_feat}                   {input_feat} │ UNet-Encoder-Advanced │  │  (B,Cout[s]/Cin[s+1],X/(2^(s+1)),Y/(2^(s+1)),Z/(2^(s+1)))
#                    inject_feats[s] ─┼──│ Extractor-Stage       │ (B,Cout[s]/Cin[s+1],X/(2^s),Y/(2^s),Z/(2^s)) │ Extractor-Downsample  │  │
#                           (Unused)  │  │ :depth=D/D[s]         │                                              └───────────────────────┘  │
#                                     │  └───────────────────────┘                                                                         │
#                                     └────────────────────────────────────────────────────────────────────────────────────────────────────┘
class UNetEncoderAdvancedExtractor(nn.Module, IODescriptive):
    """
    UNet Encoder Advanced Extractor Module - Deep Feature Extraction
    
    Extracts features from the deep (advanced) part of the encoder.
    Represents advanced semantics with lower spatial resolution but higher
    semantic abstraction.
    
    Architecture:
        For each stage s:
            stage_input → Stage → output_feats[s] → Downsample → dn_feats[s]
    
    Each stage consists of:
    - Stage: Conv-BN-ReLU layers for feature extraction
    - Downsample: 2x downsampling for next stage
    
    Attributes:
        stages: Number of extraction stages
        pipe: ModuleList of (Stage, Downsample) sequences
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: Sequence[int],  # Cin[S]: Input channels for each stage
            out_channels: Sequence[int],  # Cout[S]: Output channels for each stage
            depth: Union[int, Sequence[int]],  # D[S]: Layer depth for each stage
            reserve_io: bool = False
    ):
        """
        Initialize UNetEncoderAdvancedExtractor
        
        Args:
            in_channels: Input channels for each stage
            out_channels: Output channels for each stage
            depth: Number of ConvBNReLU layers per stage (int or sequence)
            reserve_io: If True, store I/O tensors for debugging
            
        Raises:
            AssertionError: If channel/depth sequence lengths don't match
        """
        super(UNetEncoderAdvancedExtractor, self).__init__()
        assert len(in_channels) == len(out_channels)
        if isinstance(depth, Sequence):
            assert len(in_channels) == len(depth)
        else:
            depth: List[int] = [depth] * len(in_channels)
        self.stages: int = len(in_channels)
        self.pipe: nn.ModuleList = nn.ModuleList()

        # Create stage-downsample sequence for each resolution level
        for ic, oc, dp in zip(in_channels, out_channels, depth):
            self.pipe.append(nn.Sequential(
                UNetEncoderAdvancedExtractorStage(ic, oc, dp, reserve_io),
                UNetEncoderAdvancedExtractorDownsample(reserve_io)
            ))

        self.reserve_io: bool = reserve_io

    def forward(
            self,
            input_feat: Tensor,
            inject_feats: Optional[Sequence[Tensor]] = None
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Forward pass through advanced extractor
        
        Args:
            input_feat: Input tensor of shape (B, Cin[0], X, Y, Z)
            inject_feats: Optional prior features to inject at each stage
            
        Returns:
            Tuple of (output_features, downsample_features)
            - output_features: Features from each stage for skip connections
            - downsample_features: Downsampled features for next stage
            
        Raises:
            AssertionError: If inject_feats length doesn't match stages
        """
        output_feats: List[Tensor] = []
        dn_feats: List[Tensor] = []
        if inject_feats is None:
            stage_feat: Tensor = input_feat
            for module in self.pipe:
                stage_feat = module[0](stage_feat)  # Stage
                output_feats.append(stage_feat)  # Record features for all stages
                stage_feat = module[1](stage_feat)  # Downsample
                dn_feats.append(stage_feat)  # Record features after downsample
        else:
            assert len(inject_feats) == self.stages
            stage_feat: Tensor = input_feat
            for module, inject_feat in zip(self.pipe, inject_feats):
                stage_feat = module[0](stage_feat, inject_feat)  # Stage
                output_feats.append(stage_feat)  # Record features for all stages
                stage_feat = module[1](stage_feat)  # Downsample
                dn_feats.append(stage_feat)  # Record features after downsample

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            if inject_feats is not None:
                setattr(self, 'inject_feats', [ft.cpu() for ft in inject_feats])
            setattr(self, 'output_feats', [ft.cpu() for ft in output_feats])
            setattr(self, 'dn_feats', [ft.cpu() for ft in dn_feats])
        return output_feats, dn_feats

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feats')
                        and hasattr(self, 'dn_feats')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n")
        if hasattr(self, 'inject_feats'):
            desc += f"{prefix}  inject_feats: {[tuple(ft.size()) for ft in self.inject_feats]}\n"
        desc += (f"{prefix}  O: \n"
                 f"{prefix}    output_feats: {[tuple(ft.size()) for ft in self.output_feats]}\n"
                 f"{prefix}    dn_feats: {[tuple(ft.size()) for ft in self.dn_feats]}\n")
        for module in self.pipe:
            for submodule in cast(nn.Sequential, module):
                if hasattr(submodule, 'io_description'):
                    desc += submodule.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-02-01-00 UNet-Encoder-AdvancedExtractor-Stage
# One stage within the Encoder advanced extractor, representing one specified granularity
# Pinout Diagram: [Valid]
#                ┌───────────────────────┐
#    input_feat ─│ 00-02-01-00           │─ output_feat
# (B,Cin,X,Y,Z)  │ UNet-Encoder-Advanced │  (B,Cout,X,Y,Z)
#   inject_feat ─│ Extractor-Stage       │
#      (Unused)  │                       │
#                └───────────────────────┘
# Expanded Diagram:
#                ┌──────────────┐                              ┌──────────────┐                              ┌──────────────┐
#    input_feat ─│ [0]          │───────────── ··· ────────────│ [d]          │──────────────────────────────│ [D-1]        │─ output_feat
# (B,Cin,X,Y,Z)  │ Conv-BN-ReLU │ {output_feat}   {input_feat} │ Conv-BN-ReLU │ {output_feat}   {input_feat} │ Conv-BN-ReLU │  (B,Cout,X,Y,Z)
#                └──────────────┘ (B,Cin,X,Y,Z)                └──────────────┘ (B,Cin,X,Y,Z)                └──────────────┘
class UNetEncoderAdvancedExtractorStage(nn.Module, IODescriptive):
    """
    UNet Encoder Advanced Extractor Stage Module - Single Resolution Feature Processing
    
    Processes features at a single resolution level within the advanced extractor.
    Consists of D Conv-BN-ReLU layers for feature extraction and transformation.
    
    Architecture:
        Input → [Conv-BN-ReLU] × (D-1) → Conv-BN-ReLU (Cin→Cout) → Output
    
    The first D-1 layers maintain channel dimensions, the last layer changes
    from Cin to Cout.
    
    Attributes:
        pipe: Sequential container with D ConvBNReLU layers
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: int,  # Cin: Input channels
            out_channels: int,  # Cout: Output channels
            depth: int,  # D: Number of ConvBNReLU layers
            reserve_io: bool = False
    ):
        """
        Initialize UNetEncoderAdvancedExtractorStage
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            depth: Number of ConvBNReLU layers
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetEncoderAdvancedExtractorStage, self).__init__()
        # Create D-1 layers that maintain channel dimensions
        # followed by 1 layer that changes from Cin to Cout
        self.pipe: nn.Sequential = nn.Sequential(
            *[ConvBNReLU(in_channels, in_channels, 3, padding='same', reserve_io=reserve_io)
              for _ in range(depth - 1)],
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor, inject_feat: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through encoder stage
        
        Args:
            input_feat: Input tensor of shape (B, Cin, X, Y, Z)
            inject_feat: Optional injection tensor (currently unused)
            
        Returns:
            Output tensor of shape (B, Cout, X, Y, Z)
        """
        # inject_feat is always ignored now
        output_feat: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            if inject_feat is not None:
                setattr(self, 'inject_feat', inject_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n")
        if hasattr(self, 'inject_feat'):
            desc += f"{prefix}  inject_feat: {tuple(self.inject_feat.size())}\n"
        desc += (f"{prefix}  O: \n"
                 f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        for module in self.pipe:
            if hasattr(module, 'io_description'):
                desc += module.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-02-01-01 UNet-Encoder-AdvancedExtractor-Downsample
# Downsampler after each advanced extractor stage
# Pinout Diagram: [Valid]
#                ┌───────────────────────┐
#    input_feat ─│ 00-02-01-01           │─ output_feat
#   (B,C,X,Y,Z)  │ UNet-Encoder-Advanced │  (B,C,X/2,Y/2,Z/2)
#                │ Extractor-Downsample  │
#                └───────────────────────┘
class UNetEncoderAdvancedExtractorDownsample(nn.Module, IODescriptive):
    """
    UNet Encoder Advanced Extractor Downsample Module
    
    Downsamples features by 2x in each spatial dimension using max pooling.
    This reduces spatial resolution while preserving channel dimensions.
    
    Architecture:
        Input → MaxPool3d(2x2x2, stride=2) → Output
    
    Spatial dimensions are halved: (X, Y, Z) → (X/2, Y/2, Z/2)
    
    Attributes:
        pipe: Sequential container with max pooling layer
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            reserve_io: bool = False
    ):
        """
        Initialize UNetEncoderAdvancedExtractorDownsample
        
        Args:
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetEncoderAdvancedExtractorDownsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.MaxPool3d(2, 2)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor) -> Tensor:
        """
        Forward pass through downsampling layer
        
        Args:
            input_feat: Input tensor of shape (B, C, X, Y, Z)
            
        Returns:
            Output tensor of shape (B, C, X/2, Y/2, Z/2)
        """
        output_feat: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        return desc


# 00-03 UNet-Repeater
# Collect multi-stage features from Encoder and dispatch them to proper Decoder stage
# The feature sequence will be reversed to fit Decoder
# Pinout Diagram: [Valid]
#                 ┌───────────────┐
# input_feats[S] ─│ 00-03-00      │─ output_feats[S]
#      [s]:(B,*)  │ UNet-Repeater │  [s]:(B,*)
#                 │               │
#                 └───────────────┘
# Expanded Diagram:
#                   ┌────────────────┐
#   input_feats[0] ─│ 00-03-00 [0]   │─ output_feats[S-1]
#              (*)  │ UNet-Repeater  │  (*)
#                   │ -Bridge        │
#                   └────────────────┘
#                   ⋮       ⋮        ⋮
#                   ┌────────────────┐
#   input_feats[s] ─│ 00-03-00 [s]   │─ output_feats[S-s-1]
#              (*)  │ UNet-Repeater  │  (*)
#                   │ -Bridge        │
#                   └────────────────┘
#                   ⋮       ⋮        ⋮
#                   ┌────────────────┐
# input_feats[S-2] ─│ 00-03-00 [S-2] │─ output_feats[1]
#              (*)  │ UNet-Repeater  │  (*)
#                   │ -Bridge        │
#                   └────────────────┘
#                   ┌───────────────────────┐
#     {input_feat} ─│ 00-03-01              │─ {output_feat}
# input_feats[S-1]  │ UNet-Encoder-Advanced │  output_feats[0]
#    (B,Cin,X,Y,Z)  │ Extractor-Stage       │  (B,Cout,X,Y,Z)
#                   │                       │
#   ×{inject_feat} ─│                       │
#         (Unused)  │                       │
#                   └───────────────────────┘
class UNetRepeater(nn.Module, IODescriptive):
    """
    UNet Repeater Module - Stage Feature Processing and Bottleneck
    
    Processes features from encoder stages and applies bottleneck processing
    to the deepest feature. Consists of:
    - S-1 bridge modules for skip connections
    - 1 bottleneck module for deepest feature processing
    
    Architecture:
        input_feats[0]   ──→   [Bridge]   ──→ output_feats[S-1]
        input_feats[1]   ──→   [Bridge]   ──→ output_feats[S-2]
                                   ⋮
        input_feats[S-1] ──→ [Bottleneck] ──→ output_feats[0]
    
    The repeater processes features in reverse order (deepest first).
    
    Attributes:
        pipe: ModuleList containing S-1 bridges and 1 bottleneck
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            stages: int,  # S: Total number of stages
            bottleneck_in_channels: int,  # Cin: Bottleneck input channels
            bottleneck_out_channels: int,  # Cout: Bottleneck output channels
            bottleneck_depth: int,  # D: Bottleneck layer depth
            reserve_io: bool = False
    ):
        """
        Initialize UNetRepeater
        
        Args:
            stages: Total number of stages (including bottleneck)
            bottleneck_in_channels: Input channels for bottleneck
            bottleneck_out_channels: Output channels from bottleneck
            bottleneck_depth: Number of layers in bottleneck
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetRepeater, self).__init__()
        # Create S-1 bridge modules for skip connections
        self.pipe: nn.ModuleList = nn.ModuleList(
            [UNetRepeaterBridge(reserve_io) for _ in range(stages - 1)]
        )
        # Add bottleneck module for deepest feature processing
        self.pipe.append(
            UNetRepeaterBottleneck(bottleneck_in_channels, bottleneck_out_channels, bottleneck_depth, reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feats: Sequence[Tensor]) -> List[Tensor]:
        """
        Forward pass through repeater
        
        Args:
            input_feats: List of feature tensors from encoder stages
            
        Returns:
            List of processed features (reversed order from input)
        """
        output_feats: List[Tensor] = []
        module: nn.Module
        feat: Tensor
        # Process features in reverse order (deepest first)
        for module, feat in zip(reversed(self.pipe), reversed(input_feats)):
            output_feats.append(module(feat))

        if self.reserve_io:
            setattr(self, 'input_feats', [ft.cpu() for ft in input_feats])
            setattr(self, 'output_feats', [ft.cpu() for ft in output_feats])
        return output_feats

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feats')
                        and hasattr(self, 'output_feats')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feats: {[tuple(ft.size()) for ft in self.input_feats]}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feats: {[tuple(ft.size()) for ft in self.output_feats]}\n")
        for module in self.pipe:
            if hasattr(module, 'io_description'):
                desc += module.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-03-00 UNet-Repeater-Bridge
# The feature delivery path from Encoder to Decoder stage by stage, i.e. skip connection
# Pinout Diagram: [Valid]
#                ┌───────────────┐
#    input_feat ─│ 00-03-00      │─ output_feat
#   (B,C,X,Y,Z)  │ UNet-Repeater │  (B,C,X,Y,Z)
#                │ Bridge        │
#                └───────────────┘
class UNetRepeaterBridge(nn.Module, IODescriptive):
    """
    UNet Repeater Bridge Module - Skip Connection Path
    
    A simple identity pass-through that delivers features from encoder
    to decoder stage by stage, forming skip connections.
    
    Architecture:
        Input → Identity → Output
    
    This module preserves encoder features for fusion in the decoder.
    
    Attributes:
        pipe: Sequential container with identity layer
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            reserve_io: bool = False
    ):
        """
        Initialize UNetRepeaterBridge
        
        Args:
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetRepeaterBridge, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.Identity()
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor) -> Tensor:
        """
        Forward pass through bridge (identity)
        
        Args:
            input_feat: Input tensor of shape (B, C, X, Y, Z)
            
        Returns:
            Output tensor identical to input (B, C, X, Y, Z)
        """
        output_feat: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        return desc


# 00-03-01 UNet-Repeater-Bottleneck
# The most advanced semantic feature delivery path from Encoder to Decoder at the bottom
# Pinout Diagram: [Valid]
#                ┌───────────────────────┐
#    input_feat ─│ 00-03-01              │─ output_feat
# (B,Cin,X,Y,Z)  │ UNet-Encoder-Advanced │  (B,Cout,X,Y,Z)
#   inject_feat ─│ Extractor-Stage       │
#      (Unused)  │                       │
#                └───────────────────────┘
# Expanded Diagram:
#                ┌──────────────┐                              ┌──────────────┐                              ┌──────────────┐
#    input_feat ─│ [0]          │───────────── ··· ────────────│ [d]          │──────────────────────────────│ [D-1]        │─ output_feat
# (B,Cin,X,Y,Z)  │ Conv-BN-ReLU │ {output_feat}   {input_feat} │ Conv-BN-ReLU │ {output_feat}   {input_feat} │ Conv-BN-ReLU │  (B,Cout,X,Y,Z)
#                └──────────────┘ (B,Cin,X,Y,Z)                └──────────────┘ (B,Cin,X,Y,Z)                └──────────────┘
class UNetRepeaterBottleneck(nn.Module, IODescriptive):
    """
    UNet Repeater Bottleneck Module - Deepest Feature Processing
    
    Processes the deepest features from the encoder with multiple Conv-BN-ReLU layers.
    This is the bottleneck of the U-Net architecture where the most advanced
    semantic features are extracted.
    
    Architecture:
        Input → [Conv-BN-ReLU] × (D-1) → Conv-BN-ReLU (Cin→Cout) → Output
    
    The bottleneck increases channel depth while maintaining spatial dimensions.
    
    Attributes:
        pipe: Sequential container with D ConvBNReLU layers
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: int,  # Cin: Input channels
            out_channels: int,  # Cout: Output channels
            depth: int,  # D: Number of ConvBNReLU layers
            reserve_io: bool = False
    ):
        """
        Initialize UNetRepeaterBottleneck
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            depth: Number of ConvBNReLU layers
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetRepeaterBottleneck, self).__init__()
        # Create D-1 layers that maintain channel dimensions
        # followed by 1 layer that changes from Cin to Cout
        self.pipe: nn.Sequential = nn.Sequential(
            *[ConvBNReLU(in_channels, in_channels, 3, padding='same', reserve_io=reserve_io)
              for _ in range(depth - 1)],
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor, inject_feat: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through bottleneck
        
        Args:
            input_feat: Input tensor of shape (B, Cin, X, Y, Z)
            inject_feat: Optional injection tensor (currently unused)
            
        Returns:
            Output tensor of shape (B, Cout, X, Y, Z)
        """
        # inject_feat is always ignored now
        output_feat: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            if inject_feat is not None:
                setattr(self, 'inject_feat', inject_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n")
        if hasattr(self, 'inject_feat'):
            desc += f"{prefix}    inject_feat: {tuple(self.inject_feat.size())}\n"
        desc += (f"{prefix}  O: \n"
                 f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        for module in self.pipe:
            if hasattr(module, 'io_description'):
                desc += module.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-04 UNet-DecoderPriorBank
# [Optional]
# This is for prior injection, you may inject anything to each stage of the Decoder
# Pinout Diagram: [Placeholder]
#                  ┌──────────────┐
# prior_source[S] ─│ 00-01        │─ prior_feats[S]
#   (B,Cin,X,Y,Z)  │ UNet-Decoder │  (B,Cout,X,Y,Z)
#                  │ PriorBank    │
#                  └──────────────┘
class UNetDecoderPriorBank(nn.Module, IODescriptive):
    """
    UNet Decoder Prior Bank Module - Optional Prior Feature Injection
    
    A placeholder module for injecting prior features into each decoder stage.
    This allows external information to be incorporated into the decoding process.
    
    Note:
        This is a placeholder implementation that returns None.
        Subclass and override to implement custom prior injection logic.
    
    Use Cases:
        - Inject anatomical priors
        - Add multi-modal information
        - Incorporate pre-computed features
    
    Attributes:
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            reserve_io: bool = False
    ):
        """
        Initialize UNetDecoderPriorBank
        
        Args:
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetDecoderPriorBank, self).__init__()
        self.reserve_io: bool = reserve_io

    def forward(self, prior_source: Sequence[Tensor]) -> List[Tensor]:
        """
        Forward pass through prior bank
        
        Args:
            prior_source: Sequence of prior source tensors
            
        Returns:
            None (placeholder implementation)
        """
        return None

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        return ''


# 00-04-00 UNet-DecoderPriorBank-Injector
# Injection process pipe
# Pinout Diagram: [Placeholder]
#                ┌───────────────────┐
# inject_source ─│ 00-04-00          │─ inject_feat
# (B,Cin,X,Y,Z)  │ UNet-DecoderPrior │  (B,Cout,X,Y,Z)
#                │ Bank-Injector     │
#                └───────────────────┘
class UNetDecoderPriorBankInjector(nn.Module, IODescriptive):
    """
    UNet Decoder Prior Bank Injector Module - Prior Feature Processing
    
    A placeholder module for processing prior features before injection into
    decoder stages. This allows custom transformation of prior information.
    
    Note:
        This is a placeholder implementation that returns None.
        Subclass and override to implement custom injection processing.
    
    Use Cases:
        - Transform prior features to match decoder dimensions
        - Apply learned projections to prior information
        - Normalize or scale prior features
    
    Attributes:
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            reserve_io: bool = False
    ):
        """
        Initialize UNetDecoderPriorBankInjector
        
        Args:
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetDecoderPriorBankInjector, self).__init__()
        self.reserve_io: bool = reserve_io

    def forward(self, inject_source: Tensor) -> Tensor:
        """
        Forward pass through injector
        
        Args:
            inject_source: Source tensor to inject
            
        Returns:
            None (placeholder implementation)
        """
        return None

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        return ''


# 00-05 UNet-Decoder
# Main Feature Pyramid Aggregator, which is intended to progressively aggregate multi-granularity features
# Pinout Diagram: [Valid]
#                                                                      ┌──────────────┐
#                                                          input_feat ─│ 00-05        │─ output_feats[S1+S2]
#                                                   (B,ACin[0],X,Y,Z)  │ UNet-Decoder │  # Advanced part
#                                                                      │              │  [0]:   (B,ACout[0],   X*2,          Y*2,          Z*2          )
#                                                 bridge_feats[S1+S2] ─│              │  [1]:   (B,ACout[1],   X*4,          Y*4,          Z*4          )
#                                                     # Advanced part  │              │  ⋮
#      [0]:   (B,ASC[0],   X*2,          Y*2,          Z*2          )  │              │  [S1-1]:(B,ACout[S1-1],X*(2^S1),     Y*(2^S1),     Z*(2^S1)     )
#      [1]:   (B,ASC[1],   X*4,          Y*4,          Z*4          )  │              │  # Primary part
#                                                                   ⋮  │              │  [0]:   (B,PCout[0],   X*(2^(S1+1)), Y*(2^(S1+1)), Z*(2^(S1+1)) )
#      [S1-1]:(B,ASC[S1-1],X*(2^S1),     Y*(2^S1),     Z*(2^S1)     )  │              │  [1]:   (B,PCout[1],   X*(2^(S1+2)), Y*(2^(S1+2)), Z*(2^(S1+2)) )
#                                                      # Primary part  │              │  ⋮
#      [0]:   (B,PSC[0],   X*(2^(S1+1)), Y*(2^(S1+1)), Z*(2^(S1+1)) )  │              │  [S2-1]:(B,PCout[S2-1],X*(2^(S1+S2)),Y*(2^(S1+S2)),Z*(2^(S1+S2)))
#      [1]:   (B,PSC[1],   X*(2^(S1+2)), Y*(2^(S1+2)), Z*(2^(S1+2)) )  │              │
#                                                                   ⋮  │              │
#      [S2-1]:(B,PSC[S2-1],X*(2^(S1+S2)),Y*(2^(S1+S2)),Z*(2^(S1+S2)))  │              │
#                                                                      │              │
#                                                 inject_feats[S1+S2] ─│              │
#                                                            (Unused)  │              │
#                                                                      └──────────────┘
# Expanded Diagram:
#                                                                      ┌────────────────────┐
#                                                          input_feat ─│ 00-05-00           │─ output_feats[0:S1]
#                                                   (B,ACin[0],X,Y,Z)  │ UNet-Encoder-      │  # Advanced part
#                                                                      │ AdvancedAggregator │  [0]:   (B,ACout[0],   X*2,     Y*2,     Z*2    )
#                                                  bridge_feats[0:S1] ─│                    │  [1]:   (B,ACout[1],   X*4,     Y*4,     Z*4    )
#                                                     # Advanced part  │                    │  ⋮
#                     [0]:   (B,ASC[0],   X*2,     Y*2,     Z*2     )  │                    │  [S1-1]:(B,ACout[S1-1],X*(2^S1),Y*(2^S1),Z*(2^S1))
#                     [1]:   (B,ASC[1],   X*4,     Y*4,     Z*4     )  │                    │    │
#                                                                   ⋮  │                    │    │
#                     [S1-1]:(B,ASC[S1-1],X*(2^S1),Y*(2^S1),Z*(2^S1))  │                    │    │
#                                                                      │                    │    │
#                                                  inject_feats[0:S1] ─│                    │    │
#                                                            (Unused)  │                    │    │
#                                                                      └────────────────────┘    │
#                                                                                                │       ┌────────────────────┐
#                                                                             output_feats[S1-1] └───────│ 00-05-01           │─ output_feats[S1:S1+S2]
#                                                                                                        │ UNet-Encoder       │  # Primary part
#                                                                                bridge_feats[S1:S1+S2] ─│ -PrimaryAggregator │  [0]:   (B,PCout[0],   X*(2^(S1+1)), Y*(2^(S1+1)), Z*(2^(S1+1)) )
#                                                                                        # Primary part  │                    │  [1]:   (B,PCout[1],   X*(2^(S1+2)), Y*(2^(S1+2)), Z*(2^(S1+2)) )
#                                        [0]:   (B,PSC[0],   X*(2^(S1+1)), Y*(2^(S1+1)), Z*(2^(S1+1)) )  │                    │  ⋮
#                                        [1]:   (B,PSC[1],   X*(2^(S1+2)), Y*(2^(S1+2)), Z*(2^(S1+2)) )  │                    │  [S2-1]:(B,PCout[S2-1],X*(2^(S1+S2)),Y*(2^(S1+S2)),Z*(2^(S1+S2)))
#                                                                                                     ⋮  │                    │
#                                        [S2-1]:(B,PSC[S2-1],X*(2^(S1+S2)),Y*(2^(S1+S2)),Z*(2^(S1+S2)))  │                    │
#                                                                                                        │                    │
#                                                                                inject_feats[S1:S1+S2] ─│                    │
#                                                                                              (Unused)  │                    │
#                                                                                                        └────────────────────┘
class UNetDecoder(nn.Module, IODescriptive):
    """
    UNet Decoder Module - Multi-Scale Feature Reconstruction

    The decoder reconstructs the spatial resolution while combining features
    from the encoder via skip connections. Consists of two parts:
    - Advanced Aggregator: Processes deeper features with skip connections
    - Primary Aggregator: Processes shallower features with skip connections

    Architecture:
                        bridge_feats[S1]            bridge_feats[S2]
                              ↓                         ↓
        bottleneck_feat ──→ Advanced Aggregator ──→ Primary Aggregator
                              ↓                         ↓
                        output_feats[S1]            output_feats[S2]

    Each stage upsamples by 2x and fuses with corresponding encoder features.

    Attributes:
        advanced_aggregator: Advanced feature aggregation stages
        primary_aggregator: Primary feature aggregation stages
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            advanced_in_channels: Sequence[int],  # ACin[S1]: Advanced stage input channels
            advanced_upsample_channels: Sequence[int],  # AUC[S1]: Advanced upsample channels
            advanced_bridge_channels: Sequence[int],  # ASC[S1]: Advanced bridge (skip) channels
            advanced_out_channels: Sequence[int],  # ACout[S1]: Advanced output channels
            advanced_depth: Union[int, Sequence[int]],  # AD[S1]: Advanced stage layer depth
            primary_in_channels: Sequence[int],  # PCin[S2]: Primary stage input channels
            primary_upsample_channels: Sequence[int],  # PUC[S2]: Primary upsample channels
            primary_bridge_channels: Sequence[int],  # PSC[S2]: Primary bridge (skip) channels
            primary_out_channels: Sequence[int],  # PCout[S2]: Primary output channels
            primary_depth: Union[int, Sequence[int]],  # PD[S2]: Primary stage layer depth
            reserve_io: bool = False
    ):
        """
        Initialize UNetDecoder

        Args:
            advanced_in_channels: Input channels for each advanced decoder stage
            advanced_upsample_channels: Upsample channels for each advanced stage
            advanced_bridge_channels: Bridge channels from encoder for each advanced stage
            advanced_out_channels: Output channels for each advanced decoder stage
            advanced_depth: Layer depth for each advanced stage (int or sequence)
            primary_in_channels: Input channels for each primary decoder stage
            primary_upsample_channels: Upsample channels for each primary stage
            primary_bridge_channels: Bridge channels from encoder for each primary stage
            primary_out_channels: Output channels for each primary decoder stage
            primary_depth: Layer depth for each primary stage (int or sequence)
            reserve_io: If True, store I/O tensors for debugging

        Raises:
            AssertionError: If channel/depth sequence lengths don't match
        """
        super(UNetDecoder, self).__init__()
        assert len(advanced_in_channels) == len(advanced_upsample_channels)
        assert len(advanced_bridge_channels) == len(advanced_out_channels)
        assert len(advanced_in_channels) == len(advanced_out_channels)
        if isinstance(advanced_depth, Sequence):
            assert len(advanced_in_channels) == len(advanced_depth)

        assert len(primary_in_channels) == len(primary_upsample_channels)
        assert len(primary_bridge_channels) == len(primary_out_channels)
        assert len(primary_in_channels) == len(primary_out_channels)
        if isinstance(primary_depth, Sequence):
            assert len(primary_in_channels) == len(primary_depth)

        # Initialize advanced and primary aggregators
        self.advanced_aggregator: UNetDecoderAdvancedAggregator = UNetDecoderAdvancedAggregator(
            advanced_in_channels,
            advanced_upsample_channels,
            advanced_bridge_channels,
            advanced_out_channels,
            advanced_depth,
            reserve_io
        )
        self.primary_aggregator: UNetDecoderPrimaryAggregator = UNetDecoderPrimaryAggregator(
            primary_in_channels,
            primary_upsample_channels,
            primary_bridge_channels,
            primary_out_channels,
            primary_depth,
            reserve_io
        )

        self.reserve_io: bool = reserve_io

    def forward(
            self,
            input_feat: Tensor,
            bridge_feats: Sequence[Tensor],
            inject_feats: Optional[Sequence[Tensor]] = None
    ) -> List[Tensor]:
        """
        Forward pass through decoder

        Args:
            input_feat: Bottleneck feature tensor (B, C, X, Y, Z)
            bridge_feats: List of encoder features for skip connections
            inject_feats: Optional prior features to inject at each stage

        Returns:
            List of decoded features from each stage

        Raises:
            AssertionError: If bridge_feats or inject_feats lengths don't match stages
        """
        assert len(bridge_feats) == self.advanced_aggregator.stages + self.primary_aggregator.stages
        assert (inject_feats is None or
                len(inject_feats) == self.advanced_aggregator.stages + self.primary_aggregator.stages)
        if inject_feats is None:
            advanced_feats: List[Tensor] = self.advanced_aggregator(
                input_feat,
                bridge_feats[:self.primary_aggregator.stages]
            )
            primary_feats: List[Tensor] = self.primary_aggregator(
                advanced_feats[self.advanced_aggregator.stages - 1],
                bridge_feats[self.primary_aggregator.stages:]
            )
        else:
            advanced_feats: List[Tensor] = self.advanced_aggregator(
                input_feat,
                bridge_feats[:self.primary_aggregator.stages],
                inject_feats[:self.primary_aggregator.stages]
            )
            primary_feats: List[Tensor] = self.primary_aggregator(
                advanced_feats[self.advanced_aggregator.stages - 1],
                bridge_feats[self.primary_aggregator.stages:],
                inject_feats[self.primary_aggregator.stages:]
            )

        output_feats: List[Tensor] = advanced_feats + primary_feats

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'bridge_feats', [ft.cpu() for ft in bridge_feats])
            if inject_feats is not None:
                setattr(self, 'inject_feats', [ft.cpu() for ft in inject_feats])
            setattr(self, 'output_feats', [ft.cpu() for ft in output_feats])
        return output_feats

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'bridge_feats')
                        and hasattr(self, 'output_feats')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}    bridge_feats: {[tuple(ft.size()) for ft in self.bridge_feats]}\n")
        if hasattr(self, 'inject_feats'):
            desc += f"{prefix}  inject_feats: {[tuple(ft.size()) for ft in self.inject_feats]}\n"
        desc += (f"{prefix}  O: \n"
                 f"{prefix}    output_feats: {[tuple(ft.size()) for ft in self.output_feats]}\n")
        for module in [self.advanced_aggregator, self.primary_aggregator]:
            if hasattr(module, 'io_description'):
                desc += module.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-05-00 UNet-Decoder-AdvancedAggregator
# Deep part of the Decoder, representing advanced semantic / global-view discriminative evidence aggregator
# Pinout Diagram: [Valid]
#                                              ┌──────────────┐
#                                  input_feat ─│ 00-05-00     │─ output_feats[S]
#                            (B,Cin[0],X,Y,Z)  │ UNet-Decoder │  [0]:  (B,Cout[0],  X*2,    Y*2,    Z*2    )
#                                              │ -Advanced    │  [1]:  (B,Cout[1],  X*4,    Y*4,    Z*4    )
#                             bridge_feats[S] ─│ Aggregator   │  ⋮
#   [0]:  (B,SC[0],  X*2,    Y*2,    Z*2    )  │              │  [S-1]:(B,Cout[S-1],X*(2^S),Y*(2^S),Z*(2^S))
#   [1]:  (B,SC[1],  X*4,    Y*4,    Z*4    )  │              │
#                                           ⋮  │              │
#   [S-1]:(B,SC[S-1],X*(2^S),Y*(2^S),Z*(2^S))  │              │
#                                              │              │
#                             inject_feats[S] ─│              │
#                                    (Unused)  │              │
#                                              └──────────────┘
# Expanded Diagram:
#                                     ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#                                     │  ┌───────────────────────┐                                               ┌────────────────────┐                                                     ┌───────────────────────┐  │  output_feats[0]
#                         input_feat ─┼──│ 00-05-00-00           │───────────────────────────────────────────────│ 00-05-00-01        │─────────────────────────────────────────────────────│ 00-05-00-02           │──┼─ {output_feat}→
#                   (B,Cin[0],X,Y,Z)  │  │ UNet-Decoder-Advanced │ {output_feat}                    {input_feat} │ UNet-Decoder-      │ {output_feat}                          {input_feat} │ UNet-Decoder-Advanced │  │  (B,Cout[0],X*2,Y*2,Z*2)
#                                     │  │ Aggregator-Upsample   │ (B,UC[0],X*2,Y*2,Z*2)                         │ AdvancedAggregator │ (B,UC[0]+SC[0],X*2,Y*2,Z*2)                         │ Aggregator-Stage      │  │
#                                     │  └───────────────────────┘                                {bridge_feat} ─│ -FusionPortal      │                                      {inject_feat} ─│ :depth=D/D[0]         │  │
#                                     │                                                   (B,SC[0],X*2,Y*2,Z*2)  │                    │                                    inject_feats[0]  │                       │  │
#                                     │                                                                          └────────────────────┘                                           (Unused)  └───────────────────────┘  │
#                                     └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
#                                     ⋮                                                                                       ⋮                                                                                        ⋮
#                                     ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#                  output_feats[s-1]  │  ┌───────────────────────┐                                               ┌────────────────────┐                                                     ┌───────────────────────┐  │  output_feats[s]
#                     →{output_feat} ─┼──│ 00-05-00-00           │───────────────────────────────────────────────│ 00-05-00-01        │─────────────────────────────────────────────────────│ 00-05-00-02           │──┼─ {output_feat}→
# (B,Cin[s],X*(2^s),Y*(2^s),Z*(2^s))  │  │ UNet-Decoder-Advanced │ {output_feat}                    {input_feat} │ UNet-Decoder-      │ {output_feat}                          {input_feat} │ UNet-Decoder-Advanced │  │  (B,Cout[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))
#                                     │  │ Aggregator-Upsample   │ (B,UC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1))) │ AdvancedAggregator │ (B,UC[s]+SC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1))) │ Aggregator-Stage      │  │
#                                     │  └───────────────────────┘                                {bridge_feat} ─│ -FusionPortal      │                                      {inject_feat} ─│ :depth=D/D[s]         │  │
#                                     │                           (B,SC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))  │                    │                                    inject_feats[s]  │                       │  │
#                                     │                                                                          └────────────────────┘                                           (Unused)  └───────────────────────┘  │
#                                     └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
class UNetDecoderAdvancedAggregator(nn.Module, IODescriptive):
    """
    UNet Decoder Advanced Aggregator Module - Deep Feature Aggregation
    
    Aggregates features from the deep (advanced) part of the decoder.
    Progressively upsamples and fuses features with skip connections from
    the encoder's advanced extractor.
    
    Architecture:
        For each stage s:
            input_feat[s] → Upsample → FusionPortal(+bridge) → Stage → output_feats[s]
    
    Each stage consists of:
    - Upsample: 2x upsampling using transposed convolution
    - FusionPortal: Concatenates upsampled features with skip connections
    - Stage: Conv-BN-ReLU layers for feature refinement
    
    Attributes:
        stages: Number of aggregation stages
        pipe: Sequential container of (Upsample, FusionPortal, Stage) sequences
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: Sequence[int],  # Cin[S]: Input channels for each stage
            upsample_channels: Sequence[int],  # UC[S]: Upsample channels for each stage
            bridge_channels: Sequence[int],  # SC[S]: Bridge channels for each stage
            out_channels: Sequence[int],  # Cout[S]: Output channels for each stage
            depth: Union[int, Sequence[int]],  # D[S]: Layer depth for each stage
            reserve_io: bool = False
    ):
        """
        Initialize UNetDecoderAdvancedAggregator
        
        Args:
            in_channels: Input channels for each stage
            upsample_channels: Channels after upsampling for each stage
            bridge_channels: Channels from skip connections for each stage
            out_channels: Output channels for each stage
            depth: Number of ConvBNReLU layers per stage (int or sequence)
            reserve_io: If True, store I/O tensors for debugging
            
        Raises:
            AssertionError: If channel/depth sequence lengths don't match
        """
        super(UNetDecoderAdvancedAggregator, self).__init__()
        assert len(in_channels) == len(upsample_channels)
        assert len(bridge_channels) == len(out_channels)
        assert len(in_channels) == len(out_channels)
        if isinstance(depth, Sequence):
            assert len(in_channels) == len(depth)
        else:
            depth: List[int] = [depth] * len(in_channels)
        self.stages: int = len(in_channels)
        self.pipe: nn.Sequential = nn.Sequential()

        # Create upsample-fusion-stage sequences for each resolution level
        for ic, uc, sc, oc, dp in zip(in_channels, upsample_channels, bridge_channels, out_channels, depth):
            self.pipe.append(nn.Sequential(
                UNetDecoderAdvancedAggregatorUpsample(ic, uc, reserve_io),
                UNetDecoderAdvancedAggregatorFusionPortal(reserve_io),
                UNetDecoderAdvancedAggregatorStage(uc + sc, oc, dp, reserve_io)
            ))

        self.reserve_io: bool = reserve_io

    def forward(
            self,
            input_feat: Tensor,
            bridge_feats: Sequence[Tensor],
            inject_feats: Optional[Sequence[Tensor]] = None
    ) -> List[Tensor]:
        """
        Forward pass through advanced aggregator
        
        Args:
            input_feat: Input tensor of shape (B, Cin[0], X, Y, Z)
            bridge_feats: Skip connection features from encoder
            inject_feats: Optional prior features to inject at each stage
            
        Returns:
            List of output features from each stage
            
        Raises:
            AssertionError: If bridge_feats or inject_feats length doesn't match stages
        """
        assert len(bridge_feats) == self.stages
        output_feats: List[Tensor] = []
        stage_feat: Tensor = input_feat
        if inject_feats is None:
            for module, bridge_feat in zip(self.pipe, bridge_feats):
                stage_feat = module[0](stage_feat)  # Upsample
                stage_feat = module[1](stage_feat, bridge_feat)  # FusionPortal
                stage_feat = module[2](stage_feat)  # Stage
                output_feats.append(stage_feat)
        else:
            assert len(inject_feats) == self.stages
            for module, bridge_feat, inject_feat in zip(self.pipe, bridge_feats, inject_feats):
                stage_feat = module[0](stage_feat)  # Upsample
                stage_feat = module[1](stage_feat, bridge_feat)  # FusionPortal
                stage_feat = module[2](stage_feat, inject_feat)  # Stage
                output_feats.append(stage_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'bridge_feats', [ft.cpu() for ft in bridge_feats])
            if inject_feats is not None:
                setattr(self, 'inject_feats', [ft.cpu() for ft in inject_feats])
            setattr(self, 'output_feats', [ft.cpu() for ft in output_feats])
        return output_feats

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'bridge_feats')
                        and hasattr(self, 'output_feats')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}    bridge_feats: {[tuple(ft.size()) for ft in self.bridge_feats]}\n")
        if hasattr(self, 'inject_feats'):
            desc += f"{prefix}  inject_feats: {[tuple(ft.size()) for ft in self.inject_feats]}\n"
        desc += (f"{prefix}  O: \n"
                 f"{prefix}    output_feats: {[tuple(ft.size()) for ft in self.output_feats]}\n")
        for module in self.pipe:
            for submodule in cast(nn.Sequential, module):
                if hasattr(submodule, 'io_description'):
                    desc += submodule.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-05-00-00 UNet-Decoder-AdvancedAggregator-Upsample
# Upsampler before each advanced aggregator stage
# Pinout Diagram: [Valid]
#                ┌───────────────────────┐
#    input_feat ─│ 00-05-00-00           │─ output_feat
# (B,Cin,X,Y,Z)  │ UNet-Decoder-Advanced │  (B,Cout,X*2,Y*2,Z*2)
#                │ Aggregator-Upsample   │
#                └───────────────────────┘
class UNetDecoderAdvancedAggregatorUpsample(nn.Module, IODescriptive):
    """
    UNet Decoder Advanced Aggregator Upsample Module
    
    Upsamples features by 2x in each spatial dimension using transposed convolution.
    This increases spatial resolution while changing channel dimensions.
    
    Architecture:
        Input → ConvTranspose3d(2x2x2, stride=2) → Output
    
    Spatial dimensions are doubled: (X, Y, Z) → (X*2, Y*2, Z*2)
    
    Attributes:
        pipe: Sequential container with transposed convolution layer
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: int,  # Cin: Input channels
            out_channels: int,  # Cout: Output channels
            reserve_io: bool = False
    ):
        """
        Initialize UNetDecoderAdvancedAggregatorUpsample
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetDecoderAdvancedAggregatorUpsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor) -> Tensor:
        """
        Forward pass through upsampling layer
        
        Args:
            input_feat: Input tensor of shape (B, Cin, X, Y, Z)
            
        Returns:
            Output tensor of shape (B, Cout, X*2, Y*2, Z*2)
        """
        output_feat: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        return desc


# 00-05-00-01 UNet-Decoder-AdvancedAggregator-FusionPortal
# Fuse features from the deep (advanced, higher semantic) part of Decoder and relative bridge layer (from Encoder)
# Pinout Diagram: [Valid]
#              ┌────────────────────┐
#  stage_feat ─│ 00-05-00-01        │─ output_feat
#    (B,C1,*)  │ UNet-Decoder-      │  (B,C1+C2,*)
# bridge_feat ─│ AdvancedAggregator │
#    (B,C2,*)  │ -FusionPortal      │
#              └────────────────────┘
class UNetDecoderAdvancedAggregatorFusionPortal(nn.Module, IODescriptive):
    """
    UNet Decoder Advanced Aggregator Fusion Portal Module
    
    Fuses features from the decoder's higher semantic layer with the
    corresponding encoder bridge features via concatenation.

    Architecture:
         [stage_feat (B, C1, *), bridge_feat (B, C2, *)] → Concat(dim=1) → output_feat (B, C1+C2, *)

    Attributes:
        concat: Concatenation layer along channel dimension
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            reserve_io: bool = False
    ):
        """
        Initialize UNetDecoderAdvancedAggregatorFusionPortal
        
        Args:
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetDecoderAdvancedAggregatorFusionPortal, self).__init__()
        self.concat: Concat = Concat(dim=1, reserve_io=reserve_io)

        self.reserve_io: bool = reserve_io

    def forward(self, stage_feat: Tensor, bridge_feat: Tensor) -> Tensor:
        """
        Forward pass through fusion portal
        
        Args:
            stage_feat: Upsampled decoder features of shape (B, C1, X, Y, Z)
            bridge_feat: Skip connection features of shape (B, C2, X, Y, Z)
            
        Returns:
            Concatenated features of shape (B, C1+C2, X, Y, Z)
        """
        output_feat: Tensor = self.concat(stage_feat, bridge_feat)

        if self.reserve_io:
            setattr(self, 'stage_feat', stage_feat.cpu())
            setattr(self, 'bridge_feat', bridge_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'stage_feat')
                        and hasattr(self, 'bridge_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    stage_feat: {tuple(self.stage_feat.size())}\n"
                     f"{prefix}    bridge_feat: {tuple(self.bridge_feat.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        return desc


# 00-05-00-02 UNet-Decoder-AdvancedAggregator-Stage
# One stage within the Decoder advanced aggregator, fusing one specified granularity
# Pinout Diagram: [Valid]
#                ┌───────────────────────┐
#    input_feat ─│ 00-05-00-02           │─ output_feat
# (B,Cin,X,Y,Z)  │ UNet-Decoder-Advanced │  (B,Cout,X,Y,Z)
#   inject_feat ─│ Aggregator-Stage      │
#      (Unused)  │                       │
#                └───────────────────────┘
# Expanded Diagram:
#                ┌──────────────┐                              ┌──────────────┐                              ┌──────────────┐
#    input_feat ─│ [0]          │───────────── ··· ────────────│ [d]          │──────────────────────────────│ [D-1]        │─ output_feat
# (B,Cin,X,Y,Z)  │ Conv-BN-ReLU │ {output_feat}   {input_feat} │ Conv-BN-ReLU │ {output_feat}   {input_feat} │ Conv-BN-ReLU │  (B,Cout,X,Y,Z)
#                └──────────────┘ (B,Cout,X,Y,Z)               └──────────────┘ (B,Cout,X,Y,Z)               └──────────────┘
class UNetDecoderAdvancedAggregatorStage(nn.Module, IODescriptive):
    """
    UNet Decoder Advanced Aggregator Stage Module - Single Resolution Feature Processing
    
    Processes features at a single resolution level within the advanced aggregator.
    Consists of D Conv-BN-ReLU layers for feature refinement after fusion.
    
    Architecture:
        Input → Conv-BN-ReLU (Cin→Cout) → [Conv-BN-ReLU] × (D-1) → Output
    
    The first layer changes from Cin to Cout, the remaining D-1 layers maintain
    Cout channels. This differs from encoder stages where the channel change
    happens in the last layer.
    
    Attributes:
        pipe: Sequential container with D ConvBNReLU layers
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: int,  # Cin: Input channels
            out_channels: int,  # Cout: Output channels
            depth: int,  # D: Number of ConvBNReLU layers
            reserve_io: bool = False
    ):
        """
        Initialize UNetDecoderAdvancedAggregatorStage
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            depth: Number of ConvBNReLU layers
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetDecoderAdvancedAggregatorStage, self).__init__()
        # First layer changes from Cin to Cout, remaining layers maintain Cout
        self.pipe: nn.Sequential = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io),
            *[ConvBNReLU(out_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
              for _ in range(depth - 1)],
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor, inject_feat: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through aggregator stage
        
        Args:
            input_feat: Input tensor of shape (B, Cin, X, Y, Z)
            inject_feat: Optional injection tensor (currently unused)
            
        Returns:
            Output tensor of shape (B, Cout, X, Y, Z)
        """
        # inject_feat is always ignored now
        output_feat: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            if inject_feat is not None:
                setattr(self, 'inject_feat', inject_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n")
        if hasattr(self, 'inject_feat'):
            desc += f"{prefix}  inject_feat: {tuple(self.inject_feat.size())}\n"
        desc += (f"{prefix}  O: \n"
                 f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        for module in self.pipe:
            if hasattr(module, 'io_description'):
                desc += module.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-05-01 UNet-Decoder-PrimaryAggregator
# Shallow part of the Decoder, representing primary semantic / local-view discriminative evidence aggregator
# Pinout Diagram: [Valid]
#                                              ┌──────────────┐
#                                  input_feat ─│ 00-05-01     │─ output_feats[S]
#                            (B,Cin[0],X,Y,Z)  │ UNet-Decoder │  [0]:  (B,Cout[0],  X*2,    Y*2,    Z*2    )
#                                              │ -Primary     │  [1]:  (B,Cout[1],  X*4,    Y*4,    Z*4    )
#                             bridge_feats[S] ─│ Aggregator   │  ⋮
#   [0]:  (B,SC[0],  X*2,    Y*2,    Z*2    )  │              │  [S-1]:(B,Cout[S-1],X*(2^S),Y*(2^S),Z*(2^S))
#   [1]:  (B,SC[1],  X*4,    Y*4,    Z*4    )  │              │
#                                           ⋮  │              │
#   [S-1]:(B,SC[S-1],X*(2^S),Y*(2^S),Z*(2^S))  │              │
#                                              │              │
#                             inject_feats[S] ─│              │
#                                    (Unused)  │              │
#                                              └──────────────┘
# Expanded Diagram:
#                                     ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#                                     │  ┌───────────────────────┐                                               ┌────────────────────┐                                                     ┌───────────────────────┐  │  output_feats[0]
#                         input_feat ─┼──│ 00-05-01-00           │───────────────────────────────────────────────│ 00-05-01-01        │─────────────────────────────────────────────────────│ 00-05-01-02           │──┼─ {output_feat}→
#                   (B,Cin[0],X,Y,Z)  │  │ UNet-Decoder-Primary  │ {output_feat}                    {input_feat} │ UNet-Decoder-      │ {output_feat}                          {input_feat} │ UNet-Decoder-Primary  │  │  (B,Cout[0],X*2,Y*2,Z*2)
#                                     │  │ Aggregator-Upsample   │ (B,UC[0],X*2,Y*2,Z*2)                         │ PrimaryAggregator  │ (B,UC[0]+SC[0],X*2,Y*2,Z*2)                         │ Aggregator-Stage      │  │
#                                     │  └───────────────────────┘                                {bridge_feat} ─│ -FusionPortal      │                                      {inject_feat} ─│ :depth=D/D[0]         │  │
#                                     │                                                   (B,SC[0],X*2,Y*2,Z*2)  │                    │                                    inject_feats[0]  │                       │  │
#                                     │                                                                          └────────────────────┘                                           (Unused)  └───────────────────────┘  │
#                                     └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
#                                     ⋮                                                                                       ⋮                                                                                        ⋮
#                                     ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#                  output_feats[s-1]  │  ┌───────────────────────┐                                               ┌────────────────────┐                                                     ┌───────────────────────┐  │  output_feats[s]
#                     →{output_feat} ─┼──│ 00-05-01-00           │───────────────────────────────────────────────│ 00-05-01-01        │─────────────────────────────────────────────────────│ 00-05-01-02           │──┼─ {output_feat}→
# (B,Cin[s],X*(2^s),Y*(2^s),Z*(2^s))  │  │ UNet-Decoder-Primary  │ {output_feat}                    {input_feat} │ UNet-Decoder-      │ {output_feat}                          {input_feat} │ UNet-Decoder-Primary  │  │  (B,Cout[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))
#                                     │  │ Aggregator-Upsample   │ (B,UC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1))) │ PrimaryAggregator  │ (B,UC[s]+SC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1))) │ Aggregator-Stage      │  │
#                                     │  └───────────────────────┘                                {bridge_feat} ─│ -FusionPortal      │                                      {inject_feat} ─│ :depth=D/D[s]         │  │
#                                     │                           (B,SC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))  │                    │                                    inject_feats[s]  │                       │  │
#                                     │                                                                          └────────────────────┘                                           (Unused)  └───────────────────────┘  │
#                                     └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
class UNetDecoderPrimaryAggregator(nn.Module, IODescriptive):
    """
    UNet Decoder Primary Aggregator Module - Shallow Feature Aggregation
    
    Aggregates features from the shallow (primary) part of the decoder.
    Progressively upsamples and fuses features with skip connections from
    the encoder's primary extractor.
    
    Architecture:
        For each stage s:
            input_feat[s] → Upsample → FusionPortal(+bridge) → Stage → output_feats[s]
    
    Each stage consists of:\n
    - Upsample: 2x upsampling using transposed convolution
    - FusionPortal: Concatenates upsampled features with skip connections
    - Stage: Conv-BN-ReLU layers for feature refinement
    
    Attributes:
        stages: Number of aggregation stages
        pipe: Sequential container of (Upsample, FusionPortal, Stage) sequences
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: Sequence[int],  # Cin[S]: Input channels for each stage
            upsample_channels: Sequence[int],  # UC[S]: Upsample channels for each stage
            bridge_channels: Sequence[int],  # SC[S]: Bridge channels for each stage
            out_channels: Sequence[int],  # Cout[S]: Output channels for each stage
            depth: Union[int, Sequence[int]],  # D[S]: Layer depth for each stage
            reserve_io: bool = False
    ):
        """
        Initialize UNetDecoderPrimaryAggregator
        
        Args:
            in_channels: Input channels for each stage
            upsample_channels: Channels after upsampling for each stage
            bridge_channels: Channels from skip connections for each stage
            out_channels: Output channels for each stage
            depth: Number of ConvBNReLU layers per stage (int or sequence)
            reserve_io: If True, store I/O tensors for debugging
            
        Raises:
            AssertionError: If channel/depth sequence lengths don't match
        """
        super(UNetDecoderPrimaryAggregator, self).__init__()
        assert len(in_channels) == len(upsample_channels)
        assert len(bridge_channels) == len(out_channels)
        assert len(in_channels) == len(out_channels)
        if isinstance(depth, Sequence):
            assert len(in_channels) == len(depth)
        else:
            depth: List[int] = [depth] * len(in_channels)
        self.stages: int = len(in_channels)
        self.pipe: nn.Sequential = nn.Sequential()

        # Create upsample-fusion-stage sequences for each resolution level
        for ic, uc, sc, oc, dp in zip(in_channels, upsample_channels, bridge_channels, out_channels, depth):
            self.pipe.append(nn.Sequential(
                UNetDecoderPrimaryAggregatorUpsample(ic, uc, reserve_io),
                UNetDecoderPrimaryAggregatorFusionPortal(reserve_io),
                UNetDecoderPrimaryAggregatorStage(uc + sc, oc, dp, reserve_io)
            ))

        self.reserve_io: bool = reserve_io

    def forward(
            self,
            input_feat: Tensor,  # (B,Cin[0],X,Y,Z)
            bridge_feats: Sequence[Tensor],  # [s]:(B,SC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))
            inject_feats: Optional[Sequence[Tensor]] = None  # [s]:(B,*,X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))
    ) -> List[Tensor]:
        """
        Forward pass through primary aggregator
        
        Args:
            input_feat: Input tensor of shape (B, Cin[0], X, Y, Z)
            bridge_feats: Skip connection features from encoder
            inject_feats: Optional prior features to inject at each stage
            
        Returns:
            List of output features from each stage
            
        Raises:
            AssertionError: If bridge_feats or inject_feats length doesn't match stages
        """
        assert len(bridge_feats) == self.stages
        output_feats: List[Tensor] = []
        stage_feat: Tensor = input_feat
        if inject_feats is None:
            for module, bridge_feat in zip(self.pipe, bridge_feats):
                stage_feat = module[0](stage_feat)  # Upsample
                stage_feat = module[1](stage_feat, bridge_feat)  # FusionPortal
                stage_feat = module[2](stage_feat)  # Stage
                output_feats.append(stage_feat)
        else:
            assert len(inject_feats) == self.stages
            for module, bridge_feat, inject_feat in zip(self.pipe, bridge_feats, inject_feats):
                stage_feat = module[0](stage_feat)  # Upsample
                stage_feat = module[1](stage_feat, bridge_feat)  # FusionPortal
                stage_feat = module[2](stage_feat, inject_feat)  # Stage
                output_feats.append(stage_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'bridge_feats', [ft.cpu() for ft in bridge_feats])
            if inject_feats is not None:
                setattr(self, 'inject_feats', [ft.cpu() for ft in inject_feats])
            setattr(self, 'output_feats', [ft.cpu() for ft in output_feats])
        return output_feats

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'bridge_feats')
                        and hasattr(self, 'output_feats')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}    bridge_feats: {[tuple(ft.size()) for ft in self.bridge_feats]}\n")
        if hasattr(self, 'inject_feats'):
            desc += f"{prefix}  inject_feats: {[tuple(ft.size()) for ft in self.inject_feats]}\n"
        desc += (f"{prefix}  O: \n"
                 f"{prefix}    output_feats: {[tuple(ft.size()) for ft in self.output_feats]}\n")
        for module in self.pipe:
            for submodule in cast(nn.Sequential, module):
                if hasattr(submodule, 'io_description'):
                    desc += submodule.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-05-01-00 UNet-Decoder-PrimaryAggregator-Upsample
# Upsampler before each primary aggregator stage
# Pinout Diagram: [Valid]
#                ┌──────────────────────┐
#    input_feat ─│ 00-05-01-00          │─ output_feat
# (B,Cin,X,Y,Z)  │ UNet-Decoder-Primary │  (B,Cout,X*2,Y*2,Z*2)
#                │ Aggregator-Upsample  │
#                └──────────────────────┘
class UNetDecoderPrimaryAggregatorUpsample(nn.Module, IODescriptive):
    """
    UNet Decoder Primary Aggregator Upsample Module
    
    Upsamples features by 2x in each spatial dimension using transposed convolution.
    This is used in the primary decoder stages to increase spatial resolution.
    
    Architecture:
        Input → ConvTranspose3d(2x2x2, stride=2) → Output
    
    Spatial dimensions are doubled: (X, Y, Z) → (2X, 2Y, 2Z)
    
    Attributes:
        pipe: Sequential container with transposed convolution
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: int,  # Cin: Input channels
            out_channels: int,  # Cout: Output channels
            reserve_io: bool = False
    ):
        """
        Initialize UNetDecoderPrimaryAggregatorUpsample
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetDecoderPrimaryAggregatorUpsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor) -> Tensor:
        """
        Forward pass through upsampling layer
        
        Args:
            input_feat: Input tensor of shape (B, Cin, X, Y, Z)
            
        Returns:
            Output tensor of shape (B, Cout, 2X, 2Y, 2Z)
        """
        output_feat: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        return desc


# 00-05-01-01 UNet-Decoder-PrimaryAggregator-FusionPortal
# Fuse features from the shallow (primary, lower semantic) part of Decoder and relative bridge layer (from Encoder)
# Pinout Diagram: [Valid]
#              ┌───────────────────┐
#  stage_feat ─│ 00-05-01-01       │─ output_feat
#    (B,C1,*)  │ UNet-Decoder-     │  (B,C1+C2,*)
# bridge_feat ─│ PrimaryAggregator │
#    (B,C2,*)  │ -FusionPortal     │
#              └───────────────────┘
class UNetDecoderPrimaryAggregatorFusionPortal(nn.Module, IODescriptive):
    """
    UNet Decoder Primary Aggregator Fusion Portal Module
    
    Fuses features from the decoder's lower semantic layer with the
    corresponding encoder bridge features via concatenation.
    
    Architecture:
         [stage_feat (B, C1, *), bridge_feat (B, C2, *)] → Concat(dim=1) → output_feat (B, C1+C2, *)
    
    Attributes:
        concat: Concatenation layer along channel dimension
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            reserve_io: bool = False
    ):
        """
        Initialize UNetDecoderPrimaryAggregatorFusionPortal
        
        Args:
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetDecoderPrimaryAggregatorFusionPortal, self).__init__()
        self.concat: Concat = Concat(dim=1, reserve_io=reserve_io)

        self.reserve_io: bool = reserve_io

    def forward(self, stage_feat: Tensor, bridge_feat: Tensor) -> Tensor:
        """
        Forward pass through fusion portal
        
        Args:
            stage_feat: Upsampled feature from decoder (B, C1, X, Y, Z)
            bridge_feat: Skip connection feature from encoder (B, C2, X, Y, Z)
            
        Returns:
            Concatenated feature tensor (B, C1+C2, X, Y, Z)
        """
        output_feat: Tensor = self.concat(stage_feat, bridge_feat)

        if self.reserve_io:
            setattr(self, 'stage_feat', stage_feat.cpu())
            setattr(self, 'bridge_feat', bridge_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'stage_feat')
                        and hasattr(self, 'bridge_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    stage_feat: {tuple(self.stage_feat.size())}\n"
                     f"{prefix}    bridge_feat: {tuple(self.bridge_feat.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        return desc


# 00-05-01-02 UNet-Decoder-PrimaryAggregator-Stage
# One stage within the Decoder primary aggregator, fusing one specified granularity
# Pinout Diagram: [Valid]
#                ┌──────────────────────┐
#    input_feat ─│ 00-05-01-02          │─ output_feat
# (B,Cin,X,Y,Z)  │ UNet-Decoder-Primary │  (B,Cout,X,Y,Z)
#   inject_feat ─│ Aggregator-Stage     │
#      (Unused)  │                      │
#                └──────────────────────┘
# Expanded Diagram:
#                ┌──────────────┐                              ┌──────────────┐                              ┌──────────────┐
#    input_feat ─│ [0]          │───────────── ··· ────────────│ [d]          │──────────────────────────────│ [D-1]        │─ output_feat
# (B,Cin,X,Y,Z)  │ Conv-BN-ReLU │ {output_feat}   {input_feat} │ Conv-BN-ReLU │ {output_feat}   {input_feat} │ Conv-BN-ReLU │  (B,Cout,X,Y,Z)
#                └──────────────┘ (B,Cout,X,Y,Z)               └──────────────┘ (B,Cout,X,Y,Z)               └──────────────┘
class UNetDecoderPrimaryAggregatorStage(nn.Module, IODescriptive):
    """
    UNet Decoder Primary Aggregator Stage Module
    
    One stage within the primary decoder aggregator that processes fused features.
    Consists of D Conv-BN-ReLU layers for feature refinement.
    
    Architecture:
        Input → Conv-BN-ReLU (Cin→Cout) → [Conv-BN-ReLU] × (D-1) → Output
    
    The first layer changes channel dimensions, subsequent layers maintain them.
    
    Attributes:
        pipe: Sequential container with D ConvBNReLU layers
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: int,  # Cin: Input channels
            out_channels: int,  # Cout: Output channels
            depth: int,  # D: Number of ConvBNReLU layers
            reserve_io: bool = False
    ):
        """
        Initialize UNetDecoderPrimaryAggregatorStage
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            depth: Number of ConvBNReLU layers
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetDecoderPrimaryAggregatorStage, self).__init__()
        # First layer changes from Cin to Cout, subsequent layers maintain Cout
        self.pipe: nn.Sequential = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io),
            *[ConvBNReLU(out_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
              for _ in range(depth - 1)]
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor, inject_feat: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through decoder stage
        
        Args:
            input_feat: Input tensor of shape (B, Cin, X, Y, Z)
            inject_feat: Optional injection tensor (currently unused)
            
        Returns:
            Output tensor of shape (B, Cout, X, Y, Z)
        """
        # inject_feat is always ignored now
        output_feat: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            if inject_feat is not None:
                setattr(self, 'inject_feat', inject_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n")
        if hasattr(self, 'inject_feat'):
            desc += f"{prefix}  inject_feat: {tuple(self.inject_feat.size())}\n"
        desc += (f"{prefix}  O: \n"
                 f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        for module in self.pipe:
            if hasattr(module, 'io_description'):
                desc += module.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-06 UNet-AuxiliaryClassifier
# [Optional]
# Auxiliary Classifier is for post Decoder stage auxiliary supervision tasks, such as deep supervision
# Pinout Diagram: [Valid]
#                                ┌────────────────┐
#                input_feats[S] ─│ 00-06          │─ aux_logits[S]
# [s]:(B,Cin[s],X[s],Y[s],Z[s])  │ UNet-Auxiliary │  [s]:(B,Cout[s],X[s],Y[s],Z[s])
#                                │ Classifier     │
#                                └────────────────┘
# Expanded Diagram:
#                                    ┌──────────────┐                         ┌────────┐  aux_logits[0]
#                    input_feats[0] ─│ [0]          │────────── ··· ──────────│ [0]    │─ {output}
#         (B,Cin[0],X[0],Y[0],Z[0])  │ Conv-BN-ReLU │ {output_feat}   {input} │ Conv3d │  (B,Cout[0],X[0],Y[0],Z[0])
#                                    └──────────────┘ (B,Cin[0],X,Y,Z)        └────────┘
#                                    ⋮      ⋮       ⋮
#                                    ┌──────────────┐                         ┌────────┐  aux_logits[s]
#                    input_feats[s] ─│ [s]          │────────── ··· ──────────│ [s]    │─ {output}
#         (B,Cin[s],X[s],Y[s],Z[s])  │ Conv-BN-ReLU │ {output_feat}   {input} │ Conv3d │  (B,Cout[s],X[s],Y[s],Z[s])
#                                    └──────────────┘ (B,Cin[s],X,Y,Z)        └────────┘
#                                    ⋮      ⋮       ⋮
#                                    ┌──────────────┐                         ┌────────┐  aux_logits[S-1]
#                  input_feats[S-1] ─│ [S-1]        │────────── ··· ──────────│ [S-1]  │─ {output}
# (B,Cin[S-1],X[S-1],Y[S-1],Z[S-1])  │ Conv-BN-ReLU │ {output_feat}   {input} │ Conv3d │  (B,Cout[S-1],X[S-1],Y[S-1],Z[S-1])
#                                    └──────────────┘ (B,Cin[S-1],X,Y,Z)      └────────┘
class UNetAuxiliaryClassifier(nn.Module, IODescriptive):
    """
    UNet Auxiliary Classifier Module - Deep Supervision Outputs
    
    Provides auxiliary classification outputs at multiple decoder stages
    for deep supervision during training. Each output corresponds to
    a different spatial resolution.
    
    Architecture:
        For each stage s:
            input_feats[s] → Conv-BN-ReLU(3x3x3) → Conv3d(1x1x1) → aux_logits[s]
    
    Deep supervision helps train intermediate features and improves gradient flow.
    
    Attributes:
        pipe: ModuleList with S classifiers (each: ConvBNReLU + Conv3d)
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: Sequence[int],  # Cin[S]: Input channels for each stage
            out_channels: Sequence[int],  # Cout[S]: Output classes for each stage
            reserve_io: bool = False
    ):
        """
        Initialize UNetAuxiliaryClassifier
        
        Args:
            in_channels: Input channels for each auxiliary classifier
            out_channels: Number of output classes for each auxiliary classifier
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetAuxiliaryClassifier, self).__init__()
        self.pipe: nn.ModuleList = nn.ModuleList()
        # Create a classifier for each stage
        for ic, oc in zip(in_channels, out_channels):
            self.pipe.append(nn.Sequential(
                ConvBNReLU(ic, ic, 3, padding='same', reserve_io=reserve_io),
                nn.Conv3d(ic, oc, 1)
            ))

        self.reserve_io: bool = reserve_io

    def forward(self, input_feats: Sequence[Tensor]) -> List[Tensor]:
        """
        Forward pass through auxiliary classifiers
        
        Args:
            input_feats: List of feature tensors from decoder stages
            
        Returns:
            List of auxiliary logits for deep supervision
            
        Raises:
            AssertionError: If input_feats length doesn't match number of classifiers
        """
        assert len(input_feats) == len(self.pipe)
        aux_logits: List[Tensor] = []
        for module, feat in zip(self.pipe, input_feats):
            aux_logits.append(module(feat))

        if self.reserve_io:
            setattr(self, 'input_feats', [ft.cpu() for ft in input_feats])
            setattr(self, 'aux_logits', [lt.cpu() for lt in aux_logits])
        return aux_logits

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feats')
                        and hasattr(self, 'aux_logits')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feats: {[tuple(ft.size()) for ft in self.input_feats]}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    aux_logits: {[tuple(lt.size()) for lt in self.aux_logits]}\n")
        for module in self.pipe:
            for submodule in cast(nn.Sequential, module):
                if hasattr(submodule, 'io_description'):
                    desc += submodule.io_description(max_level, indent + 1, indent_placeholder, target_level + 1)
        return desc


# 00-07 UNet-Distributor
# [Optional]
# Distributor is a Stem Feature Aggregator after Decoder, which is intended to generate or filter discriminative features
# Pinout Diagram: [Valid]
#                ┌──────────────────┐
#    input_feat ─│ 00-07            │─ output_feat
# (B,Cin,X,Y,Z)  │ UNet-Distributor │  (B,Cout,X,Y,Z)
#                └──────────────────┘
class UNetDistributor(nn.Module, IODescriptive):
    """
    UNet Distributor Module - Post-Decoder Feature Refinement
    
    A stem feature aggregator that processes decoder output before classification.
    Intended to generate or filter discriminative features for the final classifier.
    
    Architecture:
        Input → Conv3d(3x3x3) → BatchNorm → ReLU → Output
    
    Attributes:
        pipe: Sequential container with ConvBNReLU layer
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: int,  # Cin: Number of input channels
            out_channels: int,  # Cout: Number of output channels
            reserve_io: bool = False
    ):
        """
        Initialize UNetDistributor
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetDistributor, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor) -> Tensor:
        """
        Forward pass through distributor
        
        Args:
            input_feat: Input tensor of shape (B, Cin, X, Y, Z)
            
        Returns:
            Output tensor of shape (B, Cout, X, Y, Z)
        """
        output_feat: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'output_feat', output_feat.cpu())
        return output_feat

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    output_feat: {tuple(self.output_feat.size())}\n")
        return desc


# 00-08 UNet-Classifier
# Pointwise main classifier
# Pinout Diagram: [Valid]
#                ┌─────────────────┐
#    input_feat ─│ 00-08           │─ logits
# (B,Cin,X,Y,Z)  │ UNet-Classifier │  (B,Cout,X,Y,Z)
#                └─────────────────┘
class UNetClassifier(nn.Module, IODescriptive):
    """
    UNet Classifier Module - Final Segmentation Output Layer
    
    A pointwise (1x1x1) convolution that maps features to class logits.
    This is the final layer that produces the segmentation output.
    
    Architecture:
        Input → Conv3d(1x1x1) → Logits
        
    Note:
        This module returns raw logits, not probabilities.
        Apply sigmoid or softmax externally for normalization.
    
    Attributes:
        pipe: Sequential container with 1x1x1 convolution
        reserve_io: Whether to store I/O tensors for debugging
    """
    def __init__(
            self,
            in_channels: int,  # Cin: Number of input feature channels
            out_channels: int,  # Cout: Number of output classes
            reserve_io: bool = False
    ):
        """
        Initialize UNetClassifier
        
        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output classes
            reserve_io: If True, store I/O tensors for debugging
        """
        super(UNetClassifier, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1)
            # nn.Sigmoid()  # Only return logits, apply normalization latter
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: Tensor) -> Tensor:
        """
        Forward pass through classifier
        
        Args:
            input_feat: Input feature tensor of shape (B, Cin, X, Y, Z)
            
        Returns:
            Logits tensor of shape (B, Cout, X, Y, Z)
        """
        logits: Tensor = self.pipe(input_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'logits', logits.cpu())
        return logits

    def io_description(
            self,
            max_level: int = 0,
            indent: int = 0,
            indent_placeholder: str = '  ',
            target_level: int = 0
    ) -> str:
        if (not self.reserve_io or target_level > max_level
                or not (
                        hasattr(self, 'input_feat')
                        and hasattr(self, 'logits')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}  O: \n"
                     f"{prefix}    logits: {tuple(self.logits.size())}\n")
        return desc


if __name__ == "__main__":
    # Sample input
    torch.manual_seed(0)
    input_source: Tensor = torch.randn(3, 1, 64, 128, 256)
    B, C, X, Y, Z = input_source.size()

    # Params
    focuser_in_channels: int = 1
    focuser_out_channels: int = 16
    encoder_primary_in_channels: Sequence[int] = (16, 32)
    encoder_primary_out_channels: Sequence[int] = (32, 64)
    encoder_primary_depth: Union[int, Sequence[int]] = 2
    encoder_advanced_in_channels: Sequence[int] = (64, 128)
    encoder_advanced_out_channels: Sequence[int] = (128, 256)
    encoder_advanced_depth: Union[int, Sequence[int]] = 2
    bottleneck_in_channels: int = 256
    bottleneck_out_channels: int = 512
    bottleneck_depth: int = 2
    decoder_advanced_in_channels: Sequence[int] = (512, 256)
    decoder_advanced_upsample_channels: Sequence[int] = (256, 128)
    decoder_advanced_bridge_channels: Sequence[int] = (256, 128)
    decoder_advanced_out_channels: Sequence[int] = (256, 128)
    decoder_advanced_depth: Union[int, Sequence[int]] = 2
    decoder_primary_in_channels: Sequence[int] = (128, 64)
    decoder_primary_upsample_channels: Sequence[int] = (64, 32)
    decoder_primary_bridge_channels: Sequence[int] = (64, 32)
    decoder_primary_out_channels: Sequence[int] = (64, 32)
    decoder_primary_depth: Union[int, Sequence[int]] = 2
    auxiliary_classifier_in_channels: Sequence[int] = (256, 128, 64, 32)
    auxiliary_classifier_out_channels: Sequence[int] = (2, 2, 2, 2)
    distributor_in_channels: int = 32
    distributor_out_channels: int = 16
    classifier_in_channels: int = 16
    classifier_out_channels: int = 2
    reserve_io: bool = True

    # Info
    stages: int = len(auxiliary_classifier_in_channels)
    primary_stages: int = len(encoder_primary_in_channels)
    advanced_stages: int = len(encoder_advanced_in_channels)
    encoder_in_channels: List[int] = list(encoder_primary_in_channels) + list(encoder_advanced_in_channels)
    encoder_out_channels: List[int] = list(encoder_primary_out_channels) + list(encoder_advanced_out_channels)
    decoder_in_channels: List[int] = list(decoder_advanced_in_channels) + list(decoder_primary_in_channels)
    decoder_bridge_channels: List[int] = list(decoder_advanced_bridge_channels) + \
                                           list(decoder_primary_bridge_channels)
    decoder_out_channels: List[int] = list(decoder_advanced_out_channels) + list(decoder_primary_out_channels)

    # Record whether tests passed
    test_states: List[bool] = []

    # Create default UNet
    unet: UNet = UNet(
        focuser_in_channels=focuser_in_channels,
        focuser_out_channels=focuser_out_channels,
        encoder_primary_in_channels=encoder_primary_in_channels,
        encoder_primary_out_channels=encoder_primary_out_channels,
        encoder_primary_depth=encoder_primary_depth,
        encoder_advanced_in_channels=encoder_advanced_in_channels,
        encoder_advanced_out_channels=encoder_advanced_out_channels,
        encoder_advanced_depth=encoder_advanced_depth,
        bottleneck_in_channels=bottleneck_in_channels,
        bottleneck_out_channels=bottleneck_out_channels,
        bottleneck_depth=bottleneck_depth,
        decoder_advanced_in_channels=decoder_advanced_in_channels,
        decoder_advanced_upsample_channels=decoder_advanced_upsample_channels,
        decoder_advanced_bridge_channels=decoder_advanced_bridge_channels,
        decoder_advanced_out_channels=decoder_advanced_out_channels,
        decoder_advanced_depth=decoder_advanced_depth,
        decoder_primary_in_channels=decoder_primary_in_channels,
        decoder_primary_upsample_channels=decoder_primary_upsample_channels,
        decoder_primary_bridge_channels=decoder_primary_bridge_channels,
        decoder_primary_out_channels=decoder_primary_out_channels,
        decoder_primary_depth=decoder_primary_depth,
        auxiliary_classifier_in_channels=auxiliary_classifier_in_channels,
        auxiliary_classifier_out_channels=auxiliary_classifier_out_channels,
        distributor_in_channels=distributor_in_channels,
        distributor_out_channels=distributor_out_channels,
        classifier_in_channels=classifier_in_channels,
        classifier_out_channels=classifier_out_channels,
        reserve_io=reserve_io
    )


    # region Unit Test
    # 00 UNet
    def test_unet_io(seq: int, module: UNet, input_source: Tensor) -> Tuple[Tensor, List[Tensor]]:
        print(f"[{seq}] Test 00 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_source: Tuple[int, ...] = tuple(input_source.size())
        sz_input_source_expected: Tuple[int, ...] = tuple([B, C, X, Y, Z])
        # Output
        cls_logits: Tensor
        aux_cls_logits: List[Tensor]
        cls_logits, aux_cls_logits = module(input_source)
        sz_cls_logits: Tuple[int, ...] = tuple(cls_logits.size())
        sz_cls_logits_expected: Tuple[int, ...] = tuple([
            sz_input_source[0],
            classifier_out_channels,
            sz_input_source[2],
            sz_input_source[3],
            sz_input_source[4]
        ])
        sz_aux_cls_logits: List[Tuple[int, ...]] = [tuple(lt.size()) for lt in aux_cls_logits]
        sz_aux_cls_logits_expected: List[Tuple[int, ...]] = [
            tuple([sz_input_source[0], classifier_out_channels,
                   sz_input_source[2] // (2 ** (s - 1)),
                   sz_input_source[3] // (2 ** (s - 1)),
                   sz_input_source[4] // (2 ** (s - 1))
                   ]) for s in range(stages, 0, -1)
        ]

        print(f"    Input for UNet:\n"
              f"      input_source: {sz_input_source}"
              f" {'=' if sz_input_source == sz_input_source_expected else '≠'} {sz_input_source_expected} (expected)")
        print(f"    Output for UNet:\n"
              f"      cls_logits: {sz_cls_logits}"
              f" {'=' if sz_cls_logits == sz_cls_logits_expected else '≠'} {sz_cls_logits_expected} (expected)")
        print(f"      aux_cls_logits:")
        for idx in range(max(len(sz_aux_cls_logits), len(sz_aux_cls_logits_expected))):
            if idx < len(sz_aux_cls_logits_expected):
                sz_expected: Tuple[int, ...] = sz_aux_cls_logits_expected[idx]
                if idx < len(sz_aux_cls_logits):
                    sz_out: Tuple[int, ...] = sz_aux_cls_logits[idx]
                    print(f"        [{idx}] {sz_out}"
                          f" {'=' if sz_out == sz_expected else '≠'} {sz_expected} (expected)")
                else:
                    print(f"        [{idx}] None ≠ {sz_expected} (expected)")
            else:
                sz_out: Tuple[int, ...] = sz_aux_cls_logits[idx]
                print(f"        [{idx}] {sz_out} ≠ None (expected)")

        passed: bool = True
        passed = passed and (sz_input_source == sz_input_source_expected)
        passed = passed and (sz_cls_logits == sz_cls_logits_expected)
        if len(sz_aux_cls_logits) != len(sz_aux_cls_logits_expected):
            passed = False
        else:
            for idx in range(max(len(sz_aux_cls_logits), len(sz_aux_cls_logits_expected))):
                logits: Tuple[int, ...] = sz_aux_cls_logits[idx]
                expected: Tuple[int, ...] = sz_aux_cls_logits_expected[idx]
                passed = passed and (logits == expected)
        test_states.append(passed)
        print(f"[{seq}] {'PASS' if passed else 'FAILED'}\n")

        return cls_logits, aux_cls_logits


    # 00-00 UNet-Focuser
    def test_unet_focuser_io(seq: int, module: UNetFocuser, input_source: Tensor) -> Tensor:
        print(f"[{seq}] Test 00-00 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_source: Tuple[int, ...] = tuple(input_source.size())
        sz_input_source_expected: Tuple[int, ...] = tuple(tuple([B, C, X, Y, Z]))
        # Output
        output_feat: Tensor = module(input_source)
        sz_output_feat: Tuple[int, ...] = tuple(output_feat.size())
        sz_output_feat_expected: Tuple[int, ...] = tuple([B, focuser_out_channels, X, Y, Z])

        print(f"    Input for UNetFocuser:\n"
              f"      input_source: {sz_input_source}"
              f" {'=' if sz_input_source == sz_input_source_expected else '≠'} {sz_input_source_expected} (expected)")
        print(f"    Output for UNetFocuser:\n"
              f"      output_feat: {sz_output_feat}"
              f" {'=' if sz_output_feat == sz_output_feat_expected else '≠'} {sz_output_feat_expected} (expected)")

        passed: bool = True
        passed = passed and sz_input_source == sz_input_source_expected
        passed = passed and sz_output_feat == sz_output_feat_expected
        test_states.append(passed)
        print(f"[{seq}] {'PASS' if passed else 'FAILED'}\n")

        return output_feat


    # 00-02 UNet-Encoder
    def test_unet_encoder_io(
            seq: int,
            module: UNetEncoder,
            input_feat: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        print(f"[{seq}] Test 00-02 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_feat: Tuple[int, ...] = tuple(input_feat.size())
        sz_input_feat_expected: Tuple[int, ...] = tuple([B, encoder_in_channels[0], X, Y, Z])
        # Output
        output_feats: List[Tensor]
        dn_feats: List[Tensor]
        output_feats, dn_feats = module(input_feat)
        sz_output_feats: List[Tuple[int, ...]] = [tuple(ft.size()) for ft in output_feats]
        sz_output_feats_expected: List[Tuple[int, ...]] = [tuple([B, encoder_out_channels[s],
                                                                  X // (2 ** s),
                                                                  Y // (2 ** s),
                                                                  Z // (2 ** s)
                                                                  ]) for s in range(stages)]
        sz_dn_feats: List[Tuple[int, ...]] = [tuple(ft.size()) for ft in dn_feats]
        sz_dn_feats_expected: List[Tuple[int, ...]] = [tuple([B, encoder_out_channels[s],
                                                              X // (2 ** (s + 1)),
                                                              Y // (2 ** (s + 1)),
                                                              Z // (2 ** (s + 1))
                                                              ]) for s in range(stages)]

        print(f"    Input for UNetEncoder:\n"
              f"      input_feat: {sz_input_feat}"
              f" {'=' if sz_input_feat == sz_input_feat_expected else '≠'} {sz_input_feat_expected} (expected)")
        print(f"    Output for UNetEncoder:")
        print(f"      output_feats:")
        for idx in range(max(len(sz_output_feats), len(sz_output_feats_expected))):
            if idx < len(sz_output_feats_expected):
                sz_expected: Tuple[int, ...] = sz_output_feats_expected[idx]
                if idx < len(sz_output_feats):
                    sz_out: Tuple[int, ...] = sz_output_feats[idx]
                    print(f"        [{idx}] {sz_out}"
                          f" {'=' if sz_out == sz_expected else '≠'} {sz_expected} (expected)")
                else:
                    print(f"        [{idx}] None ≠ {sz_expected} (expected)")
            else:
                sz_out: Tuple[int, ...] = sz_output_feats[idx]
                print(f"        [{idx}] {sz_out} ≠ None (expected)")
        print(f"      dn_feats:")
        for idx in range(max(len(sz_dn_feats), len(sz_dn_feats_expected))):
            if idx < len(sz_dn_feats_expected):
                sz_expected: Tuple[int, ...] = sz_dn_feats_expected[idx]
                if idx < len(sz_dn_feats):
                    sz_out: Tuple[int, ...] = sz_dn_feats[idx]
                    print(f"        [{idx}] {sz_out}"
                          f" {'=' if sz_out == sz_expected else '≠'} {sz_expected} (expected)")
                else:
                    print(f"        [{idx}] None ≠ {sz_expected} (expected)")
            else:
                sz_out: Tuple[int, ...] = sz_dn_feats[idx]
                print(f"        [{idx}] {sz_out} ≠ None (expected)")

        passed: bool = True
        passed = passed and (sz_input_feat == sz_input_feat_expected)
        if len(sz_output_feats) != len(sz_output_feats_expected):
            passed = False
        else:
            for idx in range(max(len(sz_output_feats), len(sz_output_feats_expected))):
                feat: Tuple[int, ...] = sz_output_feats[idx]
                expected: Tuple[int, ...] = sz_output_feats_expected[idx]
                passed = passed and (feat == expected)
        if len(sz_dn_feats) != len(sz_dn_feats_expected):
            passed = False
        else:
            for idx in range(max(len(sz_dn_feats), len(sz_dn_feats_expected))):
                feat: Tuple[int, ...] = sz_dn_feats[idx]
                expected: Tuple[int, ...] = sz_dn_feats_expected[idx]
                passed = passed and (feat == expected)
        test_states.append(passed)
        print(f"[{seq}] {'PASS' if passed else 'FAILED'}\n")

        return output_feats, dn_feats


    # 00-03 UNet-Repeater
    def test_unet_repeater_io(
            seq: int,
            module: UNetRepeater,
            input_feats: Sequence[Tensor]
    ) -> List[Tensor]:
        print(f"[{seq}] Test 00-03 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_feats: List[Tuple[int, ...]] = [tuple(ft.size()) for ft in input_feats]
        sz_input_feats_expected: List[Tuple[int, ...]] = \
            [tuple([B, encoder_out_channels[s], X // (2 ** s), Y // (2 ** s), Z // (2 ** s)]) for s in range(stages)] + \
            [tuple([B, bottleneck_in_channels, X // (2 ** stages), Y // (2 ** stages), Z // (2 ** stages)])]
        # Output
        output_feats: List[Tensor] = module(input_feats)
        # output_feats[0] from bottleneck, output_feats[1:] are bridge features
        sz_output_feats = [tuple(ft.size()) for ft in output_feats]
        sz_output_feats_expected: List[Tuple[int, ...]] = \
            [tuple([B, bottleneck_out_channels,
                    X // (2 ** stages),
                    Y // (2 ** stages),
                    Z // (2 ** stages)])] + \
            [tuple([B, decoder_out_channels[s],
                    X // (2 ** (stages - s - 1)),
                    Y // (2 ** (stages - s - 1)),
                    Z // (2 ** (stages - s - 1))]) for s in range(stages)]

        print(f"    Input for UNetRepeater:\n"
              f"      input_feats:")
        for idx in range(max(len(sz_input_feats), len(sz_input_feats_expected))):
            if idx < len(sz_input_feats_expected):
                sz_expected = sz_input_feats_expected[idx]
                if idx < len(sz_input_feats):
                    sz_out = sz_input_feats[idx]
                    print(f"        [{idx}] {sz_out}"
                          f" {'=' if sz_out == sz_expected else '≠'} {sz_expected} (expected)")
                else:
                    print(f"        [{idx}] None ≠ {sz_expected} (expected)")
            else:
                sz_out = sz_input_feats[idx]
                print(f"        [{idx}] {sz_out} ≠ None (expected)")
        print(f"    Output for UNetRepeater:")
        print(f"      output_feats:")
        for idx in range(max(len(sz_output_feats), len(sz_output_feats_expected))):
            if idx < len(sz_output_feats_expected):
                sz_expected = sz_output_feats_expected[idx]
                if idx < len(sz_output_feats):
                    sz_out = sz_output_feats[idx]
                    print(f"        [{idx}] {sz_out}"
                          f" {'=' if sz_out == sz_expected else '≠'} {sz_expected} (expected)")
                else:
                    print(f"        [{idx}] None ≠ {sz_expected} (expected)")
            else:
                sz_out = sz_output_feats[idx]
                print(f"        [{idx}] {sz_out} ≠ None (expected)")

        passed: bool = True
        passed = passed and (sz_input_feats == sz_input_feats_expected)
        if len(sz_output_feats) != len(sz_output_feats_expected):
            passed = False
        else:
            for idx in range(max(len(sz_output_feats), len(sz_output_feats_expected))):
                feat: Tuple[int, ...] = sz_output_feats[idx]
                expected: Tuple[int, ...] = sz_output_feats_expected[idx]
                passed = passed and (feat == expected)
        test_states.append(passed)
        print(f"[{seq}] {'PASS' if passed else 'FAILED'}\n")

        return output_feats


    # 00-05 UNet-Decoder
    def test_unet_decoder_io(
            seq: int,
            module: UNetDecoder,
            input_feat: Tensor,
            bridge_feats: Sequence[Tensor]
    ) -> List[Tensor]:
        print(f"[{seq}] Test 00-05 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_feat: Tuple[int, ...] = tuple(input_feat.size())
        sz_input_feat_expected: Tuple[int, ...] = tuple([B, decoder_in_channels[0],
                                                         X // (2 ** stages),
                                                         Y // (2 ** stages),
                                                         Z // (2 ** stages)])
        sz_bridge_feats: List[Tuple[int, ...]] = [tuple(ft.size()) for ft in bridge_feats]
        sz_bridge_feats_expected: List[Tuple[int, ...]] = [
            tuple([B, decoder_bridge_channels[s],
                   X // (2 ** (stages - s - 1)),
                   Y // (2 ** (stages - s - 1)),
                   Z // (2 ** (stages - s - 1))]) for s in range(stages)]
        # Output
        output_feats: List[Tensor] = module(input_feat, bridge_feats)
        sz_output_feats = [tuple(ft.size()) for ft in output_feats]
        sz_output_feats_expected: List[Tuple[int, ...]] = [
            tuple([B, decoder_out_channels[s],
                   X // (2 ** (stages - s - 1)),
                   Y // (2 ** (stages - s - 1)),
                   Z // (2 ** (stages - s - 1))]) for s in range(stages)]

        print(f"    Input for UNetDecoder:\n"
              f"      input_feat: {sz_input_feat}"
              f" {'=' if sz_input_feat == sz_input_feat_expected else '≠'} {sz_input_feat_expected} (expected)")
        print(f"      bridge_feats:")
        for idx in range(max(len(sz_bridge_feats), len(sz_bridge_feats_expected))):
            if idx < len(sz_bridge_feats_expected):
                sz_expected = sz_bridge_feats_expected[idx]
                if idx < len(sz_bridge_feats):
                    sz_out = sz_bridge_feats[idx]
                    print(f"        [{idx}] {sz_out}"
                          f" {'=' if sz_out == sz_expected else '≠'} {sz_expected} (expected)")
                else:
                    print(f"        [{idx}] None ≠ {sz_expected} (expected)")
            else:
                sz_out = sz_bridge_feats[idx]
                print(f"        [{idx}] {sz_out} ≠ None (expected)")
        print(f"    Output for UNetDecoder:")
        print(f"      output_feats:")
        for idx in range(max(len(sz_output_feats), len(sz_output_feats_expected))):
            if idx < len(sz_output_feats_expected):
                sz_expected = sz_output_feats_expected[idx]
                if idx < len(sz_output_feats):
                    sz_out = sz_output_feats[idx]
                    print(f"        [{idx}] {sz_out}"
                          f" {'=' if sz_out == sz_expected else '≠'} {sz_expected} (expected)")
                else:
                    print(f"        [{idx}] None ≠ {sz_expected} (expected)")
            else:
                sz_out = sz_output_feats[idx]
                print(f"        [{idx}] {sz_out} ≠ None (expected)")

        passed: bool = True
        passed = passed and (sz_input_feat == sz_input_feat_expected)
        if len(sz_bridge_feats) != len(sz_bridge_feats_expected):
            passed = False
        else:
            for idx in range(max(len(sz_bridge_feats), len(sz_bridge_feats_expected))):
                feat: Tuple[int, ...] = sz_bridge_feats[idx]
                expected: Tuple[int, ...] = sz_bridge_feats_expected[idx]
                passed = passed and (feat == expected)
        if len(sz_output_feats) != len(sz_output_feats_expected):
            passed = False
        else:
            for idx in range(max(len(sz_output_feats), len(sz_output_feats_expected))):
                feat: Tuple[int, ...] = sz_output_feats[idx]
                expected: Tuple[int, ...] = sz_output_feats_expected[idx]
                passed = passed and (feat == expected)
        test_states.append(passed)
        print(f"[{seq}] {'PASS' if passed else 'FAILED'}\n")

        return output_feats


    # 00-06 UNet-Auxiliary-Classifier
    def test_unet_auxiliary_classifier_io(
            seq: int,
            module: UNetAuxiliaryClassifier,
            input_feats: Sequence[Tensor]
    ) -> List[Tensor]:
        print(f"[{seq}] Test 00-06 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_feats: List[Tuple[int, ...]] = [tuple(ft.size()) for ft in input_feats]
        sz_input_feats_expected: List[Tuple[int, ...]] = \
            [tuple([B, auxiliary_classifier_in_channels[s],
                    X // (2 ** (stages - s - 1)),
                    Y // (2 ** (stages - s - 1)),
                    Z // (2 ** (stages - s - 1))]) for s in range(stages)]
        # Output
        aux_logits: List[Tensor] = module(input_feats)
        sz_aux_logits = [tuple(ft.size()) for ft in aux_logits]
        sz_aux_logits_expected: List[Tuple[int, ...]] = \
            [tuple([B, auxiliary_classifier_out_channels[s],
                    X // (2 ** (stages - s - 1)),
                    Y // (2 ** (stages - s - 1)),
                    Z // (2 ** (stages - s - 1))]) for s in range(stages)]

        print(f"    Input for UNetAuxiliaryClassifier:\n"
              f"      input_feats:")
        for idx in range(max(len(sz_input_feats), len(sz_input_feats_expected))):
            if idx < len(sz_input_feats_expected):
                sz_expected = sz_input_feats_expected[idx]
                if idx < len(sz_input_feats):
                    sz_out = sz_input_feats[idx]
                    print(f"        [{idx}] {sz_out}"
                          f" {'=' if sz_out == sz_expected else '≠'} {sz_expected} (expected)")
                else:
                    print(f"        [{idx}] None ≠ {sz_expected} (expected)")
            else:
                sz_out = sz_input_feats[idx]
                print(f"        [{idx}] {sz_out} ≠ None (expected)")
        print(f"    Output for UNetAuxiliaryClassifier:")
        print(f"      aux_logits:")
        for idx in range(max(len(sz_aux_logits), len(sz_aux_logits_expected))):
            if idx < len(sz_aux_logits_expected):
                sz_expected = sz_aux_logits_expected[idx]
                if idx < len(sz_aux_logits):
                    sz_out = sz_aux_logits[idx]
                    print(f"        [{idx}] {sz_out}"
                          f" {'=' if sz_out == sz_expected else '≠'} {sz_expected} (expected)")
                else:
                    print(f"        [{idx}] None ≠ {sz_expected} (expected)")
            else:
                sz_out = sz_aux_logits[idx]
                print(f"        [{idx}] {sz_out} ≠ None (expected)")

        passed: bool = True
        if len(sz_input_feats) != len(sz_input_feats_expected):
            passed = False
        else:
            for idx in range(max(len(sz_input_feats), len(sz_input_feats_expected))):
                feat: Tuple[int, ...] = sz_input_feats[idx]
                expected: Tuple[int, ...] = sz_input_feats_expected[idx]
                passed = passed and (feat == expected)
        if len(sz_aux_logits) != len(sz_aux_logits_expected):
            passed = False
        else:
            for idx in range(max(len(sz_aux_logits), len(sz_aux_logits_expected))):
                logits: Tuple[int, ...] = sz_aux_logits[idx]
                expected: Tuple[int, ...] = sz_aux_logits_expected[idx]
                passed = passed and (logits == expected)
        test_states.append(passed)
        print(f"[{seq}] {'PASS' if passed else 'FAILED'}\n")

        return aux_logits


    # 00-07 UNet-Distributor
    def test_unet_distributor_io(seq: int, module: UNetDistributor, input_feat: Tensor) -> Tensor:
        print(f"[{seq}] Test 00-07 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_feat: Tuple[int, ...] = tuple(input_feat.size())
        sz_input_feat_expected: Tuple[int, ...] = tuple(tuple([B, distributor_in_channels, X, Y, Z]))
        # Output
        output_feat: Tensor = module(input_feat)
        sz_output_feat: Tuple[int, ...] = tuple(output_feat.size())
        sz_output_feat_expected: Tuple[int, ...] = tuple([B, distributor_out_channels, X, Y, Z])

        print(f"    Input for UNetDistributor:\n"
              f"      input_source: {sz_input_feat}"
              f" {'=' if sz_input_feat == sz_input_feat_expected else '≠'} {sz_input_feat_expected} (expected)")
        print(f"    Output for UNetDistributor:\n"
              f"      output_feat: {sz_output_feat}"
              f" {'=' if sz_output_feat == sz_output_feat_expected else '≠'} {sz_output_feat_expected} (expected)")

        passed: bool = True
        passed = passed and sz_input_feat == sz_input_feat_expected
        passed = passed and sz_output_feat == sz_output_feat_expected
        test_states.append(passed)
        print(f"[{seq}] {'PASS' if passed else 'FAILED'}\n")

        return output_feat


    # 00-08 UNet-Classifier
    def test_unet_classifier_io(seq: int, module: UNetClassifier, input_feat: Tensor) -> Tensor:
        print(f"[{seq}] Test 00-08 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_feat: Tuple[int, ...] = tuple(input_feat.size())
        sz_input_feat_expected: Tuple[int, ...] = tuple(tuple([B, classifier_in_channels, X, Y, Z]))
        # Output
        logits: Tensor = module(input_feat)
        sz_logits: Tuple[int, ...] = tuple(logits.size())
        sz_logits_expected: Tuple[int, ...] = tuple([B, classifier_out_channels, X, Y, Z])

        print(f"    Input for UNetClassifier:\n"
              f"      input_feat: {sz_input_feat}"
              f" {'=' if sz_input_feat == sz_input_feat_expected else '≠'} {sz_input_feat_expected} (expected)")
        print(f"    Output for UNetClassifier:\n"
              f"      logits: {sz_logits}"
              f" {'=' if sz_logits == sz_logits_expected else '≠'} {sz_logits_expected} (expected)")

        passed: bool = True
        passed = passed and sz_input_feat == sz_input_feat_expected
        passed = passed and sz_logits == sz_logits_expected
        test_states.append(passed)
        print(f"[{seq}] {'PASS' if passed else 'FAILED'}\n")

        return logits


    # Launch test
    test_seq_idx: int = 1
    unet_cls_logits: Tensor
    unet_aux_cls_logits: List[Tensor]
    unet_cls_logits, unet_aux_cls_logits = test_unet_io(test_seq_idx, unet, input_source)
    test_seq_idx += 1

    focus_output_feat: Tensor = test_unet_focuser_io(test_seq_idx, unet.focuser, input_source)
    test_seq_idx += 1

    encoder_output_feats: List[Tensor]
    encoder_dn_feats: List[Tensor]
    encoder_output_feats, encoder_dn_feats = test_unet_encoder_io(test_seq_idx, unet.encoder, focus_output_feat)
    test_seq_idx += 1

    repeater_input_feats: List[Tensor] = encoder_output_feats + [encoder_dn_feats[-1]]
    repeater_output_feats: List[Tensor]
    repeater_output_feats: List[Tensor] = test_unet_repeater_io(test_seq_idx, unet.repeater, repeater_input_feats)
    test_seq_idx += 1

    decoder_input_feats: Tensor = repeater_output_feats[0]
    decoder_bridge_feats: List[Tensor] = repeater_output_feats[1:]
    decoder_output_feats: List[Tensor] = test_unet_decoder_io(
        test_seq_idx, unet.decoder, decoder_input_feats, decoder_bridge_feats)
    test_seq_idx += 1

    auxiliary_classifier_logits: List[Tensor] = test_unet_auxiliary_classifier_io(
        test_seq_idx, unet.auxiliary_classifier, decoder_output_feats)
    test_seq_idx += 1

    distributor_output_feats: Tensor = test_unet_distributor_io(
        test_seq_idx, unet.distributor, decoder_output_feats[-1])
    test_seq_idx += 1

    classifier_logits: Tensor = test_unet_classifier_io(test_seq_idx, unet.classifier, distributor_output_feats)
    test_seq_idx += 1

    print(f"[Summary]"
          f" Overall: {'PASS' if all(test_states) else 'FAILED'}")
    for test_idx, state in enumerate(test_states):
        print(f"  [{test_idx}] {'PASS' if state else 'FAILED'}")
    # endregion

    # Module Overview
    print(f"[Module Overview]")
    print(unet)
    print()
    # endregion

    # IO Overview
    print(f"[IO Overview]")
    unet(input_source)
    for level in range(5):
        print("-" * 100)
        print(f"[MaxLevel={level}]")
        print(unet.io_description(level))
    # endregion
