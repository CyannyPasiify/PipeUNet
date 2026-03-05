# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Collection, Sequence, Union, Dict, Any, Type, Iterable, cast
from Network.block import IODescriptive, ConvNormAct, ConvBNReLU, Concat


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
#                                                              │                                              │ {shortcut_feats[S]} │              │
#                                                              │ ┌───────────────┐                            │                     │              │
#                                           {input_feats[S+1]} └─│ 00-03-00      │─ {output_feats[S+1]}       │                   ┌─│              │
#                                                                │ UNet-Repeater │       ↓                    │      {input_feat} │ │              │
#                                                                │               │  {{input_feat}, ───────────┼───────────────────┘ └──────────────┘
#                                                                └───────────────┘   {shortcut_feats[0:S]}} ──┘      (B,DACin[0],
#                                                                                                                     X/(2^(S)),
#                                                                                                                     Y/(2^(S)),
#                                                                                                                     Z/(2^(S)))

class UNet(nn.Module, IODescriptive):
    def __init__(
            self,
            focuser_in_channels: int = 1,  # FCin
            focuser_out_channels: int = 16,  # FCout
            encoder_primary_in_channels: Sequence[int] = (16, 32),  # EPCin[S1]
            encoder_primary_out_channels: Sequence[int] = (32, 64),  # EPCout[S1]
            encoder_primary_depth: Union[int, Sequence[int]] = 2,  # EPD/EPD[S1]
            encoder_advanced_in_channels: Sequence[int] = (64, 128),  # EACin[S2]
            encoder_advanced_out_channels: Sequence[int] = (128, 256),  # EACout[S2]
            encoder_advanced_depth: Union[int, Sequence[int]] = 2,  # EAD/EAD[S2]
            bottleneck_in_channels: int = 256,  # RBCin
            bottleneck_out_channels: int = 512,  # RBCout
            bottleneck_depth: int = 2,  # RBD
            decoder_advanced_in_channels: Sequence[int] = (512, 256),  # DACin[S2]
            decoder_advanced_upsample_channels: Sequence[int] = (256, 128),  # DAUC[S2]
            decoder_advanced_shortcut_channels: Sequence[int] = (256, 128),  # DASC[S2]
            decoder_advanced_out_channels: Sequence[int] = (256, 128),  # DACout[S2]
            decoder_advanced_depth: Union[int, Sequence[int]] = 2,  # DAD[S2]
            decoder_primary_in_channels: Sequence[int] = (128, 64),  # DPCin[S1]
            decoder_primary_upsample_channels: Sequence[int] = (64, 32),  # DPUC[S1]
            decoder_primary_shortcut_channels: Sequence[int] = (64, 32),  # DPSC[S1]
            decoder_primary_out_channels: Sequence[int] = (64, 32),  # DPCout[S1]
            decoder_primary_depth: Union[int, Sequence[int]] = 2,  # DPD[S1],
            auxiliary_classifier_in_channels: Sequence[int] = (256, 128, 64, 32),  # ACCin[S2+S1]
            auxiliary_classifier_out_channels: Sequence[int] = (2, 2, 2, 2),  # ACCout[S2+S1]
            distributor_in_channels: int = 32,  # DCin
            distributor_out_channels: int = 16,  # DCout
            classifier_in_channels: int = 16,  # CCin
            classifier_out_channels: int = 2,  # CCout
            reserve_io: bool = False
    ):
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
        assert len(encoder_primary_out_channels) == len(decoder_primary_shortcut_channels)
        assert len(encoder_advanced_out_channels) == len(decoder_advanced_shortcut_channels)
        # Encoder-Decoder channel match
        for oc, sc in zip(encoder_primary_out_channels, reversed(decoder_primary_shortcut_channels)):
            assert oc == sc
        for oc, sc in zip(encoder_advanced_out_channels, reversed(decoder_advanced_shortcut_channels)):
            assert oc == sc
        # Decoder layer count match
        assert len(decoder_advanced_in_channels) == len(decoder_advanced_upsample_channels)
        assert len(decoder_advanced_shortcut_channels) == len(decoder_advanced_out_channels)
        assert len(decoder_advanced_in_channels) == len(decoder_advanced_out_channels)
        if isinstance(decoder_advanced_depth, Sequence):
            assert len(decoder_advanced_depth) == len(decoder_advanced_in_channels)
        assert len(decoder_primary_in_channels) == len(decoder_primary_upsample_channels)
        assert len(decoder_primary_shortcut_channels) == len(decoder_primary_out_channels)
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
            decoder_advanced_shortcut_channels,
            decoder_advanced_out_channels,
            decoder_advanced_depth,
            decoder_primary_in_channels,
            decoder_primary_upsample_channels,
            decoder_primary_shortcut_channels,
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

    def forward(self, input_source: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        assert input_source.ndim == 5
        assert divmod(input_source.size(2), 2 ** (self.stages - 1))[1] == 0
        f_feat: torch.Tensor = self.focuser(input_source)
        enc_feats: List[torch.Tensor]
        dn_feats: List[torch.Tensor]
        enc_feats, dn_feats = self.encoder(f_feat)
        # enc_feats to skip, dn_feats[-1] to bottleneck
        rep_feats: List[torch.Tensor] = self.repeater(enc_feats + [dn_feats[-1]])
        # rep_feats[0] from bottleneck, rep_feats[1:] are shortcut features
        dec_feats: List[torch.Tensor] = self.decoder(rep_feats[0], rep_feats[1:])
        d_feat: torch.Tensor = self.distributor(dec_feats[-1])
        aux_cls_logits: List[torch.Tensor] = self.auxiliary_classifier(dec_feats)
        cls_logits: torch.Tensor = self.classifier(d_feat)

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
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            reserve_io: bool = False
    ):
        super(UNetFocuser, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_source: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_source)

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
    def __init__(
            self,
            reserve_io: bool = False
    ):
        super(UNetEncoderPriorBank, self).__init__()
        self.reserve_io: bool = reserve_io

    def forward(self, prior_source: Sequence[torch.Tensor]) -> List[torch.Tensor]:
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
    def __init__(
            self,
            reserve_io: bool = False
    ):
        super(UNetEncoderPriorBankInjector, self).__init__()
        self.reserve_io: bool = reserve_io

    def forward(self, inject_source: torch.Tensor) -> torch.Tensor:
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
    def __init__(
            self,
            primary_in_channels: Sequence[int],  # PCin[S1]
            primary_out_channels: Sequence[int],  # PCout[S1]
            primary_depth: Union[int, Sequence[int]],  # PD/PD[S1]
            advanced_in_channels: Sequence[int],  # ACin[S2]
            advanced_out_channels: Sequence[int],  # ACout[S2]
            advanced_depth: Union[int, Sequence[int]],  # AD/AD[S2]
            reserve_io: bool = False
    ):
        super(UNetEncoder, self).__init__()
        assert len(primary_in_channels) == len(primary_out_channels)
        assert len(advanced_in_channels) == len(advanced_out_channels)
        if isinstance(primary_depth, Sequence):
            assert len(primary_in_channels) == len(primary_depth)
        if isinstance(advanced_depth, Sequence):
            assert len(advanced_in_channels) == len(advanced_depth)
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
            input_feat: torch.Tensor,
            inject_feats: Optional[Sequence[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        assert (inject_feats is None or
                len(inject_feats) == self.primary_extractor.stages + self.advanced_extractor.stages)

        primary_feats: List[torch.Tensor]
        primary_dn_feats: List[torch.Tensor]
        advanced_feats: List[torch.Tensor]
        advanced_dn_feats: List[torch.Tensor]
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

        output_feats: List[torch.Tensor] = primary_feats + advanced_feats
        dn_feats: List[torch.Tensor] = primary_dn_feats + advanced_dn_feats
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
    def __init__(
            self,
            in_channels: Sequence[int],  # Cin[S]
            out_channels: Sequence[int],  # Cout[S]
            depth: Union[int, Sequence[int]],  # D/D[S]
            reserve_io: bool = False
    ):
        super(UNetEncoderPrimaryExtractor, self).__init__()
        assert len(in_channels) == len(out_channels)
        if isinstance(depth, Sequence):
            assert len(in_channels) == len(depth)
        else:
            depth: List[int] = [depth] * len(in_channels)
        self.stages: int = len(in_channels)
        self.pipe: nn.ModuleList = nn.ModuleList()

        for ic, oc, dp in zip(in_channels, out_channels, depth):
            self.pipe.append(nn.Sequential(
                UNetEncoderPrimaryExtractorStage(ic, oc, dp, reserve_io),
                UNetEncoderPrimaryExtractorDownsample(reserve_io)
            ))

        self.reserve_io: bool = reserve_io

    def forward(
            self,
            input_feat: torch.Tensor,
            inject_feats: Optional[Sequence[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        output_feats: List[torch.Tensor] = []
        dn_feats: List[torch.Tensor] = []
        if inject_feats is None:
            stage_feat: torch.Tensor = input_feat
            for module in self.pipe:
                stage_feat = module[0](stage_feat)  # Stage
                output_feats.append(stage_feat)  # Record features for all stages
                stage_feat = module[1](stage_feat)  # Downsample
                dn_feats.append(stage_feat)  # Record features after downsample
        else:
            assert len(inject_feats) == self.stages
            stage_feat: torch.Tensor = input_feat
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
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            depth: int,  # D
            reserve_io: bool = False
    ):
        super(UNetEncoderPrimaryExtractorStage, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            *[ConvBNReLU(in_channels, in_channels, 3, padding='same', reserve_io=reserve_io)
              for _ in range(depth - 1)],
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor, inject_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inject_feat is always ignored now
        output_feat: torch.Tensor = self.pipe(input_feat)

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
    def __init__(
            self,
            reserve_io: bool = False
    ):
        super(UNetEncoderPrimaryExtractorDownsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.MaxPool3d(2, 2)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_feat)

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
    def __init__(
            self,
            in_channels: Sequence[int],  # Cin[S]
            out_channels: Sequence[int],  # Cout[S]
            depth: Union[int, Sequence[int]],  # D/D[S]
            reserve_io: bool = False
    ):
        super(UNetEncoderAdvancedExtractor, self).__init__()
        assert len(in_channels) == len(out_channels)
        if isinstance(depth, Sequence):
            assert len(in_channels) == len(depth)
        else:
            depth: List[int] = [depth] * len(in_channels)
        self.stages: int = len(in_channels)
        self.pipe: nn.ModuleList = nn.ModuleList()

        for ic, oc, dp in zip(in_channels, out_channels, depth):
            self.pipe.append(nn.Sequential(
                UNetEncoderAdvancedExtractorStage(ic, oc, dp, reserve_io),
                UNetEncoderAdvancedExtractorDownsample(reserve_io)
            ))

        self.reserve_io: bool = reserve_io

    def forward(
            self,
            input_feat: torch.Tensor,
            inject_feats: Optional[Sequence[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        output_feats: List[torch.Tensor] = []
        dn_feats: List[torch.Tensor] = []
        if inject_feats is None:
            stage_feat: torch.Tensor = input_feat
            for module in self.pipe:
                stage_feat = module[0](stage_feat)  # Stage
                output_feats.append(stage_feat)  # Record features for all stages
                stage_feat = module[1](stage_feat)  # Downsample
                dn_feats.append(stage_feat)  # Record features after downsample
        else:
            assert len(inject_feats) == self.stages
            stage_feat: torch.Tensor = input_feat
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
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            depth: int,  # D
            reserve_io: bool = False
    ):
        super(UNetEncoderAdvancedExtractorStage, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            *[ConvBNReLU(in_channels, in_channels, 3, padding='same', reserve_io=reserve_io)
              for _ in range(depth - 1)],
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor, inject_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inject_feat is always ignored now
        output_feat: torch.Tensor = self.pipe(input_feat)

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
    def __init__(
            self,
            reserve_io: bool = False
    ):
        super(UNetEncoderAdvancedExtractorDownsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.MaxPool3d(2, 2)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_feat)

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
#                   │ -Shortcut      │
#                   └────────────────┘
#                   ⋮       ⋮        ⋮
#                   ┌────────────────┐
#   input_feats[s] ─│ 00-03-00 [s]   │─ output_feats[S-s-1]
#              (*)  │ UNet-Repeater  │  (*)
#                   │ -Shortcut      │
#                   └────────────────┘
#                   ⋮       ⋮        ⋮
#                   ┌────────────────┐
# input_feats[S-2] ─│ 00-03-00 [S-2] │─ output_feats[1]
#              (*)  │ UNet-Repeater  │  (*)
#                   │ -Shortcut      │
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
    def __init__(
            self,
            stages: int,  # S
            bottleneck_in_channels: int,  # Cin
            bottleneck_out_channels: int,  # Cout
            bottleneck_depth: int,  # D
            reserve_io: bool = False
    ):
        super(UNetRepeater, self).__init__()
        self.pipe: nn.ModuleList = nn.ModuleList(
            [UNetRepeaterShortcut(reserve_io) for _ in range(stages - 1)]
        )
        self.pipe.append(
            UNetRepeaterBottleneck(bottleneck_in_channels, bottleneck_out_channels, bottleneck_depth, reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        output_feats: List[torch.Tensor] = []
        module: nn.Module
        feat: torch.Tensor
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


# 00-03-00 UNet-Repeater-Shortcut
# The feature delivery path from Encoder to Decoder stage by stage, i.e. skip connection
# Pinout Diagram: [Valid]
#                ┌───────────────┐
#    input_feat ─│ 00-03-00      │─ output_feat
#   (B,C,X,Y,Z)  │ UNet-Repeater │  (B,C,X,Y,Z)
#                │ Shortcut      │
#                └───────────────┘
class UNetRepeaterShortcut(nn.Module, IODescriptive):
    def __init__(
            self,
            reserve_io: bool = False
    ):
        super(UNetRepeaterShortcut, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.Identity()
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_feat)

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
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            depth: int,  # D
            reserve_io: bool = False
    ):
        super(UNetRepeaterBottleneck, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            *[ConvBNReLU(in_channels, in_channels, 3, padding='same', reserve_io=reserve_io)
              for _ in range(depth - 1)],
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor, inject_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inject_feat is always ignored now
        output_feat: torch.Tensor = self.pipe(input_feat)

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
    def __init__(
            self,
            reserve_io: bool = False
    ):
        super(UNetDecoderPriorBank, self).__init__()
        self.reserve_io: bool = reserve_io

    def forward(self, prior_source: Sequence[torch.Tensor]) -> List[torch.Tensor]:
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
    def __init__(
            self,
            reserve_io: bool = False
    ):
        super(UNetDecoderPriorBankInjector, self).__init__()
        self.reserve_io: bool = reserve_io

    def forward(self, inject_source: torch.Tensor) -> torch.Tensor:
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
#                                               shortcut_feats[S1+S2] ─│              │  [1]:   (B,ACout[1],   X*4,          Y*4,          Z*4          )
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
#                                                shortcut_feats[0:S1] ─│                    │  [1]:   (B,ACout[1],   X*4,     Y*4,     Z*4    )
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
#                                                                              shortcut_feats[S1:S1+S2] ─│ -PrimaryAggregator │  [0]:   (B,PCout[0],   X*(2^(S1+1)), Y*(2^(S1+1)), Z*(2^(S1+1)) )
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
    def __init__(
            self,
            advanced_in_channels: Sequence[int],  # ACin[S1]
            advanced_upsample_channels: Sequence[int],  # AUC[S1]
            advanced_shortcut_channels: Sequence[int],  # ASC[S1]
            advanced_out_channels: Sequence[int],  # ACout[S1]
            advanced_depth: Union[int, Sequence[int]],  # AD[S1]
            primary_in_channels: Sequence[int],  # PCin[S2]
            primary_upsample_channels: Sequence[int],  # PUC[S2]
            primary_shortcut_channels: Sequence[int],  # PSC[S2]
            primary_out_channels: Sequence[int],  # PCout[S2]
            primary_depth: Union[int, Sequence[int]],  # PD[S2]
            reserve_io: bool = False
    ):
        super(UNetDecoder, self).__init__()
        assert len(advanced_in_channels) == len(advanced_upsample_channels)
        assert len(advanced_shortcut_channels) == len(advanced_out_channels)
        assert len(advanced_in_channels) == len(advanced_out_channels)
        if isinstance(advanced_depth, Sequence):
            assert len(advanced_in_channels) == len(advanced_depth)

        assert len(primary_in_channels) == len(primary_upsample_channels)
        assert len(primary_shortcut_channels) == len(primary_out_channels)
        assert len(primary_in_channels) == len(primary_out_channels)
        if isinstance(primary_depth, Sequence):
            assert len(primary_in_channels) == len(primary_depth)

        self.advanced_aggregator: UNetDecoderAdvancedAggregator = UNetDecoderAdvancedAggregator(
            advanced_in_channels,
            advanced_upsample_channels,
            advanced_shortcut_channels,
            advanced_out_channels,
            advanced_depth,
            reserve_io
        )
        self.primary_aggregator: UNetDecoderPrimaryAggregator = UNetDecoderPrimaryAggregator(
            primary_in_channels,
            primary_upsample_channels,
            primary_shortcut_channels,
            primary_out_channels,
            primary_depth,
            reserve_io
        )

        self.reserve_io: bool = reserve_io

    def forward(
            self,
            input_feat: torch.Tensor,
            shortcut_feats: Sequence[torch.Tensor],
            inject_feats: Optional[Sequence[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        assert len(shortcut_feats) == self.advanced_aggregator.stages + self.primary_aggregator.stages
        assert (inject_feats is None or
                len(inject_feats) == self.advanced_aggregator.stages + self.primary_aggregator.stages)
        if inject_feats is None:
            advanced_feats: List[torch.Tensor] = self.advanced_aggregator(
                input_feat,
                shortcut_feats[:self.primary_aggregator.stages]
            )
            primary_feats: List[torch.Tensor] = self.primary_aggregator(
                advanced_feats[self.advanced_aggregator.stages - 1],
                shortcut_feats[self.primary_aggregator.stages:]
            )
        else:
            advanced_feats: List[torch.Tensor] = self.advanced_aggregator(
                input_feat,
                shortcut_feats[:self.primary_aggregator.stages],
                inject_feats[:self.primary_aggregator.stages]
            )
            primary_feats: List[torch.Tensor] = self.primary_aggregator(
                advanced_feats[self.advanced_aggregator.stages - 1],
                shortcut_feats[self.primary_aggregator.stages:],
                inject_feats[self.primary_aggregator.stages:]
            )

        output_feats: List[torch.Tensor] = advanced_feats + primary_feats

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'shortcut_feats', [ft.cpu() for ft in shortcut_feats])
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
                        and hasattr(self, 'shortcut_feats')
                        and hasattr(self, 'output_feats')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}    shortcut_feats: {[tuple(ft.size()) for ft in self.shortcut_feats]}\n")
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
#                           shortcut_feats[S] ─│ Aggregator   │  ⋮
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
#                                     │  └───────────────────────┘                              {shortcut_feat} ─│ -FusionPortal      │                                      {inject_feat} ─│ :depth=D/D[0]         │  │
#                                     │                                                   (B,SC[0],X*2,Y*2,Z*2)  │                    │                                    inject_feats[0]  │                       │  │
#                                     │                                                                          └────────────────────┘                                           (Unused)  └───────────────────────┘  │
#                                     └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
#                                     ⋮                                                                                       ⋮                                                                                        ⋮
#                                     ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#                  output_feats[s-1]  │  ┌───────────────────────┐                                               ┌────────────────────┐                                                     ┌───────────────────────┐  │  output_feats[s]
#                     →{output_feat} ─┼──│ 00-05-00-00           │───────────────────────────────────────────────│ 00-05-00-01        │─────────────────────────────────────────────────────│ 00-05-00-02           │──┼─ {output_feat}→
# (B,Cin[s],X*(2^s),Y*(2^s),Z*(2^s))  │  │ UNet-Decoder-Advanced │ {output_feat}                    {input_feat} │ UNet-Decoder-      │ {output_feat}                          {input_feat} │ UNet-Decoder-Advanced │  │  (B,Cout[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))
#                                     │  │ Aggregator-Upsample   │ (B,UC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1))) │ AdvancedAggregator │ (B,UC[s]+SC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1))) │ Aggregator-Stage      │  │
#                                     │  └───────────────────────┘                              {shortcut_feat} ─│ -FusionPortal      │                                      {inject_feat} ─│ :depth=D/D[s]         │  │
#                                     │                           (B,SC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))  │                    │                                    inject_feats[s]  │                       │  │
#                                     │                                                                          └────────────────────┘                                           (Unused)  └───────────────────────┘  │
#                                     └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
class UNetDecoderAdvancedAggregator(nn.Module, IODescriptive):
    def __init__(
            self,
            in_channels: Sequence[int],  # Cin[S]
            upsample_channels: Sequence[int],  # UC[S]
            shortcut_channels: Sequence[int],  # SC[S]
            out_channels: Sequence[int],  # Cout[S]
            depth: Union[int, Sequence[int]],  # D/D[S]
            reserve_io: bool = False
    ):
        super(UNetDecoderAdvancedAggregator, self).__init__()
        assert len(in_channels) == len(upsample_channels)
        assert len(shortcut_channels) == len(out_channels)
        assert len(in_channels) == len(out_channels)
        if isinstance(depth, Sequence):
            assert len(in_channels) == len(depth)
        else:
            depth: List[int] = [depth] * len(in_channels)
        self.stages: int = len(in_channels)
        self.pipe: nn.Sequential = nn.Sequential()

        for ic, uc, sc, oc, dp in zip(in_channels, upsample_channels, shortcut_channels, out_channels, depth):
            self.pipe.append(nn.Sequential(
                UNetDecoderAdvancedAggregatorUpsample(ic, uc, reserve_io),
                UNetDecoderAdvancedAggregatorFusionPortal(reserve_io),
                UNetDecoderAdvancedAggregatorStage(uc + sc, oc, dp, reserve_io)
            ))

        self.reserve_io: bool = reserve_io

    def forward(
            self,
            input_feat: torch.Tensor,
            shortcut_feats: Sequence[torch.Tensor],
            inject_feats: Optional[Sequence[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        assert len(shortcut_feats) == self.stages
        output_feats: List[torch.Tensor] = []
        stage_feat: torch.Tensor = input_feat
        if inject_feats is None:
            for module, shortcut_feat in zip(self.pipe, shortcut_feats):
                stage_feat = module[0](stage_feat)  # Upsample
                stage_feat = module[1](stage_feat, shortcut_feat)  # FusionPortal
                stage_feat = module[2](stage_feat)  # Stage
                output_feats.append(stage_feat)
        else:
            assert len(inject_feats) == self.stages
            for module, shortcut_feat, inject_feat in zip(self.pipe, shortcut_feats, inject_feats):
                stage_feat = module[0](stage_feat)  # Upsample
                stage_feat = module[1](stage_feat, shortcut_feat)  # FusionPortal
                stage_feat = module[2](stage_feat, inject_feat)  # Stage
                output_feats.append(stage_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'shortcut_feats', [ft.cpu() for ft in shortcut_feats])
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
                        and hasattr(self, 'shortcut_feats')
                        and hasattr(self, 'output_feats')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}    shortcut_feats: {[tuple(ft.size()) for ft in self.shortcut_feats]}\n")
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
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            reserve_io: bool = False
    ):
        super(UNetDecoderAdvancedAggregatorUpsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_feat)

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
# Fuse features from higher semantic layer of Decoder and relative shortcut layer (from Encoder)
# Pinout Diagram: [Valid]
#                ┌────────────────────┐
#    stage_feat ─│ 00-05-00-01        │─ output_feat
#      (B,C1,*)  │ UNet-Decoder-      │  (B,C1+C2,*)
# shortcut_feat ─│ AdvancedAggregator │
#      (B,C2,*)  │ -FusionPortal      │
#                └────────────────────┘
class UNetDecoderAdvancedAggregatorFusionPortal(nn.Module, IODescriptive):
    def __init__(
            self,
            reserve_io: bool = False
    ):
        super(UNetDecoderAdvancedAggregatorFusionPortal, self).__init__()
        self.concat: Concat = Concat(dim=1, reserve_io=reserve_io)

        self.reserve_io: bool = reserve_io

    def forward(self, stage_feat: torch.Tensor, shortcut_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.concat(stage_feat, shortcut_feat)

        if self.reserve_io:
            setattr(self, 'stage_feat', stage_feat.cpu())
            setattr(self, 'shortcut_feat', shortcut_feat.cpu())
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
                        and hasattr(self, 'shortcut_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    stage_feat: {tuple(self.stage_feat.size())}\n"
                     f"{prefix}    shortcut_feat: {tuple(self.shortcut_feat.size())}\n"
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
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            depth: int,  # D
            reserve_io: bool = False
    ):
        super(UNetDecoderAdvancedAggregatorStage, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io),
            *[ConvBNReLU(out_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
              for _ in range(depth - 1)],
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor, inject_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inject_feat is always ignored now
        output_feat: torch.Tensor = self.pipe(input_feat)

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
#                           shortcut_feats[S] ─│ Aggregator   │  ⋮
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
#                                     │  └───────────────────────┘                              {shortcut_feat} ─│ -FusionPortal      │                                      {inject_feat} ─│ :depth=D/D[0]         │  │
#                                     │                                                   (B,SC[0],X*2,Y*2,Z*2)  │                    │                                    inject_feats[0]  │                       │  │
#                                     │                                                                          └────────────────────┘                                           (Unused)  └───────────────────────┘  │
#                                     └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
#                                     ⋮                                                                                       ⋮                                                                                        ⋮
#                                     ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#                  output_feats[s-1]  │  ┌───────────────────────┐                                               ┌────────────────────┐                                                     ┌───────────────────────┐  │  output_feats[s]
#                     →{output_feat} ─┼──│ 00-05-01-00           │───────────────────────────────────────────────│ 00-05-01-01        │─────────────────────────────────────────────────────│ 00-05-01-02           │──┼─ {output_feat}→
# (B,Cin[s],X*(2^s),Y*(2^s),Z*(2^s))  │  │ UNet-Decoder-Primary  │ {output_feat}                    {input_feat} │ UNet-Decoder-      │ {output_feat}                          {input_feat} │ UNet-Decoder-Primary  │  │  (B,Cout[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))
#                                     │  │ Aggregator-Upsample   │ (B,UC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1))) │ PrimaryAggregator  │ (B,UC[s]+SC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1))) │ Aggregator-Stage      │  │
#                                     │  └───────────────────────┘                              {shortcut_feat} ─│ -FusionPortal      │                                      {inject_feat} ─│ :depth=D/D[s]         │  │
#                                     │                           (B,SC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))  │                    │                                    inject_feats[s]  │                       │  │
#                                     │                                                                          └────────────────────┘                                           (Unused)  └───────────────────────┘  │
#                                     └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
class UNetDecoderPrimaryAggregator(nn.Module, IODescriptive):
    def __init__(
            self,
            in_channels: Sequence[int],  # Cin[S]
            upsample_channels: Sequence[int],  # UC[S]
            shortcut_channels: Sequence[int],  # SC[S]
            out_channels: Sequence[int],  # Cout[S]
            depth: Union[int, Sequence[int]],  # D/D[S]
            reserve_io: bool = False
    ):
        super(UNetDecoderPrimaryAggregator, self).__init__()
        assert len(in_channels) == len(upsample_channels)
        assert len(shortcut_channels) == len(out_channels)
        assert len(in_channels) == len(out_channels)
        if isinstance(depth, Sequence):
            assert len(in_channels) == len(depth)
        else:
            depth: List[int] = [depth] * len(in_channels)
        self.stages: int = len(in_channels)
        self.pipe: nn.Sequential = nn.Sequential()

        for ic, uc, sc, oc, dp in zip(in_channels, upsample_channels, shortcut_channels, out_channels, depth):
            self.pipe.append(nn.Sequential(
                UNetDecoderPrimaryAggregatorUpsample(ic, uc, reserve_io),
                UNetDecoderPrimaryAggregatorFusionPortal(reserve_io),
                UNetDecoderPrimaryAggregatorStage(uc + sc, oc, dp, reserve_io)
            ))

        self.reserve_io: bool = reserve_io

    def forward(
            self,
            input_feat: torch.Tensor,  # (B,Cin[0],X,Y,Z)
            shortcut_feats: Sequence[torch.Tensor],  # [s]:(B,SC[s],X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))
            inject_feats: Optional[Sequence[torch.Tensor]] = None  # [s]:(B,*,X*(2^(s+1)),Y*(2^(s+1)),Z*(2^(s+1)))
    ) -> List[torch.Tensor]:
        assert len(shortcut_feats) == self.stages
        output_feats: List[torch.Tensor] = []
        stage_feat: torch.Tensor = input_feat
        if inject_feats is None:
            for module, shortcut_feat in zip(self.pipe, shortcut_feats):
                stage_feat = module[0](stage_feat)  # Upsample
                stage_feat = module[1](stage_feat, shortcut_feat)  # FusionPortal
                stage_feat = module[2](stage_feat)  # Stage
                output_feats.append(stage_feat)
        else:
            assert len(inject_feats) == self.stages
            for module, shortcut_feat, inject_feat in zip(self.pipe, shortcut_feats, inject_feats):
                stage_feat = module[0](stage_feat)  # Upsample
                stage_feat = module[1](stage_feat, shortcut_feat)  # FusionPortal
                stage_feat = module[2](stage_feat, inject_feat)  # Stage
                output_feats.append(stage_feat)

        if self.reserve_io:
            setattr(self, 'input_feat', input_feat.cpu())
            setattr(self, 'shortcut_feats', [ft.cpu() for ft in shortcut_feats])
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
                        and hasattr(self, 'shortcut_feats')
                        and hasattr(self, 'output_feats')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    input_feat: {tuple(self.input_feat.size())}\n"
                     f"{prefix}    shortcut_feats: {[tuple(ft.size()) for ft in self.shortcut_feats]}\n")
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
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            reserve_io: bool = False
    ):
        super(UNetDecoderPrimaryAggregatorUpsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_feat)

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
# Fuse features from higher semantic layer of Decoder and relative shortcut layer (from Encoder)
# Pinout Diagram: [Valid]
#                ┌───────────────────┐
#    stage_feat ─│ 00-05-01-01       │─ output_feat
#      (B,C1,*)  │ UNet-Decoder-     │  (B,C1+C2,*)
# shortcut_feat ─│ PrimaryAggregator │
#      (B,C2,*)  │ -FusionPortal     │
#                └───────────────────┘
class UNetDecoderPrimaryAggregatorFusionPortal(nn.Module, IODescriptive):
    def __init__(
            self,
            reserve_io: bool = False
    ):
        super(UNetDecoderPrimaryAggregatorFusionPortal, self).__init__()
        self.concat: Concat = Concat(dim=1, reserve_io=reserve_io)

        self.reserve_io: bool = reserve_io

    def forward(self, stage_feat: torch.Tensor, shortcut_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.concat(stage_feat, shortcut_feat)

        if self.reserve_io:
            setattr(self, 'stage_feat', stage_feat.cpu())
            setattr(self, 'shortcut_feat', shortcut_feat.cpu())
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
                        and hasattr(self, 'shortcut_feat')
                        and hasattr(self, 'output_feat')
                )
        ):
            return ''
        prefix: str = indent_placeholder * indent
        desc: str = (f"{prefix}{self.__class__.__name__}\n"
                     f"{prefix}  I: \n"
                     f"{prefix}    stage_feat: {tuple(self.stage_feat.size())}\n"
                     f"{prefix}    shortcut_feat: {tuple(self.shortcut_feat.size())}\n"
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
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            depth: int,  # D
            reserve_io: bool = False
    ):
        super(UNetDecoderPrimaryAggregatorStage, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io),
            *[ConvBNReLU(out_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
              for _ in range(depth - 1)]
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor, inject_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inject_feat is always ignored now
        output_feat: torch.Tensor = self.pipe(input_feat)

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
    def __init__(
            self,
            in_channels: Sequence[int],  # Cin[S]
            out_channels: Sequence[int],  # Cout[S]
            reserve_io: bool = False
    ):
        super(UNetAuxiliaryClassifier, self).__init__()
        self.pipe: nn.ModuleList = nn.ModuleList()
        for ic, oc in zip(in_channels, out_channels):
            self.pipe.append(nn.Sequential(
                ConvBNReLU(ic, ic, 3, padding='same', reserve_io=reserve_io),
                nn.Conv3d(ic, oc, 1)
            ))

        self.reserve_io: bool = reserve_io

    def forward(self, input_feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        assert len(input_feats) == len(self.pipe)
        aux_logits: List[torch.Tensor] = []
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
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            reserve_io: bool = False
    ):
        super(UNetDistributor, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, padding='same', reserve_io=reserve_io)
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_feat)

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
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            reserve_io: bool = False
    ):
        super(UNetClassifier, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1)
            # nn.Sigmoid()  # Only return logits, apply normalization latter
        )

        self.reserve_io: bool = reserve_io

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.pipe(input_feat)

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
    input_source: torch.Tensor = torch.randn(3, 1, 64, 128, 256)
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
    decoder_advanced_shortcut_channels: Sequence[int] = (256, 128)
    decoder_advanced_out_channels: Sequence[int] = (256, 128)
    decoder_advanced_depth: Union[int, Sequence[int]] = 2
    decoder_primary_in_channels: Sequence[int] = (128, 64)
    decoder_primary_upsample_channels: Sequence[int] = (64, 32)
    decoder_primary_shortcut_channels: Sequence[int] = (64, 32)
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
    decoder_shortcut_channels: List[int] = list(decoder_advanced_shortcut_channels) + \
                                           list(decoder_primary_shortcut_channels)
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
        decoder_advanced_shortcut_channels=decoder_advanced_shortcut_channels,
        decoder_advanced_out_channels=decoder_advanced_out_channels,
        decoder_advanced_depth=decoder_advanced_depth,
        decoder_primary_in_channels=decoder_primary_in_channels,
        decoder_primary_upsample_channels=decoder_primary_upsample_channels,
        decoder_primary_shortcut_channels=decoder_primary_shortcut_channels,
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
    def test_unet_io(seq: int, module: UNet, input_source: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        print(f"[{seq}] Test 00 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_source: Tuple[int, ...] = tuple(input_source.size())
        sz_input_source_expected: Tuple[int, ...] = tuple([B, C, X, Y, Z])
        # Output
        cls_logits: torch.Tensor
        aux_cls_logits: List[torch.Tensor]
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
    def test_unet_focuser_io(seq: int, module: UNetFocuser, input_source: torch.Tensor) -> torch.Tensor:
        print(f"[{seq}] Test 00-00 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_source: Tuple[int, ...] = tuple(input_source.size())
        sz_input_source_expected: Tuple[int, ...] = tuple(tuple([B, C, X, Y, Z]))
        # Output
        output_feat: torch.Tensor = module(input_source)
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
            input_feat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        print(f"[{seq}] Test 00-02 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_feat: Tuple[int, ...] = tuple(input_feat.size())
        sz_input_feat_expected: Tuple[int, ...] = tuple([B, encoder_in_channels[0], X, Y, Z])
        # Output
        output_feats: List[torch.Tensor]
        dn_feats: List[torch.Tensor]
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
            input_feats: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:
        print(f"[{seq}] Test 00-03 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_feats: List[Tuple[int, ...]] = [tuple(ft.size()) for ft in input_feats]
        sz_input_feats_expected: List[Tuple[int, ...]] = \
            [tuple([B, encoder_out_channels[s], X // (2 ** s), Y // (2 ** s), Z // (2 ** s)]) for s in range(stages)] + \
            [tuple([B, bottleneck_in_channels, X // (2 ** stages), Y // (2 ** stages), Z // (2 ** stages)])]
        # Output
        output_feats: List[torch.Tensor] = module(input_feats)
        # output_feats[0] from bottleneck, output_feats[1:] are shortcut features
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
            input_feat: torch.Tensor,
            shortcut_feats: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:
        print(f"[{seq}] Test 00-05 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_feat: Tuple[int, ...] = tuple(input_feat.size())
        sz_input_feat_expected: Tuple[int, ...] = tuple([B, decoder_in_channels[0],
                                                         X // (2 ** stages),
                                                         Y // (2 ** stages),
                                                         Z // (2 ** stages)])
        sz_shortcut_feats: List[Tuple[int, ...]] = [tuple(ft.size()) for ft in shortcut_feats]
        sz_shortcut_feats_expected: List[Tuple[int, ...]] = [
            tuple([B, decoder_shortcut_channels[s],
                   X // (2 ** (stages - s - 1)),
                   Y // (2 ** (stages - s - 1)),
                   Z // (2 ** (stages - s - 1))]) for s in range(stages)]
        # Output
        output_feats: List[torch.Tensor] = module(input_feat, shortcut_feats)
        sz_output_feats = [tuple(ft.size()) for ft in output_feats]
        sz_output_feats_expected: List[Tuple[int, ...]] = [
            tuple([B, decoder_out_channels[s],
                   X // (2 ** (stages - s - 1)),
                   Y // (2 ** (stages - s - 1)),
                   Z // (2 ** (stages - s - 1))]) for s in range(stages)]

        print(f"    Input for UNetDecoder:\n"
              f"      input_feat: {sz_input_feat}"
              f" {'=' if sz_input_feat == sz_input_feat_expected else '≠'} {sz_input_feat_expected} (expected)")
        print(f"      shortcut_feats:")
        for idx in range(max(len(sz_shortcut_feats), len(sz_shortcut_feats_expected))):
            if idx < len(sz_shortcut_feats_expected):
                sz_expected = sz_shortcut_feats_expected[idx]
                if idx < len(sz_shortcut_feats):
                    sz_out = sz_shortcut_feats[idx]
                    print(f"        [{idx}] {sz_out}"
                          f" {'=' if sz_out == sz_expected else '≠'} {sz_expected} (expected)")
                else:
                    print(f"        [{idx}] None ≠ {sz_expected} (expected)")
            else:
                sz_out = sz_shortcut_feats[idx]
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
        if len(sz_shortcut_feats) != len(sz_shortcut_feats_expected):
            passed = False
        else:
            for idx in range(max(len(sz_shortcut_feats), len(sz_shortcut_feats_expected))):
                feat: Tuple[int, ...] = sz_shortcut_feats[idx]
                expected: Tuple[int, ...] = sz_shortcut_feats_expected[idx]
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
            input_feats: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:
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
        aux_logits: List[torch.Tensor] = module(input_feats)
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
    def test_unet_distributor_io(seq: int, module: UNetDistributor, input_feat: torch.Tensor) -> torch.Tensor:
        print(f"[{seq}] Test 00-07 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_feat: Tuple[int, ...] = tuple(input_feat.size())
        sz_input_feat_expected: Tuple[int, ...] = tuple(tuple([B, distributor_in_channels, X, Y, Z]))
        # Output
        output_feat: torch.Tensor = module(input_feat)
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
    def test_unet_classifier_io(seq: int, module: UNetClassifier, input_feat: torch.Tensor) -> torch.Tensor:
        print(f"[{seq}] Test 00-08 {module.__class__.__name__} IO")
        print("-" * 100)
        # Input
        sz_input_feat: Tuple[int, ...] = tuple(input_feat.size())
        sz_input_feat_expected: Tuple[int, ...] = tuple(tuple([B, classifier_in_channels, X, Y, Z]))
        # Output
        logits: torch.Tensor = module(input_feat)
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
    unet_cls_logits: torch.Tensor
    unet_aux_cls_logits: List[torch.Tensor]
    unet_cls_logits, unet_aux_cls_logits = test_unet_io(test_seq_idx, unet, input_source)
    test_seq_idx += 1

    focus_output_feat: torch.Tensor = test_unet_focuser_io(test_seq_idx, unet.focuser, input_source)
    test_seq_idx += 1

    encoder_output_feats: List[torch.Tensor]
    encoder_dn_feats: List[torch.Tensor]
    encoder_output_feats, encoder_dn_feats = test_unet_encoder_io(test_seq_idx, unet.encoder, focus_output_feat)
    test_seq_idx += 1

    repeater_input_feats: List[torch.Tensor] = encoder_output_feats + [encoder_dn_feats[-1]]
    repeater_output_feats: List[torch.Tensor]
    repeater_output_feats: List[torch.Tensor] = test_unet_repeater_io(test_seq_idx, unet.repeater, repeater_input_feats)
    test_seq_idx += 1

    decoder_input_feats: torch.Tensor = repeater_output_feats[0]
    decoder_shortcut_feats: List[torch.Tensor] = repeater_output_feats[1:]
    decoder_output_feats: List[torch.Tensor] = test_unet_decoder_io(
        test_seq_idx, unet.decoder, decoder_input_feats, decoder_shortcut_feats)
    test_seq_idx += 1

    auxiliary_classifier_logits: List[torch.Tensor] = test_unet_auxiliary_classifier_io(
        test_seq_idx, unet.auxiliary_classifier, decoder_output_feats)
    test_seq_idx += 1

    distributor_output_feats: torch.Tensor = test_unet_distributor_io(
        test_seq_idx, unet.distributor, decoder_output_feats[-1])
    test_seq_idx += 1

    classifier_logits: torch.Tensor = test_unet_classifier_io(test_seq_idx, unet.classifier, distributor_output_feats)
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
