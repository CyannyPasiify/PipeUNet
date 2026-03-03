# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any, Type, Iterable
from Network.block import ConvNormAct, ConvBNReLU, Concat


# TODO last
# 00 UNet
# UNet is an Encoder-Decoder framework for neural network building, which features multiscale representation ability and is suitable for segmentation task
class UNet(nn.Module):
    def __init__(
            self,
            normalized_shape: Union[int, List[int], Tuple[int, ...]],
            eps: float = 1e-6,
            elementwise_affine: bool = True
    ):
        super(UNet, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# 00-00 UNet-Focuser
# [Optional]
# Focuser is a Stem Feature Extractor before Encoder, which is intended to generate or enhance key region
# [Forbidden] Stem Feature shall not be used to formulate Feature Pyramid, if you wish to do so, move the pipe to Encoder
# Pinout Diagram: [Valid]
#               ┌──────────────┐
# input_source ─│ 00-00        │─ output_feat
# (B,Cin,X,Y,Z) │ UNet-Focuser │  (B,Cout,X,Y,Z)
#               └──────────────┘
class UNetFocuser(nn.Module):
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int  # Cout
    ):
        super(UNetFocuser, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, padding='same')
        )

    def forward(self, input_source: torch.Tensor) -> torch.Tensor:
        output_feat = self.pipe(input_source)
        return output_feat


# 00-01 UNet-EncoderPriorBank
# [Optional]
# This is for prior injection, you may inject anything to each stage of the Encoder
# Pinout Diagram: [Placeholder]
#                  ┌──────────────┐
# prior_source[S] ─│ 00-01        │─ prior_feats[S]
#   (B,Cin,X,Y,Z)  │ UNet-Encoder │  (B,Cout,X,Y,Z)
#                  │ PriorBank    │
#                  └──────────────┘
class UNetEncoderPriorBank(nn.Module):
    def __init__(
            self
    ):
        super(UNetEncoderPriorBank, self).__init__()

    def forward(self, prior_source: List[torch.Tensor]) -> List[torch.Tensor]:
        return None


# 00-01-00 UNet-EncoderPriorBank-Injector
# Injection process pipe
# Pinout Diagram: [Placeholder]
#                ┌───────────────────┐
# inject_source ─│ 00-01-00          │─ inject_feat
# (B,Cin,X,Y,Z)  │ UNet-EncoderPrior │  (B,Cout,X,Y,Z)
#                │ Bank-Injector     │
#                └───────────────────┘
class UNetEncoderPriorBankInjector(nn.Module):
    def __init__(
            self
    ):
        super(UNetEncoderPriorBankInjector, self).__init__()

    def forward(self, inject_source: torch.Tensor) -> torch.Tensor:
        return None


# 00-02 UNet-Encoder
# Main Feature Pyramid Extractor, which is intended to generate multi-granularity features
# Pinout Diagram: [Valid]
#                      ┌──────────────┐
#          input_feat ─│ 00-02        │─ output_feats[S1+S2]
#   (B,PCin[0],X,Y,Z)  │ UNet-Encoder │  # Primary part
#                      │              │  (B,PCout[0],X,  Y,  Z  )
# inject_feats[S1+S2] ─│              │  (B,PCout[1],X/2,Y/2,Z/2)
#            (Unused)  │              │  ⋮
#                      │              │  (B,PCout[S1-1],X/(2^(S1-1)),Y/(2^(S1-1)),Z/(2^(S1-1)))
#                      │              │  # Advanced part
#                      │              │  (B,ACout[0],X/(2^(S1)),Y/(2^(S1)),Z/(2^(S1)))
#                      │              │  (B,ACout[1],X/(2^(S1+1)),Y/(2^(S1+1)),Z/(2^(S1+1)))
#                      │              │  ⋮
#                      │              │  (B,ACout[S2-1],X/(2^(S1+S2-1)),Y/(2^(S1+S2-1)),Z/(2^(S1+S2-1)))
#                      └──────────────┘
# Expanded Diagram:
#                     ┌──────────────────┐
#         input_feat ─│ 00-02-00         │─ output_feats[0:S1]
#  (B,PCin[0],X,Y,Z)  │ UNet-Encoder-    │  # Primary part
#                     │ PrimaryExtractor │  [0]:(B,PCout[0],X,  Y,  Z  )
# inject_feats[0:S1] ─│                  │  [1]:(B,PCout[1],X/2,Y/2,Z/2)
#           (Unused)  │                  │  ⋮
#                     │                  │  [S1-1]:(B,PCout[S1-1],X/(2^(S1-1)),Y/(2^(S1-1)),Z/(2^(S1-1)))
#                     └──────────────────┘     │
#                                              │       ┌──────────────────┐
#                            output_feat[S1-1] └───────│ 00-02-01         │─ output_feats[S1:S1+S2]
#                                                      │ UNet-Encoder     │  # Advanced part
#                              inject_feats[S1:S1+S2] ─│ -Advanced        │  (B,ACout[0],X/(2^(S1)),Y/(2^(S1)),Z/(2^(S1)))
#                                            (Unused)  │ Extractor        │  (B,ACout[1],X/(2^(S1+1)),Y/(2^(S1+1)),Z/(2^(S1+1)))
#                                                      │                  │  ⋮
#                                                      │                  │  (B,ACout[S2-1],X/(2^(S1+S2-1)),Y/(2^(S1+S2-1)),Z/(2^(S1+S2-1)))
#                                                      └──────────────────┘
class UNetEncoder(nn.Module):
    def __init__(
            self,
            primary_in_channels: List[int],  # PCin[S1]
            primary_out_channels: List[int],  # PCout[S1]
            primary_depth: Union[int, List[int]],  # PD/PD[S1]
            advanced_in_channels: List[int],  # ACin[S2]
            advanced_out_channels: List[int],  # ACout[S2]
            advanced_depth: Union[int, List[int]]  # AD/AD[S2]
    ):
        super(UNetEncoder, self).__init__()
        assert len(primary_in_channels) == len(primary_out_channels)
        assert len(advanced_in_channels) == len(advanced_out_channels)
        if isinstance(primary_depth, List):
            assert len(primary_in_channels) == len(primary_depth)
        if isinstance(advanced_depth, List):
            assert len(advanced_in_channels) == len(advanced_depth)
        self.primary_extractor: UNetEncoderPrimaryExtractor = \
            UNetEncoderPrimaryExtractor(primary_in_channels, primary_out_channels, primary_depth)
        self.advanced_extractor: UNetEncoderAdvancedExtractor = \
            UNetEncoderAdvancedExtractor(advanced_in_channels, advanced_out_channels, advanced_depth)

    def forward(
            self,
            input_feat: torch.Tensor,
            inject_feats: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        assert (inject_feats is None or
                len(inject_feats) == self.primary_extractor.stages + self.advanced_extractor.stages)

        if inject_feats is None:
            primary_feats: List[torch.Tensor] = self.primary_extractor(
                input_feat)  # (S1)*(B,PCout[s1],x[s1],y[s1],z[s1])
            advanced_feats: List[torch.Tensor] = self.advanced_extractor(
                primary_feats[self.primary_extractor.stages - 1])  # (S2)*(B,ACout[s2],x[s2],y[s2],z[s2])
        else:
            primary_feats: List[torch.Tensor] = self.primary_extractor(
                input_feat,
                inject_feats[:self.primary_extractor.stages]
            )  # (S1)*(B,PCout[s1],x[s1],y[s1],z[s1])
            advanced_feats: List[torch.Tensor] = self.advanced_extractor(
                primary_feats[self.primary_extractor.stages - 1],
                inject_feats[self.primary_extractor.stages:]
            )  # (S2)*(B,ACout[s2],x[s2],y[s2],z[s2])

        output_feats: List[torch.Tensor] = primary_feats + advanced_feats
        # (S1+S2)*(B,P/ACout[s1/2],x[s1/2],y[s1/2],z[s1/2])
        return output_feats


# 00-02-00 UNet-Encoder-PrimaryExtractor
# Shallow part of the Encoder, representing primary semantics
# Pinout Diagram: [Valid]
#                   ┌──────────────────┐
#       input_feat ─│ 00-02-00         │─ output_feats[S]
# (B,Cin[0],X,Y,Z)  │ UNet-Encoder-    │  (B,Cout[0],X,  Y,  Z  )
#                   │ PrimaryExtractor │  (B,Cout[1],X/2,Y/2,Z/2)
#  inject_feats[*] ─│                  │  (B,Cout[2],X/4,Y/4,Z/4)
#         (Unused)  │                  │  ⋮
#                   └──────────────────┘
# Expanded Diagram:
#                   ┌──────────────────────────────────────────────────────────────────────────────────┐
#                   │  ┌──────────────────────┐ output_feats[0]              ┌──────────────────────┐  │
#       input_feat ─┼──│ 00-02-00-00          │──────────────────────────────│ 00-02-00-01          │──┼─ {output_feat}↓
# (B,Cin[0],X,Y,Z)  │  │ UNet-Encoder-Primary │ {output_feat}   {input_feat} │ UNet-Encoder-Primary │  │  (B,Cout[0]/Cin[1],X/2,Y/2,Z/2)
#  inject_feats[0] ─┼──│ Extractor-Stage      │ (B,Cout[0]/Cin[1],X,Y,Z)     │ Extractor-Downsample │  │
#         (Unused)  │  │ :depth=D/D[0]        │                              └──────────────────────┘  │
#                   │  └──────────────────────┘                                                        │
#                   └──────────────────────────────────────────────────────────────────────────────────┘
#                   ⋮                                        ⋮                                         ⋮
#                   ┌──────────────────────────────────────────────────────────────────────────────────┐
#                   │  ┌──────────────────────┐ output_feats[s]              ┌──────────────────────┐  │
#   ↑{output_feat} ─┼──│ 00-02-00-00          │──────────────────────────────│ 00-02-00-01          │──┼─ {output_feat}↓
# (B,Cin[s],X,Y,Z)  │  │ UNet-Encoder-Primary │ {output_feat}   {input_feat} │ UNet-Encoder-Primary │  │  (B,Cout[s]/Cin[s+1],X/(2^(s+1)),Y/(2^(s+1)),Z/(2^(s+1)))
#  inject_feats[s] ─┼──│ Extractor-Stage      │ (B,Cout[s]/Cin[s+1],X,Y,Z)   │ Extractor-Downsample │  │
#         (Unused)  │  │ :depth=D/D[s]        │                              └──────────────────────┘  │
#                   │  └──────────────────────┘                                                        │
#                   └──────────────────────────────────────────────────────────────────────────────────┘
class UNetEncoderPrimaryExtractor(nn.Module):
    def __init__(
            self,
            in_channels: List[int],  # Cin[S]
            out_channels: List[int],  # Cout[S]
            depth: Union[int, List[int]]  # D/D[S]
    ):
        super(UNetEncoderPrimaryExtractor, self).__init__()
        assert len(in_channels) == len(out_channels)
        if isinstance(depth, List):
            assert len(in_channels) == len(depth)
        self.stages: int = len(in_channels)
        self.pipe: nn.ModuleList = nn.ModuleList()
        if isinstance(depth, int):
            for ic, oc in zip(in_channels, out_channels):
                self.pipe.append(nn.Sequential(
                    UNetEncoderPrimaryExtractorStage(ic, oc, depth),
                    UNetEncoderPrimaryExtractorDownsample()
                ))
        else:
            for ic, oc, dp in zip(in_channels, out_channels, depth):
                self.pipe.append(nn.Sequential(
                    UNetEncoderPrimaryExtractorStage(ic, oc, dp),
                    UNetEncoderPrimaryExtractorDownsample()
                ))

    def forward(
            self,
            input_feat: torch.Tensor,
            inject_feats: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        output_feats: List[torch.Tensor] = []
        if inject_feats is None:
            stage_feat: torch.Tensor = input_feat
            for module in self.pipe:
                stage_feat = module[0](stage_feat)  # Stage
                output_feats.append(stage_feat)  # Record features for all stages
                stage_feat = module[1](stage_feat)  # Downsample
        else:
            assert len(inject_feats) == self.stages
            stage_feat: torch.Tensor = input_feat
            for module, inject_feat in zip(self.pipe, inject_feats):
                stage_feat = module[0](stage_feat, inject_feat)  # Stage
                output_feats.append(stage_feat)  # Record features for all stages
                stage_feat = module[1](stage_feat)  # Downsample

        return output_feats


# 00-02-00-00 UNet-Encoder-PrimaryExtractor-Stage
# One stage within the encoder primary extractor, representing one specified granularity
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
class UNetEncoderPrimaryExtractorStage(nn.Module):
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            depth: int  # D
    ):
        super(UNetEncoderPrimaryExtractorStage, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            *[ConvBNReLU(in_channels, in_channels, 3, padding='same') for _ in range(depth - 1)],
            ConvBNReLU(in_channels, out_channels, 3, padding='same')
        )

    def forward(self, input_feat: torch.Tensor, inject_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inject_feat is always ignored now
        output_feat: torch.Tensor = self.pipe(input_feat)
        return output_feat


# 00-02-00-01 UNet-Encoder-PrimaryExtractor-Downsample
# Downsampler after each primary extractor stage
# Pinout Diagram: [Valid]
#                ┌──────────────────────┐
#    input_feat ─│ 00-02-00-01          │─ output_feat
#   (B,C,X,Y,Z)  │ UNet-Encoder-Primary │  (B,C,X/2,Y/2,Z/2)
#                │ Extractor-Downsample │
#                └──────────────────────┘
class UNetEncoderPrimaryExtractorDownsample(nn.Module):
    def __init__(
            self
    ):
        super(UNetEncoderPrimaryExtractorDownsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.MaxPool3d(2, 2)
        )

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_feat)
        return output_feat


# 00-02-01 UNet-Encoder-AdvancedExtractor
# Deep part of the Encoder, representing advanced semantics
# Pinout Diagram: [Valid]
#                     ┌──────────────┐
#         input_feat ─│ 00-02-01     │─ output_feats[S]
#   (B,Cin[0],X,Y,Z)  │ UNet-Encoder │  (B,Cout[0],X,  Y,  Z  )
#                     │ -Advanced    │  (B,Cout[1],X/2,Y/2,Z/2)
#    inject_feats[*] ─│ Extractor    │  (B,Cout[2],X/4,Y/4,Z/4)
#           (Unused)  │              │  ⋮
#                     └──────────────┘
# Expanded Diagram:
#                   ┌────────────────────────────────────────────────────────────────────────────────────┐
#                   │  ┌───────────────────────┐ output_feats[0]              ┌───────────────────────┐  │
#       input_feat ─┼──│ 00-02-01-00           │──────────────────────────────│ 00-02-01-01           │──┼─ {output_feat}↓
# (B,Cin[0],X,Y,Z)  │  │ UNet-Encoder-Advanced │ {output_feat}   {input_feat} │ UNet-Encoder-Advanced │  │  (B,Cout[0]/Cin[1],X/2,Y/2,Z/2)
#  inject_feats[0] ─┼──│ Extractor-Stage       │ (B,Cout[0]/Cin[1],X,Y,Z)     │ Extractor-Downsample  │  │
#         (Unused)  │  │ :depth=D/D[0]         │                              └───────────────────────┘  │
#                   │  └───────────────────────┘                                                         │
#                   └────────────────────────────────────────────────────────────────────────────────────┘
#                   ⋮                                         ⋮                                          ⋮
#                   ┌────────────────────────────────────────────────────────────────────────────────────┐
#                   │  ┌───────────────────────┐ output_feats[s]              ┌───────────────────────┐  │
#   ↑{output_feat} ─┼──│ 00-02-01-00           │──────────────────────────────│ 00-02-01-01           │──┼─ {output_feat}↓
# (B,Cin[s],X,Y,Z)  │  │ UNet-Encoder-Advanced │ {output_feat}   {input_feat} │ UNet-Encoder-Advanced │  │  (B,Cout[s]/Cin[s+1],X/(2^(s+1)),Y/(2^(s+1)),Z/(2^(s+1)))
#  inject_feats[s] ─┼──│ Extractor-Stage       │ (B,Cout[s]/Cin[s+1],X,Y,Z)   │ Extractor-Downsample  │  │
#         (Unused)  │  │ :depth=D/D[s]         │                              └───────────────────────┘  │
#                   │  └───────────────────────┘                                                         │
#                   └────────────────────────────────────────────────────────────────────────────────────┘
class UNetEncoderAdvancedExtractor(nn.Module):
    def __init__(
            self,
            in_channels: List[int],  # Cin[S]
            out_channels: List[int],  # Cout[S]
            depth: Union[int, List[int]]  # D/D[S]
    ):
        super(UNetEncoderAdvancedExtractor, self).__init__()
        self.stages: int = len(in_channels)
        self.pipe: nn.ModuleList = nn.ModuleList()
        if isinstance(depth, int):
            for ic, oc in zip(in_channels, out_channels):
                self.pipe.append(nn.Sequential(
                    UNetEncoderAdvancedExtractorStage(ic, oc, depth),
                    UNetEncoderAdvancedExtractorDownsample()
                ))
        else:
            for ic, oc, dp in zip(in_channels, out_channels, depth):
                self.pipe.append(nn.Sequential(
                    UNetEncoderAdvancedExtractorStage(ic, oc, dp),
                    UNetEncoderAdvancedExtractorDownsample()
                ))

    def forward(
            self,
            input_feat: torch.Tensor,
            inject_feats: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        output_feats: List[torch.Tensor] = []
        if inject_feats is None:
            stage_feat: torch.Tensor = input_feat
            for module in self.pipe:
                stage_feat = module[0](stage_feat)  # Stage
                output_feats.append(stage_feat)  # Record features for all stages
                stage_feat = module[1](stage_feat)  # Downsample
        else:
            assert len(inject_feats) == self.stages
            stage_feat: torch.Tensor = input_feat
            for module, inject_feat in zip(self.pipe, inject_feats):
                stage_feat = module[0](stage_feat, inject_feat)  # Stage
                output_feats.append(stage_feat)  # Record features for all stages
                stage_feat = module[1](stage_feat)  # Downsample
        return output_feats


# 00-02-01-00 UNet-Encoder-AdvancedExtractor-Stage
# One stage within the encoder advanced extractor, representing one specified granularity
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
class UNetEncoderAdvancedExtractorStage(nn.Module):
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            depth: int  # D
    ):
        super(UNetEncoderAdvancedExtractorStage, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            *[ConvBNReLU(in_channels, in_channels, 3, padding='same') for _ in range(depth - 1)],
            ConvBNReLU(in_channels, out_channels, 3, padding='same')
        )

    def forward(self, input_feat: torch.Tensor, inject_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inject_feat is always ignored now
        output_feat: torch.Tensor = self.pipe(input_feat)
        return output_feat


# 00-02-01-01 UNet-Encoder-AdvancedExtractor-Downsample
# Downsampler after each advanced extractor stage
# Pinout Diagram: [Valid]
#                ┌───────────────────────┐
#    input_feat ─│ 00-02-01-01           │─ output_feat
#   (B,C,X,Y,Z)  │ UNet-Encoder-Advanced │  (B,C,X/2,Y/2,Z/2)
#                │ Extractor-Downsample  │
#                └───────────────────────┘
class UNetEncoderAdvancedExtractorDownsample(nn.Module):
    def __init__(
            self
    ):
        super(UNetEncoderAdvancedExtractorDownsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.MaxPool3d(2, 2)
        )

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_feat)
        return output_feat


# 00-03 UNet-Repeater
# Collect multi-stage features from Encoder and dispatch them to proper Decoder stage
# Pinout Diagram: [Valid]
#                ┌───────────────┐
#   input_feats ─│ 00-03-00      │─ output_feats
#   (B,C,X,Y,Z)  │ UNet-Repeater │  (B,C,X,Y,Z)
#                │               │
#                └───────────────┘
class UNetRepeater(nn.Module):
    def __init__(
            self,
            stages: int,
            bottleneck_in_channels: int,
            bottleneck_out_channels: int,
            bottleneck_depth: int
    ):
        super(UNetRepeater, self).__init__()
        self.pipe: nn.ModuleList = nn.ModuleList(
            [UNetRepeaterShortcut() for _ in range(stages - 1)]
        )
        self.pipe.append(
            UNetRepeaterBottleneck(
                in_channels=bottleneck_in_channels,
                out_channels=bottleneck_out_channels,
                depth=bottleneck_depth
            )
        )

    def forward(self, input_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        output_feats: List[torch.Tensor] = []
        module: nn.Module
        feat: torch.Tensor
        for module, feat in zip(self.pipe, input_feats):
            output_feats.append(module(feat))
        return output_feats


# 00-03-00 UNet-Repeater-Shortcut
# The feature delivery path from Encoder to Decoder stage by stage, i.e. skip connection
# Pinout Diagram: [Valid]
#                ┌───────────────┐
#    input_feat ─│ 00-03-00      │─ output_feat
#   (B,C,X,Y,Z)  │ UNet-Repeater │  (B,C,X,Y,Z)
#                │ Shortcut      │
#                └───────────────┘
class UNetRepeaterShortcut(nn.Module):
    def __init__(
            self
    ):
        super(UNetRepeaterShortcut, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.Identity()
        )

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        input_feat: torch.Tensor = self.pipe(input_feat)
        return input_feat


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
class UNetRepeaterBottleneck(nn.Module):
    def __init__(
            self,
            in_channels: int,  # Cin
            out_channels: int,  # Cout
            depth: int  # D
    ):
        super(UNetRepeaterBottleneck, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            *[ConvBNReLU(in_channels, in_channels, 3, padding='same') for _ in range(depth - 1)],
            ConvBNReLU(in_channels, out_channels, 3, padding='same')
        )

    def forward(self, input_feat: torch.Tensor, inject_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inject_feat is always ignored now
        output_feat: torch.Tensor = self.pipe(input_feat)
        return output_feat


# 00-04 UNet-DecoderPriorBank
# [Optional]
# This is for prior injection, you may inject anything to each stage of the Decoder
# Pinout Diagram: [Placeholder]
#                  ┌──────────────┐
# prior_source[S] ─│ 00-01        │─ prior_feats[S]
#   (B,Cin,X,Y,Z)  │ UNet-Decoder │  (B,Cout,X,Y,Z)
#                  │ PriorBank    │
#                  └──────────────┘
class UNetDecoderPriorBank(nn.Module):
    def __init__(
            self
    ):
        super(UNetDecoderPriorBank, self).__init__()

    def forward(self, prior_source: List[torch.Tensor]) -> List[torch.Tensor]:
        return None


# 00-04-00 UNet-DecoderPriorBank-Injector
# Injection process pipe
# Pinout Diagram: [Placeholder]
#                ┌───────────────────┐
# inject_source ─│ 00-04-00          │─ inject_feat
# (B,Cin,X,Y,Z)  │ UNet-DecoderPrior │  (B,Cout,X,Y,Z)
#                │ Bank-Injector     │
#                └───────────────────┘
class UNetDecoderPriorBankInjector(nn.Module):
    def __init__(
            self
    ):
        super(UNetDecoderPriorBankInjector, self).__init__()

    def forward(self, inject_source: torch.Tensor) -> torch.Tensor:
        return None


# TODO 2026-03-04
# 00-05 UNet-Decoder
class UNetDecoder(nn.Module):
    def __init__(
            self,
            primary_in_channels: List[int],
            primary_upsample_channels: List[int],
            primary_shortcut_channels: List[int],
            primary_out_channels: List[int],
            primary_depth: Union[int, List[int]],
            advanced_in_channels: List[int],
            advanced_upsample_channels: List[int],
            advanced_shortcut_channels: List[int],
            advanced_out_channels: List[int],
            advanced_depth: Union[int, List[int]]
    ):
        super(UNetDecoder, self).__init__()
        assert len(primary_in_channels) == len(primary_upsample_channels)
        assert len(primary_shortcut_channels) == len(primary_out_channels)
        assert len(primary_in_channels) == len(primary_out_channels)
        assert len(advanced_in_channels) == len(advanced_upsample_channels)
        assert len(advanced_shortcut_channels) == len(advanced_out_channels)
        assert len(advanced_in_channels) == len(advanced_out_channels)
        if isinstance(primary_depth, List):
            assert len(primary_in_channels) == len(primary_depth)
        if isinstance(advanced_depth, List):
            assert len(advanced_in_channels) == len(advanced_depth)

        self.primary_aggregator: UNetDecoderPrimaryAggregator = UNetDecoderPrimaryAggregator(
            primary_in_channels,
            primary_upsample_channels,
            primary_shortcut_channels,
            primary_out_channels,
            primary_depth
        )
        self.advanced_aggregator: UNetDecoderAdvancedAggregator = UNetDecoderAdvancedAggregator(
            advanced_in_channels,
            advanced_upsample_channels,
            advanced_shortcut_channels,
            advanced_out_channels,
            advanced_depth
        )

    def forward(
            self,
            input_feat: torch.Tensor,
            shortcut_feats: List[torch.Tensor],
            inject_feats: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        assert len(shortcut_feats) == self.primary_extractor.stages + self.advanced_extractor.stages
        assert (inject_feats is None or
                len(inject_feats) == self.primary_extractor.stages + self.advanced_extractor.stages)
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

        output_feat: List[torch.Tensor] = advanced_feats + primary_feats
        return output_feat


# 00-05-00 UNet-Decoder-PrimaryAggregator
# Shallow part of the Decoder, representing primary semantic / local-view discriminative evidence aggregator
class UNetDecoderPrimaryAggregator(nn.Module):
    def __init__(
            self,
            in_channels: List[int],
            upsample_channels: List[int],
            shortcut_channels: List[int],
            out_channels: List[int],
            depth: Union[int, List[int]]
    ):
        super(UNetDecoderPrimaryAggregator, self).__init__()
        assert len(in_channels) == len(upsample_channels)
        assert len(shortcut_channels) == len(out_channels)
        assert len(in_channels) == len(out_channels)
        if isinstance(depth, List):
            assert len(in_channels) == len(depth)
        self.stages: int = len(in_channels)
        self.pipe: nn.Sequential = nn.Sequential()
        if isinstance(depth, int):
            for ic, uc, sc, oc in zip(in_channels, upsample_channels, shortcut_channels, out_channels):
                self.pipe.append(nn.Sequential(
                    UNetDecoderPrimaryAggregatorUpsample(ic, uc),
                    UNetDecoderPrimaryAggregatorFusionPortal(),
                    UNetDecoderPrimaryAggregatorStage(uc + sc, oc, depth)
                ))
        else:
            for ic, uc, sc, oc, dp in zip(in_channels, upsample_channels, shortcut_channels, out_channels, depth):
                self.pipe.append(nn.Sequential(
                    UNetDecoderPrimaryAggregatorUpsample(ic, uc),
                    UNetDecoderPrimaryAggregatorFusionPortal(),
                    UNetDecoderPrimaryAggregatorStage(uc + sc, oc, dp)
                ))

    def forward(
            self,
            input_feat: torch.Tensor,
            shortcut_feats: List[torch.Tensor],
            inject_feats: Optional[List[torch.Tensor]] = None
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

        return output_feats


# 00-05-00-00 UNet-Decoder-PrimaryAggregator-Stage
class UNetDecoderPrimaryAggregatorStage(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            depth: int
    ):
        super(UNetDecoderPrimaryAggregatorStage, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            *[ConvBNReLU(in_channels, out_channels, 3, padding='same') for _ in range(depth - 1)],
            ConvBNReLU(out_channels, out_channels, 3, padding='same')
        )

    def forward(self, input_feat: torch.Tensor, inject_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inject_feat is always ignored now
        output_feat: torch.Tensor = self.pipe(input_feat)
        return output_feat


# 00-05-00-01 UNet-Decoder-PrimaryAggregator-Upsample
class UNetDecoderPrimaryAggregatorUpsample(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(UNetDecoderPrimaryAggregatorUpsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_feat)
        return output_feat


# 00-05-00-02 UNet-Decoder-PrimaryAggregator-FusionPortal
class UNetDecoderPrimaryAggregatorFusionPortal(nn.Module):
    def __init__(
            self
    ):
        super(UNetDecoderPrimaryAggregatorFusionPortal, self).__init__()
        self.concat: Concat = Concat(dim=1)

    def forward(self, stage_feat: torch.Tensor, shortcut_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.concat(stage_feat, shortcut_feat)
        return output_feat


# 00-05-01 UNet-Decoder-AdvancedAggregator
# Deep part of the Decoder, representing advanced semantic / global-view discriminative evidence aggregator
class UNetDecoderAdvancedAggregator(nn.Module):
    def __init__(
            self,
            in_channels: List[int],
            upsample_channels: List[int],
            shortcut_channels: List[int],
            out_channels: List[int],
            depth: Union[int, List[int]]
    ):
        super(UNetDecoderAdvancedAggregator, self).__init__()
        assert len(in_channels) == len(upsample_channels)
        assert len(shortcut_channels) == len(out_channels)
        assert len(in_channels) == len(out_channels)
        if isinstance(depth, List):
            assert len(in_channels) == len(depth)
        self.stages: int = len(in_channels)
        self.pipe: nn.Sequential = nn.Sequential()
        if isinstance(depth, int):
            for ic, uc, sc, oc in zip(in_channels, upsample_channels, shortcut_channels, out_channels):
                self.pipe.append(nn.Sequential(
                    UNetDecoderAdvancedAggregatorUpsample(ic, uc),
                    UNetDecoderAdvancedAggregatorFusionPortal(),
                    UNetDecoderAdvancedAggregatorStage(uc + sc, oc, depth)
                ))
        else:
            for ic, uc, sc, oc, dp in zip(in_channels, upsample_channels, shortcut_channels, out_channels, depth):
                self.pipe.append(nn.Sequential(
                    UNetDecoderAdvancedAggregatorUpsample(ic, uc),
                    UNetDecoderAdvancedAggregatorFusionPortal(),
                    UNetDecoderAdvancedAggregatorStage(uc + sc, oc, dp)
                ))

    def forward(
            self,
            input_feat: torch.Tensor,
            shortcut_feats: List[torch.Tensor],
            inject_feats: Optional[List[torch.Tensor]] = None
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

        return output_feats


# 00-05-01-00 UNet-Decoder-AdvancedAggregator-Stage
class UNetDecoderAdvancedAggregatorStage(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            depth: int
    ):
        super(UNetDecoderAdvancedAggregatorStage, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            *[ConvBNReLU(in_channels, out_channels, 3, padding='same') for _ in range(depth - 1)],
            ConvBNReLU(out_channels, out_channels, 3, padding='same')
        )

    def forward(self, input_feat: torch.Tensor, inject_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inject_feat is always ignored now
        output_feat: torch.Tensor = self.pipe(input_feat)
        return output_feat


# 00-05-01-01 UNet-Decoder-AdvancedAggregator-Upsample
class UNetDecoderAdvancedAggregatorUpsample(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(UNetDecoderAdvancedAggregatorUpsample, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_feat)
        return output_feat


# 00-05-01-02 UNet-Decoder-AdvancedAggregator-FusionPortal
class UNetDecoderAdvancedAggregatorFusionPortal(nn.Module):
    def __init__(
            self
    ):
        super(UNetDecoderAdvancedAggregatorFusionPortal, self).__init__()
        self.concat: Concat = Concat(dim=1)

    def forward(self, stage_feat: torch.Tensor, shortcut_feat: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.concat(stage_feat, shortcut_feat)
        return output_feat


# 00-06 UNet-DecoderAuxiliaryWrapper
# [Optional]
# Auxiliary Wrapper is for post decoder stage auxiliary supervision tasks, such as deep supervision
class UNetDecoderAuxiliaryWrapper(nn.Module):
    def __init__(
            self,
            in_channels: List[int],
            out_channels: List[int]
    ):
        super(UNetDecoderAuxiliaryWrapper, self).__init__()
        self.pipe: nn.ModuleList = nn.ModuleList()
        for ic, oc in zip(in_channels, out_channels):
            self.pipe.append(nn.Sequential(
                ConvBNReLU(ic, ic, 3, padding='same'),
                nn.Conv3d(ic, oc, 1)
            ))


    def forward(self, input_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(input_feats) == len(self.pipe)
        output_feats: List[torch.Tensor] = []
        for module, feat in zip(self.pipe, input_feats):
            output_feats.append(module(feat))
        return output_feats


# 00-07 UNet-Distributor
# [Optional]
# Distributor is a Stem Feature Aggregator after Decoder, which is intended to generate or filter discriminative features
class UNetDistributor(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(UNetDistributor, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3, padding='same')
        )

    def forward(self, input_source: torch.Tensor) -> torch.Tensor:
        output_feat: torch.Tensor = self.pipe(input_source)
        return output_feat


# 00-08 UNet-Classifier
class UNetClassifier(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(UNetClassifier, self).__init__()
        self.pipe: nn.Sequential = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1)
            # nn.Sigmoid()  # Only return logits, apply normalization latter
        )

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.pipe(input_feat)
        return logits


if __name__ == "__main__":
    pass
