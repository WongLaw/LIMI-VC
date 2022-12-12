"""LIMIVC model architecture."""

from typing import Tuple, List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .convolutional_transformer import Smoother, Extractor
from .light_dy_conv import DynamicConv, Depthwise_separable_conv


class LIMIVC(nn.Module):

    def __init__(self, d_model=512):
        super().__init__()

        self.unet = UnetBlock(d_model)

        self.smoothers = nn.Sequential(
            DynamicConv(dim_in=d_model, dim_out=d_model, kernel_size = 3, num_heads = 8),
        )

        self.mel_linear = nn.Linear(d_model, 80)

        self.post_net = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=5, padding=2),
            nn.InstanceNorm1d(256, affine=True),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.InstanceNorm1d(256, affine=True),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(256, 80, kernel_size=5, padding=2),
            nn.BatchNorm1d(80),
            nn.Dropout(0.5),
        )

    def forward(
        self,
        srcs: Tensor,
        refs: Tensor,
        src_masks: Optional[Tensor] = None,
        ref_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """Forward function.

        Args:
            srcs: (batch, src_len, 768)
            src_masks: (batch, src_len)
            refs: (batch, 80, ref_len)
            ref_masks: (batch, ref_len)
        """

        # out: (src_len, batch, d_model)
        out, attns, tgt, ref1 = self.unet(srcs, refs, src_masks=src_masks, ref_masks=ref_masks)
        # print("out1: ", out.shape)    #[t,b,d]
        # out: (src_len, batch, d_model)
        out = out.permute(1,0,2)
        out = self.smoothers(out)
        out = out.permute(1,0,2)
        # print("out2: ", out.shape)    #[t,b,d]
        # out: (src_len, batch, 80)
        out = self.mel_linear(out)

        # out: (batch, 80, src_len)
        out = out.transpose(1, 0).transpose(2, 1)
        refined = self.post_net(out)
        out = out + refined

        # out: (batch, 80, src_len)
        return out, attns, tgt, ref1


class UnetBlock(nn.Module):
    """Hierarchically attend on references."""

    def __init__(self, d_model: int):
        super(UnetBlock, self).__init__()

        self.conv1 = Depthwise_separable_conv(nin=80, nout=d_model, kernel_size = 3, padding = 1)

        self.prenet = nn.Sequential(
            nn.Linear(768, 768), nn.PReLU(), nn.Linear(768, d_model),
        )

        self.extractor = Extractor(d_model, 2, 1024)

    def forward(
        self,
        srcs: Tensor,
        refs: Tensor,
        src_masks: Optional[Tensor] = None,
        ref_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """Forward function.

        Args:
            srcs: (batch, src_len, 768)
            src_masks: (batch, src_len)
            refs: (batch, 80, ref_len)
            ref_masks: (batch, ref_len)
        """

        # tgt: (batch, tgt_len, d_model)
        tgt_ = self.prenet(srcs)
        # tgt: (tgt_len, batch, d_model)
        tgt = tgt_.transpose(0, 1)

        # ref*: (batch, d_model, mel_len)
        ref1 = self.conv1(refs)
        
        out, attn = self.extractor(
            tgt,#out
            ref1.transpose(1, 2).transpose(0, 1),
            tgt_key_padding_mask=src_masks,
            memory_key_padding_mask=ref_masks,
        )

        # out: (tgt_len, batch, d_model)
        return out, [attn], tgt_, ref1.permute(0,2,1)

