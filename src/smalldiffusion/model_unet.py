# Adapted from PNDM implmentation (https://github.com/luping-liu/PNDM)
# which is adapted from DDIM implementation (https://github.com/ermongroup/ddim)

import math
import torch
from einops import rearrange
from itertools import pairwise
from torch import nn
from .model import (
    alpha, Attention, ModelMixin, CondSequential, SigmaEmbedderSinCos,
    Conv2d, GroupNorm, Linear, Dropout, Rngs
)

def Normalize(ch, rngs: Rngs):
    return GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True, rngs=rngs)

def Upsample(ch, rngs: Rngs):
    return nn.Sequential(
        nn.Upsample(scale_factor=2.0, mode='nearest'),
        Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, rngs=rngs)
    )

def Downsample(ch, rngs: Rngs):
    return nn.Sequential(
        nn.ConstantPad2d((0, 1, 0, 1), 0),
        Conv2d(ch, ch, kernel_size=3, stride=2, padding=0, rngs=rngs)
    )

class ResnetBlock(nn.Module):
    def __init__(self, *, in_ch, out_ch=None, conv_shortcut=False,
                 dropout, temb_channels=512, rngs: Rngs):
        super().__init__()
        self.in_ch = in_ch
        out_ch = in_ch if out_ch is None else out_ch
        self.out_ch = out_ch
        self.use_conv_shortcut = conv_shortcut

        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            Linear(temb_channels, out_ch, rngs=rngs)
        )
        self.layer1 = nn.Sequential(
            Normalize(in_ch, rngs=rngs),
            nn.SiLU(),
            Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, rngs=rngs)
        )
        self.layer2 = nn.Sequential(
            Normalize(out_ch, rngs=rngs),
            nn.SiLU(),
            Dropout(dropout, rngs=rngs),
            Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, rngs=rngs)
        )
        if self.in_ch != self.out_ch:
            kernel_stride_padding = (3,1,1) if self.use_conv_shortcut else (1,1,0)
            self.shortcut = Conv2d(in_ch, out_ch, *kernel_stride_padding, rngs=rngs)

    def forward(self, x, temb):
        h = x
        h = self.layer1(h)
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.layer2(h)
        if self.in_ch != self.out_ch:
            x = self.shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=1, rngs: Rngs | None = None):
        super().__init__()
        # Normalize input along the channel dimension
        self.norm = Normalize(ch, rngs=rngs)
        # Attention over D: (B, N, D) -> (B, N, D)
        self.attn = Attention(head_dim=ch // num_heads, num_heads=num_heads, rngs=rngs)

    def forward(self, x, temb):
        # temb is currently not used, but included for CondSequential to work
        B, C, H, W = x.shape
        h_ = self.norm(x)
        h_ = rearrange(h_, 'b c h w -> b (h w) c')
        h_ = self.attn(h_)
        h_ = rearrange(h_, 'b (h w) c -> b c h w', h=H, w=W)
        return x + h_

class Unet(ModelMixin, nn.Module):
    def __init__(self, in_dim, in_ch, out_ch,
                 ch               = 128,
                 ch_mult          = (1,2,2,2),
                 embed_ch_mult    = 4,
                 num_res_blocks   = 2,
                 attn_resolutions = (16,),
                 dropout          = 0.1,
                 resamp_with_conv = True,
                 sig_embed        = None,
                 cond_embed       = None,
                 *,
                 rngs: Rngs,
                 ):
        super().__init__()

        self.ch = ch
        self.in_dim = in_dim
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.input_dims = (in_ch, in_dim, in_dim)
        self.temb_ch = self.ch * embed_ch_mult

        # Embeddings
        self.sig_embed = sig_embed or SigmaEmbedderSinCos(self.temb_ch, rngs=rngs)
        make_block = lambda in_ch, out_ch: ResnetBlock(
            in_ch=in_ch, out_ch=out_ch, temb_channels=self.temb_ch, dropout=dropout, rngs=rngs
        )
        self.cond_embed = cond_embed

        # Downsampling
        curr_res = in_dim
        in_ch_dim = [ch * m for m in (1,)+ch_mult]
        self.conv_in = Conv2d(in_ch, self.ch, kernel_size=3, stride=1, padding=1, rngs=rngs)
        self.downs = nn.ModuleList()
        block_in, block_out = 0, 0
        for i, (block_in, block_out) in enumerate(pairwise(in_ch_dim)):
            down = nn.Module()
            down.blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                block = [make_block(block_in,block_out)]
                if curr_res in attn_resolutions:
                    block.append(AttnBlock(block_out, rngs=rngs))
                down.blocks.append(CondSequential(*block))
                block_in = block_out
            if i < self.num_resolutions - 1: # Not last iter
                down.downsample = Downsample(block_in, rngs=rngs)
                curr_res = curr_res // 2
            self.downs.append(down)

        # Middle
        self.mid = CondSequential(
            make_block(block_in, block_in),
            AttnBlock(block_in, rngs=rngs),
            make_block(block_in, block_in)
        )

        # Upsampling
        self.ups = nn.ModuleList()
        for i_level, (block_out, next_skip_in) in enumerate(pairwise(reversed(in_ch_dim))):
            up = nn.Module()
            up.blocks = nn.ModuleList()
            skip_in = block_out
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = next_skip_in
                block: list = [make_block(block_in+skip_in, block_out)]
                if curr_res in attn_resolutions:
                    block.append(AttnBlock(block_out, rngs=rngs))
                up.blocks.append(CondSequential(*block))
                block_in = block_out
            if i_level < self.num_resolutions - 1: # Not last iter
                up.upsample = Upsample(block_in, rngs=rngs)
                curr_res = curr_res * 2
            self.ups.append(up)

        # Out
        self.out_layer = nn.Sequential(
            Normalize(block_in, rngs=rngs),
            nn.SiLU(),
            Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1, rngs=rngs)
        )

    def fwd_emb(self, x, emb, cond):
        assert x.shape[2] == x.shape[3] == self.in_dim

        if self.cond_embed is not None:
            assert cond is not None and x.shape[0] == cond.shape[0], \
                'Conditioning must have same batches as x!'
            emb += self.cond_embed(cond)

        # downsampling
        hs = [self.conv_in(x)]
        for down in self.downs:
            for block in down.blocks: # type: ignore
                h = block(hs[-1], emb)
                hs.append(h)
            if hasattr(down, 'downsample'):
                hs.append(down.downsample(hs[-1])) # type: ignore

        # middle
        h = self.mid(hs[-1], emb)

        # upsampling
        for up in self.ups:
            for block in up.blocks: # type: ignore
                h = block(torch.cat([h, hs.pop()], dim=1), emb)
            if hasattr(up, 'upsample'):
                h = up.upsample(h) # type: ignore

        # out
        return self.out_layer(h)

    def forward(self, x, sigma, cond=None):
        emb = self.sig_embed(x.shape[0], sigma.squeeze())
        return self.fwd_emb(x, emb, cond)
