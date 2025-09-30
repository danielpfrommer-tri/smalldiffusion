import math
import typing as tp
import numpy as np
import torch
import torch.nn.functional as F

import jax.random
import jax.numpy

from torch import nn
from einops import rearrange, repeat
from itertools import pairwise

type RngSeed = int | tp.Any

class RngStream:
    def __init__(self, key: RngSeed, *, tag: str):
        self.key = jax.random.key(key) if isinstance(key, int) else key
        self.count = jax.numpy.array(0, dtype=jax.numpy.uint32)
        self.tag = tag

    def __call__(self) -> jax.Array:
        key = jax.random.fold_in(self.key, self.count)
        self.count = self.count + 1
        return key

    def last(self) -> jax.Array:
        return jax.random.fold_in(self.key, self.count - 1)

    def fork(self, *, split: int | tuple[int, ...] | None = None):
        key = self()
        if split is not None:
            key = jax.random.split(key, split)
        return type(self)(key, tag=self.tag)

class Rngs:
    def __init__(
        self,
        default: (
            RngSeed | RngStream | tp.Mapping[str, RngSeed | RngStream] | None
        ) = None,
        **rngs: RngSeed | RngStream,
    ):
        if default is not None:
            if isinstance(default, tp.Mapping):
                rngs = {**default, **rngs}
            else:
                rngs["default"] = default

        self.streams = {}
        for tag, key in rngs.items():
            if isinstance(key, RngStream):
                key = key.key.value[...]
            self.streams[tag] = RngStream(
                key=key,
                tag=tag,
            )

    def _get_stream(self, name: str, error_type: type[Exception]) -> RngStream:
        if name not in self.streams:
            if "default" not in self.streams:
                raise error_type(f"No RngStream named '{name}' found in Rngs.")
            stream = self.streams["default"]
        else:
            stream = self.streams[name]
        return stream

    def __getitem__(self, name: str):
        return self._get_stream(name, KeyError)

    def __getattr__(self, name: str):
        if name == "streams":
            super().__getattribute__(name)
        return self._get_stream(name, AttributeError)

    def __call__(self):
        return self.default()

    def __contains__(self, name: tp.Any) -> bool:
        return name in vars(self)

    def items(self):
        for name, stream in vars(self).items():
            if isinstance(stream, RngStream):
                yield name, stream

## Basic functions used by all models

class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None,
                 *, rngs: Rngs):
        self.kernel_init_key = rngs.params()
        if bias:
            self.bias_init_key = rngs.params()
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
    
    def reset_parameters(self):
        scale = 1/ math.sqrt(self.in_features)
        kernel = jax.random.uniform(self.kernel_init_key, self.weight.shape[::-1], minval=-scale, maxval=scale).T
        self.weight.data.copy_(torch.from_numpy(np.array(kernel)))

        if self.bias is not None:
            bias = jax.random.uniform(self.bias_init_key, self.bias.shape, minval=-scale, maxval=scale)
            self.bias.data.copy_(torch.from_numpy(np.array(bias)))
    
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] | str = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 *, rngs: Rngs):
        self.kernel_init_key = rngs.params()
        if bias:
            self.bias_init_key = rngs.params()
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         device=device, dtype=dtype)
    
    def reset_parameters(self):
        scale = 1 / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        jax_shape = tuple(self.weight.shape[i] for i in (2, 3, 1, 0))
        # kernel = initializers.variance_scaling(1.0, "fan_in", "uniform")(self.kernel_init_key, jax_shape)
        kernel = jax.random.uniform(self.kernel_init_key, jax_shape, minval=-scale, maxval=scale)
        kernel = kernel.transpose(3, 2, 0, 1) # to PyTorch shape
        self.weight.data.copy_(torch.from_numpy(np.array(kernel)))

        if self.bias is not None:
            bias = jax.random.uniform(self.bias_init_key, self.bias.shape, minval=-scale, maxval=scale)
            self.bias.data.copy_(torch.from_numpy(np.array(bias)))
    
class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False, *, rngs: Rngs):
        self.rng_stream = rngs.dropout.fork()
        super().__init__(p, inplace)
    
    # use the jax dropout rng
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)
        # TODO: Make not super slow
        if not self.training:
            return input
        else:
            mask_shape = input.shape
            if len(mask_shape) == 4:
                # turn NCHW to NHWC for jax compatbility
                mask_shape = tuple(mask_shape[i] for i in (0, 2, 3, 1))
                mask = jax.random.bernoulli(self.rng_stream(), self.p, mask_shape)
                mask = jax.numpy.transpose(mask, (0, 3, 1, 2)) # back to NCHW
            else:
                mask = jax.random.bernoulli(self.rng_stream(), self.p, mask_shape)
            mask = torch.from_numpy(np.array(mask))
            mask = mask.to(input.device)
            output = torch.where(mask, 0, input)
            scale = 1/(1 - self.p)
            return output * scale


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, *, num_groups=32, eps=1e-06,
                affine=True, device=None, dtype=None, rngs: Rngs):
        if affine:
            self.scale_init_key = rngs.params()
            self.bias_init_key = rngs.params()
        super().__init__(
            num_groups, num_channels,
            eps, affine, device, dtype
        )

class ModelMixin:
    def rand_input(self, batchsize) -> torch.Tensor:
        assert hasattr(self, 'input_dims'), 'Model must have "input_dims" attribute!'
        return torch.randn((batchsize,) + self.input_dims) # type: ignore

    # Currently predicts eps, override following methods to predict, for example, x0
    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):
        return loss()(eps, self(x0 + sigma * eps, sigma, cond=cond)) # type: ignore

    def predict_eps(self, x, sigma, cond=None):
        return self(x, sigma, cond=cond) # type: ignore

    def predict_eps_cfg(self, x, sigma, cond, cfg_scale):
        if cond is None or cfg_scale == 0:
            return self.predict_eps(x, sigma, cond=cond)
        assert sigma.shape == tuple(), 'CFG sampling only supports singleton sigma!'
        uncond = torch.full_like(cond, self.cond_embed.null_cond) # (B,) # type: ignore
        eps_cond, eps_uncond = self.predict_eps(                  # (B,), (B,)
            torch.cat([x, x]), sigma, torch.cat([cond, uncond])   # (2B,)
        ).chunk(2)
        return eps_cond + cfg_scale * (eps_cond - eps_uncond)

def get_sigma_embeds(batches, sigma, scaling_factor=0.5, log_scale=True):
    if sigma.shape == torch.Size([]):
        sigma = sigma.unsqueeze(0).repeat(batches)
    else:
        assert sigma.shape == (batches,), 'sigma.shape == [] or [batches]!'
    if log_scale:
        sigma = torch.log(sigma)
    s = sigma.unsqueeze(1) * scaling_factor
    return torch.cat([torch.sin(s), torch.cos(s)], dim=1)

# A simple embedding that works just as well as usual sinusoidal embedding
class SigmaEmbedderSinCos(nn.Module):
    def __init__(self, hidden_size, scaling_factor=0.5, log_scale=True, *, rngs: Rngs):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.log_scale = log_scale
        self.mlp = nn.Sequential(
            Linear(2, hidden_size, bias=True, rngs=rngs),
            nn.SiLU(),
            Linear(hidden_size, hidden_size, bias=True, rngs=rngs)
        )

    def forward(self, batches, sigma):
        sig_embed = get_sigma_embeds(batches, sigma,
                                     self.scaling_factor,
                                     self.log_scale)                      # (B, 2)
        return self.mlp(sig_embed)                                        # (B, D)


## Modifiers for models, such as including scaling or changing model predictions

def alpha(sigma):
    return 1/(1+sigma**2)

# Scale model input so that its norm stays constant for all sigma
def Scaled[T: type[ModelMixin]](cls: T) -> T:
    def forward(self, x, sigma, cond=None):
        return cls.forward(self, x * alpha(sigma).sqrt(), sigma, cond=cond) # type: ignore
    return type(cls.__name__ + 'Scaled', (cls,), dict(forward=forward)) # type: ignore

# Train model to predict x0 instead of eps
def PredX0(cls: type[ModelMixin]):
    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):
        return loss()(x0, self(x0 + sigma * eps, sigma, cond=cond))
    def predict_eps(self, x, sigma, cond=None):
        x0_hat = self(x, sigma, cond=cond)
        return (x - x0_hat)/sigma
    return type(cls.__name__ + 'PredX0', (cls,),
                dict(get_loss=get_loss, predict_eps=predict_eps))

# Train model to predict v (https://arxiv.org/pdf/2202.00512.pdf) instead of eps
def PredV(cls: type[ModelMixin]):
    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):
        xt = x0 + sigma * eps
        v = alpha(sigma).sqrt() * eps - (1-alpha(sigma)).sqrt() * x0
        return loss()(v, self(xt, sigma, cond=cond))
    def predict_eps(self, x, sigma, cond=None):
        v_hat = self(x, sigma, cond=cond)
        return alpha(sigma).sqrt() * (v_hat + (1-alpha(sigma)).sqrt() * x)
    return type(cls.__name__ + 'PredV', (cls,),
                dict(get_loss=get_loss, predict_eps=predict_eps))

## Common functions for other models

class CondSequential(nn.Sequential):
    def forward(self, x, cond): # type: ignore
        for module in self._modules.values():
            x = module(x, cond)
        return x

class Attention(nn.Module):
    def __init__(self, head_dim, num_heads=8, qkv_bias=False, *, rngs: Rngs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        dim = head_dim * num_heads
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias, rngs=rngs)
        self.proj = Linear(dim, dim, rngs=rngs)

    def forward(self, x):
        # (B, N, D) -> (B, N, D)
        # N = H * W / patch_size**2, D = num_heads * head_dim
        q, k, v = rearrange(self.qkv(x), 'b n (qkv h k) -> qkv b h n k',
                            h=self.num_heads, k=self.head_dim)
        x = rearrange(F.scaled_dot_product_attention(q, k, v),
                      'b h n k -> b n (h k)')
        return self.proj(x)

# Embedding table for conditioning on labels assumed to be in [0, num_classes),
# unconditional label encoded as: num_classes
class CondEmbedderLabel(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_prob=0.1):
        super().__init__()
        self.embeddings = nn.Embedding(num_classes + 1, hidden_size)
        self.null_cond = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels): # (B,) -> (B, D)
        if self.training:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, self.null_cond, labels)
        return self.embeddings(labels)

## Simple MLP for toy examples

class TimeInputMLP(ModelMixin, nn.Module):
    sigma_dim = 2
    def __init__(self, dim=2, output_dim=None, hidden_dims=(16,128,256,128,16)):
        super().__init__()
        layers = []
        for in_dim, out_dim in pairwise((dim + self.sigma_dim,) + hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], output_dim or dim))

        self.net = nn.Sequential(*layers)
        self.input_dims = (dim,)

    def forward(self, x, sigma, cond=None):
        # x     shape: b x dim
        # sigma shape: b x 1 or scalar
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze()) # shape: b x 2
        nn_input = torch.cat([x, sigma_embeds], dim=1)               # shape: b x (dim + 2)
        return self.net(nn_input)

class ConditionalMLP(TimeInputMLP):
    def __init__(self, dim=2, hidden_dims=(16,128,256,128,16),
                 cond_dim=4, num_classes=10, dropout_prob=0.1):
        super().__init__(dim=dim+cond_dim, output_dim=dim, hidden_dims=hidden_dims)
        self.input_dims = (dim,)
        self.cond_embed = CondEmbedderLabel(cond_dim, num_classes, dropout_prob)

    def forward(self, # type: ignore
                x,     # shape: b x dim
                sigma, # shape: b x 1 or scalar
                cond,  # shape: b
                ):
        cond_embeds = self.cond_embed(cond)                          # shape: b x cond_dim
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze()) # shape: b x sigma_dim
        nn_input = torch.cat([x, sigma_embeds, cond_embeds], dim=1)  # shape: b x (dim + sigma_dim + cond_dim)
        return self.net(nn_input)

## Ideal denoiser defined by a dataset

def sq_norm(M, k):
    # M: b x n --(norm)--> b --(repeat)--> b x k
    return (torch.norm(M, dim=1)**2).unsqueeze(1).repeat(1,k)

class IdealDenoiser(ModelMixin, nn.Module):
    def __init__(self, dataset: torch.utils.data.Dataset):
        super().__init__()
        self.data = torch.stack([dataset[i] for i in range(len(dataset))]) # type: ignore
        self.input_dims = self.data.shape[1:]

    def forward(self, x, sigma, cond=None):
        data = self.data.to(x)                                                         # shape: db x c1 x ... x cn
        x_flat = x.flatten(start_dim=1)
        d_flat = data.flatten(start_dim=1)
        xb, xr = x_flat.shape
        db, dr = d_flat.shape
        assert xr == dr, 'Input x must have same dimension as data!'
        assert sigma.shape == tuple() or sigma.shape[0] == xb, \
            f'sigma must be singleton or have same batch dimension as x! {sigma.shape}'
        # sq_diffs: ||x - x0||^2
        sq_diffs = sq_norm(x_flat, db).T + sq_norm(d_flat, xb) - 2 * d_flat @ x_flat.T # shape: db x xb
        weights = torch.nn.functional.softmax(-sq_diffs/2/sigma.squeeze()**2, dim=0)             # shape: db x xb
        eps = torch.einsum('ij,i...->j...', weights, data)                             # shape: xb x c1 x ... x cn
        return (x - eps) / sigma