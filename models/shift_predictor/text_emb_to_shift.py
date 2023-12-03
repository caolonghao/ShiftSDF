import math
import copy
from random import random
from tqdm.auto import tqdm
from functools import partial, wraps
from contextlib import contextmanager, nullcontext
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import nn, einsum
from torch.cuda.amp import autocast
from torch.special import expm1
import torchvision.transforms as T

import kornia.augmentation as K

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from .openai_model_3d import UNet3DModel

# helper functions
def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def divisible_by(numer, denom):
    return (numer % denom) == 0

def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]

def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# helper classes

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

# tensor helpers

def log(t, eps: float = 1e-12):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, dim = -1)

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class TextShiftPredictor(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = config["input_dim"]
        self.image_channel = config["image_channel"] # the channel of latent space
        self.image_size = config["image_size"] # the dimention of latent space
        self.cond_model = config["cond_model"]
        
        # self.predictor = nn.Linear(self.input_dim, self.image_channel * self.image_size * self.image_size * self.image_size)
        
        self.predictor = nn.Sequential(
            nn.Linear(self.input_dim, 32 * 4 * 4 * 4),
            Swish(),
            View((-1, 32, 4, 4, 4)),
            nn.ConvTranspose3d(32, 16, 4, 2, 1), # 8
            Swish(),
            nn.ConvTranspose3d(16, 16, 1, 1, 0), # 8
            Swish(),
            nn.ConvTranspose3d(16, self.image_channel, 4, 2, 1) # 16
        )
        
    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, -1).contiguous()
        return self.predictor(x).reshape(-1, self.image_channel, self.image_size, self.image_size, self.image_size)

class LayerNorm(nn.Module):
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., use_timestep=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.use_timestep = use_timestep

        if self.use_timestep:
            time_embed_dim = 4 * dim_head
            self.time_emb_proj = nn.Sequential(
                nn.Linear(dim_head, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, 3 * inner_dim),
            )

    def forward(self, x, context=None, t=None, mask=None):
        h = self.heads
        
        bs, c = x.shape[:2]
        # if context is not None:
        #     if torch.isnan(context).any():
        #         import pdb; pdb.set_trace()
        # print("input x.shape: ", x.shape)
        q = self.to_q(x)
        context = default(context, x)

        if context is not None:
            if torch.isnan(context).any():
                import pdb; pdb.set_trace()

        k = self.to_k(context)
        v = self.to_v(context)

        # if torch.isnan(q).any():
        #     import pdb; pdb.set_trace()

        # if torch.isnan(k).any():
        #     import pdb; pdb.set_trace()

        # if torch.isnan(v).any():
        #     import pdb; pdb.set_trace()

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if torch.isnan(sim).any():
            import pdb; pdb.set_trace()

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        if self.use_timestep:
            time_step = timestep_embedding(t, self.dim_head, repeat_only=False)
            time_emb = self.time_emb_proj(time_step).reshape(bs, c, -1)
            out = out + time_emb
        
        return self.to_out(out)



class CrossAttentionShiftPredictor(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_channel = config["image_channel"] # the channel of latent space
        self.image_size = config["image_size"] # the dimention of latent space
        
        self.input_dim = self.image_size ** 3
        self.dim_head = config["dim_head"]
        self.heads = config["heads"]
        self.context_dim = config["context_dim"]
        self.use_timestep = config["use_timestep"]
        
        self.cross_attn = CrossAttention(self.input_dim, context_dim=self.context_dim, heads=self.heads,
                                         dim_head=self.dim_head, use_timestep=self.use_timestep)
    

    def forward(self, x, t=None, context=None):
        B, C = x.shape[:2]
        x = x.reshape(B, C, -1).contiguous()
        context_channel = context.shape[1]
        context = context.reshape(B, context_channel, -1).contiguous()
        
        output = self.cross_attn(x, context=context, t=t)
        output = output.reshape(-1, self.image_channel, self.image_size, self.image_size, self.image_size)
        return output
        
class UNetShiftPredictor(nn.Module):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = ShiftUNet(params)
        
    def forward(self, x, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # eps
        out = self.model(x, t, cond)
        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out
        
class ShiftUNet(nn.Module):
    def __init__(self, unet_params, vq_conf=None, conditioning_key=None):
        """ init method """
        super().__init__()

        self.diffusion_net = UNet3DModel(**unet_params)
        self.conditioning_key = conditioning_key # default for lsun_bedrooms


    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        # x: should be latent code. shape: (bs X z_dim X d X h X w)
        # print("c_concat: ", c_concat, "c_crossattn: ", c_crossattn)
        # print("self.conditioning_key: ", self.conditioning_key)
        if self.conditioning_key is None:
            out = self.diffusion_net(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_net(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_net(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_net(xc, t, context=cc)
            # import pdb; pdb.set_trace()
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_net(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out