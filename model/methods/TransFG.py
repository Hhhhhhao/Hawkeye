import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

from torch.hub import load_state_dict_from_url

from model.utils import load_state_dict, download_cached_file
from model.registry import BACKBONE, MODEL
from model.backbone.vision_transformer import VisionTransformer, model_urls, resize_pos_embed_vit


__all__ = [
    'TransFG_ViT_Small_P16',
    'TransFG_ViT_Small_P16_IN21k',
    'TransFG_ViT_Base_P16',
    'TransFG_ViT_Base_P16_IN21k'
]


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransFGBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        h = x 
        x, a = self.attn(self.norm1(x))
        x = h + self.ls1(x)

        h = x 
        x = self.mlp(self.norm2(x))
        x = h + self.ls2(x)
        return x, a


class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:,:,0,1:]

        _, max_inx = last_map.max(2)
        return _, max_inx


class TransFG(VisionTransformer):
    def __init__(self, global_pool=False, **kwargs):
        kwargs['block_fn'] = TransFGBlock
        if "depth" in kwargs:
            kwargs['depth'] = kwargs['depth'] - 1
        super().__init__(global_pool, **kwargs)
        self.part_select = Part_Attention()
        self.part_layer = TransFGBlock(
                dim=kwargs['embed_dim'],
                num_heads=kwargs['num_heads'],
                mlp_ratio=kwargs['mlp_ratio'],
                qkv_bias=kwargs['qkv_bias'])
        self.part_norm = nn.LayerNorm(kwargs['embed_dim'], eps=1e-6)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_weights = []
        for blk in self.blocks:
            x, a = blk(x)
            attn_weights.append(a)

        part_num, part_inx = self.part_select(attn_weights)
        part_inx = part_inx + 1
        parts = []
        B, num = part_inx.shape
        for i in range(B):
            parts.append(x[i, part_inx[i,:]])
        parts = torch.stack(parts).squeeze(1)
        concat = torch.cat((x[:,0].unsqueeze(1), parts), dim=1)
        part_states, part_weights = self.part_layer(concat)

        if self.global_pool:
            part_states = part_states[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            part_states = self.part_norm(part_states)
            outcome = part_states[:, 0]
        
        outcome = self.fc_norm(outcome)
        return outcome

def _transFG(
    arch: str,
    pretrained: bool,
    progress: bool,
    **model_kwargs: Any
    ) -> TransFG:

    model = TransFG(**model_kwargs)
    
    if pretrained:
        model_url = model_urls[arch]
        if model_url.endswith('.npz'):
            np_path = download_cached_file(model_url, progress=True)
            model.load_pretrained(np_path)
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            posemb_new = model.pos_embed.data
            posemb = state_dict['pos_embed']
            state_dict['pos_embed'] = resize_pos_embed_vit(posemb, posemb_new)
            load_state_dict(model, state_dict)
    del model.norm
    return model


# @BACKBONE.register
def transfg_vit_small_patch16_224(pretrained=False, progress=True, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return _transFG('vit_small_patch16_224', pretrained, progress, **model_kwargs)

# @BACKBONE.register
def transfg_vit_small_patch16_224_in21k(pretrained=False, progress=True, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return _transFG('vit_small_patch16_224_in21k', pretrained, progress, **model_kwargs)

# @BACKBONE.register
def transfg_vit_base_patch16_224(pretrained=False, progress=True, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return _transFG('vit_base_patch16_224', pretrained, progress, **model_kwargs)

# @BACKBONE.register
def transfg_vit_base_patch16_224_in21k(pretrained=False, progress=True, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return _transFG('vit_base_patch16_224_in21k', pretrained, progress, **model_kwargs)

@MODEL.register
def TransFG_ViT_Small_P16(config):
    pretrained = config.pretrained if 'pretrained' in config else True
    return transfg_vit_small_patch16_224(pretrained=pretrained, img_size=config.img_size, num_classes=config.num_classes)


@MODEL.register
def TransFG_ViT_Small_P16_IN21K(config):
    pretrained = config.pretrained if 'pretrained' in config else True
    return transfg_vit_small_patch16_224_in21k(pretrained=pretrained, img_size=config.img_size, num_classes=config.num_classes)

@MODEL.register
def TransFG_ViT_Base_P16(config):
    pretrained = config.pretrained if 'pretrained' in config else True
    return transfg_vit_base_patch16_224(pretrained=pretrained, img_size=config.img_size, num_classes=config.num_classes)


@MODEL.register
def TransFG_ViT_Base_P16_IN21K(config):
    pretrained = config.pretrained if 'pretrained' in config else True
    return transfg_vit_base_patch16_224_in21k(pretrained=pretrained, img_size=config.img_size, num_classes=config.num_classes)



if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    config = CN()
    config.img_size = 224
    config.num_classes = 200
    model = TransFG_ViT_Base_P16_IN21K(config)

    x = torch.randn((2, 3, 224, 224))
    o = model(x)