# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

import timm.models.vision_transformer

from model.utils import load_state_dict, download_cached_file
from model.registry import BACKBONE, MODEL


__all__ = [
    'vit_small_patch16_224',
    'vit_base_patch16_224',
    'vit_small_patch16_224_in21k',
    'vit_base_patch16_224_in21k',
    'ViT_Small_P16',
    'ViT_Small_P16_IN21K',
    'ViT_Base_P16',
    'ViT_Base_P16_IN21K'
]


model_urls = {
    'vit_small_patch16_224': 'https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
    'vit_base_patch16_224': 'https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
    'vit_small_patch16_224_in21k': 'https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
    'vit_base_patch16_224_in21k': 'https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
    'vit_large_patch16_224_in21k': 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz'
}




class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.num_features = self.embed_dim

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        else:
            self.fc_norm = nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        
        outcome = self.fc_norm(outcome)
        return outcome
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x



def resize_pos_embed_vit(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    import math
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    # _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb




def _vit(
    arch: str,
    pretrained: bool,
    progress: bool,
    **model_kwargs: Any
    ) -> VisionTransformer:

    model = VisionTransformer(**model_kwargs)
    
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
    return model




@BACKBONE.register
def vit_small_patch16_224(pretrained=False, progress=True, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return _vit('vit_small_patch16_224', pretrained, progress, **model_kwargs)

@BACKBONE.register
def vit_small_patch16_224_in21k(pretrained=False, progress=True, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return _vit('vit_small_patch16_224_in21k', pretrained, progress, **model_kwargs)

@BACKBONE.register
def vit_base_patch16_224(pretrained=False, progress=True, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return _vit('vit_base_patch16_224', pretrained, progress, **model_kwargs)

@BACKBONE.register
def vit_base_patch16_224_in21k(pretrained=False, progress=True, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return _vit('vit_base_patch16_224_in21k', pretrained, progress, **model_kwargs)


@MODEL.register
def ViT_Small_P16(config):
    pretrained = config.pretrained if 'pretrained' in config else True
    return vit_small_patch16_224(pretrained=pretrained, img_size=config.img_size, num_classes=config.num_classes)


@MODEL.register
def ViT_Small_P16_IN21K(config):
    pretrained = config.pretrained if 'pretrained' in config else True
    return vit_small_patch16_224_in21k(pretrained=pretrained, img_size=config.img_size, num_classes=config.num_classes)

@MODEL.register
def ViT_Base_P16(config):
    pretrained = config.pretrained if 'pretrained' in config else True
    return vit_base_patch16_224(pretrained=pretrained, img_size=config.img_size, num_classes=config.num_classes)


@MODEL.register
def ViT_Base_P16_IN21K(config):
    pretrained = config.pretrained if 'pretrained' in config else True
    return vit_base_patch16_224_in21k(pretrained=pretrained, img_size=config.img_size, num_classes=config.num_classes)