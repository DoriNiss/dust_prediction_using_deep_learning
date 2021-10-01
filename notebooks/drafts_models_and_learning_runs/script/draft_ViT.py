#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn


B,N,C = 4,9,12
x = torch.rand([B,N,C])
x.shape


dim = 12


qkv = nn.Linear(dim,3*dim,bias=False)


qkv_result = qkv(x)
qkv_result.shape


num_heads = 6


qkv_result_reshaped = qkv_result.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
q,k,v = qkv_result_reshaped[0],qkv_result_reshaped[1],qkv_result_reshaped[2]
q.shape


var = None
# var = 6
a = var or 5
a


from torch import nn as nn
# Patch Embed: https://github.com/rwightman/pytorch-image-models/blob/d3f744065088ca9b6b3a0f968c70e90ed37de75b/timm/models/layers/patch_embed.py#L15
# input: [batch_size,channels_in,H,W] -> output: [batch_size,num_patches,embedding_size] (learnable embedding)
x = torch.rand([10,17,81,81])
img_size = (81,81)
patch_size = (9,9)
grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
print("grid_size:",grid_size)
num_patches = grid_size[0] * grid_size[1]
print("num_patches:",num_patches)
embed_dim = 32 #512
in_chans = 17
proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
x_projected = proj(x)
print(x_projected.shape)
x_projected_flatten = x_projected.flatten(2).transpose(1, 2)
print(x_projected_flatten.shape)


b = None
if b: print("Yes")





x = torch.rand([3,7,81,81])





def patch_embed(x, embed_dim=32, patch_size=(9,9)):
    in_chans = x.shape[1]
    img_size = (x.shape[-2],x.shape[-1])
    grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
    print("grid_size:",grid_size)
    num_patches = grid_size[0] * grid_size[1]
    print("num_patches:",num_patches)
    proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    x_projected = proj(x)
    x_projected_flatten = x_projected.flatten(2).transpose(1, 2)
    print(f"x: {x.shape}, x_prjected: {x_projected.shape}, x_projected_flatten: {x_projected_flatten.shape}")
    return x_projected_flatten

def attention(x,dim=32,num_heads=8):
    B, N, C = x.shape
    head_dim = dim // num_heads
    scale = head_dim ** -0.5
    qkv = nn.Linear(dim,3*dim,bias=False)
    qkv_x = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q,k,v = qkv_x[0],qkv_x[1],qkv_x[2]
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    print(f"attention: out: {x.shape},v: {v.shape}")
    return x

def mlp(x, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    fc1 = nn.Linear(in_features, hidden_features)
    act = act_layer()
    fc2 = nn.Linear(hidden_features, out_features)
    out = fc1(x)
    out = act(out)
    out = fc2(out)
    priunt(f"Mlp: in: {x.shape}, out: {out.shape}")
    return x


x_embed = patch_embed(x)


x_attn = attention(x_embed)







