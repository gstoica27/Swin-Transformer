# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from audioop import bias
from grpc import xds_server_credentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.hybrid_sa import AugmentedWindowAttention
import pdb


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    mechanism: type of attention to do within a window
        - forward: normal self-attention
        - reverse: our reversed self-attention
        - forward_and_reverse: forward and reversed attention done in parallel. Different softmax for each branch.
        - shared_forward_and_reverse: Same as previous but with sharing of the softmax
    """
    def __init__(
        self, 
        dim, 
        window_size, 
        num_heads, 
        qkv_bias=True, 
        qk_scale=None, 
        attn_drop=0., 
        proj_drop=0.,
        mechanism_instructions={'type': 'forward'}
    ):
        super().__init__()
        
        self.qkv_bias = qkv_bias
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.embed_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.mechanism = mechanism_instructions['type']
        self.reduce_reverse = mechanism_instructions.get('reduce_reverse', False)
        self.reverse_dim = head_dim if self.reduce_reverse else dim
        self.reverse_activation = self.reverse_activation_fn(mechanism_instructions.get('reverse_activation', 'none'))
        self.hypernetwork_bias = mechanism_instructions.get('hypernetwork_bias', False)
        self.mechanism_instructions = mechanism_instructions

        self.reverse_parameters = []
        self.forward_parameters = []
        self.attention_parameters = []
        self.output_parameters = []

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.attention_parameters.append(self.relative_position_bias_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.instantiate_scoring_weights()
        self.instantiate_proj_weights()
        self.instantiate_generator_weights()
        self.instantiate_output_weights()

        if self.mechanism_instructions.get('weigh_directions', False):
            self.lmbda = nn.Parameter(torch.zeros(1, dtype=torch.float32, requires_grad=True))
        else:
            self.lmbda = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        # if torch.cuda.is_available(): 
        #         self.lmbda = self.lmbda.cuda()
        self.lmbda_activation = nn.Sigmoid()

        self.orthogonal_loss = nn.L1Loss()

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def instantiate_scoring_weights(self):
        self.qk = nn.Linear(self.dim, self.dim * 2, bias=self.qkv_bias)
        if self.mechanism == 'forward':
            self.forward_qk = self.qk
            # self.forward_parameters.append(self.forward_qk)
        elif self.mechanism == 'reverse':
            self.reverse_qk = self.qk
            # self.reverse_parameters.append(self.reverse_qk)
        elif self.mechanism == 'shared_forward_and_reverse':
            self.forward_qk = self.reverse_qk = self.qk
            # self.reverse_parameters.append(self.reverse_qk)
            # self.forward_parameters.append(self.forward_qk)
        else:
            raise ValueError(f'Unknown attention type: {self.mechanism}')
        self.attention_parameters.append(self.qk)
    
    def instantiate_proj_weights(self):
        if self.mechanism == 'forward':
            self.forward_v = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)
            self.forward_parameters.append(self.forward_v)
        elif self.mechanism == 'reverse':
            if self.mechanism_instructions.get('project_values', True):
                self.reverse_v = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)
                self.reverse_parameters.append(self.reverse_v)
        elif self.mechanism == 'shared_forward_and_reverse':
            self.forward_v = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)
            self.reverse_v = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)
            self.forward_parameters.append(self.forward_v)
            self.reverse_parameters.append(self.reverse_v)
        else:
            raise ValueError(f'Unknown attention type: {self.mechanism}')
        
    def instantiate_generator_weights(self):
        if 'reverse' in self.mechanism:
            if self.mechanism_instructions.get('single_weight_matrix', False):
                self.weight = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.weight_generator = nn.Linear(self.reverse_dim, self.embed_dim * self.embed_dim, bias=False)
            self.reverse_parameters.append(self.weight_generator)
            if self.hypernetwork_bias:
                self.bias_generator = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
                self.reverse_parameters.append(self.bias_generator)
            if self.reduce_reverse and self.mechanism_instructions.get('project_input', True):
                # pdb.set_trace()
                if self.mechanism_instructions.get('value_is_input', False):

                    self.reverse_reducer = self.reverse_v
                else:
                    self.reverse_reducer = nn.Linear(self.dim, self.dim)
                self.reverse_parameters.append(self.reverse_reducer)
    
    def instantiate_output_weights(self):
        self.attn_drop = nn.Dropout(self.attn_drop)
        self.output_proj = nn.Linear(self.dim, self.dim)
        self.output_drop = nn.Dropout(self.proj_drop)
        if self.mechanism == 'forward':
            self.forward_attn_drop = self.attn_drop
            self.forward_proj = self.output_proj
            self.forward_proj_drop = self.output_drop
        elif self.mechanism == 'reverse':
            self.reverse_attn_drop = self.attn_drop
            self.reverse_proj = self.output_proj
            self.reverse_proj_drop = self.output_drop
        elif self.mechanism == 'shared_forward_and_reverse':
            self.forward_attn_drop = self.reverse_attn_drop = self.attn_drop
            self.forward_proj = self.reverse_proj = self.output_proj
            self.reverse_proj_drop = self.forward_proj_drop = self.output_drop
        self.output_parameters += [self.attn_drop, self.output_proj, self.output_drop]

    def reverse_activation_fn(self, reverse_activation):
        if reverse_activation == 'none':
            return lambda x: x
        elif reverse_activation == 'relu':
            return nn.ReLU()
        elif reverse_activation == 'gelu':
            return nn.GELU()
        elif reverse_activation == 'sigmoid':
            return nn.Sigmoid()
        elif reverse_activation == 'tanh':
            return nn.Tanh()

    def apply_forward_attention(self, x, attn):
        BW, K2, C = x.shape
        v = self.forward_v(x).reshape(BW, K2, self.num_heads, C // self.num_heads).transpose(2, 1)
        x = (attn @ v).transpose(1, 2).reshape(BW, K2, C)
        return x

    def apply_reverse_attention(self, x, attn):
        BW, K2, C = x.shape
        if self.mechanism_instructions.get('project_values', True):
            v = self.reverse_v(x)
        else:
            v = x
        v = v.reshape(BW, K2, self.num_heads, C // self.num_heads).transpose(2, 1)
        
        if self.mechanism_instructions.get('transpose_softmax', True):
            attn = attn.transpose(-2, -1)
        if not self.mechanism_instructions.get('activate_hyper_weights', False):
            v = self.reverse_activation(v)
        expert_mixture = (attn @ v) # [BW, h, K2, C/h]
        # pdb.set_trace()
        if self.reduce_reverse:
            if self.mechanism_instructions.get('gen_indiv_hyper_weights', False):
                # pdb.set_trace()
                v_weights = self.weight_generator(v).reshape(BW, self.num_heads, K2, self.embed_dim, self.embed_dim)
                if self.mechanism_instructions.get('activate_hyper_weights', False):
                    v_weights = self.reverse_activation(v_weights)
                weights = (attn @ v_weights.flatten(-2)).reshape(BW, self.num_heads, K2, self.embed_dim, self.embed_dim)
            else:
                weights = self.weight_generator(expert_mixture).reshape(BW, self.num_heads, K2, self.embed_dim, self.embed_dim)
                # pdb.set_trace()
                if self.mechanism_instructions.get('activate_hyper_weights', False):
                    weights = self.reverse_activation(weights)
            
            if self.mechanism_instructions.get('project_input', True):
                v_r = self.reverse_reducer(x)
            else:
                v_r = x

            if self.mechanism_instructions.get('activate_input', True):
                v_r = self.reverse_activation(v_r)
            v_r = v_r.reshape(BW, K2, self.num_heads, self.embed_dim, 1).transpose(2, 1)
            
            if self.mechanism_instructions.get('single_weight_matrix', False):
                output = self.weight(v_r.squeeze(-1)).transpose(2, 1).flatten(2)
            else:
                output = (weights @ v_r).squeeze(-1)
                if self.hypernetwork_bias:
                    biases = self.bias_generator(expert_mixture)
                    output = biases + output
                output = output.transpose(2,1).flatten(2)
        else:
            # import pdb;pdb.set_trace()
            proj_weight = self.weight_generator(x).\
                reshape(BW, K2, self.embed_dim, self.embed_dim)
            output = (expert_mixture.transpose(2, 1) @ proj_weight).reshape(BW, K2, C)

        if self.hypernetwork_bias:
            proj_bias = self.bias_generator(expert_mixture).\
            transpose(2, 1).\
                reshape(BW, K2, C)
            output = output + proj_bias
        
        return output

    def forward(self, x, mask=None):
        # pdb.set_trace()
        BW, K2, C = x.shape
        qk = self.qk(x).reshape(BW, K2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(BW // nW, nW, self.num_heads, K2, K2) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, K2, K2)
            reverse_attn = F.softmax(attn, -2)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            reverse_attn = F.softmax(attn, -2)

        attn = self.attn_drop(attn) # [BW, h, K2, K2]

        if self.mechanism == 'forward':
            output = self.apply_forward_attention(x, attn)
        elif self.mechanism == 'reverse':
            output = self.apply_reverse_attention(x, reverse_attn)
        elif self.mechanism in {'forward_and_reverse', 'shared_forward_and_reverse'}:
            # pdb.set_trace()
            forward_attn = self.apply_forward_attention(x, attn)
            reverse_attn = self.apply_reverse_attention(x, reverse_attn)
            # forward_weight = self.lmbda_activation(self.lmbda)
            # reverse_weight = 1 - forward_weight
            output = forward_attn + reverse_attn
        
        x = self.output_proj(output)
        x = self.output_drop(x)
        return x
    
    def concatenate_linear_parameters(self, layer):
        param = layer.weight.transpose(1, 0)
        if self.qkv_bias:
            param = torch.cat((param, layer.bias.reshape(1, -1)))
        return param
    
    def compute_all_orthogonality_loss(self, weight):
        # pdb.set_trace()
        H_WC = weight.flatten(1)
        HW_C = weight.flatten(0,1)
        CH_W = weight.permute(2, 0, 1).flatten(0,1)

        inner_H_WC = H_WC @ H_WC.transpose(1,0) # [H,WC] x [WC,H] -> [H,H]
        inner_HW_C = HW_C.transpose(1,0) @ HW_C # [C,C]
        inner_CH_W = CH_W.transpose(1,0) @ CH_W # [W,W]

        H_WC_loss = self.orthogonal_loss(inner_H_WC, torch.eye(inner_H_WC.shape[0]).cuda())
        HW_C_loss = self.orthogonal_loss(inner_HW_C, torch.eye(inner_HW_C.shape[0]).cuda())
        CH_W_loss = self.orthogonal_loss(inner_CH_W, torch.eye(inner_CH_W.shape[0]).cuda())
        return H_WC_loss + HW_C_loss + CH_W_loss

    def compute_reversed_orthognality_norms(self):
        weight = self.weight_generator.weight.reshape(self.embed_dim, self.embed_dim, self.embed_dim)
        A = self.concatenate_linear_parameters(self.reverse_v)
        C = self.concatenate_linear_parameters(self.reverse_reducer)

        h_A = torch.stack(torch.split(A, self.embed_dim, dim=1), dim=0) # [h,C+1,C/h]
        h_C = torch.stack(torch.split(C, self.embed_dim, dim=1), dim=0) # [h,C+1,C/h]

        inner_A = torch.bmm(h_A.transpose(2,1), h_A) # [h,C/h,C/h]
        inner_C = torch.bmm(h_C.transpose(2,1), h_C) # [h,C/h,C/h]

        h_Identity = torch.eye(self.embed_dim).unsqueeze(0).tile(self.num_heads, 1, 1)
        if torch.cuda.is_available(): h_Identity = h_Identity.cuda()
        A_loss = self.orthogonal_loss(inner_A, h_Identity)
        C_loss = self.orthogonal_loss(inner_C, h_Identity)
        # pdb.set_trace()
        weight_loss = self.compute_all_orthogonality_loss(weight)

        return {
            'weight': weight_loss,
            'A': A_loss,
            'C': C_loss
        }

    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
    
    def forward_flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

    def reverse_flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 2 * self.dim
        flops += N * self.embed_dim ** 3
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.embed_dim) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.embed_dim)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    
    def flops(self, N):
        return self.reverse_flops(N) + self.forward_flops(N)

class WindowReverseAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0., mechanism_instructions={}):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.embed_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_weight_generator = nn.Linear(self.dim, self.embed_dim * self.embed_dim, bias=False)
        self.proj_bias_generator = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        BW, K2, C = x.shape
        qkv = self.qkv(x).reshape(BW, K2, 3, self.num_heads, C //self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(BW // nW, nW, self.num_heads, K2, K2) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, K2, K2)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).transpose(-2, -1) # [BW, h, K2, K2]
        # pdb.set_trace()
        expert_mixture = (attn @ v) # [BW, h, K2, C/h]
        # import pdb;pdb.set_trace()
        proj_weight = self.proj_weight_generator(x).\
            reshape(BW, K2, self.embed_dim, self.embed_dim)
        proj_bias = self.proj_bias_generator(expert_mixture).\
            transpose(2, 1).\
                reshape(BW, K2, C)
        output_projed = (expert_mixture.transpose(2, 1) @ proj_weight).reshape(BW, K2, C)
        output = output_projed + proj_bias
        
        x = self.proj(output)
        x = self.proj_drop(x)
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
    
    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 2 * self.dim
        flops += N * self.embed_dim ** 3
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.embed_dim) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.embed_dim)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class WindowAttentionOld(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., mechanism_instructions={}):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # v = self.v(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        # q = k = F.normalize(x.reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0], -1)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, mechanism_instructions={'type': 'forward'}):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            mechanism_instructions=mechanism_instructions)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.non_attention_parameters = [
            self.norm1, self.norm2, self.mlp
        ]

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # pdb.set_trace()
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        try:
            assert L == H * W, "input feature has wrong size"
        except:
            pdb.set_trace()

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # if self.attn_mask is not None:
        #     pdb.set_trace()
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, approach_args=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 approach_args={}, mechanism_instructions={'type': 'forward'}):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 mechanism_instructions=mechanism_instructions)
            for i in range(depth)])
        # pdb.set_trace()
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, approach_args=approach_args
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class BiAttnSwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False,
                 reverse_attention_locations=[],
                 mechanism_instructions={'type': 'forward'},
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        # pdb.set_trace()
        self.reverse_attention_layers = []
        for i_layer in range(self.num_layers):
            if i_layer in reverse_attention_locations:
                layer_mechanism_instructions = mechanism_instructions
            else:
                layer_mechanism_instructions = {'type': 'forward'}

            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                    patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                mechanism_instructions=layer_mechanism_instructions
            )
            self.layers.append(layer)

            if i_layer in reverse_attention_locations:
                for block in layer.blocks:
                    self.reverse_attention_layers.append(block.attn)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
    
    def get_reverse_attention_locations(self):
        return self.reverse_attention_layers

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, use_amp=True, return_reverse_layers=False):
        if use_amp:
            with torch.cuda.amp.autocast():
                x = self.forward_features(x)
                x = self.head(x)
        else:
            x = self.forward_features(x)
            x = self.head(x)
        
        if return_reverse_layers:
            reverse_layers = self.reverse_attention_layers
            return x, reverse_layers
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
