
from asyncio.log import logger
from audioop import bias
from grpc import xds_server_credentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pdb


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


class OrthogonalWindowAttention(nn.Module):
    r""" Orthogonal Window based multi-head self attention (W-MSA) module with relative position bias.
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., lambda_value=0.):

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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.lambda_value = lambda_value
        self.selection_lambda = nn.Parameter(torch.tensor(self.lambda_value, requires_grad=True))
        self.q_basis_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # Legacy-Named Arguments. Need to fix naming scheme later
        self.reverse_parameters = [self.q_basis_proj]
        self.forward_parameters = []
        self.attention_parameters = [self.qkv, self.attn_drop]
        self.output_parameters = [self.proj, self.proj_drop]

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q_basis = self.q_basis_proj(x).reshape(B_, N, self.num_heads, C // self.num_heads).transpose(2, 1)

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
            k_on_q = self.attn_drop(self.softmax(attn))
            q_on_k = self.attn_drop(F.softmax(attn, dim=-2))
        else:
            k_on_q = self.attn_drop(self.softmax(attn))
            q_on_k = self.attn_drop(F.softmax(attn, dim=-2))

        # k_on_q = self.attn_drop(k_on_q)
        # x = (k_on_q @ v).transpose(1, 2).reshape(B_, N, C)
        # pdb.set_trace()
        # Compute Standard Orthogonal Attention
        q_basis_squared_norm = torch.square(torch.linalg.vector_norm(q_basis, dim=-1, keepdim=True))    # [B,H,N,1]
        q_basis_unit_norm = (q_basis / q_basis_squared_norm)                                            # [B,N,H,E]/[B,H,N,1]->[B,H,N,E]
        v_on_q_scale = q_basis @ v.transpose(3, 2)                                                      # [B,H,N,E]x([B,H,N,E]->[B,H,E,N])->[B,H,N,N]
        vk_on_q = (k_on_q * v_on_q_scale).sum(-1, keepdim=True)                                         # [B,H,N,N]x[B,H,N,N]->[B,H,N,1]
        q_basis_scaled = vk_on_q * q_basis_unit_norm                                                    # [B,H,N,1]*[B,H,N,E]->[B,H,N,E]
        v_comb = (k_on_q @ v)                                                                           # [B,H,N,N]x[B,H,N,E]->[B,H,N,E]
        v_orths = (v_comb - q_basis_scaled).transpose(1,2).reshape(B_, N, C)                            # [B,H,N,E]-[B,H,N,E]->[B,H,N,E]->[B,N,H,E]->[B,N,C]
        # pdb.set_trace()
        # Compute Inverse Orthogonal Attention
        v_squared_norm = torch.square(torch.linalg.vector_norm(v, dim=-1, keepdim=True))                # [B,H,N,1]
        v_unit_norm = (v / v_squared_norm)                                                              # [B,H,N,E]/[B,H,N,1]->[B,H,N,E]
        q_on_v_scale = v_on_q_scale                                                                     # [B,H,N,N]
        q_on_vk = q_on_k * q_on_v_scale                                                                 # [B,H,N,N]*[B,H,N,N]->[B,H,N,N]
        v_scaled = q_on_vk @ v_unit_norm                                                                # [B,H,N,N]x[B,H,N,E]->[B,H,N,E]
        q_basis_weighted = q_basis * q_on_k.sum(-1, keepdim=True)                                       # [B,H,N,E]*([B,H,N,N]->[B,H,N,1])->[B,H,N,E]
        q_orths = (q_basis_weighted - v_scaled).transpose(2, 1).reshape(B_, N, C)                       # [B,H,N,E]-[B,H,N,E]->[B,H,N,E]->[B,N,H,E]->[B,N,C]
        # pdb.set_trace()
        # Combine Attentions
        convex_weight = F.sigmoid(self.selection_lambda)
        x = convex_weight * v_orths + (1 - convex_weight) * q_orths
        
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
