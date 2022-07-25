
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


class GeneralizedWindowAttention(nn.Module):
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
        is_bidirectional=False,
        add_layer_norms=False,
        lambda_value=0.
    ):
        super().__init__()
        
        self.qkv_bias = qkv_bias
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.embed_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.is_bidirectional = is_bidirectional
        self.activation = nn.Identity() # nn.GELU()
        self.add_layer_norms = add_layer_norms
        self.lambda_value = lambda_value

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
        self.instantiate_generator_weights()
        self.instantiate_output_weights()
        
        self.orthogonal_loss = nn.L1Loss()

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def instantiate_scoring_weights(self):
        self.qk = nn.Linear(self.dim, self.dim * 2, bias=self.qkv_bias)
        self.attention_parameters.append(self.qk)
        
    def instantiate_generator_weights(self):
        # if self.is_bidirectional:
        self.G = nn.Linear(1, self.dim * self.embed_dim * self.num_heads, bias=False)
        self.bottleneck = nn.Linear(self.dim, 1, bias=self.qkv_bias)
        if self.qkv_bias:
            self.B = nn.Linear(1, self.dim, bias=False)

        # if self.add_layer_norms:
        #     self.msa_norm = nn.LayerNorm(normalized_shape=[
        #             # self.window_size[0] * self.window_size[1], 
        #             self.dim
        #         ], elementwise_affine=False
        #     )
        #     self.isa_norm = nn.LayerNorm(normalized_shape=[
        #             # self.window_size[0] * self.window_size[1], 
        #             self.dim
        #         ], elementwise_affine=False
        #     )
        #     self.reverse_parameters += [
        #         self.msa_norm,
        #         self.isa_norm
        #     ]

        self.reverse_parameters += [
            # self.selection_lambda,
            self.G, 
            self.B, 
            self.bottleneck,
            # self.input_encoder, 
            # self.generator_encoder,
            # self.output_decoder,
        ]
    
    def instantiate_output_weights(self):
        self.attn_drop = nn.Dropout(self.attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(self.proj_drop)
        
        self.output_parameters += [
            self.attn_drop, 
            self.proj, 
            self.proj_drop,
            # self.msa_drop,
            ]
        
        # if self.is_bidirectional:
        #     self.misa_proj = nn.Linear(self.dim, self.dim)
        #     self.misa_drop = nn.Dropout(self.proj_drop_num)
        #     self.reverse_parameters += [
        #         self.misa_proj,
        #         self.misa_drop,
        #     ]


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

    def complete_projection_weight(self, partial_weights):
        remaining_weight_component = self.output_decoder.weight.reshape(self.num_heads, self.embed_dim, self.embed_dim)
        return torch.einsum('abcde,bef->abcdf', partial_weights, remaining_weight_component)
    
    def forward(self, x, mask=None):
        # pdb.set_trace()
        BW, K2, C = x.shape
        qkv = self.qk(x).reshape(BW, K2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]  # make torchscript happy (cannot use tensor as tuple)
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
        attn = self.attn_drop(attn) # [BW, h, K2, K2]

        queryheaded_x = torch.einsum('abcd,ade->abce', attn, x) # [B, H, Q, D]
        bottleneck = self.bottleneck(x)
        # pdb.set_trace()
        value_proj = self.G(bottleneck).reshape(BW, K2, self.dim, self.num_heads, self.embed_dim).permute(0, 3, 1, 2, 4)
        values = torch.einsum('abcd,abcde->abce', queryheaded_x, value_proj) # [B, H, Q, E]
        if self.qkv_bias:
            value_bias = self.B(bottleneck).reshape(BW, K2, self.num_heads, self.embed_dim).permute(0, 2, 1, 3)
            values += value_bias
        output = values.transpose(1, 2).flatten(2)
            
        x = self.proj(output)
        x = self.proj_drop(x)
        return x
    
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
    
    # self.orthogonal_regularization()
