
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


class AugmentedWindowAttention(nn.Module):
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
        self.embed_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.mechanism = mechanism_instructions['type']
        self.activation = self.reverse_activation_fn(mechanism_instructions.get('reverse_activation', 'none'))
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
        self.instantiate_generator_weights()
        self.instantiate_output_weights()

        self.orthogonal_loss = nn.L1Loss()

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def instantiate_scoring_weights(self):
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=self.qkv_bias)
        self.attention_parameters.append(self.qkv)
        
    def instantiate_generator_weights(self):
        if 'reverse' in self.mechanism:
            self.G = nn.Linear(self.embed_dim, self.embed_dim * self.embed_dim, bias=False)
            self.reverse_parameters.append(self.G)
            if self.hypernetwork_bias:
                self.bias_generator = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
                self.reverse_parameters.append(self.bias_generator)
            self.local_proj = nn.Linear(self.dim, self.dim)
            self.global_proj = nn.Linear(self.dim, self.dim)
            self.reverse_parameters += [self.local_proj, self.global_proj]
    
    def instantiate_output_weights(self):
        self.attn_drop = nn.Dropout(self.attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(self.proj_drop)
        self.output_parameters += [self.attn_drop, self.proj, self.proj_drop]

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

    def apply_reverse_attention(self, local_inputs, global_inputs):
        BW, K2, C = local_inputs.shape
        global_groups = self.activation(self.global_proj(global_inputs).reshape(BW, K2, self.num_heads, self.embed_dim))
        global_weights = self.G(global_groups).reshape(BW, K2, self.num_heads, self.embed_dim, self.embed_dim)
        local_groups = self.activation(self.local_proj(local_inputs)).reshape(BW, K2, self.num_heads, self.embed_dim)
        output = (local_groups.view(BW, K2, self.num_heads, 1, self.embed_dim) @ global_weights).view(BW, K2, self.num_heads, self.embed_dim)
        if self.hypernetwork_bias:
            biases = self.bias_generator(global_groups)
            output = output + biases
        output = output.flatten(2)
        return output

    def forward(self, x, mask=None):
        # pdb.set_trace()
        BW, K2, C = x.shape
        qkv = self.qkv(x).reshape(BW, K2, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
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
        sa_outputs = (attn @ v).transpose(1, 2).reshape(BW, K2, C)

        if self.mechanism == 'reverse':
            output = self.apply_reverse_attention(x, sa_outputs)
        elif self.mechanism in {'forward_and_reverse', 'shared_forward_and_reverse'}:
            # pdb.set_trace()
            local_refinements = self.apply_reverse_attention(x, sa_outputs)
            output = sa_outputs + local_refinements
        else:
            output = sa_outputs
        
        x = self.proj(output)
        x = self.proj_drop(x)
        return x
    
    def concatenate_linear_parameters(self, layer):
        pdb.set_trace()
        param = layer.weight.transpose(1, 0)
        if self.qkv_bias:
            param = torch.cat((param, layer.bias.reshape(1, -1)))
        return param
    
    def compute_all_orthogonality_loss(self, weight):
        pdb.set_trace()
        H_WC = weight.flatten(1)
        HW_C = weight.flatten(0,1)
        CH_W = weight.permute(2, 0, 1).flatten(0,1)

        inner_H_WC = H_WC @ H_WC.transpose(1,0) # [H,WC] x [WC,H] -> [H,H]
        inner_HW_C = HW_C.transpose(1,0) @ HW_C # [C,C]
        inner_CH_W = CH_W.transpose(1,0) @ CH_W # [W,W]

        H_WC_loss = self.orthogonal_loss(inner_H_WC, torch.eye(inner_H_WC.shape[0]))
        HW_C_loss = self.orthogonal_loss(inner_HW_C, torch.eye(inner_HW_C.shape[0]))
        CH_W_loss = self.orthogonal_loss(inner_CH_W, torch.eye(inner_CH_W.shape[0]))
        return H_WC_loss + HW_C_loss + CH_W_loss

    def compute_reversed_orthognality_norms(self):
        pdb.set_trace()
        weight = self.G.weight.reshape(self.embed_dim, self.embed_dim, self.embed_dim)
        A = self.concatenate_linear_parameters(self.global_proj)
        C = self.concatenate_linear_parameters(self.local_proj)

        h_A = torch.stach(torch.split(A, self.num_heads, dim=1), dim=0) # [h,C+1,C/h]
        h_C = torch.stach(torch.split(C, self.num_heads, dim=1), dim=0) # [h,C+1,C/h]

        inner_A = torch.bmm(h_A.transpose(2,1), h_A) # [h,C/h,C/h]
        inner_C = torch.bmm(h_C.transpose(2,1), h_C) # [h,C/h,C/h]

        h_Identity = torch.eye(self.embed_dim).unsqueeze(0).tile(self.num_heads, 1, 1)
        A_loss = self.orthogonal_loss(inner_A, h_Identity)
        C_loss = self.orthogonal_loss(inner_C, h_Identity)
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
