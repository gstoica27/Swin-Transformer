from __future__ import print_function
from calendar import c
import random
import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from positional_encodings import PositionalEncoding2D
"""
Modified from: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

def is_nan(x):
    nans = x.isnan().sum()
    infs = x.isinf().sum()
    flag = nans + infs
    if flag:
        print(nans, infs)
        pdb.set_trace()

class ConvolutionalSelfAttention(nn.Module):
    def __init__(self,
        spatial_shape,
        filter_size,
        approach_args={'approach_name': 'Four', 'padding': 'valid', 'stride': 1}
    ):
        super(ConvolutionalSelfAttention, self).__init__()
        self.spatial_shape = spatial_shape
        self.input_H, self.input_W, self.spatial_C = spatial_shape
        self.approach_args = approach_args
        self.approach_name = approach_args.APPROACH_NAME
        self.padding_type = approach_args.PADDING
        self.apply_stochastic_stride = approach_args.APPLY_STOCHASTIC_STRIDE
        self.stride = approach_args.STRIDE
        self.softmax_temp = approach_args.SOFTMAX_TEMP
        self.use_residual_connection = approach_args.USE_RESIDUAL_CONNECTION
        self.output_dim = int(approach_args.OUTPUT_SCALAR * self.spatial_C)

        self.filter_K = filter_size
        self.filter_size = self.filter_K * self.filter_K

        self.input_padder = self.compute_padding(self.padding_type)
        self.spatial_H = int(self.input_H + 2 * self.padding_tuple[2])
        self.spatial_W = int(self.input_W + 2 * self.padding_tuple[0])
        self.bn_flag = hasattr(self.approach_args, 'ADD_BATCH_NORM') and self.approach_args.ADD_BATCH_NORM
        self.ln_flag = hasattr(self.approach_args, 'ADD_LAYER_NORM') and self.approach_args.ADD_LAYER_NORM
        if self.bn_flag:
            self.bn = nn.BatchNorm2d(self.spatial_C)
        if not self.bn_flag and self.ln_flag:
            self.bn = nn.LayerNorm([self.spatial_C, self.input_H, self.input_W])
        
        if 'self_attention' in self.approach_name or self.approach_args.POS_EMB_DIM < 0:
            self.pos_emb_dim = self.spatial_C
        else:
            self.pos_emb_dim = self.approach_args.POS_EMB_DIM
        # self.pos_emb_dim = self.spatial_C if 'self_attention' in self.approach_name else self.approach_args.get('pos_emb_dim', 0)

        h_convs, w_convs = self.get_num_convs()
        output_shrunk = h_convs < self.input_H or w_convs < self.input_W
        if self.use_residual_connection and output_shrunk:
            self.upsampler = torch.nn.Upsample(size=(self.input_H, self.input_W), mode='bilinear')
        else:
            self.upsampler = torch.nn.Identity()
        self.local_mask = self.compute_input_mask()
        self.setup_approach()
        self.name2approach = {
            'One': self.efficient_approach1,
            'Two': self.efficient_approach2,
            'Three': self.approach3,
            'Three_kq': self.approach3,
            'Three_unmasked': self.approach3_v2,
            'Three_unmasked_kq': self.approach3_v2,
            'Three_unmasked_cls': self.approach3_v2_cls,
            'Three_unmasked_cls_proj': self.approach3_v2_cls_proj,
            'Three_unmasked_avgHW': self.approach3_v2_avgHW,
            'Three_unmasked_avgU': self.approach3_v2_avgU,
            'Three_unmasked_base': self.approach3_v2_base,
            'Four': self.approach4,
            '5': self.approach5,
            'Four_mem_efficient': self.approach4_mem_efficient,
            'self_attention': self.self_attention,
            'self_attention_no_kq': self.self_attention,
            'local_window_self_attention': self.local_window_self_attention,
            'local_window_self_attention_unmasked': self.local_window_self_attention_unmasked,
            'pytorch_self_attention': self.pytorch_self_attention,
            'reversed_self_attention_elemwise': self.reverse_self_attention,
            'reversed_self_attention_normal': self.reverse_self_attention,
            'background_self_attention': self.background_self_attention,
            'self_attention_cpg': self.self_attention_cpg,
            'kl_div_self_attention': self.kl_div_self_attention,
            'gru_self_attention': self.gru_self_attention,
            'gaussian_blur': self.gaussian_blur,
            'box_filter': self.box_filter
        }

        self.local_mask = self.compute_input_mask()
        if self.approach_name in {
            'Three', 'Three_kq', 'Three_unmasked', 'Four', 'Four_mem_efficient', 
            'Three_unmasked_cls', 'Three_unmasked_cls_proj', 'Three_unmasked_avgU',
            'Three_unmasked_kq'
        }:
            new_shape = [self.num_convs, 1, self.spatial_H, self.spatial_W, 1]                                      # [Nc,1,H,W,1]
            self.local_mask = self.local_mask.reshape(*new_shape)
            self.padding_mask = self.padding_mask.reshape([1, 1, self.spatial_H, self.spatial_W, 1])

        if torch.cuda.is_available():
            self.local_mask = self.local_mask.cuda()
            self.local_indices = self.local_indices.cuda()
            self.padding_mask = self.padding_mask.cuda()

    def compute_padding(self, padding_type):
        if padding_type.lower() == 'valid' or ('self_attention' in self.approach_name and 'local_window' not in self.approach_name):
            padding_tuple = (0, 0, 0, 0)
        elif padding_type.lower() == 'same':
            padding_x = int((self.stride * (self.input_W - 1) - self.input_W + self.filter_K) / 2)
            padding_y = int((self.stride * (self.input_H - 1) - self.input_H + self.filter_K) / 2)
            padding_tuple = (padding_x, padding_x, padding_y, padding_y)
        else:
            raise ValueError('Unknown padding type requested')
        self.padding_tuple = padding_tuple
        input_padder = nn.ConstantPad2d(padding_tuple, 0.)
        # Compute padding indicator mask
        padding_mask = torch.ones((self.input_H, self.input_W))
        self.padding_mask = 1 - input_padder(padding_mask.unsqueeze(0)).squeeze(0)
        return input_padder

    # expects image to be [B,C,H,W]
    def undo_padding(self, image):
        _, _, H, W = image.shape
        left_pad, right_pad, top_pad, bottom_pad = self.padding_tuple
        return image[:, :, top_pad:H-bottom_pad, left_pad:W-right_pad]

    def get_num_convs(self):
        h_dim = int((self.spatial_H - self.filter_K) / self.stride) + 1
        w_dim = int((self.spatial_W - self.filter_K) / self.stride) + 1
        return h_dim, w_dim

    def get_output_shape(self):
        if self.use_residual_connection:
            h_dim, w_dim = self.input_H, self.input_W
        else:
            h_dim, w_dim = self.get_num_convs()
        return [h_dim, w_dim, self.spatial_C]

    def setup_approach(self):
        self.X_encoding_dim = self.spatial_C                                                                            # Call this E
        # Account for positional encodings
        self.maybe_create_positional_encodings()
        if 'self_attention' not in self.approach_name and self.approach_args.get('pos_emb_dim', 0) > 0:
            self.X_encoding_dim += self.pos_emb_dim
        if self.approach_name == 'One':
            self.global_transform = nn.Linear(self.X_encoding_dim, self.filter_K * self.filter_K)
        elif self.approach_name == 'Two':
            self.global_transform = nn.Linear(self.X_encoding_dim, 1)
            self.key_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
            self.query_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
        elif self.approach_name == 'Three_unmasked_base':
            self.global_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
            if self.approach_args.SIMILARITY_METRIC == 'dot_product':
                self.key_proj = nn.Linear(self.X_encoding_dim, self.spatial_C)
                self.query_proj = nn.Linear(self.X_encoding_dim, self.spatial_C)
            if self.approach_args.OUTPUT_PROJECTION:
                self.output_proj = nn.Linear(self.X_encoding_dim, self.spatial_C)
            if self.approach_args.FORGET_GATE_TYPE == 'sigmoid':
                self.forget_gate_nonlinearity = self.apply_local_sigmoid
            elif self.approach_args.FORGET_GATE_TYPE == 'softmax':
                self.forget_gate_nonlinearity = self.apply_local_softmax
        elif self.approach_name in {
            'Three', 'Three_kq', 'Three_unmasked', 'Three_unmasked_cls', 'Three_unmasked_cls_proj', 
            'Three_unmasked_avgHW', 'Three_unmasked_avgU', 'Three_unmasked_kq'
        }:
        # self.approach_name == 'Three' or self.approach_name == 'Three_kq' or self.approach_name == 'Three_unmasked':
            # pdb.set_trace()
            self.global_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
            self.local_transform = nn.Linear(self.X_encoding_dim, self.output_dim)
            # self.key_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
            # self.query_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
            self.indices = np.array([(i, j) for i in range(self.convs_height) for j in range(self.convs_width)])
            self.random_k = self.approach_args.RANDOM_K
            if 'kq' in self.approach_name:
                self.key_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
                self.query_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
            if self.approach_name in {'Three_unmasked_cls', 'Three_unmasked_cls_proj'}:
                self.cls = nn.Parameter(torch.rand(1, self.spatial_C, requires_grad=True))
                if self.approach_name == 'Three_unmasked_cls_proj':
                    self.conv_proj = nn.Linear(self.spatial_C, self.spatial_C * self.spatial_C)
            if self.approach_args.FORGET_GATE_TYPE == 'sigmoid':
                self.forget_gate_nonlinearity = self.apply_local_sigmoid
            elif self.approach_args.FORGET_GATE_TYPE == 'softmax':
                self.forget_gate_nonlinearity = self.apply_local_softmax
            
        elif self.approach_name == 'Four' or self.approach_name == 'Four_mem_efficient':
            self.key_transform = nn.Linear(self.X_encoding_dim, self.X_encoding_dim)
            self.query_transform = nn.Linear(self.X_encoding_dim, self.X_encoding_dim)
            self.value_transform = nn.Linear(self.X_encoding_dim, 1)
        elif self.approach_name == '5':
            self.key_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
            self.query_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
            self.value_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
        elif 'self_attention' in self.approach_name:
            self.key_transform = nn.Linear(self.spatial_C, self.spatial_C)
            self.query_transform = nn.Linear(self.spatial_C, self.spatial_C)
            self.value_transform = nn.Linear(self.spatial_C, self.spatial_C)
            if self.approach_name in [
                'background_self_attention', 
                'local_window_self_attention', 
                'local_window_self_attention_unmasked'
            ]:
                elem_scales_init = torch.rand(self.spatial_H, self.spatial_W, 1, requires_grad=True)
                with torch.no_grad():
                    elem_scales_init = elem_scales_init.view(self.spatial_H * self.spatial_W, 1)
                    elem_scales_init = F.normalize(elem_scales_init, dim=0)
                    elem_scales_init = elem_scales_init.view(self.spatial_H, self.spatial_W, 1)
                self.elem_scales = nn.Parameter(elem_scales_init)
            # elif self.approach_name == 'self_attention_cpg':
            #     value_dim = self.approach_args['value_dim']
            #     if value_dim < 0: value_dim = self.spatial_C
            #     self.value_transform = nn.Linear(self.spatial_C, value_dim)
            #     self.cpg = nn.Linear(self.spatial_C, self.spatial_C * value_dim)
        elif self.approach_name in {'gaussian_blur', 'box_filter'}: pass
        else:
            raise ValueError('Invalid Approach type')

    def maybe_create_positional_encodings(self):
        if self.pos_emb_dim > 0:
            self.positional_encodings = PositionalEncoding2D(self.pos_emb_dim)
        else:
            self.positional_encodings = None

    def masked_softmax(self, vec, mask, dim=1, epsilon=1e-5):
        # Adapted from: https://discuss.pytorch.org/t/apply-mask-softmax/14212/14
        vec = vec * mask
        exps = torch.exp(vec - vec.max(dim, keepdim=True)[0])
        masked_exps = exps * mask
        masked_sums = masked_exps.sum(dim, keepdim=True) #+ epsilon
        return (masked_exps/masked_sums)

    def cudaify_module(self):
        self.local_mask = self.local_mask.cuda()
        self.local_indices = self.local_indices()

    def compute_cell_lengths(self, num_cells, total_length):
        cell_lengths = self.stride * torch.ones(num_cells, dtype=torch.int)

        # convs_height and convs_width are not necessarily divisible by stride
        if self.apply_stochastic_stride:
            leftover_length = total_length - self.stride * num_cells
            for _ in range(leftover_length):
                i = random.randrange(len(cell_lengths))
                cell_lengths[i] += 1

        return cell_lengths

    def compute_input_mask(self):
        unit_stride_convs_height = self.spatial_H - self.filter_K + 1
        unit_stride_convs_width = self.spatial_W - self.filter_K + 1
        
        self.convs_height = int(math.ceil(unit_stride_convs_height / self.stride))
        self.convs_width = int(math.ceil(unit_stride_convs_width / self.stride))
        self.num_convs = self.convs_height * self.convs_width
        # if self.convs_height == 6: pdb.set_trace()
        cell_heights = self.compute_cell_lengths(self.convs_height, unit_stride_convs_height)
        cell_widths = self.compute_cell_lengths(self.convs_width, unit_stride_convs_width)

        cell_row_starts = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int), cell_heights[:-1])), dim=0)
        cell_col_starts = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int), cell_widths[:-1])), dim=0)

        input_mask = torch.zeros(
            (
                self.num_convs,
                self.spatial_H,
                self.spatial_W,
            ),
            dtype=torch.float32)

        self.local_indices = torch.zeros((self.num_convs, self.filter_K * self.filter_K))

        # if self.approach_name == 'Three_unmasked':
        #     return input_mask

        conv_idx = 0
        for i_idx, i in enumerate(cell_row_starts):
            for j_idx, j in enumerate(cell_col_starts):
                offset_i = i + (random.randrange(cell_heights[i_idx]) if self.apply_stochastic_stride else 0)
                offset_j = j + (random.randrange(cell_widths[j_idx]) if self.apply_stochastic_stride else 0)

                input_mask[conv_idx, offset_i:offset_i+self.filter_K, offset_j:offset_j+self.filter_K] = 1.
                self.local_indices[conv_idx] = input_mask[conv_idx].reshape(-1).nonzero().sort()[0].reshape(-1)
                input_mask[conv_idx] = F.relu(input_mask[conv_idx] - self.padding_mask)
                conv_idx += 1
        return input_mask.flatten(1)                                                                                    # [Nc, HW]

    def split_input(self, batch, mask_X_g=True):
        batch_r = batch.unsqueeze(0)                                                                                    # [1,B,H,W,E]
        if mask_X_g:
            X_g = batch_r * (1 - self.local_mask)                                                                       # [F,B,H,W,E]

            X_g = X_g.reshape(self.num_convs, -1, self.spatial_H * self.spatial_W, self.X_encoding_dim)                 # [F,B,HW,E]
            X_gi = torch.argsort(
                (X_g.sum(-1) != 0).type(torch.float32),
                dim=-1
            )[:, :, self.filter_size:].sort(dim=-1)[0]                                                                  # [F,B,(HW-K^2)]
            X_g_flat = X_g.reshape(-1, X_g.shape[2], X_g.shape[3])                                                      # [FB,HW,E]
            X_gi_flat = X_gi.reshape(-1, X_gi.shape[2])                                                                 # [FB,(HW-K^2)]
            X_g_flat_ = X_g_flat[torch.arange(X_g_flat.shape[0]).unsqueeze(-1), X_gi_flat]                              # [FB,(HW-K^2),E]
            X_g_ = X_g_flat_.reshape(
                self.num_convs, -1, X_g_flat_.shape[1], X_g_flat_.shape[2]
            )
        else:
            X_g_ = batch_r.reshape(1, -1, self.spatial_H * self.spatial_W, self.X_encoding_dim)                         # [1,B,HW,E]
        X_l = (batch_r * self.local_mask).reshape(                                                                      # [F,B,H,W,E]
            self.num_convs, -1, self.spatial_H * self.spatial_W, self.X_encoding_dim                                    # [F,B,HW,E]
        )
        X_li = torch.argsort(
            (X_l.sum(-1) != 0).type(torch.float32),
            dim=-1
        )[:, :, -self.filter_size:].sort(dim=-1)[0]                                                                     # [F,B,K^2]
        X_li_flat = X_li.reshape(-1, self.filter_size)                                                                  # [FB,K^2]
        X_l_flat = X_l.reshape(-1, X_l.shape[2], X_l.shape[3])                                                          # [FB,HW,E]
        X_l_flat_ = X_l_flat[torch.arange(X_l_flat.shape[0]).unsqueeze(-1), X_li_flat]                                  # [FB,K^2,E]
        X_l_ = X_l_flat_.reshape(self.num_convs, -1, self.filter_size, self.X_encoding_dim)                             # [F,B,K^2,E]
        return X_l_, X_g_

    def cosine_similarity(self, x, y):
        # Assume x, y are [F,B,*,E]
        x_normed = torch.nn.functional.normalize(x, dim=-1)
        y_normed = torch.nn.functional.normalize(y, dim=-1)
        if len(x.shape) == 3:
            y_permuted = y_normed.transpose(2, 1)
            res = torch.bmm(x_normed, y_permuted)
        else:
            y_permuted = y_normed.transpose(3, 2)
            res = torch.einsum('ijkl,ijla->ijka', x_normed, y_permuted)
        return res

    def efficient_approach1(self, batch):
        batch = self.maybe_add_positional_encodings(batch)
        batch_flat = batch.flatten(1,2)                                                                                 # [B,H,W,C] -> [B,HW,C]
        gX = self.global_transform(batch_flat)                                                                          # [B,HW,C] -> [B,HW,K^2]
        global_mask = (1 - self.padding_mask.reshape(1, -1)) - self.local_mask                                          # [Nc,HW]
        convolutions = torch.einsum(
            'ijk,kl->ijl',
            gX.transpose(2, 1),                                                                                         # [B,HW,K^2] -> [B,K^2,HW]
            global_mask.transpose(1,0)                                                                                  # [Nc,HW] -> [HW,Nc]
            ).transpose(2,1)                                                                                            # [B,K^2,HW]x[HW,Nc] -> [B,K^2,Nc] -> [B,Nc,K^2]
        local_idxs_onehot = F.one_hot(self.local_indices.type(torch.int64))                                             # OneHot([Nc,HW]) -> [Nc,K^2,HW]
        Xl = torch.einsum('ijk,lmj->ilmk', batch_flat, local_idxs_onehot.type(torch.float))                             # [B,HW,C],[Nc,K^2,HW] -> [B,Nc,K^2,C]
        result = torch.einsum('abcd,abc->abd', Xl, convolutions)                                                        # [B,Nc,K^2,C]x[B,Nc,K^2] -> [B,Nc,C]
        return result[:,:,:self.spatial_C].reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
            )                                                                                                           # [B,F,F,C]

    def maybe_add_positional_encodings(self, batch):
        if self.positional_encodings is not None:
            positional_encodings = self.positional_encodings(
                batch[:, :, :, :self.pos_emb_dim]                                                                       # [B,H,W,P]
            )
            if 'self_attention' in self.approach_name or self.approach_args.get('pos_emb_dim', 0) < 0:
                return batch + positional_encodings
            return torch.cat((batch, positional_encodings), dim=-1)                                                     # [[B,H,W,C];[B,H,W,P]] -> [B,H,W,C+P]
        return batch

    def efficient_approach2(self, batch):
        # Add positional Encodings
        batch_pos = self.maybe_add_positional_encodings(batch)
        # Get global mask
        global_mask = (1 - self.padding_mask.reshape(1, -1)) - self.local_mask                                          # [Nc,HW]
        batch_flat = batch_pos.flatten(1, 2)                                                                            # [B,H,W,C] -> [B,HW,C]
        keys = self.key_transform(batch_flat)                                                                           # [B,HW,C]
        queries = self.query_transform(batch_flat)                                                                      # [B,HW,C]
        values = self.global_transform(batch_flat)                                                                      # [B,HW,C] -> [B,HW,1]
        # score = self.cosine_similarity(batch_flat, batch_flat)                                                        # [B,HW,C]x[B,HW,C] -> [B,HW,HW]
        score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])                                 # [B,HW,C] x [B,C,HW] -> [B,HW,HW]
        
        """
        ------------------------------------------------ Explanation ------------------------------------------------
        |       The next several lines compute a memory efficient masked-softmax using our                          |
        |       local and global masks in special ways. We will try to explain what is                              |
        |       happening in the context of the sotmax equation applied in this approach:                           |
        |       z = Σ_i^{HW-K^2} V_i \frac{ e^{ matmul(Q, [K.T]_i) } }{ Σ_j^{HW-K^2} e^{ matmul(Q, [K.T]_j) } }     |
        |         = Σ_i^{HW-K^2} \frac{ V_i x e^{ matmul(Q, [K.T]_i) } }{ Σ_j^{HW-K^2} e^{ matmul(Q, [K.T]_j) } }   |
        |         = \frac{ Σ_i^{HW-K^2} V_i x e^{ matmul(Q, [K.T]_i) } }{ Σ_j^{HW-K^2} e^{ matmul(Q, [K.T]_j) } }   |
        |       F = sigmoid(z)                                                                                      |
        |       Where V_i is a scalar, Q is R^{K^2 x C}, V.T is R^{C x (HW-K^2)}, softmax is R^{K^2 x (HW-K^2)},    |
        |       Z is R^{K^2 x 1}                                                                                    |
        -------------------------------------------------------------------------------------------------------------
        """

        # Subtract maximum for softmax stability & compute softmax numerator: e^{ matmul(Q, [K.T]) }
        # Note however that the computed tensor is R^{HW x HW} rather than R^{K^2 x (HW - K^2)} 
        # as in the above master equation. We take care of this in line 357.
        exp_sim = torch.exp(score - score.max(dim=-1, keepdim=True)[0])                                                 # [B,HW,HW]
        # This computes the softmax denominator: Σ_j^{HW-K^2} e^{ matmul(Q, [K.T]_j) }. 
        # However, it does this for each element in HW, rather than just K^2. We address this in line 357
        exp_sum = torch.matmul(exp_sim, global_mask.transpose(1, 0))                                                    # [B,HW,HW]x[HW,Nc] -> [B,HW,Nc]
        # Invert denominator to obtain \frac{1}{ Σ_j^{HW-K^2} e^{ matmul(Q, [K.T]_j) } } 
        # & add epsilon for numerical stability
        inverted_sum = 1. / (exp_sum + 6.1e-5)                                                                          # [B,HW,Nc]
        # Softly transform denominator from all HW to just K^2 by masking out all sums not in X_l 
        masked_denominator = inverted_sum * self.local_mask.transpose(1,0).unsqueeze(0)                                 # [B,HW,Nc]x([Nc,Hw] -> [HW,Nc] -> [1,HW,Nc]) -> [B,HW,Nc]
        # Compute numerator x value: V_i x e^{ matmul(Q, [K.T]_i) }
        g_sim = values.transpose(2, 1) * exp_sim                                                                        # ([B,HW,1] -> [B,1,HW])x[B,HW,HW] -> [B,HW,HW]
        # Compute numerator sum: Σ_i^{HW-K^2} V_i x e^{ matmul(Q, [K.T]_i) }
        g_sum = torch.matmul(g_sim, global_mask.transpose(1, 0))                                                        # [B,HW,HW]x([B,Nc,HW] -> [B,HW,Nc]) -> [B,HW,Nc]
        # g_sum x masked_denominator gives us z (line 339), sigmoid gives us F (line 340)
        extended_filter = F.sigmoid(g_sum * masked_denominator)                                                         # [B,HW,Nc]
        # Apply filter on X_l. Note that our F is only non-zero for elements corresponding to those in X_l
        result = torch.bmm(batch_flat.transpose(2, 1), extended_filter).transpose(2, 1)                                 # ([B,HW,C] -> [B,C,HW])x[B,HW,Nc] -> [B,C,Nc] -> [B,Nc,C]

        return result[:, :, :self.spatial_C].reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
        )                                                                                                                # [B,Nc,C] -> [B,F,F,C]

    def approach3_v2(self, batch):
        # local_mask = self.local_mask.flatten(1).transpose(1, 0)
        valid_elements = (1 - self.padding_mask).flatten(2, 4)
        X = self.maybe_add_positional_encodings(batch)                                                                  # [B,H,W,E]
        batch_size, H, W, _ = X.shape
        X = X.view(-1, H * W, X.shape[-1])                                                                              # [B,HW,E]
        values = self.global_transform(X)                                                                               # [B,HW,C]
        local_values = self.local_transform(X)
        
        if not self.approach_args.SIMILARITY_METRIC == 'avg_pool':
            if self.approach_args.SIMILARITY_METRIC == 'cosine_similarity':
                X_normed = F.normalize(X, dim=-1)                                                                           # [B,HW,C]
                scores = torch.bmm(X_normed, X_normed.transpose(2, 1))                                                      # [B,HW,HW]
            elif self.approach_args.SIMILARITY_METRIC == 'dot_product':
                if 'kq' in self.approach_name:
                    keys = self.key_transform(X)
                    queries = self.query_transform(X)
                    denom = math.sqrt(queries.shape[-1])
                else:
                    keys = queries = X
                    denom = 1.
                scores = torch.bmm(queries, keys.transpose(2, 1) / denom)                                                           # [B,HW,HW]
            attn = self.masked_softmax(                                                                                     # [B,HW,HW]
                scores, 
                mask=valid_elements,                                                                                        # Mask out padding indices [1, 1, HW]
                dim=-1, epsilon=1e-5 
            )                                                                                                               # [B,HW,HW]

            filter_vecs = torch.bmm(attn, values)                                                                           # [B,HW,C]
        else:
            filter_vecs = values.mean(1, keepdim=True)
        filter_vals = (filter_vecs * X).sum(-1, keepdim=True)                                                           # [B,HW,C] x [B,HW,C] -> [B,HW,1]
        # weighted_X = self.forget_gate_nonlinearity(filter_vals) * X                                                   # [B,HW,C] x [B,HW,1] -> [B,HW,C]
        # output = torch.matmul(weighted_X.transpose(2, 1), local_mask).transpose(2, 1)                                 # [B,C,HW] x [HW,Nc] -> [B,C,Nc]
        output = self.forget_gate_nonlinearity(filter_raw=filter_vals, pooling_features=local_values)
        # pdb.set_trace()
        return output.reshape(
            batch_size, self.convs_height, self.convs_width, self.output_dim
        )
    
    def approach3_v2_base(self, batch):
        # local_mask = self.local_mask.flatten(1).transpose(1, 0)
        # print(self.padding_mask.shape)
        valid_elements = (1 - self.padding_mask)#.flatten(2, 4)
        X = self.maybe_add_positional_encodings(batch)                                                                  # [B,H,W,E]
        batch_size, H, W, _ = X.shape
        X = X.view(-1, H * W, X.shape[-1])                                                                              # [B,HW,E]
        values = self.global_transform(X)                                                                               # [B,HW,C]
        if not self.approach_args.SIMILARITY_METRIC == 'avg_pool':
            X_normed = F.normalize(X, dim=-1)                                                                               # [B,HW,C]
            if self.approach_args.SIMILARITY_METRIC == 'cosine_similarity':
                scores = torch.bmm(X_normed, X_normed.transpose(2, 1))                                                      # [B,HW,HW]
            elif self.approach_args.SIMILARITY_METRIC == 'dot_product':
                key = self.key_proj(X)
                query = self.query_proj(X)
                query = query / np.sqrt(query.shape[-1])
                scores = torch.bmm(query, key.transpose(2, 1))                                                              # [B,HW,HW]
            attn = self.masked_softmax(                                                                                     # [B,HW,HW]
                scores, 
                mask=valid_elements.flatten().unsqueeze(0).unsqueeze(0),                                                    # Mask out padding indices [1, 1, HW]
                dim=-1, epsilon=1e-5 
            )                                                                                                               # [B,HW,HW]

            filter_vecs = torch.bmm(attn, values)                                                                           # [B,HW,C]
        else:
            filter_vecs = (values.sum(1, keepdim=True) * valid_elements.flatten().unsqueeze(0).unsqueeze(-1)) / valid_elements.sum()
        filter_vals = (filter_vecs * X).sum(-1, keepdim=True)                                                           # [B,HW,C] x [B,HW,C] -> [B,HW,1]
        # weighted_X = self.forget_gate_nonlinearity(filter_vals) * X                                                   # [B,HW,C] x [B,HW,1] -> [B,HW,C]
        # output = torch.matmul(weighted_X.transpose(2, 1), local_mask).transpose(2, 1)                                 # [B,C,HW] x [HW,Nc] -> [B,C,Nc]
        output = self.forget_gate_nonlinearity(filter_raw=filter_vals, pooling_features=X)
        if hasattr(self.approach_args, 'OUTPUT_PROJECTION') and self.approach_args.OUTPUT_PROJECTION:
            output = self.output_proj(output)
        return output.reshape(
            batch_size, self.convs_height, self.convs_width, self.output_dim
        )

    def approach3_v2_cls(self, batch):
        # local_mask = self.local_mask.flatten(1).transpose(1, 0)
        valid_elements = (1 - self.padding_mask).flatten(2, 4)
        X = self.maybe_add_positional_encodings(batch)                                                                  # [B,H,W,E]
        batch_size, H, W, _ = X.shape
        X = X.view(-1, H * W, X.shape[-1])                                                                              # [B,HW,E]
        values = self.global_transform(X)                                                                               # [B,HW,C]

        X_normed = F.normalize(X, dim=-1)                                                                               # [B,HW,C]
        cls_normed = F.normalize(self.cls, dim=-1)                                                                      # [B,1,C]
        if self.approach_args.SIMILARITY_METRIC == 'cosine_similarity':
            scores = torch.matmul(X_normed, cls_normed.transpose(1, 0)).transpose(2, 1)                                 # [B,HW,C] x ([1,HW] -> [HW,1]) -> [B,HW,1] -> [B,1,HW]
        elif self.approach_args.SIMILARITY_METRIC == 'dot_product':
            scores = torch.matmul(X, self.cls.transpose(1, 0)).transpose(2, 1)                                          # [B,HW,C] x ([1,HW] -> [HW,1]) -> [B,HW,1] -> [B,1,HW]
        attn = self.masked_softmax(                                                                                     # [B,1,HW]
            scores, 
            mask=valid_elements,                                                                                        # Mask out padding indices [1, 1, HW]
            dim=-1, epsilon=1e-5 
        )                                                                                                               # [B,1,HW]
        
        filter_vecs = torch.bmm(attn, values)                                                                           # [B,1,C]
        filter_vals = (filter_vecs * X).sum(-1, keepdim=True)                                                           # [B,1,C] x [B,HW,C] -> [B,HW,1]
        output = self.forget_gate_nonlinearity(filter_raw=filter_vals, pooling_features=X)
        return output.reshape(
            batch_size, self.convs_height, self.convs_width, self.spatial_C
        )

    def approach3_v2_avgU(self, batch):
        valid_elements = (1 - self.padding_mask).flatten(2, 4)
        X = self.maybe_add_positional_encodings(batch)                                                                  # [B,H,W,E]
        batch_size, H, W, _ = X.shape
        X = X.view(-1, H * W, X.shape[-1])                                                                              # [B,HW,E]
        values = self.global_transform(X)                                                                               # [B,HW,C]

        X_normed = F.normalize(X, dim=-1)                                                                               # [B,HW,C]
        if self.approach_args.SIMILARITY_METRIC == 'cosine_similarity':
            scores = torch.bmm(X_normed, X_normed.transpose(2, 1))                                                      # [B,HW,HW]
        elif self.approach_args.SIMILARITY_METRIC == 'dot_product':
            scores = torch.bmm(X, X.transpose(2, 1))                                                                    # [B,HW,HW]
        
        attn = self.masked_softmax(                                                                                     # [B,HW,HW]
            scores, 
            mask=valid_elements,                                                                                        # Mask out padding indices [1, 1, HW]
            dim=-1, epsilon=1e-5 
        )                                                                                                               # [B,HW,HW]

        filter_vecs = torch.bmm(attn, values)                                                                           # [B,HW,C]
        filter_vecs = filter_vecs.mean(dim=1, keepdim=True)                                                             # [B,HW,C] -> [B,1,C]
        filter_vals = (filter_vecs * X).sum(-1, keepdim=True)                                                           # [B,1,C] x [B,HW,C] -> [B,HW,1]
        output = self.forget_gate_nonlinearity(filter_raw=filter_vals, pooling_features=X)
        return output.reshape(
            batch_size, self.convs_height, self.convs_width, self.spatial_C
        )
    
    def approach3_v2_cls_proj(self, batch):
         # local_mask = self.local_mask.flatten(1).transpose(1, 0)
        valid_elements = (1 - self.padding_mask).flatten(2, 4)
        X = self.maybe_add_positional_encodings(batch)                                                                  # [B,H,W,E]
        batch_size, H, W, _ = X.shape
        X = X.view(-1, H * W, X.shape[-1])                                                                              # [B,HW,E]
        values = self.global_transform(X)                                                                               # [B,HW,C]

        X_normed = F.normalize(X, dim=-1)                                                                               # [B,HW,C]
        cls_normed = F.normalize(self.cls, dim=-1)
        if self.approach_args.SIMILARITY_METRIC == 'cosine_similarity':
            scores = torch.matmul(X_normed, cls_normed.transpose(1, 0)).transpose(2, 1)                                 # [B,HW,C] x ([1,HW] -> [HW,1]) -> [B,HW,1] -> [B,1,HW]
        elif self.approach_args.SIMILARITY_METRIC == 'dot_product':
            scores = torch.matmul(X, self.cls.transpose(1, 0)).transpose(2, 1)                                          # [B,HW,C] x ([1,HW] -> [HW,1]) -> [B,HW,1] -> [B,1,HW]
        attn = self.masked_softmax(                                                                                     # [B,1,HW]
            scores, 
            mask=valid_elements,                                                                                        # Mask out padding indices [1, 1, HW]
            dim=-1, epsilon=1e-5 
        )                                                                                                               # [B,1,HW]

        filter_vec = torch.bmm(attn, values)                                                                            # [B,1,C]
        filter_vals = (filter_vec * X).sum(-1, keepdim=True)                                                            # [B,1,C] x [B,HW,C] -> [B,HW,1]
        projection_matrices = self.conv_proj(filter_vec.squeeze(1)).reshape(-1, self.spatial_C, self.spatial_C)         # f([B,1,C] -> [B,C]) -> [B,CC] -> [B,C,C]
        X_projected = torch.bmm(X, projection_matrices)                                                                 # [B,HW,C] x [B,C,C]
        output = self.forget_gate_nonlinearity(filter_raw=filter_vals, pooling_features=X_projected)                    # [B,Nc,C]
        return output.reshape(
            batch_size, self.convs_height, self.convs_width, self.spatial_C
        )

    def approach3_v2_avgHW(self, batch):
        X = self.maybe_add_positional_encodings(batch)                                                                  # [B,H,W,E]
        batch_size, H, W, _ = X.shape
        X = X.view(-1, H * W, X.shape[-1])                                                                              # [B,HW,E]
        values = self.global_transform(X)                                                                               # [B,HW,C]
        filter_vecs = values.mean(dim=1, keepdim=True)
        filter_vals = (filter_vecs * X).sum(-1, keepdim=True)                                                           # [B,1,C] x [B,HW,C] -> [B,HW,1]
        output = self.forget_gate_nonlinearity(filter_raw=filter_vals, pooling_features=X)
        return output.reshape(
            batch_size, self.convs_height, self.convs_width, self.spatial_C
        )

    def gaussian_blur(self, batch):
        X = batch.permute(0, 3, 1, 2)
        # output = self.gaussian_blur(X)
        # pdb.set_trace()
        output = torchvision.transforms.functional.gaussian_blur(X, kernel_size=self.filter_K)
        residual = self.undo_padding(output)
        return residual.permute(0, 2, 3, 1)
    
    def box_filter(self, batch):
        X = batch.permute(0, 3, 1, 2)
        output = F.avg_pool2d(X, kernel_size=self.filter_K, stride=self.stride)
        # output = self.avg_filter(X)
        return output.permute(0, 2, 3, 1)

    def apply_local_softmax(self, filter_raw, pooling_features):
        """
        Apply a softmax filter only over each receptive field. 
        The logic here closely follows that of approach2. 
        Please see that documentation for a better understanding of what is happening here.
        """
        local_mask = self.local_mask.flatten(1).transpose(1, 0)                                                         # [Nc,HW] -> [HW,Nc]

        filter_exp = torch.exp(filter_raw - filter_raw.max(dim=1, keepdim=True)[0]).squeeze(-1)                         # [B,HW]
        filter_exp_sum = torch.matmul(filter_exp, local_mask)                                                           # [B,Nc]
        filter_exp_sum_inv = 1. / (filter_exp_sum + 6.1e-5)                                                             # [B,Nc]
        exp_weighted_features = pooling_features * filter_exp.unsqueeze(-1)                                             # [B,HW,C] x [B,HW,1] -> [B,HW,C]
        exp_weighted_features_sum = torch.matmul(exp_weighted_features.transpose(2, 1), local_mask).transpose(2, 1)     # ([B,HW,C] -> [B,C,HW]) x [HW,Nc] -> [B,C,Nc] -> [B,Nc,C]
        weighted_features = exp_weighted_features_sum * filter_exp_sum_inv.unsqueeze(-1)                                # [B,Nc,C] x ([B,Nc] -> [B,Nc,1]) -> [B,Nc,C]
        return weighted_features
    
    def apply_local_sigmoid(self, filter_raw, pooling_features):
        local_mask = self.local_mask.flatten(1).transpose(1, 0)
        weighted_X = F.sigmoid(filter_raw) * pooling_features                                                           # [B,HW,C] x [B,HW,1] -> [B,HW,C]
        output = torch.matmul(weighted_X.transpose(2, 1), local_mask).transpose(2, 1)                                   # [B,C,HW] x [HW,Nc] -> [B,C,Nc] -> [B,Nc,C]
        return output
        
    def approach3(self, batch):
        if self.approach_name == 'Three_unmasked':
            local_mask = torch.zeros(
                (
                    self.num_convs,
                    self.spatial_H,
                    self.spatial_W,
                ),
                dtype=torch.float32).cuda().reshape(
                    self.num_convs, 1, self.spatial_H, self.spatial_W, 1
                )
        else:
            local_mask = self.local_mask
        global_mask = (1 - self.padding_mask - local_mask).flatten(
            start_dim=1, end_dim=-1).reshape(
                self.convs_height, self.convs_width, -1)                                                                # [Nc,1,H,W,1]
        X = self.maybe_add_positional_encodings(batch)                                                                  # [B,H,W,E]
        batch_size, H, W, _ = X.shape
        X_flat_spatial = X.view(-1, H * W, X.shape[-1])                                                                 # [B,HW,E]
        X_g_vectors = self.global_transform(X_flat_spatial)                                                             # [B,HW,C]

        # convs_height = self.convs_height
        # convs_width = self.convs_width
        # For cosine similarity
        X_normed = torch.nn.functional.normalize(X, dim=-1)

        keys = X_normed
        queries = X_normed
        denom = 1.
        if self.approach_name == 'Three_kq':
            keys = self.key_transform(X)
            queries = self.query_transform(X)
            denom = math.sqrt(queries.shape[-1])
        elif self.approach_args.SIMILARITY_METRIC == 'dot_product':
            keys = X
            queries = X
            denom = 1.

        # output = torch.zeros(batch_size, convs_height, convs_width, self.spatial_C, dtype=torch.float).cuda()           # [B,F,F,C]
        # TODO: Hack. Assumes Padding is 'same'
        # pdb.set_trace()
        output = torch.clone(self.undo_padding(batch.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        indices = self.indices
        if 'random_k' in self.approach_args and self.random_k > 0:
            indices = np.random.permutation(indices)[:self.random_k]
        for i, j in indices:
        # for i in range(0, convs_height, self.stride):
        #     for j in range(0, convs_width, self.stride):

                X_l = X[:, i:i+self.filter_K, j:j+self.filter_K] / denom                                                # [B,K,K,C]
                raw_compatibilities = torch.einsum(
                    'bhwc,bkjc->bhwkj', 
                    keys, 
                    queries[:, i:i+self.filter_K, j:j+self.filter_K] / denom
                )                                                                                                       # [B,H,W,K,K]
                raw_compatibilities = raw_compatibilities.view(-1, H * W, self.filter_size)                             # [B,HW,K^2]
                compatabilities = self.masked_softmax(
                    self.softmax_temp * raw_compatibilities,
                    global_mask[i, j].unsqueeze(0).unsqueeze(-1),
                    dim=1,
                    epsilon=1e-5
                    )
                W_g = torch.bmm(compatabilities.transpose(2, 1), X_g_vectors)                                           # ([B,HW,K^2] -> [B,K^2,HW]) x [B,HW,C] -> [B,K^2,C]
                X_l_flat_spatial = X_l[:,:,:,:self.spatial_C].reshape(-1, self.filter_size, self.spatial_C)             # [B,K^2,C]
                forget_gate = self.forget_gate_nonlinearity(torch.sum(W_g * X_l_flat_spatial, dim=-1, keepdim=True))    # [B,K^2,1]
                output[:, i, j] = (forget_gate * X_l_flat_spatial).sum(dim=1)                                           # [B,K^2,1] x [B,K^2,C] -> [B,K^2,C] -> [B,C]
        return output

    def approach4(self, batch):
        global_mask = (1 - self.padding_mask - self.local_mask).flatten(start_dim=1, end_dim=-1)                        # [Nc,1,H,W,1] -> [Nc,HW,1]
        batch_pos = self.maybe_add_positional_encodings(batch)
        X_l, X_g = self.split_input(batch_pos, mask_X_g=False)

        keys = self.key_transform(X_g)                                                                                  # [1,B,HW,E]
        queries = self.query_transform(X_l)                                                                             # [F,B,K^2,E]
        values = self.value_transform(X_g)                                                                              # [1,B,HW,1]

        raw_compatibilities = self.cosine_similarity(
            keys, queries                                                                                               # [1,B,HW,E], [F,B,K^2,E]
        )                                                                                                               # [F,B,HW,K^2]
        compatabilities = self.masked_softmax(
            self.softmax_temp * raw_compatibilities,
            global_mask.unsqueeze(1).unsqueeze(-1),
            dim=2
        )
        elem_mul = values * compatabilities                                                                             # [1,B,HW,1] x [F,B,HW,K^2] -> [F,B,HW,K^2]
        W_g = elem_mul.sum(dim=2).unsqueeze(-1)                                                                         # [F,B,HW,K^2] -> [F,B,K^2] -> [F,B,K^2,1]
        X_l = X_l[:, :, :, :self.spatial_C]                                                                             # [F,B,K^2,E] -> [F,B,K^2,C]
        convolved_X = (W_g * X_l).sum(dim=2).permute(1,0,2)                                                             # [F,B,K^2,1] x [F,B,K^2,C] -> [F,B,K^2,C] -> [F,B,C] -> [B,F,C]
        output = convolved_X.reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
        )                                                                                                               # [B,F,C] -> [B,F_H,F_W,C]
        return output

    def approach4_mem_efficient(self, batch):
        global_mask = (1 - self.padding_mask - self.local_mask).flatten(                                                # [Nc,1,H,W,1]
            start_dim=1, end_dim=-1).reshape(                                                                           # [Nc, HW]
                self.convs_height, self.convs_width, -1)                                                                # [F,F,HW]
        X = self.maybe_add_positional_encodings(batch)                                                                  # [B,H,W,E]
        batch_size, H, W, _ = X.shape
        keys = torch.nn.functional.normalize(self.key_transform(X), dim=-1)
        all_queries = torch.nn.functional.normalize(self.query_transform(X), dim=-1)
        X_flat_spatial = X.view(-1, H * W, X.shape[-1])                                                                 # [B,HW,E]
        values = self.value_transform(X_flat_spatial)                                                                   # [B,HW,1]

        convs_height = (H - self.filter_K) // self.stride + 1
        convs_width = (W - self.filter_K) // self.stride + 1

        output = torch.zeros(batch_size, convs_height, convs_width, self.spatial_C, dtype=torch.float).cuda()           # [B,F,F,C]
        for i in range(0, convs_height, self.stride):
            for j in range(0, convs_width, self.stride):
                queries = all_queries[:, i:i+self.filter_K, j:j+self.filter_K]                                          # [B,K,K,E]
                raw_compatibilities = torch.einsum('bhwe,bkje->bhwkj', keys, queries)                                   # [B,H,W,K,K]
                raw_compatibilities[:, i:i+self.filter_K, j:j+self.filter_K, :, :] = 0
                raw_compatibilities = raw_compatibilities.view(-1, H * W, self.filter_size)                             # [B,HW,K^2]
                compatabilities = self.masked_softmax(
                    self.softmax_temp * raw_compatibilities,
                    global_mask[i, j].unsqueeze(0).unsqueeze(-1),
                    dim=1,
                    epsilon=1e-5
                    )
                elem_mul = values * compatabilities                                                                     # [B,HW,1] x [B,HW,K^2] -> [B,HW,K^2]
                W_g = elem_mul.sum(dim=1).view(-1, self.filter_K, self.filter_K).unsqueeze(-1)                          # [B,HW,K^2] -> [B,K^2] -> [B,K,K] -> [B,K,K,1]
                X_l = X[:, i:i+self.filter_K, j:j+self.filter_K, :self.spatial_C]                                       # [B,K,K,C]
                output[:, i, j] = (W_g * X_l).sum(dim=(1,2))                                                            # [B,K,K,1] x [B,K,K,C] -> [B,K,K,C] -> [B,C]
        return output

    def approach5(self, batch):
        batch = self.maybe_add_positional_encodings(batch)
        global_mask = (1 - self.padding_mask.reshape(1, -1)) - self.local_mask                                          # [Nc,HW]
        queries = self.query_transform(batch).flatten(1, 2)                                                             # [B,HW,C]
        values = self.value_transform(batch).flatten(1, 2)                                                              # [B,HW,C]
        keys = self.key_transform(batch).flatten(1, 2)                                                                  # [B,HW,C]

        cos_sim = self.cosine_similarity(queries, values)                                                               # [B,HW,HW]
        global_sims = torch.einsum('ijk,kl->ijl', cos_sim, global_mask.T)                                               # [B,HW,Nc]
        nand_indicator = F.sigmoid(-global_sims)                                                                        # [B,HW,Nc]
        local_nands = nand_indicator * self.local_mask.transpose(1, 0).unsqueeze(0)                                     # [B,HW,Nc]
        grouped_localities = torch.bmm(keys.permute(0, 2, 1), local_nands)                                              # [B,C,HW]x[B,HW,F]
        output = grouped_localities.permute(0, 2, 1).reshape(-1, self.convs_height, self.convs_width, self.spatial_C)   # [B,C,F] -> [B,F,C] -> [B,F,F,]
        return output

    # adapted from https://github.com/sooftware/attentions/blob/master/attentions.py
    def self_attention(self, batch):
        B = batch.shape[0]
        batch_pos = self.maybe_add_positional_encodings(batch)
        values = self.value_transform(batch_pos).flatten(1, 2)                                                          # [B,HW,C]
        if self.approach_name == 'self_attention_no_kq':
            queries = batch_pos.flatten(1, 2)                                                                           # [B,HW,C]
            keys = batch_pos.flatten(1, 2)                                                                              # [B,HW,C]
        else:
            queries = self.query_transform(batch_pos).flatten(1, 2)                                                     # [B,HW,C]
            keys = self.key_transform(batch_pos).flatten(1, 2)                                                          # [B,HW,C]

        score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])                                 # [B,HW,C] x [B,C,HW] -> [B,HW,HW]
        attn = F.softmax(score, -1)                                                                                     # [B,HW,HW]
        context = torch.bmm(attn, values)                                                                               # [B,HW,HW] x [B,HW,C] -> [B,HW,C]
        try:
            return context.reshape(B, self.input_H, self.input_W, -1)                                                       # [B,HW,C] -> [B,H,W,C]
        except:
            pdb.set_trace()

    def kl_div_self_attention(self, batch):
        B = batch.shape[0]
        batch_pos = batch
        queries = self.query_transform(batch_pos).flatten(1, 2)                                                         # [B,HW,C]
        values = self.value_transform(batch_pos).flatten(1, 2)                                                          # [B,HW,C]
        keys = self.key_transform(batch_pos).flatten(1, 2)                                                              # [B,HW,C]

        score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])                                 # [B,HW,C] x [B,C,HW] -> [B,HW,HW]
        probs = F.log_softmax(score, -1)                                                                                # [B,HW,HW]
        targets = torch.ones_like(probs).cuda().log_softmax(-1)

        div = F.kl_div(probs, targets, reduction='none', log_target=True).mean(-1, keepdim=True)                                         # [B,HW]
        keep_val = F.sigmoid(div)
        context = keep_val * values
        return context.reshape(B, self.input_H, self.input_W, -1)

    def gru_self_attention(self, batch):
        B = batch.shape[0]
        # batch_pos = batch
        batch_pos = self.maybe_add_positional_encodings(batch)
        queries = self.query_transform(batch_pos).flatten(1, 2)                                                         # [B,HW,C]
        values = self.value_transform(batch_pos).flatten(1, 2)                                                          # [B,HW,C]
        keys = self.key_transform(batch_pos).flatten(1, 2)                                                              # [B,HW,C]

        score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])                                 # [B,HW,C] x [B,C,HW] -> [B,HW,HW]
        attn = F.softmax(score, -1)                                                                                     # [B,HW,HW]
        context = torch.bmm(attn, values)                                                                               # [B,HW,HW] x [B,HW,C] -> [B,HW,C]
        
        probs = F.log_softmax(score, -1)                                                                                # [B,HW,HW]
        targets = torch.ones_like(probs).cuda().log_softmax(-1)
        div = F.kl_div(probs, targets, reduction='none', log_target=True).mean(-1, keepdim=True)                                         # [B,HW]
        keep_val = F.sigmoid(div).reshape(B, self.input_H, self.input_W, -1)
        context = context.reshape(B, self.input_H, self.input_W, -1)                                                    # [B,HW,C] -> [B,H,W,C]
        context = context * keep_val + batch * (1-keep_val)

        return context
        
    def local_window_self_attention(self, batch, include_Xl_in_Xg=False):
        global_mask = (1 - self.padding_mask.reshape(1, -1))                                                            # [1,HW]
        if not include_Xl_in_Xg:
            global_mask = global_mask - self.local_mask                                                                 # [Nc,HW]

        batch_pos = self.maybe_add_positional_encodings(batch)
        # scaled_batch = (batch_pos * self.elem_scales).flatten(1, 2)                                                   # [B,H,W,C] x [B,H,W,C] -> [B,HW,C]
        scaled_batch = (batch_pos).flatten(1, 2)                                                                        # [B,H,W,C] -> [B,HW,C]

        scaled_local = torch.matmul(scaled_batch.permute(0, 2, 1), self.local_mask.transpose(1,0)).permute(0, 2, 1)     # ([B,HW,C] -> [B,C,HW]) x ([Nc,HW] -> [HW,Nc]) -> [B,C,Nc] -> [B,Nc,C]

        local_queries = self.query_transform(scaled_local)                                                               # [B,Nc,C]
        global_values = self.value_transform(batch_pos).flatten(1, 2)                                                    # [B,HW,C]
        global_keys = self.key_transform(batch_pos).flatten(1, 2)                                                        # [B,HW,C]

        local_queries = F.normalize(local_queries, dim=-1)
        global_keys = F.normalize(global_keys, dim=-1)
        score = torch.bmm(local_queries, global_keys.transpose(1, 2)) / math.sqrt(local_queries.shape[-1])              # [B,Nc,C] x ([B,HW,C] -> [B,C,HW]) -> [B,Nc,HW]
        attn = self.masked_softmax(vec=score, mask=global_mask.unsqueeze(0), dim=-1)                                    # [B,Nc,HW]
        convolution = torch.bmm(attn, global_values)                                                                    # [B,Nc,HW] x [B,HW,C] -> [B,Nc,C]
        if convolution.isnan().sum() > 0:
            #pdb.set_trace()
            print('Crap!')
        return convolution.reshape(-1, self.convs_height, self.convs_width, self.spatial_C)

    def local_window_self_attention_unmasked(self, batch):
        return self.local_window_self_attention(batch, include_Xl_in_Xg=True)

    def pytorch_self_attention(self, batch):
        batch_pos = self.maybe_add_positional_encodings(batch).flatten(1, 2).transpose(1, 0)
        in_proj_bias = torch.cat((self.query_transform._parameters['bias'], self.key_transform._parameters['bias'], self.value_transform._parameters['bias']))
        attn_output, attn_output_weights = F.multi_head_attention_forward(
                batch_pos, batch_pos, batch_pos, self.spatial_C, 1, # q, k, v, dim, heads
                None, in_proj_bias, # in_proj and in_proj bias
                None, None, False, # bias_k, bias_v, add_zero_attn
                0, torch.eye(self.spatial_C).cuda(), torch.zeros(self.spatial_C).cuda(), # dropout, out proj weight, out proj bias
                training=True,
                key_padding_mask=None, need_weights=False,
                attn_mask=None, use_separate_proj_weight=True,
                q_proj_weight=self.query_transform._parameters['weight'], k_proj_weight=self.key_transform._parameters['weight'],
                v_proj_weight=self.value_transform._parameters['weight'])

        return attn_output.transpose(1, 0).reshape(-1, self.input_H, self.input_W, self.spatial_C)

    def reverse_self_attention(self, batch):
        batch_pos = self.maybe_add_positional_encodings(batch)
        queries = self.query_transform(batch_pos).flatten(1, 2)                                                         # [B,HW,C]
        values = self.value_transform(batch_pos).flatten(1, 2)                                                          # [B,HW,C]
        keys = self.key_transform(batch_pos).flatten(1, 2)                                                              # [B,HW,C]
        # pdb.set_trace()
        score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])                                 # [B,HW,C] x [B,C,HW] -> [B,HW,HW]
        if self.approach_name == 'reversed_self_attention_normal':
            joint_scores = score.mean(-1, keepdim=True)                                                                 # [B,HW,1]
            expectation_fn = F.softmax(joint_scores, 1)                                                                 # [B,HW,1]
        elif self.approach_name == 'reversed_self_attention_elemwise':
            attn = F.softmax(score, 1)                                                                                  # [B,HW,HW]
            expectation_fn = torch.bmm(attn, values)                                                                    # [B,HW,HW] x [B,HW,C] -> [B,HW,C]

        context = expectation_fn * values                                                                               # [B,HW,C] x [B,HW,C] -> [B,HW,C]
        return context.reshape(-1, self.input_H, self.input_W, self.spatial_C)                                          # [B,H,W,C]

    def background_self_attention(self, batch):
        batch_pos = self.maybe_add_positional_encodings(batch)

        global_mask = (1 - self.padding_mask.reshape(1, -1)) - self.local_mask                                          # [Nc,HW
        scaled_batch = (batch_pos * self.elem_scales).flatten(1, 2)                                                     # [B,H,W,C] x [B,H,W,C] -> [B,HW,C]
        scaled_global = torch.matmul(scaled_batch.permute(0, 2, 1), global_mask.transpose(1,0)).permute(0, 2, 1)        # ([B,HW,C] -> [B,C,HW]) x ([Nc,HW] -> [HW,Nc]) -> [B,C,Nc] -> [B,Nc,C]

        global_queries = self.query_transform(scaled_global)                                                            # [B,Nc,C]
        local_values = self.value_transform(batch_pos).flatten(1, 2)                                                    # [B,HW,C]
        local_keys = self.key_transform(batch_pos).flatten(1, 2)                                                        # [B,HW,C]

        global_queries = F.normalize(global_queries, dim=-1)
        local_keys = F.normalize(local_keys, dim=-1)
        score = torch.bmm(global_queries, local_keys.transpose(1, 2)) / math.sqrt(global_queries.shape[-1])             # [B,Nc,C] x ([B,HW,C] -> [B,C,HW]) -> [B,Nc,HW]
        attn = self.masked_softmax(vec=score, mask=self.local_mask.unsqueeze(0), dim=-1)                                # [B,Nc,HW]
        convolution = torch.bmm(attn, local_values)                                                                     # [B,Nc,HW] x [B,HW,C] -> [B,Nc,C]
        if convolution.isnan().sum() > 0:
            #pdb.set_trace()
            print('Crap!')
        return convolution.reshape(-1, self.input_H, self.input_W, self.spatial_C)

    def self_attention_cpg(self, batch):
        B, H, W, C = batch.shape
        sa_encodings = self.self_attention(batch)                                                                       # [B,HW,D]
        cpg_matrix = self.cpg(batch)                                                                                    # [B,HW,C] x [C,CD] -> [B,HW,CD]
        cpg_matrix = cpg_matrix.reshape(B, H, W, -1, C)                                                                 # [B,HW,CD] -> [B,HW,D,C]
        output = torch.einsum('abcd,abcde->abce', sa_encodings, cpg_matrix)                                             # [B,HW,D] x [B,HW,D,C] -> [B,HW,C]
        return output

    def forward(self, batch):
        """
        Input shape expected to be [B,C,H,W]
        """
        return_3 = False
        if len(batch.shape) == 3:
            return_3 = True
            batch = batch.reshape(
                batch.shape[0], 
                self.spatial_shape[0], 
                self.spatial_shape[1],
                self.spatial_shape[2]
            )
        # print('BATCH SHAPE: {}'.format(batch.shape))
        # print('EXPECTED INPUT SHAPE: {}'.format(self.spatial_shape))
        batch = batch.permute(0, 3, 1, 2)
        batch = self.input_padder(batch)                                                                                # Pad batch for resolution reduction/preservation
        batch = batch.permute(0, 2, 3, 1)                                                                               # [B,C,H,W] -> [B,H,W,C]
        output = self.name2approach[self.approach_name](batch)                                                          # [B,C,F,F] -> [B,F,F,C]
        if self.bn_flag or self.ln_flag:
            output = output.permute(0, 3, 1, 2)
            output = self.bn(output)
            output = output.permute(0, 2, 3, 1)
        if self.use_residual_connection:
            residual = self.undo_padding(batch.permute(0, 3, 1, 2))
            output = output.permute(0, 3, 1, 2)                                                                         # [B,F,F,C] -> [B,C,F,F]
            output = (self.upsampler(output) + residual).permute(0, 2, 3, 1)
        if return_3:
            output = output.flatten(1,2)
        return output

    def flops(self, N=None):

        if 'Three_unmasked_base' == self.approach_name:
            return 0
        elif 'Three' in self.approach_name:
            flops = 0
            # Apply Global Transform
            flops += self.spatial_H * self.spatial_W * self.spatial_C * self.spatial_C
            # Apply Window Attention
            # Input Normalization
            window_flops = self.spatial_H * self.spatial_W * self.spatial_C * 2
            # Similarity Creation
            window_flops += N * self.spatial_W * self.spatial_H * self.spatial_C * self.spatial_C
            # Softmax
            window_flops += N * self.spatial_W * self.spatial_H * 2
            # Filter Creation
            window_flops += N * self.spatial_H * self.spatial_H * self.spatial_H * self.spatial_W
            # Sum Projections
            window_flops += N * self.spatial_H * self.spatial_W
            # Apply Forget Gate & Sum & Sigmoid
            window_flops += N * self.spatial_C * 3
            # Apply Filter & Sum
            window_flops += N * self.spatial_C * 3
            flops += self.num_convs * window_flops
        
        elif self.approach_name == 'Two':
            flops = 0
            # Apply Global Transform
            flops += self.spatial_H * self.spatial_W * self.spatial_C
            # Cosine Similarity
            flops += (self.spatial_H * self.spatial_W * self.spatial_C * 2) + \
                (self.spatial_H * self.spatial_W * self.spatial_C * self.spatial_W * self.spatial_H)
            # Global Mask Applicator
            flops += self.spatial_C * self.num_convs * self.spatial_H * self.spatial_W
            # Inverse & Mask
            flops += self.spatial_W * self.spatial_H * self.spatial_C * 2
            # Apply Filter
            flops += self.spatial_H * self.spatial_W * self.spatial_C
            # Second Global Mask Applicator
            flops += self.spatial_C * self.num_convs * self.spatial_H * self.spatial_W
            # Finish product
            flops += self.spatial_H * self.spatial_W * self.spatial_C * self.num_convs

        return flops


class ConvAttnWrapper(nn.Module):
    def __init__(self, backbone, variant_kwargs):
        super().__init__()
        self.variant_kwargs = variant_kwargs
        # instantiate the backbone
        self.backbone = backbone
        # Obtain ordered list of backbone layers | Layer spatial information
        self.backbone_layers, self.backbone_spatial_shapes = self.backbone.get_network_structure()

        self.network_structure = self.inject_variant()

    def inject_variant(self):
        injection_instructions = self.variant_kwargs['injection_info']
        network_structure = []
        start_point = 0
        for injection_instruction in injection_instructions:
            inject_layer, inject_number, filter_size = injection_instruction
            # Add backbone layers up to injection point
            network_structure += self.backbone_layers[start_point: inject_layer]
            spatial_shape = self.backbone_spatial_shapes[inject_layer-1]
            # Stack network modules
            for i in range(inject_number):
                variant_module = ConvolutionalSelfAttention(
                    spatial_shape=spatial_shape,
                    filter_size=filter_size,
                    approach_args=self.variant_kwargs
                )
                variant_bn = nn.BatchNorm2d(spatial_shape[-1])
                network_structure += [variant_module, variant_bn]
                spatial_shape = variant_module.get_output_shape()

            start_point = inject_layer
        network_structure += self.backbone_layers[start_point:]
        network_structure = nn.Sequential(*network_structure)
        return network_structure

    def forward(self, batch):
        return self.network_structure(batch)
    

    
