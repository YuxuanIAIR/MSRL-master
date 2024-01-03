import torch.nn
from torch import nn
from torch.nn.init import xavier_uniform_


class SparseAttentionGate(nn.Module):

    def __init__(self, num_conv_layer=7, channel_num=8):
        super().__init__()

        self.channel_num = channel_num

        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layer):
            self.conv_layers.append(nn.Sequential(            # may add bias
                nn.Conv2d(self.channel_num, self.channel_num, kernel_size=(1, 3), padding=(0, 1)), nn.PReLU()
            ))

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, primitive_weights, bsz, cross_range=0, attn_mask=None, seq_mask=False, eps=1e-5):

        # cross: cross_range > 0 and need attn_mask (may have seq_mask)
        # spatial: cross_range == 0 and attn_mask is given
        # temporal: cross_range == 0 and no attn_mask [self attn (may have seq_mask) or future-past attn]

        # arrange input from (bsz * num_heads, tgt_len, src_len)

        assert self.channel_num == primitive_weights.shape[0] // bsz

        if cross_range > 0:
            assert attn_mask is not None

            agent_num = len(attn_mask)
            query_len = primitive_weights.shape[1] // agent_num
            key_len = primitive_weights.shape[2] // agent_num

            primitive_weights = primitive_weights.reshape([bsz, self.channel_num, query_len, agent_num,
                                                           key_len, agent_num]).transpose(3, 4)
            primitive_weights = primitive_weights.reshape([bsz, self.channel_num,
                                                           query_len * key_len, agent_num, agent_num])

            # select t (need to change due to input crop)
            cross_index = []
            for i in range(query_len):
                for j in range(key_len):
                    if abs(i - j) <= cross_range and (not i == j):
                        if (not seq_mask) or i > j:
                            cross_index.append(key_len * i + j)

            cross_indices = torch.tensor(cross_index, device=primitive_weights.device)
            selected_cross_weights = torch.index_select(primitive_weights, 2, cross_indices).contiguous()
            rearranged_primitive_weights = selected_cross_weights.transpose(1, 2)\
                .reshape([bsz * len(cross_indices), self.channel_num, agent_num, agent_num])

        else:
            rearranged_primitive_weights = primitive_weights.reshape([bsz, self.channel_num, primitive_weights.shape[1],
                                                                      primitive_weights.shape[2]])

        # conv for high-level features
        conv_output = rearranged_primitive_weights
        for i, layer in enumerate(self.conv_layers):
            conv_output = layer(conv_output) + conv_output

        # obligated connections (set some positions as zero in gate) temporal or spatial

        # gate for sparse matrix
        sparse_weights = torch.relu(rearranged_primitive_weights - torch.sigmoid(conv_output))  # sigmoid + relu as gate

        # remove self-self connections for spatial or cross interaction modeling
        if cross_range > 0 or attn_mask is not None:  # imitate agent-aware attention in AgentFormer
            assert rearranged_primitive_weights.shape[-1] == rearranged_primitive_weights.shape[-2]
            self_mask_size = rearranged_primitive_weights.shape[-1]
            self_mask = torch.ones([self_mask_size, self_mask_size], device=conv_output.device) - \
                        torch.eye(self_mask_size, device=conv_output.device)
            self_mask = self_mask.unsqueeze(0).unsqueeze(0)
            sparse_weights = sparse_weights * self_mask

        # obligated connections (set some positions as zero in gate) temporal or spatial

        # reshape back
        if cross_range > 0:
            updated_weights = torch.zeros_like(primitive_weights, device=primitive_weights.device)

            filled_sparse_weights = sparse_weights.reshape([bsz, len(cross_indices), self.channel_num,
                                                            agent_num, agent_num]).transpose(1, 2)  # bsz = 1 or 20
            updated_weights[:, :, cross_indices, :, :] = filled_sparse_weights

            updated_weights = updated_weights.reshape([bsz, self.channel_num, query_len, key_len,
                                                       agent_num, agent_num]).transpose(3, 4)
            updated_weights = updated_weights.reshape(bsz * self.channel_num, query_len * agent_num, key_len * agent_num)

            sparse_weights = updated_weights
        else:
            if attn_mask is not None:  # spatial
                attn_mask = torch.ones_like(attn_mask, device=attn_mask.device)
                sparse_weights = sparse_weights * attn_mask
            else:  # temporal, note that seq_mask only works in self attn rather than future-past attn
                if seq_mask and primitive_weights.shape[1] == primitive_weights.shape[2]:
                    sparse_weights = sparse_weights * torch.tril(torch.ones_like(sparse_weights,  # OK right
                                                                                 device=sparse_weights.device))
            sparse_weights = sparse_weights.reshape([bsz * self.channel_num,
                                                     primitive_weights.shape[1], primitive_weights.shape[2]])

        # zero-softmax
        weight_exp = torch.exp(sparse_weights) - 1
        exp_sum = torch.sum(weight_exp, dim=-1, keepdim=True)
        norm_sparse_weights = weight_exp / (exp_sum + eps)  # eps is for avoiding occurrence of nan

        return norm_sparse_weights
