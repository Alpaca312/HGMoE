
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import GCNConv, GATConv, NNConv

class EA(GCNConv):
    def __init__(self, in_channels: int, out_channels: int,
                 num_categories: int, improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, num_experts=1, noisy_gating=True, k=1, **kwargs):
        super(EA, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                                    improved=improved, cached=cached, add_self_loops=add_self_loops,
                                                    normalize=normalize, bias=bias, **kwargs)
        self.lin = MoE(input_size=in_channels, output_size=out_channels, num_experts=num_experts,
                       noisy_gating=noisy_gating, k=k)
        self.attn = nn.Parameter(torch.Tensor(num_categories, in_channels))
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:


        category_attn_scores = torch.matmul(x, self.attn.t())  # (num_nodes, num_categories)
        category_attn_scores = torch.softmax(category_attn_scores, dim=-1)

        out = None
        expert_lists = []


        for i in range(self.attn.shape[0]):
            mask = edge_attr[:, 0] == i
            if mask.sum() == 0:
                continue
            edge_index_i = edge_index[:, mask]
            edge_weight_i = edge_attr[mask, 1]


            edge_attention = category_attn_scores[edge_index_i[0]]


            edge_weight_i = edge_weight_i * edge_attention[:, i]

            x_i, expert_list, moe_loss = self.lin(x)
            out_i = self.propagate(edge_index_i, x=x_i, edge_weight=edge_weight_i, size=None)

            if out is None:
                out = out_i
            else:
                out += out_i

            expert_lists.append(expert_list)

        if self.bias is not None:
            out += self.bias

        return out, expert_lists, moe_loss


import torch
import torch.nn as nn


class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, noisy_gating=True, k=1, edge_attr_size=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.k = k
        self.w_gating = nn.Parameter(torch.Tensor(input_size, num_experts))
        self.edge_proj = nn.Linear(edge_attr_size, num_experts)


        #Selection of a suitable expert network based on the type of dataset
        # if expert_type == 'gcn':
        #     self.experts = nn.ModuleList([GCNConv(self.input_size, self.output_size) for _ in range(self.num_experts)])
        # elif expert_type == 'gat':
        #     self.experts = nn.ModuleList([GATConv(self.input_size, self.output_size) for _ in range(self.num_experts)])
        # elif expert_type == 'nnconv':
        #     assert edge_attr_dim is not None, "edge_attr_dim must be provided for NNConv experts"
        #     self.experts = nn.ModuleList(
        #         [NNConv(self.input_size, self.output_size, nn.Linear(edge_attr_dim, self.input_size * self.output_size))
        #          for _ in range(self.num_experts)])
        # else:
        #     raise ValueError("Unsupported expert type")

        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if i % 3 == 0:  # Expert 0, 3, 6...
                self.experts.append(nn.Sequential(
                    nn.Linear(input_size, output_size)
                ))
            elif i % 3 == 1:  # Expert 1, 4, 7...
                self.experts.append(nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.ReLU(),
                    nn.Linear(output_size, output_size)
                ))
            else:  # Expert 2, 5, 8...
                self.experts.append(nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(output_size, output_size),
                    nn.ReLU(),
                    nn.Linear(output_size, output_size)
                ))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_gating)
        nn.init.xavier_uniform_(self.edge_proj.weight)

    def cv_squared(self, x):
        eps = 1e-10
        mean = torch.mean(x)
        variance = torch.var(x)
        return variance / (mean ** 2 + eps)

    def noisy_top_k_gating(self, x, edge_attr=None, train=True):
        clean_logits = torch.einsum('nd,de->ne', x, self.w_gating)
        if edge_attr is not None:
            edge_logits = self.edge_proj(edge_attr)
            clean_logits += edge_logits
        if self.noisy_gating and train:
            noise = torch.randn_like(clean_logits) * 1e-2
            logits = clean_logits + noise
        else:
            logits = clean_logits


        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        top_k_gates = torch.softmax(top_k_logits, dim=-1)
        return top_k_gates, top_k_indices

    def forward(self, x, edge_attr=None):
        gates, indices = self.noisy_top_k_gating(x, edge_attr, train=self.training)
        output_dim = self.experts[0][-1].out_features if isinstance(self.experts[0][-1], nn.Linear) else \
        self.experts[0][0].out_features
        final_output = torch.zeros(x.size(0), output_dim, device=x.device)

        expert_outputs = []

        importance = torch.zeros(self.num_experts, device=x.device)
        load = torch.zeros(self.num_experts, device=x.device)

        for expert_id in range(self.num_experts):
            expert_out = self.experts[expert_id](x)
            expert_outputs.append(expert_out)

        for i in range(self.k):
            idx = indices[:, i]
            gate = gates[:, i].unsqueeze(-1)

            for expert_id in range(self.num_experts):
                mask = (idx == expert_id)
                if mask.sum() == 0:
                    continue
                expert_out = expert_outputs[expert_id][mask]
                final_output[mask] += gate[mask] * expert_out

                importance[expert_id] += gate[mask].sum()
                load[expert_id] += mask.sum()

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= 1e-2

        return final_output, expert_outputs, loss

