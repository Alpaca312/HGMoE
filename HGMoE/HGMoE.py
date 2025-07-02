
import torch
from torch import nn
from MoEs import EA

class HGMoE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=256, num_experts=4, noisy_gating=True, k=2, dropout=0.3,
                 use_batch_norm=True, expert_diversity=False, num_categories=3):
        super(HGMoE, self).__init__()
        self.conv1 = EA(in_channels, hidden_dim, num_categories=num_categories, num_experts=num_experts, noisy_gating=noisy_gating, k=k)
        self.conv_mid = EA(hidden_dim, hidden_dim, num_categories=num_categories, num_experts=num_experts, noisy_gating=noisy_gating, k=k)
        self.conv2 = EA(hidden_dim, out_channels, num_categories=num_categories, num_experts=num_experts, noisy_gating=noisy_gating, k=k)
        self.use_batch_norm = use_batch_norm
        self.expert_diversity = expert_diversity
        self.norm_layer_1_output = None
        self.norm_layer_2_output = None
        self.norm_layer_3_output = None
        if use_batch_norm:
            self.norm_layer_1 = nn.BatchNorm1d(in_channels)
            self.norm_layer_2 = nn.BatchNorm1d(hidden_dim)
            self.norm_layer_3 = nn.BatchNorm1d(hidden_dim)

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, edge_index, edge_attr):
        intermediate_features = []
        moe_losses = []

        if self.use_batch_norm:
            x = self.norm_layer_1(x)
        x, expert_list1, moe_loss1 = self.conv1(x, edge_index, edge_attr)
        moe_losses.append(moe_loss1)

        if self.use_batch_norm:
            x = self.norm_layer_2(x)
        x, expert_list2, moe_loss2 = self.conv2(x, edge_index, edge_attr)
        moe_losses.append(moe_loss2)

        intermediate_features.append(x)
        if self.dropout is not None:
            x = self.dropout(x)


        return x, expert_list1, expert_list2, moe_losses

