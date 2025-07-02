
import torch.nn as nn
import torch
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.1, beta=0.1, gamma=0.01):
        """
        gamma：用于调节MoE load balancing loss权重
        """
        super(CombinedLoss, self).__init__()
        self.class_weights = class_weights
        self.alpha = alpha  # 专家多样性损失权重
        self.beta = beta    # 注意力多样性损失权重
        self.gamma = gamma  # MoE load balancing loss权重

    def forward(self, predictions, targets, expert_outputs, attention_weights, moe_losses):
        # 1. 分类交叉熵损失
        weighted_cross_entropy_loss = F.cross_entropy(predictions, targets, weight=self.class_weights)

        # 2. 专家输出多样性损失
        expert_diversity_loss = 0.0
        for experts in expert_outputs:
            experts = torch.stack(experts)  # (num_experts, ...)
            mean_expert = experts.mean(dim=0, keepdim=True)
            diversity_loss = ((experts - mean_expert) ** 2).mean()
            expert_diversity_loss += diversity_loss

        # 3. 注意力权重多样性损失
        attention_diversity_loss = 0.0
        for att_weight in attention_weights:
            att_weight = torch.stack(att_weight)  # (num_heads, ...)
            mean_weight = att_weight.mean(dim=0, keepdim=True)
            diversity_loss = ((att_weight - mean_weight) ** 2).mean()
            attention_diversity_loss += diversity_loss

        # 4. MoE Load Balancing Loss
        moe_load_balance_loss = 0.0
        for moe_loss in moe_losses:
            moe_load_balance_loss += moe_loss  # 每一层的MoE loss加总

        # 5. 总损失
        total_loss = (weighted_cross_entropy_loss +
                      self.alpha * expert_diversity_loss +
                      self.beta * attention_diversity_loss +
                      self.gamma * moe_load_balance_loss)

        return total_loss

