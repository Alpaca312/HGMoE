
import torch.nn as nn
import torch
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.1, beta=0.1, gamma=0.01):
        """
        gamma：MoE load balancing loss
        """
        super(CombinedLoss, self).__init__()
        self.class_weights = class_weights
        self.alpha = alpha  
        self.beta = beta    
        self.gamma = gamma  # MoE load balancing loss

    def forward(self, predictions, targets, expert_outputs, attention_weights, moe_losses):
        
        weighted_cross_entropy_loss = F.cross_entropy(predictions, targets, weight=self.class_weights)

        
        expert_diversity_loss = 0.0
        for experts in expert_outputs:
            experts = torch.stack(experts)  # (num_experts, ...)
            mean_expert = experts.mean(dim=0, keepdim=True)
            diversity_loss = ((experts - mean_expert) ** 2).mean()
            expert_diversity_loss += diversity_loss

        
        attention_diversity_loss = 0.0
        for att_weight in attention_weights:
            att_weight = torch.stack(att_weight)  # (num_heads, ...)
            mean_weight = att_weight.mean(dim=0, keepdim=True)
            diversity_loss = ((att_weight - mean_weight) ** 2).mean()
            attention_diversity_loss += diversity_loss

        #  MoE Load Balancing Loss
        moe_load_balance_loss = 0.0
        for moe_loss in moe_losses:
            moe_load_balance_loss += moe_loss  
        
        total_loss = (weighted_cross_entropy_loss +
                      self.alpha * expert_diversity_loss +
                      self.beta * attention_diversity_loss +
                      self.gamma * moe_load_balance_loss)

        return total_loss

