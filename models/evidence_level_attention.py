# models/evidence_level_attention.py

import torch
import torch.nn as nn
from typing import List


class EvidenceLevelAttention(nn.Module):
    """
    Evidence-level attention layer để propagate thông tin giữa các evidences.
    Mỗi evidence là một node, tạo graph để các evidences interact với nhau.
    Theo công thức (19), (20), (21) trong paper.
    """
    
    def __init__(self, hidden_dim: int = 300):
        """
        Args:
            hidden_dim: Dimension của evidence representations (từ word-level attention)
        """
        super(EvidenceLevelAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # MLP để tính attention coefficient p_ij (công thức 19)
        # Input: concatenation của 2 evidence nodes [h_ei; h_ej]
        self.W_0 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_1 = nn.Linear(hidden_dim, 1)
    
    def forward(self, h_e_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Một layer của evidence-level attention
        
        Args:
            h_e_list: List of [hidden_dim] - n evidence representations
                      (h_e1, ..., h_en) từ word-level attention
        
        Returns:
            h_e_updated: [n, hidden_dim] - Updated evidence representations
        """
        # Accept either list [n, H] or tensor [n, H] or batched tensor [B, n, H]
        if isinstance(h_e_list, list):
            h_prev = torch.stack(h_e_list, dim=0)
        else:
            h_prev = h_e_list

        if h_prev.dim() == 3:
            # Batched: [B, n, H]
            B, n, H = h_prev.size()

            # h_i: [B, n, n, H], h_j same
            h_i = h_prev.unsqueeze(2).expand(-1, -1, n, -1)
            h_j = h_prev.unsqueeze(1).expand(-1, n, -1, -1)

            concat_ij = torch.cat([h_i, h_j], dim=-1)  # [B, n, n, 2H]
            hidden = torch.relu(self.W_0(concat_ij))  # [B, n, n, H]
            p_ij = self.W_1(hidden).squeeze(-1)  # [B, n, n]

            a_ij = torch.softmax(p_ij, dim=-1)  # [B, n, n]

            # Batch matmul: a_ij [B, n, n] @ h_prev [B, n, H] -> [B, n, H]
            h_updated = torch.matmul(a_ij, h_prev)
            return h_updated
        else:
            # [n, H]
            n = h_prev.size(0)
            h_i = h_prev.unsqueeze(1).expand(n, n, self.hidden_dim)
            h_j = h_prev.unsqueeze(0).expand(n, n, self.hidden_dim)
            concat_ij = torch.cat([h_i, h_j], dim=-1)
            hidden = torch.relu(self.W_0(concat_ij))
            p_ij = self.W_1(hidden).squeeze(-1)
            a_ij = torch.softmax(p_ij, dim=-1)
            h_updated = torch.matmul(a_ij, h_prev)
            return h_updated


class EvidenceLevelAttentionStack(nn.Module):
    """
    Stack T layers của evidence-level attention
    Theo mô tả "By stacking T layers..."
    """
    
    def __init__(self, hidden_dim: int = 300, num_layers: int = 2):
        """
        Args:
            hidden_dim: Dimension của evidence representations
            num_layers: Số layers T để stack
        """
        super(EvidenceLevelAttentionStack, self).__init__()
        
        self.num_layers = num_layers
        
        # T attention layers
        self.attention_layers = nn.ModuleList([
            EvidenceLevelAttention(hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, h_e_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Propagate qua T layers
        
        Args:
            h_e_list: List of [hidden_dim] - Initial evidence representations
                      từ word-level attention
        
        Returns:
            H_T_e: [n, hidden_dim] - Final evidence representations (công thức 22)
        """
        # Accept list of tensors or tensor [n, H] or batched [B, n, H]
        if isinstance(h_e_list, list):
            h_t = torch.stack(h_e_list, dim=0)
        else:
            h_t = h_e_list

        # Propagate through layers
        for t, layer in enumerate(self.attention_layers):
            if h_t.dim() == 3:
                # Batched [B, n, H]
                B, n, H = h_t.size()
                h_t = layer(h_t)  # layer handles batched
            else:
                # [n, H], layer expects list or tensor
                h_list = [h_t[i] for i in range(h_t.size(0))]
                h_t = layer(h_list)

        H_T_e = h_t
        return H_T_e
