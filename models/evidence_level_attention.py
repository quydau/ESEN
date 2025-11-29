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
        n = len(h_e_list)
        
        # Stack thành tensor để xử lý
        # h^(t-1): [n, hidden_dim]
        h_prev = torch.stack(h_e_list, dim=0)
        
        # Tính attention coefficients cho tất cả cặp (i, j)
        # p_ij cho mọi i, j
        
        # Expand để tạo tất cả các cặp
        h_i = h_prev.unsqueeze(1).expand(n, n, self.hidden_dim)  # [n, n, hidden_dim]
        h_j = h_prev.unsqueeze(0).expand(n, n, self.hidden_dim)  # [n, n, hidden_dim]
        
        # Công thức (19): p_ij = W_1(ReLU(W_0([h_ei; h_ej])))
        concat_ij = torch.cat([h_i, h_j], dim=-1)  # [n, n, 2*hidden_dim]
        
        # MLP
        hidden = torch.relu(self.W_0(concat_ij))  # [n, n, hidden_dim]
        p_ij = self.W_1(hidden).squeeze(-1)  # [n, n]
        
        # Công thức (20): a_ij = softmax(p_ij)
        # Normalize theo dimension j (neighbors của node i)
        a_ij = torch.softmax(p_ij, dim=-1)  # [n, n]
        
        # Công thức (21): h^t_ei = Σ_j a_ij * h^(t-1)_ej
        # Aggregation: [n, n] @ [n, hidden_dim] = [n, hidden_dim]
        h_updated = torch.matmul(a_ij, h_prev)  # [n, hidden_dim]
        
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
        # Convert list sang tensor
        h_t = torch.stack(h_e_list, dim=0)  # [n, hidden_dim]
        
        # Propagate qua T layers
        for t, layer in enumerate(self.attention_layers):
            # Convert tensor về list cho layer
            h_list = [h_t[i] for i in range(h_t.size(0))]
            
            # Apply attention layer
            h_t = layer(h_list)  # [n, hidden_dim]
        
        # Công thức (22): H^T_e = [h^T_e1; ...; h^T_en]
        # Đã có dạng [n, hidden_dim], mỗi row là một h^T_ei
        H_T_e = h_t
        
        return H_T_e
