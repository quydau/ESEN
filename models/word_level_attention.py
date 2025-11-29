# models/word_level_attention.py

import torch
import torch.nn as nn
from typing import List


class WordLevelAttention(nn.Module):
    """
    Word-level attention layer để tính attention weights cho từng từ trong evidence
    dựa trên claim representation.
    Theo công thức (15), (16), (17), (18) trong paper.
    """
    
    def __init__(self, hidden_dim: int = 300, publisher_dim: int = 100):
        """
        Args:
            hidden_dim: Dimension của node embeddings
            publisher_dim: Dimension của publisher embedding
        """
        super(WordLevelAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.publisher_dim = publisher_dim
        
        # W_c: project [w_k_e; h_t_c] để tính attention score (công thức 16)
        # Input dimension = hidden_dim (word) + hidden_dim (claim avg)
        self.W_c = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # W_w: project để tính final attention weights (công thức 17)
        self.W_w = nn.Linear(hidden_dim, 1)
        
        # Publisher embedding (nếu có thông tin publisher)
        # Trong dataset có thể có publisher ID -> embed thành vector
        # Tạm thời giả sử publisher đã được embed sẵn
    
    def forward(self, 
                H_e: torch.Tensor,
                H_c: torch.Tensor,
                P_e: torch.Tensor = None) -> torch.Tensor:
        """
        Tính word-level attention cho một evidence graph
        
        Args:
            H_e: [num_evi_nodes, hidden_dim] - Evidence node embeddings (w_k_e)
            H_c: [num_claim_nodes, hidden_dim] - Claim node embeddings
            P_e: [publisher_dim] - Publisher embedding (optional)
            
        Returns:
            h_e: [hidden_dim + publisher_dim] - Evidence representation với publisher info
        """
        num_evi_nodes = H_e.size(0)
        
        # Công thức (15): h_t_c = (1/n) * Σ H_i_c
        # Tính average của claim nodes
        h_t_c = torch.mean(H_c, dim=0)  # [hidden_dim]
        
        # Expand để concat với mỗi evidence word
        h_t_c_expanded = h_t_c.unsqueeze(0).expand(num_evi_nodes, -1)  # [num_evi_nodes, hidden_dim]
        
        # Công thức (16): p_k = tanh([w_k_e; h_t_c] * W_c)
        # Concatenate evidence word embeddings với claim average
        concat_input = torch.cat([H_e, h_t_c_expanded], dim=-1)  # [num_evi_nodes, 2*hidden_dim]
        p_k = torch.tanh(self.W_c(concat_input))  # [num_evi_nodes, hidden_dim]
        
        # Công thức (17): a_k = exp(p_k * W_w) / Σ exp(p_i * W_w)
        # Tính attention scores
        scores = self.W_w(p_k)  # [num_evi_nodes, 1]
        scores = scores.squeeze(-1)  # [num_evi_nodes]
        
        # Softmax để normalize thành attention weights
        a_k = torch.softmax(scores, dim=0)  # [num_evi_nodes]
        
        # Công thức (18): h_t_e = Σ a_k * w_k_e
        # Weighted sum của evidence words
        a_k_expanded = a_k.unsqueeze(-1)  # [num_evi_nodes, 1]
        h_t_e = torch.sum(a_k_expanded * H_e, dim=0)  # [hidden_dim]
        
        # Thêm publisher information
        if P_e is not None:
            # h_e = [h_t_e; P_e]
            h_e = torch.cat([h_t_e, P_e], dim=0)  # [hidden_dim + publisher_dim]
        else:
            h_e = h_t_e
        
        return h_e


class WordLevelAttentionWrapper(nn.Module):
    """
    Wrapper để apply word-level attention cho tất cả evidences
    """
    
    def __init__(self, hidden_dim: int = 300, publisher_dim: int = 100):
        super(WordLevelAttentionWrapper, self).__init__()
        
        self.word_attention = WordLevelAttention(hidden_dim, publisher_dim)
    
    def forward(self,
                H_c: torch.Tensor,
                H_e_list: List[torch.Tensor],
                P_e_list: List[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Apply word-level attention cho tất cả evidences
        
        Args:
            H_c: [num_claim_nodes, hidden_dim] - Claim representation
            H_e_list: List of [num_evi_nodes, hidden_dim] - Evidence representations
            P_e_list: List of [publisher_dim] - Publisher embeddings (optional)
            
        Returns:
            h_e_list: List of [hidden_dim + publisher_dim] - Updated evidence representations
                      Denoted as (h_e1, ..., h_en) trong paper
        """
        h_e_list = []
        
        for i, H_e in enumerate(H_e_list):
            # Lấy publisher embedding nếu có
            P_e = P_e_list[i] if P_e_list is not None else None
            
            # Apply word-level attention
            h_e = self.word_attention(H_e, H_c, P_e)
            h_e_list.append(h_e)
        
        return h_e_list
