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
        # Support single evidence [Ne, H] or batched [B, Ne, H]
        if H_e.dim() == 3:
            # H_e: [B, Ne, H], H_c: [B, Nc, H], P_e: [B, pub] or [B, Ne, pub]
            B, Ne, H = H_e.size()

            # h_t_c: average of claim nodes per batch -> [B, H]
            h_t_c = torch.mean(H_c, dim=1)  # [B, H]

            # Expand to [B, Ne, H]
            h_t_c_expanded = h_t_c.unsqueeze(1).expand(-1, Ne, -1)  # [B, Ne, H]

            # Concatenate and project
            concat_input = torch.cat([H_e, h_t_c_expanded], dim=-1)  # [B, Ne, 2H]
            p_k = torch.tanh(self.W_c(concat_input))  # [B, Ne, H]

            scores = self.W_w(p_k).squeeze(-1)  # [B, Ne]
            a_k = torch.softmax(scores, dim=-1)  # [B, Ne]

            a_k_expanded = a_k.unsqueeze(-1)  # [B, Ne, 1]
            h_t_e = torch.sum(a_k_expanded * H_e, dim=1)  # [B, H]

            if P_e is not None:
                # P_e may be [B, pub] (same publisher for evidence) or [B, Ne, pub]
                if P_e.dim() == 2:
                    P_expand = P_e.unsqueeze(1).expand(-1, Ne, -1)[:, 0, :]
                    h_e = torch.cat([h_t_e, P_expand], dim=-1)
                else:
                    # if P_e per-evidence: select first dimension
                    h_e = torch.cat([h_t_e, P_e[:, 0, :]], dim=-1)
            else:
                h_e = h_t_e

            return h_e  # [B, H] or [B, H+pub]
        else:
            # Single sample behavior
            num_evi_nodes = H_e.size(0)

            h_t_c = torch.mean(H_c, dim=0)  # [H]
            h_t_c_expanded = h_t_c.unsqueeze(0).expand(num_evi_nodes, -1)

            concat_input = torch.cat([H_e, h_t_c_expanded], dim=-1)
            p_k = torch.tanh(self.W_c(concat_input))

            scores = self.W_w(p_k).squeeze(-1)
            a_k = torch.softmax(scores, dim=0)
            a_k_expanded = a_k.unsqueeze(-1)
            h_t_e = torch.sum(a_k_expanded * H_e, dim=0)

            if P_e is not None:
                h_e = torch.cat([h_t_e, P_e], dim=0)
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
        # Support batched inputs: H_c [B, Nc, H], H_e_list [B, E, Ne, H]
        if isinstance(H_e_list, dict):
            # In batched prepared format we may receive dict; assume keys
            H_e_nodes = H_e_list['node_features']
            P_e = None
            if P_e_list is not None:
                P_e = P_e_list
            # H_e_nodes: [B, E, Ne, H]
            B, E, Ne, H = H_e_nodes.size()
            h_e_out = []
            for e in range(E):
                H_e_b = H_e_nodes[:, e, :, :]  # [B, Ne, H]
                P_e_b = P_e[:, e, :] if (P_e is not None and P_e.dim() == 3) else (P_e if P_e is not None else None)
                h_e = self.word_attention(H_e_b, H_c, P_e_b)  # [B, H(+pub)]
                h_e_out.append(h_e)
            # Stack into [B, E, H']
            h_e_list = torch.stack(h_e_out, dim=1)
            return h_e_list
        elif isinstance(H_e_list, list):
            # Original behavior: list of tensors per evidence
            h_e_list = []
            for i, H_e in enumerate(H_e_list):
                P_e = P_e_list[i] if P_e_list is not None else None
                h_e = self.word_attention(H_e, H_c, P_e)
                h_e_list.append(h_e)
            return h_e_list
        else:
            # If H_e_list is tensor [B,E,Ne,H]
            H_e_nodes = H_e_list
            B, E, Ne, H = H_e_nodes.size()
            h_e_out = []
            for e in range(E):
                H_e_b = H_e_nodes[:, e, :, :]
                P_e_b = P_e_list[:, e, :] if (P_e_list is not None and P_e_list.dim() == 3) else (P_e_list if P_e_list is not None else None)
                h_e = self.word_attention(H_e_b, H_c, P_e_b)
                h_e_out.append(h_e)
            h_e_list = torch.stack(h_e_out, dim=1)
            return h_e_list
