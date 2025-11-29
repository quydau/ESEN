# models/sentence_level_attention.py

import torch
import torch.nn as nn
from typing import Tuple, List


class SentenceLevelAttention(nn.Module):
    """
    Sentence-level attention layer để tính interaction giữa claim và evidences.
    Mỗi evidence có impact khác nhau lên claim veracity.
    Theo công thức (23), (24), (25) trong paper.
    """
    
    def __init__(self, hidden_dim: int = 300, publisher_dim: int = 100):
        """
        Args:
            hidden_dim: Dimension của evidence và claim representations
            publisher_dim: Dimension của publisher embeddings
        """
        super(SentenceLevelAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.publisher_dim = publisher_dim
        
        # W_c: project concatenation [H_q_e; h_c] (công thức 23)
        # Note: `hidden_dim` passed here may already include `publisher_dim` when
        # this module is constructed by `InformationInteractionNetwork`.
        # Therefore the input dimension should be `hidden_dim * 2` (concatenation
        # of two vectors of size `hidden_dim`), not `hidden_dim * 2 + publisher_dim`.
        self.W_c = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # W_s: tính attention weights (công thức 24)
        self.W_s = nn.Linear(hidden_dim, 1)
    
    def forward(self, 
                H_T_e: torch.Tensor,
                H_c: torch.Tensor,
                P_c: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính sentence-level attention giữa claim và evidences
        
        Args:
            H_T_e: [n, hidden_dim] - Evidence representations từ evidence-level attention
            H_c: [num_claim_nodes, hidden_dim] - Claim node embeddings
            P_c: [publisher_dim] - Claim publisher embedding (optional)
        
        Returns:
            H_r_e: [hidden_dim] - Weighted evidence representation
            H_final: [hidden_dim * 2 + publisher_dim] - Final representation [h_c; H_r_e]
        """
        n = H_T_e.size(0)  # Số evidences
        
        # Tính h_c = [h_t_c; P_c]
        # h_t_c: average của claim nodes (như công thức 15)
        h_t_c = torch.mean(H_c, dim=0)  # [hidden_dim]
        
        if P_c is not None:
            h_c = torch.cat([h_t_c, P_c], dim=0)  # [hidden_dim + publisher_dim]
        else:
            h_c = h_t_c  # [hidden_dim]
        
        # Expand h_c để concat với mỗi evidence
        h_c_expanded = h_c.unsqueeze(0).expand(n, -1)  # [n, hidden_dim + publisher_dim]
        
        # Công thức (23): p_q = tanh([H_q_e; h_c] * W_c)
        concat_input = torch.cat([H_T_e, h_c_expanded], dim=-1)  # [n, 2*hidden_dim + publisher_dim]
        p_q = torch.tanh(self.W_c(concat_input))  # [n, hidden_dim]
        
        # Công thức (24): a_q = exp(p_q * W_s) / Σ exp(p_j * W_s)
        scores = self.W_s(p_q)  # [n, 1]
        scores = scores.squeeze(-1)  # [n]
        a_q = torch.softmax(scores, dim=0)  # [n]
        
        # Công thức (25): H_r_e = Σ a_q * H_q_e
        a_q_expanded = a_q.unsqueeze(-1)  # [n, 1]
        H_r_e = torch.sum(a_q_expanded * H_T_e, dim=0)  # [hidden_dim]
        
        # Final representation: H_final = [h_c; H_r_e]
        H_final = torch.cat([h_c, H_r_e], dim=0)  # [2*hidden_dim + publisher_dim]
        
        return H_r_e, H_final


class InformationInteractionNetwork(nn.Module):
    """
    Complete Information Interaction Network for Evidence Perception (IINEP)
    Kết hợp word-level, evidence-level, và sentence-level attention
    """
    
    def __init__(self, 
                 hidden_dim: int = 300,
                 publisher_dim: int = 100,
                 num_evi_layers: int = 2):
        super(InformationInteractionNetwork, self).__init__()
        
        from models.word_level_attention import WordLevelAttentionWrapper
        from models.evidence_level_attention import EvidenceLevelAttentionStack
        
        # Word-level attention
        self.word_attention = WordLevelAttentionWrapper(hidden_dim, publisher_dim)
        
        # Evidence-level attention (T layers)
        self.evidence_attention = EvidenceLevelAttentionStack(
            hidden_dim + publisher_dim,  # Vì word-level đã concat publisher
            num_evi_layers
        )
        
        # Sentence-level attention
        self.sentence_attention = SentenceLevelAttention(
            hidden_dim + publisher_dim,
            publisher_dim
        )
    
    def forward(self,
                H_c: torch.Tensor,
                H_e_list: List[torch.Tensor],
                P_c: torch.Tensor = None,
                P_e_list: List[torch.Tensor] = None) -> torch.Tensor:
        """
        Complete IINEP forward pass
        
        Args:
            H_c: [num_claim_nodes, hidden_dim] - Claim từ semantic enhancement
            H_e_list: List of [num_evi_nodes, hidden_dim] - Evidences từ semantic refinement
            P_c: [publisher_dim] - Claim publisher
            P_e_list: List of [publisher_dim] - Evidence publishers
        
        Returns:
            H_final: [2*hidden_dim + publisher_dim] - Final representation cho classifier
        """
        # Step 1: Word-level attention
        h_e_list = self.word_attention(H_c, H_e_list, P_e_list)
        # → (h_e1, ..., h_en), mỗi h_e: [hidden_dim + publisher_dim]
        
        # Step 2: Evidence-level attention
        H_T_e = self.evidence_attention(h_e_list)
        # → [n, hidden_dim + publisher_dim]
        
        # Step 3: Sentence-level attention
        H_r_e, H_final = self.sentence_attention(H_T_e, H_c, P_c)
        # → H_final: [2*hidden_dim + publisher_dim]
        
        return H_final
