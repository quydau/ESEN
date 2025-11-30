# models/semantic_refinement.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from models.ggnn import GGNN


class SemanticRefinement(nn.Module):
    """
    Semantic Refinement Module để loại bỏ redundant information trong evidences
    Theo công thức (11), (12), (13), (14) trong paper
    """
    
    def __init__(self, 
                 hidden_dim: int = 300,
                 alpha: float = 0.5,
                 top_k_ratio: float = 0.3,
                 num_ggnn_steps: int = 3):
        """
        Args:
            hidden_dim: Dimension của node features
            alpha: Trọng số cân bằng giữa score1 và score2 (công thức 13)
            top_k_ratio: Tỷ lệ nodes cần loại bỏ (k = top_k_ratio * num_nodes)
            num_ggnn_steps: Số steps cho GGNN refinement
        """
        super(SemanticRefinement, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.top_k_ratio = top_k_ratio
        
        # W_score1 cho công thức (11)
        self.W_score1 = nn.Linear(hidden_dim, 1, bias=False)
        
        # GGNN cho evidence-aware refinement (công thức 14)
        self.ggnn = GGNN(hidden_dim=1, num_steps=num_ggnn_steps)
    
    def compute_self_redundancy(self, H_e: torch.Tensor) -> torch.Tensor:
        """
        Công thức (11): score1 = H_e * W_score1
        
        Args:
            H_e: [num_nodes, hidden_dim] - Evidence node representations
            
        Returns:
            score1: [num_nodes, 1] - Self-redundancy scores
        """
        # Supports H_e: [num_nodes, hidden_dim] or [B, num_nodes, hidden_dim]
        score1 = self.W_score1(H_e)  # [..., num_nodes, 1] or [num_nodes,1]
        return score1
    
    def compute_claim_relevance_redundancy(self, 
                                       H_e: torch.Tensor,
                                       H_c: torch.Tensor) -> torch.Tensor:
        """
        Công thức (12): score_i^2 = -log(Σ_j∈c W_ij · H_cj)
        
        W_ij là relevance giữa evidence node i và claim node j
        Node có relevance CAO với claim → score THẤP → GIỮ LẠI
        Node có relevance THẤP với claim → score CAO → LOẠI BỎ
        
        Args:
            H_e: [num_evi_nodes, hidden_dim] - Evidence representations
            H_c: [num_claim_nodes, hidden_dim] - Claim representations
            
        Returns:
            score2: [num_evi_nodes, 1] - Claim-relevance redundancy scores
        """
        # W_ij = relevance between evidence node i and claim node j
        # [num_evi_nodes, hidden_dim] @ [hidden_dim, num_claim_nodes]
        # Support batched inputs: H_e [num_evi_nodes, H] or [B, num_evi_nodes, H]
        if H_e.dim() == 3:
            # H_e: [B, Ne, H], H_c: [B, Nc, H]
            W = torch.matmul(H_e, H_c.transpose(-2, -1))  # [B, Ne, Nc]
            W = torch.softmax(W, dim=-1)
            weighted_claim = torch.matmul(W, H_c)  # [B, Ne, H]
            relevance_sum = torch.sum(weighted_claim, dim=-1, keepdim=True)  # [B, Ne, 1]
            score2 = -torch.log(relevance_sum + 1e-10)
            return score2
        else:
            W = torch.matmul(H_e, H_c.transpose(0, 1))  # [Ne, Nc]
            W = torch.softmax(W, dim=-1)
            weighted_claim = torch.matmul(W, H_c)  # [Ne, H]
            relevance_sum = torch.sum(weighted_claim, dim=-1, keepdim=True)  # [Ne,1]
            score2 = -torch.log(relevance_sum + 1e-10)
            return score2
    
    def combine_scores(self, 
                       score1: torch.Tensor, 
                       score2: torch.Tensor) -> torch.Tensor:
        """
        Công thức (13): score = α · score1 + (1 - α) · score2
        
        Args:
            score1: [num_nodes, 1] - Self-redundancy
            score2: [num_nodes, 1] - Claim-relevance redundancy
            
        Returns:
            score: [num_nodes, 1] - Combined redundancy score
        """
        score = self.alpha * score1 + (1 - self.alpha) * score2
        return score
    
    def refine_with_ggnn(self, 
                         score: torch.Tensor,
                         adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Công thức (14): S = GGNN(Ã, score)
        
        Args:
            score: [num_nodes, 1] - Combined redundancy scores
            adj_matrix: [num_nodes, num_nodes] - Evidence graph adjacency
            
        Returns:
            S: [num_nodes, 1] - Evidence-aware redundancy scores
        """
        # GGNN supports batched score and adj
        S = self.ggnn(score, adj_matrix)  # [..., num_nodes, 1]
        return S
    
    def forward(self, 
                H_e: torch.Tensor,
                H_c: torch.Tensor,
                adj_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete semantic refinement process
        
        Args:
            H_e: [num_nodes, hidden_dim] - Evidence node representations
            H_c: [num_claim_nodes, hidden_dim] - Claim representations
            adj_e: [num_nodes, num_nodes] - Evidence adjacency matrix
            
        Returns:
            H_e_refined: [num_remaining_nodes, hidden_dim] - Refined evidence
            keep_mask: [num_nodes] - Boolean mask of kept nodes
        """
        # Support batched and single inputs
        if H_e.dim() == 3:
            # H_e: [B, Ne, H], H_c: [B, Nc, H], adj_e: [B, Ne, Ne]
            B, Ne, H = H_e.size()

            score1 = self.compute_self_redundancy(H_e)  # [B, Ne, 1]
            score2 = self.compute_claim_relevance_redundancy(H_e, H_c)  # [B, Ne, 1]
            score = self.combine_scores(score1, score2)  # [B, Ne, 1]

            S = self.refine_with_ggnn(score, adj_e)  # [B, Ne, 1]

            k = (self.top_k_ratio * Ne).long().item() if isinstance(self.top_k_ratio, torch.Tensor) else int(self.top_k_ratio * Ne)

            H_e_refined_list = []
            keep_masks = []
            for b in range(B):
                S_b = S[b].squeeze(-1)  # [Ne]
                sorted_indices = torch.argsort(S_b, descending=False)
                keep_indices = sorted_indices[:-k] if k > 0 else sorted_indices
                keep_mask = torch.zeros(Ne, dtype=torch.bool, device=H_e.device)
                keep_mask[keep_indices] = True
                keep_masks.append(keep_mask)
                H_e_refined_list.append(H_e[b][keep_mask])

            return H_e_refined_list, keep_masks
        else:
            num_nodes = H_e.size(0)
            score1 = self.compute_self_redundancy(H_e)
            score2 = self.compute_claim_relevance_redundancy(H_e, H_c)
            score = self.combine_scores(score1, score2)
            S = self.refine_with_ggnn(score, adj_e)

            k = int(self.top_k_ratio * num_nodes)
            sorted_indices = torch.argsort(S.squeeze(), descending=False)
            keep_indices = sorted_indices[:-k] if k > 0 else sorted_indices
            keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=H_e.device)
            keep_mask[keep_indices] = True
            H_e_refined = H_e[keep_mask]
            return H_e_refined, keep_mask


class SemanticRefinementWrapper(nn.Module):
    """
    Wrapper để apply semantic refinement cho tất cả evidences
    """
    
    def __init__(self, 
                 hidden_dim: int = 300,
                 alpha: float = 0.5,
                 top_k_ratio: float = 0.3,
                 num_ggnn_steps: int = 3):
        super(SemanticRefinementWrapper, self).__init__()
        
        self.refinement = SemanticRefinement(
            hidden_dim, alpha, top_k_ratio, num_ggnn_steps
        )
    
    def forward(self, 
                H_c: torch.Tensor,
                H_e_list: List[torch.Tensor],
                adj_e_list: List[torch.Tensor]) -> Dict:
        """
        Apply refinement cho tất cả evidences
        
        Args:
            H_c: [num_claim_nodes, hidden_dim] - Claim representation
            H_e_list: List of [num_evi_nodes, hidden_dim] - Evidence representations
            adj_e_list: List of [num_evi_nodes, num_evi_nodes] - Evidence adjacency matrices
            
        Returns:
            {
                'H_e_refined': List of refined evidence representations,
                'keep_masks': List of boolean masks
            }
        """
        # Support batched inputs: H_c [B, Nc, H], H_e_list: [B, E, Ne, H], adj_e_list: [B, E, Ne, Ne]
        if isinstance(H_e_list, dict):
            # Expect dict with 'node_features' and 'adj_matrix'
            evi_nodes = H_e_list['node_features']  # [B, E, Ne, H]
            evi_adj = H_e_list['adj_matrix']      # [B, E, Ne, Ne]

            B, E, Ne, H = evi_nodes.size()
            H_e_refined = []
            keep_masks = []

            for b in range(B):
                H_e_refined_b = []
                keep_masks_b = []
                for e in range(E):
                    H_e = evi_nodes[b, e]
                    adj_e = evi_adj[b, e]
                    H_e_ref, mask = self.refinement(H_e, H_c[b], adj_e)
                    H_e_refined_b.append(H_e_ref)
                    keep_masks_b.append(mask)
                H_e_refined.append(H_e_refined_b)
                keep_masks.append(keep_masks_b)

            return {'H_e_refined': H_e_refined, 'keep_masks': keep_masks}
        elif isinstance(H_e_list, torch.Tensor):
            # H_e_list: [B, E, Ne, H]
            B, E, Ne, H = H_e_list.size()
            H_e_refined = []
            keep_masks = []
            for b in range(B):
                H_e_refined_b = []
                keep_masks_b = []
                for e in range(E):
                    H_e_ref, mask = self.refinement(H_e_list[b, e], H_c[b], adj_e_list[b, e])
                    H_e_refined_b.append(H_e_ref)
                    keep_masks_b.append(mask)
                H_e_refined.append(H_e_refined_b)
                keep_masks.append(keep_masks_b)

            return {'H_e_refined': H_e_refined, 'keep_masks': keep_masks}
        else:
            H_e_refined_list = []
            keep_masks = []
            for H_e, adj_e in zip(H_e_list, adj_e_list):
                H_e_ref, mask = self.refinement(H_e, H_c, adj_e)
                H_e_refined_list.append(H_e_ref)
                keep_masks.append(mask)
            return {'H_e_refined': H_e_refined_list, 'keep_masks': keep_masks}
