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
        score1 = self.W_score1(H_e)  # [num_nodes, 1]
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
        W = torch.matmul(H_e, H_c.transpose(0, 1))  # [num_evi_nodes, num_claim_nodes]
        
        # Normalize
        W = torch.softmax(W, dim=-1)  # [num_evi_nodes, num_claim_nodes]
        
        # Σ_j W_ij · H_cj (weighted sum of claim features)
        # [num_evi_nodes, num_claim_nodes] @ [num_claim_nodes, hidden_dim]
        weighted_claim = torch.matmul(W, H_c)  # [num_evi_nodes, hidden_dim]
        
        # Sum over hidden_dim
        relevance_sum = torch.sum(weighted_claim, dim=-1, keepdim=True)  # [num_evi_nodes, 1]
        
        # score_i^2 = -log(relevance_sum)
        # High relevance → low score → KEEP
        # Low relevance → high score → DISCARD
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
        S = self.ggnn(score, adj_matrix)  # [num_nodes, 1]
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
        num_nodes = H_e.size(0)
        
        # Step 1: Compute self-redundancy (công thức 11)
        score1 = self.compute_self_redundancy(H_e)
        
        # Step 2: Compute claim-relevance redundancy (công thức 12)
        score2 = self.compute_claim_relevance_redundancy(H_e, H_c)
        
        # Step 3: Combine scores (công thức 13)
        score = self.combine_scores(score1, score2)
        
        # Step 4: GGNN refinement (công thức 14)
        S = self.refine_with_ggnn(score, adj_e)
        
        # Step 5: Sort và discard top k highest redundancy nodes
        k = int(self.top_k_ratio * num_nodes)
        
        # Get indices sorted by redundancy (ascending: low redundancy first)
        sorted_indices = torch.argsort(S.squeeze(), descending=False)
        
        # Keep nodes with lowest redundancy (discard top k)
        keep_indices = sorted_indices[:-k] if k > 0 else sorted_indices
        
        # Create boolean mask
        keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=H_e.device)
        keep_mask[keep_indices] = True
        
        # Filter evidence representations
        H_e_refined = H_e[keep_mask]  # [num_remaining_nodes, hidden_dim]
        
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
        H_e_refined_list = []
        keep_masks = []
        
        for H_e, adj_e in zip(H_e_list, adj_e_list):
            H_e_ref, mask = self.refinement(H_e, H_c, adj_e)
            H_e_refined_list.append(H_e_ref)
            keep_masks.append(mask)
        
        return {
            'H_e_refined': H_e_refined_list,
            'keep_masks': keep_masks
        }
