# models/fusion.py

import torch
import torch.nn as nn
from typing import Dict, List


class SemanticSyntacticFusion(nn.Module):
    """
    Fusion semantic và syntactic representations
    Theo công thức (7), (8), (9), (10) trong paper
    """
    
    def __init__(self, hidden_dim: int = 300, ffn_hidden_dim: int = 512):
        """
        Args:
            hidden_dim: Dimension của H_sem và H_syn
            ffn_hidden_dim: Hidden dimension cho FFN
        """
        super(SemanticSyntacticFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # FFN để tính attention weights α_syn (công thức 7)
        # Input: [H_syn; H_sem] có dimension = 2 * hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(2 * hidden_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, hidden_dim),
            nn.Softmax(dim=-1)  # α_syn = softmax(FFN([H_syn; H_sem]))
        )
        
        # W_sem và b_sem cho công thức (9)
        # Input: [H_sem; M] có dimension = 2 * hidden_dim
        self.gate_linear = nn.Linear(2 * hidden_dim, hidden_dim)
    
    def forward(self, H_sem: torch.Tensor, H_syn: torch.Tensor) -> torch.Tensor:
        """
        Fuse semantic và syntactic representations
        
        Args:
            H_sem: [num_nodes, hidden_dim] - Semantic representation
            H_syn: [num_nodes, hidden_dim] - Syntactic representation
            
        Returns:
            H_enhanced: [num_nodes, hidden_dim] - Enhanced semantic representation (H'_sem)
        """
        # Công thức (7): α_syn = softmax(FFN([H_syn; H_sem]))
        concat_input = torch.cat([H_syn, H_sem], dim=-1)  # [num_nodes, 2*hidden_dim]
        alpha_syn = self.ffn(concat_input)  # [num_nodes, hidden_dim]
        
        # Công thức (8): M = α_syn ⊙ H_syn (element-wise multiplication)
        M = alpha_syn * H_syn  # [num_nodes, hidden_dim]
        
        # Công thức (9): f = σ(W_sem · [H_sem; M] + b_sem)
        concat_sem_M = torch.cat([H_sem, M], dim=-1)  # [num_nodes, 2*hidden_dim]
        f = torch.sigmoid(self.gate_linear(concat_sem_M))  # [num_nodes, hidden_dim]
        
        # Công thức (10): H'_sem = f ⊙ H_sem + (1 - f) ⊙ M
        H_enhanced = f * H_sem + (1 - f) * M  # [num_nodes, hidden_dim]
        
        return H_enhanced


class SemanticEnhancementNetwork(nn.Module):
    """
    Complete Semantic Enhancement Network
    Kết hợp semantic extraction, syntactic extraction, và fusion
    """
    
    def __init__(self, 
                 input_dim: int = 300,
                 hidden_dim: int = 300,
                 num_semantic_steps: int = 5,
                 num_attention_heads: int = 4,
                 num_dense_sublayers: int = 3,
                 ffn_hidden_dim: int = 512):
        super(SemanticEnhancementNetwork, self).__init__()
        
        from models.semantic_extraction import SemanticInformationExtraction
        from models.syntactic_extraction import SyntacticInformationExtraction
        
        # Semantic extraction module
        self.semantic_extractor = SemanticInformationExtraction(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_steps=num_semantic_steps
        )
        
        # Syntactic extraction module
        self.syntactic_extractor = SyntacticInformationExtraction(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            num_sublayers=num_dense_sublayers
        )
        
        # Fusion module
        self.fusion = SemanticSyntacticFusion(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim
        )
    
    def forward(self, prepared_sample: Dict) -> Dict:
        """
        Complete semantic enhancement process
        
        Args:
            prepared_sample: Output từ ESENDataPreprocessor.prepare_sample()
            
        Returns:
            {
                'H_c': [num_claim_nodes, hidden_dim] - Enhanced claim representation,
                'H_e': List[[num_evi_nodes, hidden_dim], ...] - Enhanced evidence representations
            }
        """
        # Step 1: Extract semantic representations
        semantic_outputs = self.semantic_extractor(prepared_sample)
        H_c_sem = semantic_outputs['H_c_sem']
        H_e_sem_list = semantic_outputs['H_e_sem']
        
        # Step 2: Extract syntactic representations
        syntactic_outputs = self.syntactic_extractor(prepared_sample)
        H_c_syn = syntactic_outputs['H_c_syn']
        H_e_syn_list = syntactic_outputs['H_e_syn']
        
        # Step 3: Fuse semantic and syntactic for claim
        H_c = self.fusion(H_c_sem, H_c_syn)
        
        # Step 4: Fuse semantic and syntactic for each evidence
        H_e_list = []
        for H_e_sem, H_e_syn in zip(H_e_sem_list, H_e_syn_list):
            H_e = self.fusion(H_e_sem, H_e_syn)
            H_e_list.append(H_e)
        
        return {
            'H_c': H_c,        # Enhanced claim representation
            'H_e': H_e_list    # Enhanced evidence representations
        }
