# models/semantic_extraction.py

import torch
import torch.nn as nn
from typing import Dict
from models.ggnn import GGNN


class SemanticEncoder(nn.Module):
    """
    Encoder sử dụng GGNN để extract semantic representation
    """
    
    def __init__(self, 
                 input_dim: int = 300,
                 hidden_dim: int = 300,
                 num_steps: int = 5):
        super(SemanticEncoder, self).__init__()
        
        self.ggnn = GGNN(hidden_dim, num_steps)
        
        # Transform nếu input_dim != hidden_dim
        if input_dim != hidden_dim:
            self.input_transform = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_transform = nn.Identity()
    
    def forward(self, 
                node_features: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, input_dim] - từ preprocessor
            adj_matrix: [num_nodes, num_nodes]
            
        Returns:
            H_sem: [num_nodes, hidden_dim]
        """
        # Support both single-graph [N, D] and batched [B, N, D]
        batched = node_features.dim() == 3

        if batched:
            # node_features: [B, N, input_dim]
            B, N, _ = node_features.size()
            # Apply input transform (works on last dim)
            node_features = self.input_transform(node_features)

            # Extract semantic info via GGNN (supports batched)
            H_sem = self.ggnn(node_features, adj_matrix)  # [B, N, hidden_dim]
        else:
            node_features = self.input_transform(node_features)
            H_sem = self.ggnn(node_features, adj_matrix)

        return H_sem


class SemanticInformationExtraction(nn.Module):
    """
    Module extract H_c_sem và H_e_sem cho claim và evidences
    """
    
    def __init__(self, 
                 input_dim: int = 300,
                 hidden_dim: int = 300,
                 num_steps: int = 5):
        super(SemanticInformationExtraction, self).__init__()
        
        self.semantic_encoder = SemanticEncoder(input_dim, hidden_dim, num_steps)
    
    def forward(self, prepared_sample: Dict) -> Dict:
        """
        Args:
            prepared_sample: Output từ ESENDataPreprocessor.prepare_sample()
                {
                    'claim': {
                        'semantic': {'node_features', 'adj_matrix', ...}
                    },
                    'evidences': {
                        'semantic': [{'node_features', 'adj_matrix'}, ...]
                    }
                }
        
        Returns:
            {
                'H_c_sem': [num_claim_nodes, hidden_dim],
                'H_e_sem': List[[num_evi_nodes, hidden_dim], ...]
            }
        """
        # Determine if input is batched
        claim_data = prepared_sample['claim']['semantic']
        if claim_data['node_features'].dim() == 3:
            # Batched: claim node_features [B, Nc, D], claim adj [B, Nc, Nc]
            H_c_sem = self.semantic_encoder(
                claim_data['node_features'],
                claim_data['adj_matrix']
            )  # [B, Nc, H]

            # Evidences: node_features [B, E, Ne, D] -> reshape to [B*E, Ne, D]
            evi_data = prepared_sample['evidences']['semantic']
            B, E, Ne, D = evi_data['node_features'].size()
            evi_nodes = evi_data['node_features'].reshape(B * E, Ne, D)
            evi_adj = evi_data['adj_matrix'].reshape(B * E, Ne, Ne)

            H_e = self.semantic_encoder(evi_nodes, evi_adj)  # [B*E, Ne, H]
            H_e = H_e.reshape(B, E, Ne, -1)  # [B, E, Ne, H]

            return {
                'H_c_sem': H_c_sem,
                'H_e_sem': H_e
            }
        else:
            # Single sample (unchanged)
            H_c_sem = self.semantic_encoder(
                claim_data['node_features'],
                claim_data['adj_matrix']
            )

            H_e_sem = []
            for evi_data in prepared_sample['evidences']['semantic']:
                H_e = self.semantic_encoder(
                    evi_data['node_features'],
                    evi_data['adj_matrix']
                )
                H_e_sem.append(H_e)

            return {
                'H_c_sem': H_c_sem,
                'H_e_sem': H_e_sem
            }
