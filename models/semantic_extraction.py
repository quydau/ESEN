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
        # Transform input
        node_features = self.input_transform(node_features)
        
        # Extract semantic info
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
        # Extract claim semantic representation
        claim_data = prepared_sample['claim']['semantic']
        H_c_sem = self.semantic_encoder(
            claim_data['node_features'],
            claim_data['adj_matrix']
        )
        
        # Extract evidences semantic representations
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
