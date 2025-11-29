# models/esen.py

import torch
import torch.nn as nn
from typing import Dict
from models.fusion import SemanticEnhancementNetwork
from models.semantic_refinement import SemanticRefinementWrapper
from models.sentence_level_attention import InformationInteractionNetwork


class ESENClassifier(nn.Module):
    """
    Classifier cho ESEN
    Công thức (26), (27)
    """
    
    def __init__(self, input_dim: int, num_classes: int = 2):
        super(ESENClassifier, self).__init__()
        
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, H_final: torch.Tensor) -> torch.Tensor:
        """
        Công thức (26): ŷ = Softmax(W * H_final + b)
        """
        # Accept either a single-sample 1D `H_final` or a batched 2D tensor.
        if not isinstance(H_final, torch.Tensor):
            H_final = torch.tensor(H_final, dtype=torch.float32)

        if H_final.dim() == 1:
            # Single sample: add batch dim, apply fc, then remove batch dim
            logits = self.fc(H_final.unsqueeze(0)).squeeze(0)
        elif H_final.dim() == 2:
            # Batched input
            logits = self.fc(H_final)
        else:
            raise ValueError(f"Unexpected H_final dimensions: {H_final.dim()}")

        return logits  # Return logits (for single sample: 1D tensor of size num_classes)


class ESEN(nn.Module):
    """
    Complete ESEN Model
    End-to-end fake news detection
    """
    
    def __init__(self,
                 input_dim: int = 300,
                 hidden_dim: int = 300,
                 publisher_dim: int = 100,
                 num_semantic_steps: int = 5,
                 num_attention_heads: int = 4,
                 num_dense_sublayers: int = 3,
                 ffn_hidden_dim: int = 512,
                 alpha: float = 0.5,
                 top_k_ratio: float = 0.3,
                 num_ggnn_steps: int = 3,
                 num_evi_layers: int = 2,
                 num_classes: int = 2):
        super(ESEN, self).__init__()
        
        # 1. Semantic Enhancement Network
        self.semantic_enhancement = SemanticEnhancementNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_semantic_steps=num_semantic_steps,
            num_attention_heads=num_attention_heads,
            num_dense_sublayers=num_dense_sublayers,
            ffn_hidden_dim=ffn_hidden_dim
        )
        
        # 2. Semantic Refinement
        self.semantic_refinement = SemanticRefinementWrapper(
            hidden_dim=hidden_dim,
            alpha=alpha,
            top_k_ratio=top_k_ratio,
            num_ggnn_steps=num_ggnn_steps
        )
        
        # 3. Information Interaction Network for Evidence Perception
        self.iinep = InformationInteractionNetwork(
            hidden_dim=hidden_dim,
            publisher_dim=publisher_dim,
            num_evi_layers=num_evi_layers
        )
        
        # 4. Classifier
        # Note: IINEP returns a final vector of size 2*(hidden_dim + publisher_dim)
        final_dim = 2 * (hidden_dim + publisher_dim)
        self.classifier = ESENClassifier(final_dim, num_classes)
    
    def forward(self, batch: Dict) -> torch.Tensor:
        """
        Args:
            batch: {
                'prepared_sample': từ ESENDataPreprocessor,
                'P_c': [publisher_dim] - claim publisher,
                'P_e_list': List of [publisher_dim] - evidence publishers,
                'adj_e_list': List of adjacency matrices cho refinement
            }
        
        Returns:
            logits: [num_classes]
        """
        prepared_sample = batch['prepared_sample']
        P_c = batch.get('P_c')
        P_e_list = batch.get('P_e_list')
        adj_e_list = batch['adj_e_list']
        
        # Step 1: Semantic Enhancement
        enhanced = self.semantic_enhancement(prepared_sample)
        H_c = enhanced['H_c']  # [num_claim_nodes, hidden_dim]
        H_e_list = enhanced['H_e']  # List of [num_evi_nodes, hidden_dim]
        
        # Step 2: Semantic Refinement (loại bỏ redundant info)
        refined = self.semantic_refinement(H_c, H_e_list, adj_e_list)
        H_e_refined = refined['H_e_refined']
        
        # Step 3: Information Interaction (IINEP)
        H_final = self.iinep(H_c, H_e_refined, P_c, P_e_list)
        
        # Step 4: Classification
        logits = self.classifier(H_final)
        
        return logits


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Công thức (27): Cross Entropy Loss
    L = -(y*log(ŷ) + (1-y)*log(1-ŷ))
    """
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.unsqueeze(0), labels.unsqueeze(0))
    return loss
