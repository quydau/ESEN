# models/syntactic_extraction.py

import torch
import torch.nn as nn
from typing import Dict, List
import math


class MultiHeadAttentionGraph(nn.Module):
    """
    Transform syntactic graph thành N weighted graphs bằng multi-head attention
    Theo công thức (2) trong paper
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        """
        Args:
            hidden_dim: Dimension của node features
            num_heads: Số attention heads (N trong paper)
        """
        super(MultiHeadAttentionGraph, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim  # dimension cho mỗi head
        
        # Projection matrices cho mỗi head
        # W_Q_i và W_K_i cho từng head
        self.W_Q = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) 
            for _ in range(num_heads)
        ])
        self.W_K = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_heads)
        ])
    
    def forward(self, h: torch.Tensor) -> List[torch.Tensor]:
        """
        Transform syntactic graph representation thành N weighted adjacency matrices
        
        Args:
            h: [num_nodes, hidden_dim] - syntactic graph representation h^(t-1)
            
        Returns:
            List of Ã^(t) cho mỗi head: [num_heads, num_nodes, num_nodes]
        """
        adj_matrices = []
        
        for i in range(self.num_heads):
            # Q = h * W_Q_i
            Q = self.W_Q[i](h)  # [num_nodes, hidden_dim]
            
            # K = h * W_K_i
            K = self.W_K[i](h)  # [num_nodes, hidden_dim]
            
            # Attention scores: Q * K^T / sqrt(d)
            # [num_nodes, hidden_dim] @ [hidden_dim, num_nodes]
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            # scores: [num_nodes, num_nodes]
            
            # Ã^(t) = softmax(scores)
            A_t = torch.softmax(scores, dim=-1)  # [num_nodes, num_nodes]
            
            adj_matrices.append(A_t)
        
        return adj_matrices


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer cho mỗi weighted graph
    Theo công thức (3) trong paper
    """
    
    def __init__(self, hidden_dim: int):
        super(GraphConvLayer, self).__init__()
        
        # W^(t) và b^(t)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [num_nodes, hidden_dim] - h^(t-1)
            adj: [num_nodes, num_nodes] - Ã^(t)
            
        Returns:
            h^(t): [num_nodes, hidden_dim]
        """
        # h^(t)_i = ρ(Σ_j Ã^(t)_ij * W^(t) * h^(t-1)_j + b^(t))
        # [num_nodes, num_nodes] @ [num_nodes, hidden_dim]
        aggregated = torch.matmul(adj, h)  # [num_nodes, hidden_dim]
        
        # Apply linear transformation
        h_new = self.W(aggregated)  # includes bias b^(t)
        
        # Activation function ρ (ReLU)
        h_new = torch.relu(h_new)
        
        return h_new


class DenselyConnectedLayer(nn.Module):
    """
    Densely Connected Layer với M sub-layers
    Theo công thức (4) và (5) trong paper
    """
    
    def __init__(self, hidden_dim: int, num_sublayers: int = 3):
        """
        Args:
            hidden_dim: Dimension của features
            num_sublayers: Số sub-layers M
        """
        super(DenselyConnectedLayer, self).__init__()
        
        self.num_sublayers = num_sublayers
        
        # Graph conv layers cho mỗi sub-layer
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim) 
            for _ in range(num_sublayers)
        ])
        
        # Final projection để giữ dimension
        # Vì concatenate M+1 representations (x + M sub-layer outputs)
        self.output_proj = nn.Linear(
            hidden_dim * (num_sublayers + 1), 
            hidden_dim
        )
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, hidden_dim] - input representation
            adj: [num_nodes, num_nodes] - Ã^(t) cho graph này
            
        Returns:
            D^(M)_syn: [num_nodes, hidden_dim]
        """
        # Lưu tất cả outputs để concatenate
        outputs = [x]  # Bắt đầu với input x
        
        h = x
        for i in range(self.num_sublayers):
            # D^(i)_syn = GCN(concatenation of all previous layers, adj)
            # Concatenate tất cả previous outputs
            h_concat = torch.cat(outputs, dim=-1)
            
            # Apply graph convolution (với internal projection nếu cần)
            # Projection về hidden_dim trước khi apply GCN
            if i > 0:
                h_concat = nn.Linear(
                    h_concat.size(-1), 
                    h.size(-1)
                ).to(h.device)(h_concat)
            
            h = self.conv_layers[i](h_concat if i == 0 else h, adj)
            outputs.append(h)
        
        # D^(M)_syn = [x; D^(1)_syn; ...; D^(M-1)_syn]
        D_M = torch.cat(outputs, dim=-1)  # [num_nodes, hidden_dim * (M+1)]
        
        # Project về hidden_dim
        # Công thức (5): h^(t)_syn = W^(t) * Ã^(t) * D^(t)_syn + b^(t)
        # (đã tích hợp vào output_proj)
        h_syn = self.output_proj(D_M)  # [num_nodes, hidden_dim]
        
        return h_syn


class SyntacticEncoder(nn.Module):
    """
    Complete Syntactic Encoder với:
    - Multi-head attention để tạo N weighted graphs
    - Densely connected layers cho mỗi weighted graph
    - Linear combination để merge outputs
    """
    
    def __init__(self, 
                 input_dim: int = 300,
                 hidden_dim: int = 300,
                 num_heads: int = 4,
                 num_sublayers: int = 3):
        super(SyntacticEncoder, self).__init__()
        
        self.num_heads = num_heads
        
        # Input transformation
        if input_dim != hidden_dim:
            self.input_transform = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_transform = nn.Identity()
        
        # Multi-head attention để tạo N weighted graphs
        self.multi_head_attn = MultiHeadAttentionGraph(hidden_dim, num_heads)
        
        # Densely connected layer cho MỖI head
        self.dense_layers = nn.ModuleList([
            DenselyConnectedLayer(hidden_dim, num_sublayers)
            for _ in range(num_heads)
        ])
        
        # Linear combination layer (công thức 6)
        # W_syn và b_syn
        self.linear_combination = nn.Linear(
            hidden_dim * num_heads,  # concatenate N heads
            hidden_dim
        )
    
    def forward(self, 
                node_features: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, input_dim]
            adj_matrix: [num_nodes, num_nodes] - Ã^(0) (original syntactic adj)
            
        Returns:
            H_syn: [num_nodes, hidden_dim]
        """
        # Transform input
        h = self.input_transform(node_features)  # h^(0)
        
        # Step 1: Multi-head attention tạo N weighted graphs
        # Ã^(1), ..., Ã^(N)
        # Support batched input: either [num_nodes, D] or [B, num_nodes, D]
        if h.dim() == 3:
            B, N, D = h.size()

            # For multi-head attention and dense layers, process per-batch
            weighted_adjs = []
            for i in range(self.num_heads):
                # Use the multi-head modules defined in MultiHeadAttentionGraph
                Q = self.multi_head_attn.W_Q[i](h)  # [B, N, H]
                K = self.multi_head_attn.W_K[i](h)  # [B, N, H]
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.multi_head_attn.d_k)  # [B,N,N]
                A_t = torch.softmax(scores, dim=-1)
                weighted_adjs.append(A_t)

            # Apply dense layers per head: each dense layer expects [N, H] and adj [N,N]
            h_syn_list = []
            for t in range(self.num_heads):
                # For each batch, apply DenselyConnectedLayer separately and stack
                h_t_list = []
                adj_t = weighted_adjs[t]
                for b in range(B):
                    h_b = h[b]
                    adj_b = adj_t[b]
                    h_t_syn = self.dense_layers[t](h_b, adj_b)  # [N, H]
                    h_t_list.append(h_t_syn)
                h_t_stack = torch.stack(h_t_list, dim=0)  # [B, N, H]
                h_syn_list.append(h_t_stack)

            # Concatenate across heads on last dim
            h_out = torch.cat(h_syn_list, dim=-1)  # [B, N, H * N_heads]
            H_syn = self.linear_combination(h_out)  # [B, N, H]
            return H_syn
        else:
            weighted_adjs = self.multi_head_attn(h)  # List of [num_nodes, num_nodes]

            # Step 2: Apply densely connected layer cho mỗi weighted graph
            h_syn_list = []
            for t in range(self.num_heads):
                # h^(t)_syn từ mỗi weighted graph
                h_t_syn = self.dense_layers[t](h, weighted_adjs[t])
                h_syn_list.append(h_t_syn)

            # Step 3: Concatenate và linear combination
            # h_out = [h^(1)_syn, ..., h^(N)_syn]
            h_out = torch.cat(h_syn_list, dim=-1)  # [num_nodes, hidden_dim * N]

            # H_syn = W_syn * h_out + b_syn (công thức 6)
            H_syn = self.linear_combination(h_out)  # [num_nodes, hidden_dim]
            return H_syn


class SyntacticInformationExtraction(nn.Module):
    """
    Module extract H_c_syn và H_e_syn cho claim và evidences
    """
    
    def __init__(self, 
                 input_dim: int = 300,
                 hidden_dim: int = 300,
                 num_heads: int = 4,
                 num_sublayers: int = 3):
        super(SyntacticInformationExtraction, self).__init__()
        
        self.syntactic_encoder = SyntacticEncoder(
            input_dim, hidden_dim, num_heads, num_sublayers
        )
    
    def forward(self, prepared_sample: Dict) -> Dict:
        """
        Args:
            prepared_sample: Output từ ESENDataPreprocessor.prepare_sample()
                {
                    'claim': {
                        'syntactic': {'node_features', 'adj_matrix', ...}
                    },
                    'evidences': {
                        'syntactic': [{'node_features', 'adj_matrix'}, ...]
                    }
                }
        
        Returns:
            {
                'H_c_syn': [num_claim_nodes, hidden_dim],
                'H_e_syn': List[[num_evi_nodes, hidden_dim], ...]
            }
        """
        # Extract claim syntactic representation
        claim_data = prepared_sample['claim']['syntactic']

        if claim_data['node_features'].dim() == 3:
            # Batched: claim node_features [B, Nc, D], claim adj [B, Nc, Nc]
            H_c_syn = self.syntactic_encoder(
                claim_data['node_features'],
                claim_data['adj_matrix']
            )  # [B, Nc, H]

            # Evidences: node_features [B, E, Ne, D] -> reshape to [B*E, Ne, D]
            evi_data = prepared_sample['evidences']['syntactic']
            B, E, Ne, D = evi_data['node_features'].size()
            evi_nodes = evi_data['node_features'].reshape(B * E, Ne, D)
            evi_adj = evi_data['adj_matrix'].reshape(B * E, Ne, Ne)

            H_e = self.syntactic_encoder(evi_nodes, evi_adj)  # [B*E, Ne, H]
            H_e = H_e.reshape(B, E, Ne, -1)  # [B, E, Ne, H]

            return {
                'H_c_syn': H_c_syn,
                'H_e_syn': H_e
            }
        else:
            H_c_syn = self.syntactic_encoder(
                claim_data['node_features'],
                claim_data['adj_matrix']
            )

            H_e_syn = []
            for evi_data in prepared_sample['evidences']['syntactic']:
                H_e = self.syntactic_encoder(
                    evi_data['node_features'],
                    evi_data['adj_matrix']
                )
                H_e_syn.append(H_e)

            return {
                'H_c_syn': H_c_syn,
                'H_e_syn': H_e_syn
            }
