import torch
import torch.nn as nn


class GGNN(nn.Module):
    """
    Graph Gated Neural Network để extract semantic information
    Theo công thức (1) trong paper
    """
    
    def __init__(self, 
                 hidden_dim: int = 300,
                 num_steps: int = 5):
        """
        Args:
            hidden_dim: Chiều của hidden representation
            num_steps: Số bước propagation (T trong paper)
        """
        super(GGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        # Trainable parameters từ công thức (1)
        # Cho update gate z_t
        self.W_z = nn.Linear(hidden_dim, hidden_dim)
        self.U_z = nn.Linear(hidden_dim, hidden_dim)
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))
        
        # Cho reset gate r_t
        self.W_r = nn.Linear(hidden_dim, hidden_dim)
        self.U_r = nn.Linear(hidden_dim, hidden_dim)
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))
        
        # Cho candidate activation l_t
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.U_h = nn.Linear(hidden_dim, hidden_dim)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        
        # Cho aggregation a_t (optional, có thể dùng trực tiếp adjacency matrix)
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, 
                node_features: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Propagate thông tin trên semantic graph
        
        Args:
            node_features: [num_nodes, hidden_dim] - h^0
            adj_matrix: [num_nodes, num_nodes] - Ã
            
        Returns:
            H_sem: [num_nodes, hidden_dim] - Final semantic representation
        """
        # Support both single-graph [N, H] and batched graphs [B, N, H]
        h_t = node_features

        batched = h_t.dim() == 3

        for t in range(self.num_steps):
            # Aggregate neighbor info: works for both batched and single
            # If batched: adj_matrix [B,N,N], self.W_a(h_t) -> [B,N,H]
            a_t = torch.matmul(adj_matrix, self.W_a(h_t))

            # Update gate
            z_t = torch.sigmoid(self.W_z(a_t) + self.U_z(h_t) + self.b_z)

            # Reset gate
            r_t = torch.sigmoid(self.W_r(a_t) + self.U_r(h_t) + self.b_r)

            # Candidate activation
            l_t = torch.tanh(self.W_h(a_t) + self.U_h(r_t * h_t) + self.b_h)

            # Update
            h_t = l_t * z_t + h_t * (1 - z_t)

        return h_t
