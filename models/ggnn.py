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
        # Khởi tạo h_0
        h_t = node_features  # [num_nodes, hidden_dim]
        
        # Propagation qua T steps
        for t in range(self.num_steps):
            # Bước 1: Aggregate thông tin từ neighbors
            # a_t = Σ Ã * h_j * W_a
            # [num_nodes, num_nodes] @ [num_nodes, hidden_dim] @ [hidden_dim, hidden_dim]
            a_t = torch.matmul(adj_matrix, self.W_a(h_t))  # [num_nodes, hidden_dim]
            
            # Bước 2: Update gate
            # z_t = σ(W_z * a_t + U_z * h_t + b_z)
            z_t = torch.sigmoid(
                self.W_z(a_t) + self.U_z(h_t) + self.b_z
            )  # [num_nodes, hidden_dim]
            
            # Bước 3: Reset gate
            # r_t = σ(W_r * a_t + U_r * h_t + b_r)
            r_t = torch.sigmoid(
                self.W_r(a_t) + self.U_r(h_t) + self.b_r
            )  # [num_nodes, hidden_dim]
            
            # Bước 4: Candidate activation
            # l_t = tanh(W_h * a_t + U_h * (r_t ⊙ h_t) + b_h)
            l_t = torch.tanh(
                self.W_h(a_t) + self.U_h(r_t * h_t) + self.b_h
            )  # [num_nodes, hidden_dim]
            
            # Bước 5: Update node representation (GRU-style)
            # H_t = l_t ⊙ z_t + h_t ⊙ (1 - z_t)
            h_t = l_t * z_t + h_t * (1 - z_t)  # [num_nodes, hidden_dim]
        
        # Return final representation
        return h_t  # H_sem
