import torch
from torch import nn

class GraphEmbedding(nn.Module):
    def __init__(self, inp_dim, E_node_dim, E_edge_dim, out_dim):
        super().__init__()
        self.node_embed = nn.Parameter(torch.randn(1, 1, E_node_dim))
        self.edge_embed = nn.Parameter(torch.randn(1, 1, E_edge_dim))

        # node_features_dim + E_node_dim = edge_features_dim + E_edges_dim
        self.p = nn.Linear(inp_dim , out_dim)
    def forward(self, node_features, edge_features):
        # node_features: [batch_size, seq_len_1, node_features_dim + 2*D_dim]
        # edges_features: [batch_size, seq_len_2, edge_features_dim + 2*D_dim]
        # final_dim = node_features_dim + 2*D_dim + E_node_dim = edge_features_dim + 2*D_dim + E_edge_dim 

        m_n = torch.cat([
            node_features,
            self.node_embed.expand(node_features.size(0), node_features.size(1), -1)
        ], dim=-1)

        m_e = torch.cat([
            edge_features,
            self.edge_embed.expand(edge_features.size(0), edge_features.size(1), -1)
        ], dim=-1)
        x = torch.cat([m_n, m_e], dim=1) # [batch_size, seq_len_1 + seq_len_2, final_dim]
        return self.p(x)  # [batch_size, seq_len_1 + seq_len_2, out_dim]