import torch
from torch import nn
from src.python.models.favor_plus_attention import FAVORPlusAttention
from src.python.models.transmap.graph_embedding import GraphEmbedding

class TFEMLayer(nn.Module):
    def __init__(self, out_dim, ff_dim, n_heads, nb_features, ortho_scaling, dropout, activation_fn):
        super().__init__()
        self.self_attn = FAVORPlusAttention(out_dim, n_heads, nb_features, ortho_scaling, dropout)
        self.F = nn.Sequential(
            nn.Linear(out_dim, ff_dim),
            activation_fn,
            nn.Linear(ff_dim, out_dim)
        )
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x, mask):
        m = self.self_attn(x, mask)
        f = self.F(m) 
        return self.norm2(m + f)

class TFEM(nn.Module):
    def __init__(self, num_mhd_layers,max_features_dim, D_dim, E_node_dim, E_edge_dim,
                 out_dim, ff_dim, n_heads, nb_features, ortho_scaling,
                 dropout, activation_fn=nn.ReLU()):
        super().__init__()
        inp_dim = max_features_dim + 2 * D_dim + min(E_node_dim, E_edge_dim)
        self.graph_embedding = GraphEmbedding(inp_dim, E_node_dim, E_edge_dim, out_dim)
        nb_features = None if (nb_features is None or nb_features <= 0) else nb_features
        self.layers = nn.ModuleList([
            TFEMLayer(out_dim, ff_dim, n_heads, nb_features, ortho_scaling, dropout, activation_fn)
            for _ in range(num_mhd_layers)
        ])

    def forward(self, node_features, edge_features, mask):
        xp = self.graph_embedding(node_features, edge_features)  

        empty_vector = torch.zeros(xp.size(0), 1, xp.size(2), device=xp.device)
        mask = torch.cat([mask, torch.ones(mask.size(0), 1, device=mask.device)], dim=1)
        xp = torch.cat([xp, empty_vector], dim=1)

        outputs = []
        for layer in self.layers:
            xp = layer(xp, mask)
            outputs.append(xp)

        z = torch.stack(outputs, dim=0).mean(dim=0)
        return z  
