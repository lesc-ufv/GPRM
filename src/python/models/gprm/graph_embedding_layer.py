import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

class GraphEmbeddingLayer(nn.Module):
    def __init__(
        self, in_dim_nodes, in_dim_edges, hidden_dim, out_dim,
        n_heads, n_layers, use_gnn=True, residual=True,
        negative_slope=0.2, dropout=0.0, activation_fn=nn.ReLU()
    ):
        super().__init__()

        self.activation_fn = activation_fn
        dims = [hidden_dim] * (n_layers - 1) + [out_dim]
        node_dims = [in_dim_nodes] + dims
        edge_dims = [in_dim_edges] + dims
        self.n_layers = n_layers
        self.use_gnn = use_gnn

        self.node_enc = nn.ModuleList()
        self.edge_enc = nn.ModuleList()
        self.node_norm = nn.ModuleList()
        self.edge_norm = nn.ModuleList()

        for i in range(n_layers):

            if use_gnn:
                self.node_enc.append(
                    GATv2Conv(
                        node_dims[i], node_dims[i+1],
                        heads=n_heads, concat=False,
                        edge_dim=edge_dims[i],
                        residual=residual,           
                        negative_slope=negative_slope,
                        dropout=dropout
                    )
                )
            else:
                self.node_enc.append(nn.Linear(node_dims[i], node_dims[i+1]))

            self.edge_enc.append(nn.Linear(edge_dims[i], edge_dims[i+1]))

            self.node_norm.append(nn.LayerNorm(node_dims[i]))
            self.edge_norm.append(nn.LayerNorm(edge_dims[i]))

        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, node_features, edge_features, edge_indices):

        for i in range(self.n_layers):
            if i == 0:
                n_in = node_features 
                e_in = edge_features if edge_features.numel() > 0 else None
            else:
                n_in = self.node_norm[i](node_features)
                e_in = self.edge_norm[i](edge_features) if edge_features.numel() > 0 else None
                  
            if self.use_gnn:
                n_out = self.node_enc[i](n_in, edge_indices, e_in)
            else:
                n_out = self.node_enc[i](n_in)

            e_out = self.edge_enc[i](e_in) if e_in is not None else None


            if i < self.n_layers - 1:
                n_out = self.activation_fn(n_out)
                if e_out is not None:
                    e_out = self.activation_fn(e_out)

            node_features = n_out
            if e_out is not None:
                edge_features = e_out

        return self.dropout(node_features), (
            self.dropout(edge_features) if edge_features.numel() > 0 else edge_features
        )
