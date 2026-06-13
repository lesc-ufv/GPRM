import torch
from torch import nn
from torch_geometric.nn import GIN
from src.python.models.gprm.positional_encoders import *


class EmbeddingGenerationLayer(nn.Module):
    def __init__(self, lpe_dim, len_tokens, out_dim, n_GIN_layers_list, dropout = 0.0, max_len = 5000, use_gin = True, 
                  use_mobility_info = True, use_coord_info =True, use_placement_order = True, ignore_LPE = False):
        super().__init__()
        self.out_dim = out_dim
        self.token_enc = nn.Embedding(len_tokens, out_dim)
        pe = get_positional_encoding(max_len, out_dim)
        self.placement_order_enc = PlacementOrderEncoder(pe, out_dim) if use_placement_order else None
        self.mobility_encoder = MobilityEncoder(pe, out_dim) if use_mobility_info else None
        self.pe_pos_enc = PEPositionalEncoder(pe, out_dim) if use_coord_info else None
        self.use_gin = use_gin
        self.ignore_LPE = ignore_LPE

        self.gin_enc_list = nn.ModuleList([
            GIN(1, out_dim, n_layers, dropout = dropout, train_eps = True) for n_layers in n_GIN_layers_list
        ])
        self.n_gin_models = len(n_GIN_layers_list)

        self.dfg_lpe_enc = LPEEncoder(lpe_dim, out_dim)
        self.adg_lpe_enc = LPEEncoder(lpe_dim, out_dim)

        dfg_max_sources = (0 if ignore_LPE else 1) + 1 + self.n_gin_models + (1 if use_mobility_info else 0) + (1 if use_placement_order else 0)
        adg_max_sources = (0 if ignore_LPE else 1) + 1 + self.n_gin_models + (1 if use_coord_info else 0)
        self.dfg_final_node_proj = nn.Linear(dfg_max_sources * out_dim, out_dim)
        self.adg_final_node_proj = nn.Linear((adg_max_sources) * out_dim , out_dim)
        self.node_ln = nn.LayerNorm(out_dim) # not used currently;
        self.gin_proj_list = nn.ModuleList([nn.Identity() for _ in range(self.n_gin_models)])
        self.dfg_edge_from_nodes_proj = nn.Linear(2 * dfg_max_sources * out_dim, out_dim)
        self.adg_edge_from_nodes_proj = nn.Linear(2 * adg_max_sources * out_dim, out_dim)


    def forward(self, node_features, edge_indices, placement_order_ids, op_ids, adg_pos, node_temporal_ids):
        device = node_features.device
        node_feature_list = []
        is_dfg = (adg_pos is None)

        if not self.ignore_LPE and node_features is not None:
            node_in = self.dfg_lpe_enc(node_features) if is_dfg else self.adg_lpe_enc(node_features)
            node_feature_list.append(node_in)

        if op_ids is not None:
            op_emb = self.token_enc(op_ids)  
            node_feature_list.append(op_emb)

        if self.use_gin:
            ones = torch.ones(node_features.size(0), 1, device=device)
            for i, gin in enumerate(self.gin_enc_list):
                gin_out = gin(ones, edge_indices)  
                gin_out = self.gin_proj_list[i](gin_out)
                node_feature_list.append(gin_out)

        if node_temporal_ids is not None and self.mobility_encoder is not None:
            sched_enc = self.mobility_encoder(node_temporal_ids)
            node_feature_list.append(sched_enc)

        if placement_order_ids is not None and self.placement_order_enc is not None:
            placement_order_emb = self.placement_order_enc(placement_order_ids)
            node_feature_list.append(placement_order_emb)

        if adg_pos is not None and self.pe_pos_enc is not None:
            pos_x, pos_y, pos_z = adg_pos[:,0].long(), adg_pos[:,1].long(), adg_pos[:,2].long()
            pos_emb = self.pe_pos_enc(pos_x, pos_y, pos_z)
            node_feature_list.append(pos_emb)

    
        node_cat = torch.cat(node_feature_list, dim=-1) 
        node_features = self.dfg_final_node_proj(node_cat) if is_dfg else self.adg_final_node_proj(node_cat)

        if edge_indices is not None and node_features.numel() > 0:
            edge_concat = torch.cat([node_cat[edge_indices[0]], node_cat[edge_indices[1]]], dim=-1)
            edge_features = self.dfg_edge_from_nodes_proj(edge_concat) if is_dfg else self.adg_edge_from_nodes_proj(edge_concat)
        return node_features, edge_features
