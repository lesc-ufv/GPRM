import torch
from torch import nn
from src.python.models.gprm.gprm_mlp import GPRMMLP
from src.python.models.gprm.embedding_generation_layer import EmbeddingGenerationLayer
from src.python.models.gprm.graph_embedding_layer import GraphEmbeddingLayer
from python.models.gprm.placement_order_encoder import OriginalPositionalEncoding
class ED_GPRM(nn.Module):
    def calc_node_dim(self,lpe_dim, emb_dim, n_gin_layers, use_map_info, is_dfg):
        dim = lpe_dim
        if n_gin_layers:
            dim += n_gin_layers*emb_dim
        if use_map_info:
            dim += 5*emb_dim if is_dfg else 6*emb_dim
        return dim
    
    def __init__(self, lpe_dim, vocab_len, feat_out_dim,
                 out_dim, ff_out_dim, GIN_layers_list, n_graph_embed_layers, n_transformer_enc_layers,
                 n_transformer_dec_layers, n_heads,
                 activation_fn_str, use_mlp = False,
                 dropout = 0.0, use_rms_norm = True, use_norm_before_pred = True, negative_slope= 0.02, max_len = 5000,
                 norm_initial_emb = True, use_gin_emb = True, 
                 use_map_emb_info = True, use_spatio_temporal_info = True, use_gnn = True, concat_state_action = True):
        super().__init__()
        n_gin_models = len(GIN_layers_list)
        if activation_fn_str == "gelu":
            activation_fn = nn.GELU()
        elif activation_fn_str == "relu":
            activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {activation_fn_str}")
        dfg_node_dim = self.calc_node_dim(lpe_dim, feat_out_dim, n_gin_models, use_map_emb_info, True)
        dfg_edge_dim = 2*dfg_node_dim

        adg_node_dim = self.calc_node_dim(lpe_dim, feat_out_dim, n_gin_models, use_map_emb_info, False)
        adg_edge_dim = 2*adg_node_dim

        mapping_node_dim = adg_node_dim + dfg_node_dim
        mapping_edge_dim = 2*mapping_node_dim
       
        self.embedding_generation = EmbeddingGenerationLayer(vocab_len, feat_out_dim, GIN_layers_list, dropout, max_len, use_gin_emb= use_gin_emb,
                                                             use_map_info_emb=use_map_emb_info, use_spatio_temporal_info=use_spatio_temporal_info)
         

        self.dfg_enc = GraphEmbeddingLayer(dfg_node_dim,dfg_edge_dim, ff_out_dim, out_dim, n_heads, n_graph_embed_layers,
                                        dropout=dropout, activation_fn = activation_fn,negative_slope=negative_slope, use_gnn=use_gnn)
        
        self.adg_enc = GraphEmbeddingLayer(adg_node_dim,adg_edge_dim, ff_out_dim, out_dim, n_heads, n_graph_embed_layers,
                                        dropout = dropout, activation_fn = activation_fn, negative_slope=negative_slope, use_gnn=use_gnn)
        

        self.transformer_enc = nn.TransformerEncoder(out_dim, n_heads, n_transformer_enc_layers, n_transformer_enc_layers, ff_out_dim, 
                                             dropout=dropout, activation = activation_fn )
        self.transformer_dec = nn.TransformerDecoder(out_dim, n_heads, n_transformer_dec_layers, n_transformer_dec_layers, ff_out_dim, 
                                             dropout=dropout, activation = activation_fn )
        
        self.decode_inp_proj = nn.Linear(2*out_dim, out_dim)
        if use_norm_before_pred:
            if use_rms_norm:
                self.emb_layer_norm = nn.RMSNorm(out_dim)
            else:
                self.emb_layer_norm = nn.LayerNorm(out_dim)

        self.norm_initial_emb = norm_initial_emb
        self.use_norm_before_pred = use_norm_before_pred
        self.positional_encoding = OriginalPositionalEncoding(out_dim, dropout, max_len)
        self.concat_state_action = concat_state_action
        if norm_initial_emb:
            self.dfg_node_emb_norm =  nn.LayerNorm(dfg_node_dim)
            self.adg_node_emb_norm = nn.LayerNorm(adg_node_dim)
            self.mapping_node_emb_norm = nn.LayerNorm(mapping_node_dim )
            
            self.dfg_edge_emb_norm = nn.LayerNorm(dfg_edge_dim)
            self.adg_edge_emb_norm = nn.LayerNorm(adg_edge_dim)
            self.mapping_edge_emb_norm = nn.LayerNorm(mapping_edge_dim)

        self.map_emb = nn.Parameter(torch.randn(1, 1, out_dim), requires_grad = True)
        self.out_dim = out_dim

        self.policy = GPRMMLP(2*out_dim if  self.concat_state_action else out_dim ,1) if use_mlp else nn.Linear(out_dim, 1)
        self.value = GPRMMLP(out_dim,1) if use_mlp else nn.Linear(out_dim, 1)

    def forward(self, dfg_node_features, dfg_edge_indices, dfg_placement_type_ids, dfg_routing_type_ids, dfg_node_scheduling_ids, 
                dfg_ops_ids,
                adg_node_features, adg_edge_indices, adg_placement_type_ids, adg_routing_type_ids, adg_ops_ids, adg_pos,
                placement_idx, batch_idx, seq_idx,
                contextual_action_seq_idx, action_batch_idx, action_seq_idx,
                state_idx, max_seq_len, max_legal_actions):

        max_seq_len = max_seq_len.to("cpu").item()
        max_legal_actions = max_legal_actions.to("cpu").item()

        batch_size = state_idx.size(0)
        device = adg_node_features.device
        state_emb = torch.ones((batch_size, self.out_dim), device=device)

        dfg_node_features, dfg_edge_features = self.embedding_generation(
            dfg_node_features, None, dfg_edge_indices, dfg_ops_ids, None,
            dfg_placement_type_ids, dfg_routing_type_ids, dfg_node_scheduling_ids
        ) 

        adg_node_features, adg_edge_features = self.embedding_generation(
            adg_node_features, None, adg_edge_indices, adg_ops_ids, adg_pos,
            adg_placement_type_ids, adg_routing_type_ids, None
        )

        if placement_idx.numel() != 0:
            decoder_input = torch.cat([
                dfg_node_features[placement_idx[0]],
                adg_node_features[placement_idx[1]]
            ], dim=-1)
        else:
            decoder_input = torch.tensor([], device=device)

        if self.norm_initial_emb:
            dfg_node_features = self.dfg_node_emb_norm(dfg_node_features)
            adg_node_features = self.adg_node_emb_norm(adg_node_features)
            dfg_edge_features = self.dfg_edge_emb_norm(dfg_edge_features)
            adg_edge_features = self.adg_edge_emb_norm(adg_edge_features)
   
        dfg_node_features, dfg_edge_features = self.dfg_enc(dfg_node_features, dfg_edge_features, dfg_edge_indices)
        adg_node_features, adg_edge_features = self.adg_enc(adg_node_features, adg_edge_features, adg_edge_indices)

        src_key_padding_mask = torch.ones(batch_size, max_seq_len, device=device, dtype=torch.bool)
        final_embedding = torch.zeros(batch_size, max_seq_len, self.out_dim, device=device)

        temp_seq = [dfg_node_features, dfg_edge_features, adg_node_features, adg_edge_features, state_emb]
        embed_seq = [value for value in temp_seq if self.is_valid_tensor(value)]
        embeddings = torch.stack(embed_seq, dim=0).reshape(-1, self.out_dim)

        expanded_indices = seq_idx.unsqueeze(1).expand(-1, self.out_dim)  
        final_embedding.scatter_(1, expanded_indices.unsqueeze(1), embeddings.unsqueeze(1))

        src_key_padding_mask.scatter_(0, batch_idx.unsqueeze(1), torch.tensor(False, device=device))

        contextual_mapping_embedding = self.transformer_enc(final_embedding, src_key_padding_mask = src_key_padding_mask)
        tgt_key_padding_mask = ...

        self.outs = self.transformer_dec(decoder_input, contextual_mapping_embedding, tgt_key_padding_mask = tgt_key_padding_mask, memory_key_padding_mask = src_key_padding_mask, 
                                         tgt_is_causal = True)
        if self.use_norm_before_pred:
            final_embedding = self.emb_layer_norm(final_embedding)

        legal_actions_emb = contextual_mapping_embedding[action_batch_idx, contextual_action_seq_idx]

        final_legal_actions_emb = torch.zeros((batch_size, max_tgt_seq_len, max_legal_actions, final_embedding.size(-1)), device=device)
        expanded_action_indices = action_seq_idx.unsqueeze(1).expand(-1, legal_actions_emb.size(-1))
        final_legal_actions_emb.scatter_(1, expanded_action_indices.unsqueeze(1), legal_actions_emb.unsqueeze(1))

        valid_actions_mask = torch.zeros((batch_size, max_legal_actions), dtype=torch.bool, device=device)
        valid_actions_mask.scatter_(0, action_batch_idx.unsqueeze(1), torch.tensor(True, device=device))

        masked_policy_logits = self.policy(final_legal_actions_emb).squeeze(-1)
        masked_policy_logits.masked_fill_(~valid_actions_mask, -float('inf'))

        batch_state_idx = torch.arange(batch_size, device=device)
        states_emb = contextual_mapping_embedding[batch_state_idx, state_idx]

        return masked_policy_logits, self.value(states_emb)

    def is_valid_tensor(self,x):
        return isinstance(x, torch.Tensor) and x.numel() > 0


# # Inputs do DFG (Dataflow Graph)
# dfg_node_features = torch.tensor([
#     [0.3162,  0.4472],
#     [0.4472,  0.5117],
#     [0.4472,  0.1954],
#     [0.4472, -0.1954],
#     [0.3162, -0.4472],
#     [0.4472, -0.5117]
# ])

# dfg_edge_indices = torch.tensor([
#     [1, 1, 3, 5, 5],
#     [2, 0, 2, 3, 4]
# ])

# dfg_placement_type_ids = torch.tensor([
#     [4, 3, 4, 4, 4, 4],
#     [6, 6, 6, 6, 6, 5]
# ], dtype=torch.int32)

# dfg_routing_type_ids = torch.tensor([
#     [4, 4, 4, 4, 4],
#     [6, 6, 6, 6, 6]
# ], dtype=torch.int32)

# dfg_node_scheduling_ids = torch.tensor([
#     [1, 0, 2, 1, 1, 0],
#     [2, 1, 2, 1, 2, 0]
# ], dtype=torch.int32)

# dfg_ops_ids = torch.tensor([9, 9, 9, 9, 9, 9])

# # Inputs do ADG (Architecture Design Graph)
# adg_node_features = torch.tensor([
#     [-0.2303, -0.0075],
#     [-0.2462,  0.1043],
#     [-0.2462,  0.3409],
#     [-0.2303,  0.4113],
#     [-0.2462, -0.1168],
#     [-0.2752, -0.0048],
#     [-0.2752,  0.2665],
#     [-0.2462,  0.3449],
#     [-0.2462, -0.3449],
#     [-0.2752, -0.2665],
#     [-0.2752,  0.0048],
#     [-0.2462,  0.1168],
#     [-0.2303, -0.4113],
#     [-0.2462, -0.3409],
#     [-0.2462, -0.1043],
#     [-0.2303,  0.0075]
# ])

# adg_edge_indices = torch.tensor([[ 0,  1,  1,  2,  2,  3,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,
#           8,  8,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 12, 13, 13, 14, 14, 15,
#           0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,
#           9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,  0,  0,  1,  2,
#           3,  3,  4,  7,  8, 11, 12, 12, 13, 14, 15, 15,  0,  0,  1,  1,  1,  2,
#           2,  2,  3,  3,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,
#           8,  8,  8,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13,
#          13, 14, 14, 14, 15, 15],
#         [ 5,  4,  6,  5,  7,  6,  1,  9,  0,  2,  8, 10,  1,  3,  9, 11,  2, 10,
#           5, 13,  4,  6, 12, 14,  5,  7, 13, 15,  6, 14,  9,  8, 10,  9, 11, 10,
#           8,  2,  9,  3, 10,  0, 11,  1, 12,  6, 13,  7, 14,  4, 15,  5,  0, 10,
#           1, 11,  2,  8,  3,  9,  4, 14,  5, 15,  6, 12,  7, 13,  3, 12, 13, 14,
#           0, 15,  7,  4, 11,  8, 15,  0,  1,  2, 12,  3,  4,  1,  5,  2,  0,  6,
#           3,  1,  7,  2,  0,  8,  5,  1,  9,  6,  4,  2, 10,  7,  5,  3, 11,  6,
#           4, 12,  9,  5, 13, 10,  8,  6, 14, 11,  9,  7, 15, 10,  8, 13,  9, 14,
#          12, 10, 15, 13, 11, 14]])

# adg_placement_type_ids = torch.tensor([
#     [4]*16,
#     [5]*16
# ], dtype=torch.int32)

# adg_routing_type_ids = torch.tensor([[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#          4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#          4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#          4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#          4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#          4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
#         [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#          6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#          6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#          6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#          6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#          6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]], dtype=torch.int32)

# adg_ops_ids = torch.tensor([0]*16, dtype=torch.int32)

# adg_pos = torch.tensor([
#     [0, 0, 0],
#     [0, 0, 1],
#     [0, 0, 2],
#     [0, 0, 3],
#     [0, 1, 0],
#     [0, 1, 1],
#     [0, 1, 2],
#     [0, 1, 3],
#     [0, 2, 0],
#     [0, 2, 1],
#     [0, 2, 2],
#     [0, 2, 3],
#     [0, 3, 0],
#     [0, 3, 1],
#     [0, 3, 2],
#     [0, 3, 3]
# ], dtype=torch.int32)

# # Mapping
# mapping_edge_indices = torch.empty((2, 0), dtype=torch.int64)
# # mapping_edge_indices = torch.tensor([[0],[0]], dtype=torch.int64)

# placement_idx = torch.tensor([[1], [11]], dtype=torch.int32)
# routing_idx = torch.empty((2, 0), dtype=torch.int32)

# # Batch and sequence
# batch_idx = torch.zeros(161, dtype=torch.int32)
# seq_idx = torch.arange(161, dtype=torch.int32)

# # Actions
# enc_contextual_action_seq_idx = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=torch.int32)
# enc_action_batch_idx = torch.zeros(16, dtype=torch.int32)
# action_seq_idx = torch.arange(16, dtype=torch.int32)

# # State
# state_idx = torch.tensor([[160]], dtype=torch.int32)

# # Meta-info
# max_seq_len = torch.tensor(161)
# max_legal_actions = torch.tensor(16)

# model = GPRM(
#     lpe_dim=2,                 
#     vocab_len=18,             
#     feat_out_dim=654,          
#     out_dim=24,               
#     ff_out_dim=43,           
#     GIN_layers_list=[32, 32],  
#     n_graph_embed_layers=2,    
#     n_transformer_layers=4,    
#     n_heads=8,                 
#     activation_fn_str="relu", 
#     use_mlp=True,             
#     dropout=0.1,               
#     use_rms_norm=True,        
#     use_norm_before_pred=True,
#     negative_slope=0.02,       
#     max_len=5000,              
#     norm_initial_emb=True,     
#     use_gin_emb=True,          
#     use_map_emb_info=True,     
#     use_gnn=True,              
#     use_transformer=True       
# ).to("cpu")

# out = model( dfg_node_features, dfg_edge_indices, dfg_placement_type_ids, dfg_routing_type_ids, dfg_node_scheduling_ids , 
#                 dfg_ops_ids,
#                 adg_node_features, adg_edge_indices, adg_placement_type_ids, adg_routing_type_ids, adg_ops_ids, adg_pos,
#                 mapping_edge_indices, placement_idx, routing_idx, batch_idx, seq_idx,
#                 enc_contextual_action_seq_idx, enc_action_batch_idx, action_seq_idx,
#                 state_idx, max_seq_len, max_legal_actions)
# print(out)
