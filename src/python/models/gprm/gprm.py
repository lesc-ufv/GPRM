import torch
from torch import nn
from src.python.models.gprm.gprm_mlp import GPRMMLP
from src.python.models.gprm.embedding_generation_layer import EmbeddingGenerationLayer
from src.python.models.gprm.graph_embedding_layer import GraphEmbeddingLayer

class GPRM(nn.Module):
    def calc_node_dim(self, emb_dim,):
        return emb_dim
    
    def __init__(self, lpe_dim, vocab_len, feat_out_dim,
                 out_dim, ff_out_dim, GIN_layers_list, n_graph_embed_layers, n_transformer_layers, n_heads,
                 activation_fn_str, use_mlp = False,
                 dropout = 0.0, use_rms_norm = True, use_norm_before_pred = True, negative_slope= 0.02, max_len = 5000,
                 norm_initial_emb = True, use_gin_emb = True, 
                 use_mobility_info = True, use_coord_info =True, use_gnn = True,
                 use_transformer = True, use_placement_order = True, ignore_LPE = False,
                 scale = True):
        super().__init__()
        self.use_placement_order = use_placement_order
        
        if activation_fn_str == "gelu":
            activation_fn = nn.GELU()
        elif activation_fn_str == "relu":
            activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {activation_fn_str}")
        dfg_node_dim = self.calc_node_dim(feat_out_dim)
        dfg_edge_dim = (dfg_node_dim)

        adg_node_dim = self.calc_node_dim(feat_out_dim)
        adg_edge_dim = (adg_node_dim)

        mapping_node_dim = feat_out_dim
        mapping_edge_dim = feat_out_dim
       
        self.embedding_generation = EmbeddingGenerationLayer(lpe_dim, vocab_len, feat_out_dim, GIN_layers_list, dropout, max_len, use_gin= use_gin_emb,
                                                             use_mobility_info = use_mobility_info, use_coord_info = use_coord_info, use_placement_order = use_placement_order,
                                                             ignore_LPE = ignore_LPE)
         

        self.dfg_enc = GraphEmbeddingLayer(dfg_node_dim ,dfg_edge_dim, ff_out_dim, out_dim, n_heads, n_graph_embed_layers,
                                        dropout=dropout, activation_fn = activation_fn,negative_slope=negative_slope, use_gnn=use_gnn)
        
        self.adg_enc = GraphEmbeddingLayer(adg_node_dim,adg_edge_dim, ff_out_dim, out_dim, n_heads, n_graph_embed_layers,
                                        dropout = dropout, activation_fn = activation_fn, negative_slope=negative_slope, use_gnn=use_gnn)
        
        self.mapping_enc = GraphEmbeddingLayer(mapping_node_dim, mapping_edge_dim, ff_out_dim, out_dim, n_heads, n_graph_embed_layers,        
                                                dropout = dropout, activation_fn = activation_fn, negative_slope=negative_slope, use_gnn=use_gnn)
        
        self.use_transformer_layer = use_transformer
        self.policy = GPRMMLP(out_dim,1,dropout,activation_fn) if use_mlp else nn.Linear(out_dim, 1)
        self.value = GPRMMLP(out_dim,1, dropout,activation_fn) if use_mlp else nn.Linear(out_dim, 1)

        self.attn_outputs = []
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=n_heads, dim_feedforward=ff_out_dim,
                                               dropout=dropout, activation=activation_fn, batch_first=True, norm_first=True)
            self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers,
                                                        norm = nn.RMSNorm(out_dim) if use_rms_norm else nn.LayerNorm(out_dim))

            def hook(module, input, output):
                self.attn_outputs = output[1]
                
            self.transformer_enc.layers[-1].self_attn.register_forward_hook(hook)
            
        else: 
            self.mlp = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(out_dim, ff_out_dim),
                        activation_fn,
                        nn.Dropout(dropout),
                        nn.Linear(ff_out_dim, out_dim)
                    )
                    for _ in range(n_transformer_layers)
                ]
            )
        if use_norm_before_pred:
            if use_rms_norm:
                self.pre_norm = nn.RMSNorm(out_dim)
                self.post_norm = nn.RMSNorm(out_dim)
            else:
                self.pre_norm = nn.LayerNorm(out_dim)
                self.post_norm = nn.LayerNorm(out_dim)

        self.norm_initial_emb = norm_initial_emb
        self.use_norm_before_pred = use_norm_before_pred

        if norm_initial_emb:
            Norm = nn.RMSNorm if use_rms_norm else nn.LayerNorm

            self.dfg_node_emb_norm     = Norm(dfg_node_dim)
            self.adg_node_emb_norm     = Norm(adg_node_dim)
            self.mapping_node_emb_norm = Norm(mapping_node_dim)

            self.dfg_edge_emb_norm     = Norm(dfg_edge_dim)
            self.adg_edge_emb_norm     = Norm(adg_edge_dim)
            self.mapping_edge_emb_norm = Norm(mapping_edge_dim)

        self.out_dim = out_dim
        self.log_path = "log.log"
        self.state_readout_emb = nn.Parameter(torch.randn(out_dim), requires_grad=True)
        self.map_node_proj = nn.Linear(dfg_node_dim + adg_node_dim, feat_out_dim)
        self.map_edge_proj = nn.Linear(dfg_edge_dim + adg_edge_dim, feat_out_dim)
        self.scale = scale
        
    def get_critic(self):
        return self.value
    
    def get_actor(self):
        return self.policy
    
    def forward(self, 
                dfg_node_features, 
                dfg_edge_indices, 
                dfg_placement_order_ids,
                dfg_node_scheduling_ids , 
                dfg_ops_ids,
                adg_node_features, adg_edge_indices, 
                adg_ops_ids, adg_pos,
                mapping_edge_indices, placement_idx, routing_idx, batch_idx, seq_idx,
                contextual_action_seq_idx, action_batch_idx, action_seq_idx,
                state_idx, max_seq_len, max_legal_actions):
        max_seq_len = max_seq_len.to("cpu").item()
        max_legal_actions = max_legal_actions.to("cpu").item()


        batch_size = state_idx.size(0)
        device = adg_node_features.device
        states_emb = self.state_readout_emb.unsqueeze(0).expand(batch_size, -1)

        dfg_node_features, dfg_edge_features = self.embedding_generation(
            dfg_node_features , dfg_edge_indices,
            dfg_placement_order_ids if self.use_placement_order else None, dfg_ops_ids, None,
            dfg_node_scheduling_ids
        )

        adg_node_features, adg_edge_features = self.embedding_generation(
            adg_node_features, adg_edge_indices,
            None, adg_ops_ids, adg_pos, None
        )

        if placement_idx.numel() != 0:
            mapping_node_features = torch.cat([
                dfg_node_features[placement_idx[0]],
                adg_node_features[placement_idx[1]]
            ], dim=-1)
            mapping_node_features = self.map_node_proj(mapping_node_features)
        else:
            mapping_node_features = torch.tensor([], device=dfg_node_features.device)


        if routing_idx.numel() != 0:
            mapping_edge_features = torch.cat([
                dfg_edge_features[routing_idx[0]],
                adg_edge_features[routing_idx[1]]
            ], dim=-1)
            mapping_edge_features = self.map_edge_proj(mapping_edge_features)
        else:
            mapping_edge_features = torch.tensor([], device=dfg_node_features.device)

  
        if self.norm_initial_emb:
            dfg_node_features = self.dfg_node_emb_norm(dfg_node_features)
 
            adg_node_features = self.adg_node_emb_norm(adg_node_features)

            dfg_edge_features = self.dfg_edge_emb_norm(dfg_edge_features)

            adg_edge_features = self.adg_edge_emb_norm(adg_edge_features)

            if mapping_node_features.numel() > 0:
                mapping_node_features = self.mapping_node_emb_norm(mapping_node_features)
            if mapping_edge_features.numel() > 0:
                mapping_edge_features = self.mapping_edge_emb_norm(mapping_edge_features)

    
        dfg_node_features, dfg_edge_features = self.dfg_enc(dfg_node_features, dfg_edge_features, dfg_edge_indices)
        
        adg_node_features, adg_edge_features = self.adg_enc(adg_node_features, adg_edge_features, adg_edge_indices)
        if mapping_node_features.numel() > 0:
            mapping_node_features, mapping_edge_features = self.mapping_enc(mapping_node_features, mapping_edge_features, mapping_edge_indices)
            
        padding_mask = torch.ones(batch_size, max_seq_len).to(device)

        final_embedding = torch.zeros(batch_size, max_seq_len, self.out_dim).to(device)
        
        temp_seq = [dfg_node_features, dfg_edge_features, adg_node_features, adg_edge_features,mapping_node_features, 
                            mapping_edge_features, states_emb]
        
        embed_seq = [value for value in temp_seq if self.is_valid_tensor(value)]
        
        batch_indices = batch_idx
        seq_indices = seq_idx

        
        embeddings = torch.cat(embed_seq,dim=0)

        final_embedding[batch_indices, seq_indices] = embeddings
        
        padding_mask[batch_indices, seq_indices] = 0 
        padding_mask = padding_mask.bool()

        if self.use_norm_before_pred:
            final_embedding = self.pre_norm(final_embedding)

        if self.scale:
            final_embedding = final_embedding * (self.out_dim ** 0.5)

        if self.use_transformer_layer:
            contextual_mapping_embedding = self.transformer_enc(final_embedding, src_key_padding_mask=padding_mask)
        else:
            contextual_mapping_embedding = self.mlp(final_embedding)

        legal_actions_emb = contextual_mapping_embedding[action_batch_idx,contextual_action_seq_idx]
        
        if self.use_norm_before_pred:
            legal_actions_emb = self.post_norm(legal_actions_emb)

        final_legal_actions_emb = torch.zeros((batch_size, max_legal_actions, final_embedding.size(-1)), device=legal_actions_emb.device)

        final_legal_actions_emb[action_batch_idx, action_seq_idx] = legal_actions_emb

        policy_logits = self.policy(final_legal_actions_emb).squeeze(-1)  
        valid_actions_mask = torch.zeros((batch_size, max_legal_actions), dtype=torch.bool, device=legal_actions_emb.device)
        valid_actions_mask[action_batch_idx, action_seq_idx] = True

        masked_policy_logits = policy_logits.masked_fill(~valid_actions_mask, -float('inf'))

        batch_state_idx = torch.arange(0,batch_size)
        states_emb = contextual_mapping_embedding[batch_state_idx, state_idx]
        if self.use_norm_before_pred:
            states_emb = self.post_norm(states_emb)

        return masked_policy_logits, self.value(states_emb)


    def is_valid_tensor(self,x):
        return isinstance(x, torch.Tensor) and x.numel() > 0