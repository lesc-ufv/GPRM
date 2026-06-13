import torch
from torch import nn
from src.python.models.transmap.tfem import TFEM
class TransMap(nn.Module):
    def __init__(self, num_mhd_layers, max_dfg_feat, max_adg_feat, lpe_dim, E_dfg_node_dim, E_dfg_edge_dim, E_adg_node_dim,E_adg_edge_dim,
                  out_dim_policy, state_dim, out_dim, ff_dim, n_heads, nb_features, hidden_dim, ortho_scaling = 0, dropout = 0, activation_fn = "relu"):
        super().__init__()
        if activation_fn == "relu":
            activation_fn = nn.ReLU()
        else:
            raise ValueError("[TransMap] Invalid activation fn.")
        
        self.dfg_transformer = TFEM(num_mhd_layers,max_dfg_feat, lpe_dim, E_dfg_node_dim, E_dfg_edge_dim,out_dim,
                                     ff_dim,n_heads, nb_features,ortho_scaling,dropout,activation_fn)
        self.adg_transformer = TFEM(num_mhd_layers,max_adg_feat, lpe_dim, E_adg_node_dim, E_adg_edge_dim, out_dim,
                                     ff_dim,n_heads, nb_features, ortho_scaling, dropout,activation_fn)
        self.fc = nn.Linear(2*out_dim + max_dfg_feat + 2*lpe_dim, state_dim)   

        self.policy_network = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.Linear(hidden_dim, out_dim_policy))
        self.value_network = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                            nn.Linear(hidden_dim, hidden_dim),
                                    nn.Linear(hidden_dim, 1))
    
    def get_critic(self):
        return self.value_network
    
    def get_actor(self):
        return self.policy_network
    
    def forward(self, dfg_node_features, dfg_edge_features, adg_node_features, adg_edge_features, node_to_be_mapped_features,
                 dfg_padding_mask, adg_padding_mask, pe_mask ):
        # [batch_size, seq_len_2]
        # dfg_node_features: [batch_size, seq_len_1, dfg_node_dim + 2*D_dim]
        # dfg_edge_features: [batch_size, seq_len_1, dfg_edge_dim + 2*D_dim]
        # adg_node_features: [batch_size, seq_len_2, adg_node_dim + 2*D_dim]
        # adg_edge_features: [batch_size, seq_len_2, adg_edge_dim + 2*D_dim] 
        # node_to_be_mapped_features: [batch_size, dfg_node_dim] 
        dfg_state = self.dfg_transformer(dfg_node_features, dfg_edge_features, dfg_padding_mask)[:,-1,:] # [batch_size, out_dim]
        adg_state = self.adg_transformer(adg_node_features, adg_edge_features, adg_padding_mask)[:,-1,:] # [batch_size, out_dim]
        mapping_state = torch.cat([dfg_state, adg_state, node_to_be_mapped_features], dim=-1) # [batch_size, 2*out_dim + dfg_node_dim]
        mapping_state = self.fc(mapping_state) # [batch_size, state_dim]
        policy_logits = self.policy_network(mapping_state)  # [batch_size, out_dim_policy]
        masked_policy_logits = torch.where(pe_mask != 0, policy_logits, torch.tensor(-float('inf')).to(policy_logits.device)) # [batch_size, out_dim_policy]
        values = self.value_network(mapping_state) # [batch_size, 1]
        return masked_policy_logits, values # [batch_size, out_dim_policy], [batch_size, 1]

# if __name__ != "__main__":
#     dfg_node_dim = 7
#     adg_node_dim = 5
#     dfg_edge_features_dim = 4
#     adg_edge_features_dim = 2
#     D_dfg_dim = D_adg_dim = 4
#     E_dfg_node_dim = E_adg_node_dim = 32
#     E_dfg_edge_dim = E_dfg_node_dim + dfg_node_dim - dfg_edge_features_dim
#     E_adg_edge_dim = E_adg_node_dim + adg_node_dim - adg_edge_features_dim
#     out_dim_policy = 16
#     state_dim = 32
#     out_dim = 128
#     ff_dim = 256
#     n_heads = 4
#     nb_features = 64
#     ortho_scaling = 1
#     model = TransMap(dfg_node_dim,adg_node_dim, D_dfg_dim, D_adg_dim,E_dfg_node_dim, E_dfg_edge_dim, E_adg_node_dim,E_adg_edge_dim,
#                   out_dim_policy, state_dim, out_dim, ff_dim, n_heads, nb_features, ortho_scaling=ortho_scaling)
#     batch_size = 2
#     seq_len_1 = seq_len2 = 10
#     dfg_node_features = torch.randn(batch_size, seq_len_1, dfg_node_dim + 2*D_dfg_dim)
#     dfg_edge_features = torch.randn(batch_size, seq_len_1, dfg_edge_features_dim + 2*D_dfg_dim)
#     adg_node_features = torch.randn(batch_size, seq_len2, adg_node_dim + 2*D_adg_dim)
#     adg_edge_features = torch.randn(batch_size, seq_len2, adg_edge_features_dim + 2*D_adg_dim)
#     node_to_be_mapped_features = torch.randn(batch_size, dfg_node_dim)
#     padding_mask = torch.randint(0,2,(batch_size, seq_len_1 + seq_len2))
#     pe_mask = torch.randint(0,2,(batch_size, out_dim_policy))
#     masked_policy_logits, values = model(dfg_node_features, dfg_edge_features, adg_node_features, adg_edge_features, node_to_be_mapped_features, padding_mask, pe_mask)
#     print(masked_policy_logits)
#     print(values)