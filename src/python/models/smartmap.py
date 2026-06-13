import torch
from src.python.models.graph_embedding_generation import GraphEmbeddingGeneration
from src.python.models.general_policy_net import GeneralPolicyNet
from src.python.models.value_net import ValueNet
from torch_geometric.data import Data, Batch
import numpy as np

class SmartMap(torch.nn.Module):
    def __init__(self,dim_dfg_feat,dim_cgra_feat,out_dim,n_heads, negative_slope = 0.02):
        super().__init__()
        self.graph_embed_generation = GraphEmbeddingGeneration(dim_dfg_feat,dim_cgra_feat,out_dim,negative_slope,n_heads)
        self.value_net = ValueNet(out_dim,out_dim,negative_slope)
        self.policy_net = GeneralPolicyNet(out_dim,dim_cgra_feat,out_dim,out_dim,negative_slope)
    def get_critic(self):
        return self.value_net
    
    def get_actor(self):
        return self.policy_net
    
    def get_shared_paremeters(self):
        return self.graph_embed_generation.parameters()
    
    def forward(self,dfg_features, dfg_edge_indices, cgra_features, cgra_edge_indices,vertex_to_be_mapped_features,actions,pad_mask_actions,pad_dfg_mask,pad_cgra_mask):
        state_vector = self.graph_embed_generation( dfg_features, dfg_edge_indices, cgra_features, cgra_edge_indices,
                                                    vertex_to_be_mapped_features,
                                                    pad_dfg_mask,
                                                    pad_cgra_mask)
        p = self.policy_net(state_vector,actions)
        v = self.value_net(state_vector)
        p = torch.where(pad_mask_actions == 0,-torch.inf,p)
        return p,v
