from torch_geometric.nn import GATConv
import torch
from typing import Union
from torch_geometric.data import Data, Batch
from torch import Tensor

class GraphEmbeddingGeneration(torch.nn.Module):
    def __init__(self, dim_feat_dfg,dim_feat_cgra, dim_out,
                slope_leaky_relu, n_heads):
        super(GraphEmbeddingGeneration,self).__init__()
        self.dim_out = dim_out
        self.gat_dfg = GATConv(in_channels=dim_feat_dfg,out_channels=dim_out,heads=n_heads,concat=False) 
        self.gat_cgra = GATConv(in_channels=dim_feat_cgra,out_channels=dim_out,heads=n_heads,concat=False)
        self.fc_metadata = torch.nn.Linear(dim_feat_dfg,dim_out)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(3*dim_out,dim_out),
                                       torch.nn.Linear(dim_out,dim_out))
        self.slope_leaky_relu = slope_leaky_relu 

    def forward(self, dfg_features: Tensor, dfg_edge_indices: Tensor, cgra_features: Tensor, cgra_edge_indices: Tensor,
                vertex_to_be_mapped_feature: Tensor, dfg_padding_mask: Tensor, cgra_padding_mask: Tensor):
        
        batch_size = vertex_to_be_mapped_feature.size(0)
        #(all_nodes,features)
        embeddings_dfg = self.gat_dfg(dfg_features, dfg_edge_indices)


        #(batch,nodes,features)
        embeddings_dfg = torch.reshape(embeddings_dfg,(batch_size,-1,self.dim_out))
        final_embeddings_dfg = (embeddings_dfg * dfg_padding_mask.unsqueeze(-1)).sum(-2) / dfg_padding_mask.sum(dim=-1).unsqueeze(-1)


        #(all_nodes,features)
        embeddings_cgra = self.gat_cgra(cgra_features, cgra_edge_indices)
        
        #(batch,nodes,features)
        embeddings_cgra = torch.reshape(embeddings_cgra,(batch_size, -1, self.dim_out))
        
        final_embeddings_cgra = (embeddings_cgra * cgra_padding_mask.unsqueeze(-1)).sum(-2) / cgra_padding_mask.sum(dim=-1).unsqueeze(-1)
        
        #(batch,features)
        embedding_vertex_to_be_mapped = self.fc_metadata(vertex_to_be_mapped_feature)
        
        #(batch,3*features)
        state_vector = torch.concatenate([final_embeddings_dfg,final_embeddings_cgra,embedding_vertex_to_be_mapped],dim=-1)
        
        #(batch,3*features)
        final_embedding = self.mlp(state_vector)

        return torch.nn.functional.leaky_relu(final_embedding,self.slope_leaky_relu)