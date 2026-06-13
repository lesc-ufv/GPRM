from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter)
import numpy as np
import torch
import torch.nn.functional as F

def get_laplacian_emb(edge_index, number_of_nodes, is_undirected, D_dim):
        if not is_undirected:
            undir_edge_index = to_undirected(edge_index)
        else:
            undir_edge_index = edge_index
        L = to_scipy_sparse_matrix(
                *get_laplacian(undir_edge_index, normalization='sym',
                          )
            )
        evals, evects = np.linalg.eigh(L.toarray())
        idx = evals.argsort()[:D_dim]
        evals, evects = evals[idx], np.real(evects[:, idx])
        evals = torch.from_numpy(np.real(evals)).clamp_min(0)

        evects = torch.from_numpy(evects).float()
        denom = evects.norm(p=2, dim=0, keepdim=True)
        denom = denom.clamp_min(1e-12).expand_as(evects)
        evects = evects / denom

        if number_of_nodes < D_dim:
            EigVecs = F.pad(evects, (0, D_dim - number_of_nodes), value=float(0))
        else:
            EigVecs = evects

        if number_of_nodes < D_dim:
            EigVals = F.pad(evals, (0, D_dim - number_of_nodes ), value=float(0)).unsqueeze(0)
        else:
            EigVals = evals.unsqueeze(0)
        EigVals = EigVals.repeat(number_of_nodes, 1).unsqueeze(2)
        return EigVals, EigVecs
