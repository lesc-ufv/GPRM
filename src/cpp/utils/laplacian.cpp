// #include "src/hpp/utils/laplacian.hpp"

// calc_
// torch::Tensor laplacian_matrix(const int& num_nodes, const std::vector<std:vector<int>>& edge_indices) {
//     auto adjacency_matrix = torch::zeros({num_nodes, num_nodes});
//     auto edge_indices_tensor = matrix_to_tensor(edge_indices, torch::kInt32); 
//     auto inv_edge_indices_tensor = torch::flip(edge_indices_tensor, {1});
    
//     adjacency_matrix.index_put_({edge_indices_tensor}, 1);
//     adjacency_matrix.index_put_({inv_edge_indices_tensor}, 1);

//     auto identity = torch::eye(num_nodes);

//     auto degree_matrix = torch::diag(adjacency_matrix.sum(1));

//     auto norm_degree_matrix = torch::sqrt(torch::inverse(degree_matrix));

//     auto laplacian_matrix = identity - torch::matmul(torch::matmul(norm_degree_matrix, adjacency_matrix), norm_degree_matrix);
//     return laplacian_matrix;
// }

// torch::Tensor laplacian_embeddings(torch::Tensor laplacian_matrix, const int& max_embeddings) {
//     auto [evals, evects] = torch::linalg::eigh(laplacian_matrix);

//     int num_nodes = laplacian_matrix.size(0);

//     auto idxs = torch::argsort(evals, 0, false).slice(0, 0, max_embeddings);

//     evals = evals.index_select(0, idxs);
//     #ifdef DEBUG
//         std::cout << "Ordered evals" <<evals << std::endl;
//     #endif
//     evects = evects.index_select(1, idxs); 

//     if (num_nodes < max_embeddings) {
//         auto diff = max_embeddings - num_nodes;
//         auto padding = torch::zeros({num_nodes, diff});
//         evects = torch::cat({evects, padding}, 1);
//     }

//     return evects;
// }

// torch::Tensor gen_laplacian_embeddings(const int& num_nodes, const std::vector<std::pair<int, int>>& edge_indices, const int& max_embeddings) {
//     auto laplacian = laplacian_matrix(num_nodes, edge_indices);
//     return laplacian_embeddings(laplacian, max_embeddings);
// }

