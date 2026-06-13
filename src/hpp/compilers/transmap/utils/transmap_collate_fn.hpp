#ifndef TRANSMAP_COLLATE_FN_HPP
#define TRANSMAP_COLLATE_FN_HPP

#include <iostream>
#include "src/hpp/compilers/transmap/rl/transmap_state.hpp"

inline void add_and_pad_features(torch::Tensor& batch, const torch::Tensor& features, int max_size, int batch_idx) {
    int current_size = features.size(0);
    torch::Tensor padded;
    if (current_size < max_size) {
        auto padding = torch::zeros({max_size - current_size, features.size(1)}, torch::dtype(features.dtype()).device(features.device()));
        padded = torch::cat({features, padding}, 0);
    } else {
        padded = features;
    }
    batch[batch_idx] = padded;
}

inline std::vector<c10::IValue> transmap_collate_fn(const std::vector<std::shared_ptr<TransMapState>>& states, const std::vector<int>& indices) {
    #ifdef DEBUG
    std::cout << "\n=== Starting TransMap collate fn ===" << std::endl;
    std::cout << "Processing batch of " << indices.size() << " states\n";
    #endif

    int max_dfg_node_size = 0;
    int max_dfg_edge_size = 0;

    auto first_state = states[0];

    int max_adg_node_size = first_state->get_number_of_spatial_PEs();
    int max_adg_edge_size = 0;
    for (int idx : indices) {
        const auto& state = states[idx];
        max_dfg_node_size = std::max(max_dfg_node_size, state->get_dfg_size());
        max_dfg_edge_size = std::max(max_dfg_edge_size, state->get_num_dfg_edges());
        max_adg_edge_size = std::max(max_adg_edge_size, state->get_num_cgra_spatial_edges());

    }

    int batch_size = indices.size();

    int dfg_node_dim = first_state->get_dfg_node_features_size();
    int dfg_edge_dim = first_state->get_dfg_edge_features_size();
    int adg_node_dim = first_state->get_adg_node_features_size();
    int adg_edge_dim = first_state->get_adg_edge_features_size();

    torch::Tensor batch_dfg_node_features = torch::zeros({batch_size, max_dfg_node_size, dfg_node_dim}, torch::kFloat32);
    torch::Tensor batch_dfg_edge_features = torch::zeros({batch_size, max_dfg_edge_size, dfg_edge_dim}, torch::kFloat32);
    torch::Tensor batch_adg_node_features = torch::zeros({batch_size, max_adg_node_size, adg_node_dim}, torch::kFloat32);
    torch::Tensor batch_adg_edge_features = torch::zeros({batch_size, max_adg_edge_size, adg_edge_dim}, torch::kFloat32);
    torch::Tensor batch_node_to_be_mapped_features = torch::zeros({batch_size, dfg_node_dim}, torch::kFloat32);
    torch::Tensor batch_pe_masks = torch::empty({batch_size, max_adg_node_size}, torch::kBool);
    torch::Tensor batch_dfg_mask_pad = torch::zeros({batch_size, max_dfg_node_size + max_dfg_edge_size}, torch::kBool);
    torch::Tensor batch_adg_mask_pad = torch::zeros({batch_size, max_adg_node_size + max_adg_edge_size}, torch::kBool);

    #ifdef DEBUG
    std::cout << "Max DFG node size: " << max_dfg_node_size << ", edge size: " << max_dfg_edge_size << std::endl;
    std::cout << "Max ADG node size: " << max_adg_node_size << ", edge size: " << max_adg_edge_size << std::endl;
    std::cout << "--- Processing states ---" << std::endl;
    #endif

    for (int batch_idx = 0; batch_idx < static_cast<int>(indices.size()); ++batch_idx) {
        const int idx = indices[batch_idx];
        const auto& state = states[idx];
    
        #ifdef DEBUG
        std::cout << "Processing state " << idx << std::endl;
        #endif
    
        auto dfg_node_feat = state->get_dfg_node_features();
        auto dfg_edge_feat = state->get_dfg_edge_features();
        int dfg_nodes = state->get_dfg_size();
        int dfg_edges = state->get_num_dfg_edges();
    
        add_and_pad_features(batch_dfg_node_features, dfg_node_feat, max_dfg_node_size, batch_idx);
        add_and_pad_features(batch_dfg_edge_features, dfg_edge_feat, max_dfg_edge_size, batch_idx);
    
        auto adg_node_feat = state->get_adg_node_features();
        auto adg_edge_feat = state->get_adg_edge_features();
        int adg_nodes = state->get_number_of_spatial_PEs();
        int adg_edges = state->get_num_cgra_spatial_edges();
    
        batch_adg_node_features[batch_idx] = adg_node_feat;
        add_and_pad_features(batch_adg_edge_features, adg_edge_feat, max_adg_edge_size, batch_idx);
    
        batch_node_to_be_mapped_features[batch_idx] = state->get_node_to_be_mapped_features().squeeze();
        batch_pe_masks[batch_idx] = state->get_mask().squeeze();
    
        batch_dfg_mask_pad[batch_idx].slice(0, 0, dfg_nodes).fill_(true);
        batch_dfg_mask_pad[batch_idx].slice(0, max_dfg_node_size, max_dfg_node_size + dfg_edges).fill_(true);
    
        batch_adg_mask_pad[batch_idx].slice(0, 0, adg_nodes).fill_(true);
        batch_adg_mask_pad[batch_idx].slice(0, max_adg_node_size, max_adg_node_size + adg_edges).fill_(true);
    }

    // assert(batch_adg_node_features.size(1) + batch_adg_edge_features.size(1) == batch_adg_mask_pad.size(1));
    return {
        batch_dfg_node_features,
        batch_dfg_edge_features,
        batch_adg_node_features,
        batch_adg_edge_features,
        batch_node_to_be_mapped_features,
        batch_dfg_mask_pad,
        batch_adg_mask_pad,
        batch_pe_masks
    };
}

#endif
