#include "src/hpp/compilers/mapzero/utils/mapzero_collate_fn.hpp"
#include <iostream>

std::vector<c10::IValue> mapzero_collate_fn(const std::vector<std::shared_ptr<MapZeroState>> &states, const std::vector<int> &indices)
{
    #ifdef DEBUG
    std::cout << "\n=== Starting mapzero_collate_fn ===" << std::endl;
    std::cout << "Processing batch of " << indices.size() << " states\n";
    #endif

    torch::Tensor batch_dfg_features = torch::tensor({}, torch::dtype(torch::kFloat32)).requires_grad_(false);
    std::vector<torch::Tensor> batch_dfg_edge_indices;
    torch::Tensor batch_cgra_features = torch::tensor({},torch::dtype(torch::kFloat32)).requires_grad_(false);
    std::vector<torch::Tensor> batch_cgra_edge_indices;
    torch::Tensor batch_node_to_be_mapped_features = torch::tensor({},torch::dtype(torch::kFloat32)).requires_grad_(false);
    torch::Tensor batch_masks = torch::tensor({},torch::kBool).requires_grad_(false);
    torch::Tensor batch_target_value = torch::tensor({},torch::dtype(torch::kFloat32)).requires_grad_(false);
    torch::Tensor batch_dfg_mask_pad = torch::tensor({},torch::kBool).requires_grad_(false);
    torch::Tensor batch_cgra_mask_pad = torch::tensor({},torch::kBool).requires_grad_(false);
    
    int max_dfg_size = 0;
    int max_cgra_size = 0;

    for (auto indice: indices) {
        auto state = states[indice];        
        int current_dfg = state->get_dfg_size();
        int current_cgra = state->get_cgra_input_size();
        
        max_dfg_size = std::max(max_dfg_size, current_dfg);
        max_cgra_size = std::max(max_cgra_size, current_cgra);

        #ifdef DEBUG
        std::cout << "State " << indice << ": DFG size=" << current_dfg 
                  << ", CGRA size=" << current_cgra << std::endl;
        #endif
    }

    #ifdef DEBUG
    std::cout << "Max DFG size: " << max_dfg_size << std::endl;
    std::cout << "Max CGRA size: " << max_cgra_size << std::endl;
    std::cout << "\n--- Processing states ---" << std::endl;
    #endif

    for (auto& indice: indices) {
        auto& state = states[indice];

        #ifdef DEBUG
        std::cout << "\nProcessing state " << indice << std::endl;
        #endif
        
        auto dfg_features = state->get_dfg_features();
        auto dfg_pad_mask = state->get_dfg_pad_masks();
        
        #ifdef DEBUG
        std::cout << "Original DFG features shape: " << dfg_features.sizes() << std::endl;
        #endif
        
        add_and_pad_graph(dfg_features, dfg_pad_mask, max_dfg_size, batch_dfg_features, batch_dfg_mask_pad);

        auto cgra_features = state->get_cgra_features();
        auto cgra_pad_mask = state->get_cgra_pad_masks();

        #ifdef DEBUG
        std::cout << "Original CGRA features shape: " << cgra_features.sizes() << std::endl;
        #endif
        
        add_and_pad_graph(cgra_features, cgra_pad_mask, max_cgra_size, batch_cgra_features, batch_cgra_mask_pad);
        
        auto dfg_edge_indices = state->get_dfg_edge_indices();
        batch_dfg_edge_indices.push_back(dfg_edge_indices);

        #ifdef DEBUG
        std::cout << "DFG edge indices shape: " << dfg_edge_indices.sizes() << std::endl;
        #endif
        
        auto cgra_edge_indices = state->get_cgra_edge_indices();
        batch_cgra_edge_indices.push_back(cgra_edge_indices);

        #ifdef DEBUG
        std::cout << "CGRA edge indices shape: " << cgra_edge_indices.sizes() << std::endl;
        #endif

        auto node_to_be_mapped_features = state->get_node_to_be_mapped_features();
        batch_node_to_be_mapped_features = torch::cat({batch_node_to_be_mapped_features, node_to_be_mapped_features}, 0);

        auto pe_mask = state->get_mask();
        batch_masks = torch::cat({batch_masks, pe_mask}, 0);
    }

    auto dfg_edge_indice_tensor = create_batch_edge_indices(max_dfg_size, batch_dfg_edge_indices);
    auto cgra_edge_indice_tensor = create_batch_edge_indices(max_cgra_size, batch_cgra_edge_indices);

    #ifdef DEBUG
    std::cout << "Final DFG edge indices shape: " << dfg_edge_indice_tensor.sizes() << std::endl;
    std::cout << "Final CGRA edge indices shape: " << cgra_edge_indice_tensor.sizes() << std::endl;
    std::cout << "\n--- Final output shapes ---" << std::endl;
    #endif

    std::vector<c10::IValue> inputs;
    inputs.emplace_back(batch_dfg_features);
    
    #ifdef DEBUG
    std::cout << "DFG features output shape: " << batch_dfg_features.sizes() << std::endl;
    #endif

    inputs.emplace_back(dfg_edge_indice_tensor);
    inputs.emplace_back(batch_cgra_features);

    #ifdef DEBUG
    std::cout << "CGRA features output shape: " << batch_cgra_features.sizes() << std::endl;
    #endif

    inputs.emplace_back(cgra_edge_indice_tensor);
    inputs.emplace_back(batch_node_to_be_mapped_features);

    #ifdef DEBUG
    std::cout << "Node features output shape: " << batch_node_to_be_mapped_features.sizes() << std::endl;
    #endif

    inputs.emplace_back(batch_masks);

    #ifdef DEBUG
    std::cout << "PE mask output shape: " << batch_masks.sizes() << std::endl;
    #endif

    inputs.emplace_back(batch_dfg_mask_pad);
    inputs.emplace_back(batch_cgra_mask_pad);

    #ifdef DEBUG
    std::cout << "\n=== Finished mapzero_collate_fn ===" << std::endl;
    #endif

    return inputs;
}
