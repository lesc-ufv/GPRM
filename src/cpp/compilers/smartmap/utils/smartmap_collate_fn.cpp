#include "src/hpp/compilers/smartmap/utils/smartmap_collate_fn.hpp"
#include <iostream>

std::vector<c10::IValue> smartmap_collate_fn(const std::vector<std::shared_ptr<SmartMapState>>& states, 
                                             const std::vector<int>& indices)
{
    
#ifdef DEBUG
    std::cout << "\n=== Starting smartmap_collate_fn ===" << std::endl;
    std::cout << "Number of states: " << indices.size() << std::endl;
#endif

    torch::Tensor batch_dfg_features = torch::tensor({}, torch::dtype(torch::kFloat32)).requires_grad_(false);
    std::vector<torch::Tensor> batch_dfg_edge_indices;
    torch::Tensor batch_cgra_features = torch::tensor({},torch::dtype(torch::kFloat32)).requires_grad_(false);
    std::vector<torch::Tensor> batch_cgra_edge_indices;
    torch::Tensor batch_node_to_be_mapped_features = torch::tensor({},torch::dtype(torch::kFloat32)).requires_grad_(false);
    torch::Tensor batch_masks = torch::tensor({},torch::kBool).requires_grad_(false);
    torch::Tensor batch_dfg_mask_pad = torch::tensor({},torch::dtype(torch::kBool)).requires_grad_(false);
    torch::Tensor batch_cgra_mask_pad = torch::tensor({},torch::dtype(torch::kBool)).requires_grad_(false);
    torch::Tensor batch_actions_features = torch::tensor({},torch::dtype(torch::kFloat32)).requires_grad_(false);
    torch::Tensor batch_actions_mask_pad = torch::tensor({},torch::kBool).requires_grad_(false);
    
    int max_actions_size = 0;
    int max_dfg_size = 0;
    int max_cgra_size = 0;

#ifdef DEBUG
    std::cout << "\n--- Calculating max sizes ---" << std::endl;
#endif

    for (auto state: states) {
        max_actions_size = std::max(max_actions_size,state->get_legal_actions_size());
    }
    for (auto indice : indices) {
        auto state = states[indice];
        
        int current_dfg = state->get_num_dfg_nodes();
        int current_cgra = state->get_num_cgra_nodes();
        
        max_dfg_size = std::max(max_dfg_size, current_dfg);
        max_cgra_size = std::max(max_cgra_size, current_cgra);

#ifdef DEBUG
        std::cout << "State " << indice  << ", DFG=" << current_dfg << ", CGRA=" << current_cgra << std::endl;
#endif
    }

    #ifdef DEBUG
    std::cout << "Max sizes - Actions: " << max_actions_size 
              << ", DFG: " << max_dfg_size << ", CGRA: " << max_cgra_size << std::endl;
    std::cout << "\n--- Processing states ---" << std::endl;
    #endif 

    for (auto indice : indices) {
#ifdef DEBUG
        std::cout << "\nProcessing state " << indice << std::endl;
#endif
        auto& state = states[indice];
        
        auto dfg_features = state->get_dfg_features();
        auto dfg_pad_mask = state->get_dfg_pad_masks();
#ifdef DEBUG
        std::cout << "DFG features before padding: " << dfg_features.sizes() << std::endl;
#endif
        add_and_pad_graph(dfg_features, dfg_pad_mask, max_dfg_size, batch_dfg_features, batch_dfg_mask_pad);
#ifdef DEBUG
        std::cout << "DFG features after padding: " << batch_dfg_features.sizes() << std::endl;
#endif

        auto cgra_features = state->get_cgra_features();
        auto cgra_pad_mask = state->get_cgra_pad_masks();
#ifdef DEBUG
        std::cout << "CGRA features before padding: " << cgra_features.sizes() << std::endl;
#endif
        add_and_pad_graph(cgra_features, cgra_pad_mask, max_cgra_size, batch_cgra_features, batch_cgra_mask_pad);
#ifdef DEBUG
        std::cout << "CGRA features after padding: " << batch_cgra_features.sizes() << std::endl;
#endif

        auto dfg_edge_indices = state->get_dfg_edge_indices();
        batch_dfg_edge_indices.push_back(dfg_edge_indices);
#ifdef DEBUG
        std::cout << "Added DFG edge indices: " << dfg_edge_indices.sizes() << std::endl;
#endif
        
        auto cgra_edge_indices = state->get_cgra_edge_indices();
        batch_cgra_edge_indices.push_back(cgra_edge_indices);
#ifdef DEBUG
        std::cout << "Added CGRA edge indices: " << cgra_edge_indices.sizes() << std::endl;
#endif

        auto node_to_be_mapped_features = state->get_node_to_be_mapped_features();
        batch_node_to_be_mapped_features = torch::cat({batch_node_to_be_mapped_features, node_to_be_mapped_features}, 0);
#ifdef DEBUG
        std::cout << "Accumulated node features: " << batch_node_to_be_mapped_features.sizes() << std::endl;
#endif

        auto actions_features_tensor = state->get_legal_actions_features_tensor();
        auto actions_mask_tensor = state->get_action_pad_mask();
        int cur_actions_size = actions_features_tensor.size(-2);
        
#ifdef DEBUG
        std::cout << "Actions before padding: " << actions_features_tensor.sizes() 
                  << ", Mask: " << actions_mask_tensor.sizes() << std::endl;
#endif

        if (cur_actions_size < max_actions_size) {
            auto diff = max_actions_size - cur_actions_size;
            actions_features_tensor = torch::cat({
                actions_features_tensor, 
                torch::zeros({1,diff, actions_features_tensor.size(-1)}, torch::kFloat32)
            }, 1);

            actions_mask_tensor = torch::cat({
                actions_mask_tensor, 
                torch::zeros({1, diff}, torch::kBool)
            }, -1);
#ifdef DEBUG
            std::cout << "Actions after padding: " << actions_features_tensor.sizes() 
                      << ", Mask: " << actions_mask_tensor.sizes() << std::endl;
#endif
        }

        batch_actions_features = torch::cat({batch_actions_features, actions_features_tensor}, 0);
        batch_actions_mask_pad = torch::cat({batch_actions_mask_pad, actions_mask_tensor}, 0);
        
#ifdef DEBUG
        std::cout << "Accumulated actions: " << batch_actions_features.sizes() 
                  << ", Accumulated mask: " << batch_actions_mask_pad.sizes() << std::endl;
#endif
    }

#ifdef DEBUG
    std::cout << "\n--- Creating batch edge indices ---" << std::endl;
#endif
    auto dfg_edge_indice_tensor = create_batch_edge_indices(max_dfg_size, batch_dfg_edge_indices);
    auto cgra_edge_indice_tensor = create_batch_edge_indices(max_cgra_size, batch_cgra_edge_indices);
#ifdef DEBUG
    std::cout << "Final DFG edge indices: " << dfg_edge_indice_tensor.sizes() << std::endl;
    std::cout << "Final CGRA edge indices: " << cgra_edge_indice_tensor.sizes() << std::endl;
#endif

#ifdef DEBUG
    std::cout << "\n--- Preparing final output ---" << std::endl;
#endif
    batch_actions_features = batch_actions_features.reshape({-1, max_actions_size, 7});
#ifdef DEBUG
    std::cout << "Final actions shape: " << batch_actions_features.sizes() << std::endl;
#endif

    std::vector<c10::IValue> inputs;
    inputs.emplace_back(batch_dfg_features);
    inputs.emplace_back(dfg_edge_indice_tensor);
    inputs.emplace_back(batch_cgra_features);
    inputs.emplace_back(cgra_edge_indice_tensor);
    inputs.emplace_back(batch_node_to_be_mapped_features);
    inputs.emplace_back(batch_actions_features);
    inputs.emplace_back(batch_actions_mask_pad);
    inputs.emplace_back(batch_dfg_mask_pad);
    inputs.emplace_back(batch_cgra_mask_pad);

#ifdef DEBUG
    std::cout << "\n=== Finished smartmap_collate_fn ===" << std::endl;
    std::cout << "Output list size: " << inputs.size() << std::endl;
#endif

    return inputs;
}

// std::vector<c10::IValue> smartmap_collate_fn2(const std::vector<std::shared_ptr<SmartMapState>>& states, 
//         const std::vector<int>& indices)
// {
//         std::vector<c10::IValue> inputs;

//         int max_actions_size = 0;
//         int max_dfg_size = 0;
//         int max_cgra_size = 0;

//         for (auto state : states) {
//         max_actions_size = std::max(max_actions_size, state->get_legal_actions_size());
//         }

//         for (auto indice : indices) {
//                 auto state = states[indice];
//                 max_dfg_size = std::max(max_dfg_size, state->get_num_dfg_nodes());
//                 max_cgra_size = std::max(max_cgra_size, state->get_num_cgra_nodes());
//         }

//         int64_t total_elements = 0;

//         std::vector<torch::Tensor> all_dfg_features;
//         for (auto indice : indices) {
//         auto& state = states[indice];
//         torch::Tensor dfg_features = state->get_dfg_features();
//         all_dfg_features.push_back(dfg_features);
//         }
//         inputs.insert(inputs.end(), all_dfg_features.begin(), all_dfg_features.end());

//         std::vector<torch::Tensor> all_dfg_edge_indices;
//         for (auto indice : indices) {
//         auto& state = states[indice];
//         auto dfg_edge_indices = state->get_dfg_edge_indices();
//         all_dfg_edge_indices.push_back(dfg_edge_indices);
//         }
//         inputs.insert(inputs.end(), all_dfg_edge_indices.begin(), all_dfg_edge_indices.end());

//         std::vector<torch::Tensor> all_cgra_features;
//         for (auto indice : indices) {
//         auto& state = states[indice];
//         torch::Tensor cgra_features = state->get_cgra_features();
//         all_cgra_features.push_back(cgra_features);
//         }
//         inputs.insert(inputs.end(), all_cgra_features.begin(), all_cgra_features.end());

//         std::vector<torch::Tensor> all_cgra_edge_indices;
//         for (auto indice : indices) {
//         auto& state = states[indice];
//         auto cgra_edge_indices = state->get_cgra_edge_indices();
//         all_cgra_edge_indices.push_back(cgra_edge_indices);
//         }
//         inputs.insert(inputs.end(), all_cgra_edge_indices.begin(), all_cgra_edge_indices.end());

//         std::vector<torch::Tensor> all_node_to_be_mapped;
//         for (auto indice : indices) {
//         auto& state = states[indice];
//         auto node_to_be_mapped = state->get_node_to_be_mapped_features();
//         all_node_to_be_mapped.push_back(node_to_be_mapped);
//         }
//         inputs.insert(inputs.end(), all_node_to_be_mapped.begin(), all_node_to_be_mapped.end());

//         std::vector<torch::Tensor> all_actions_features;
//         for (auto indice : indices) {
//         auto& state = states[indice];
//         auto actions_features = state->get_legal_actions_features_tensor();
//         all_actions_features.push_back(actions_features);
//         }
//         inputs.insert(inputs.end(), all_actions_features.begin(), all_actions_features.end());

//         std::vector<torch::Tensor> all_actions_mask;
//         for (auto indice : indices) {
//         auto& state = states[indice];
//         auto actions_mask = state->get_action_pad_mask();
//         all_actions_mask.push_back(actions_mask);
//         }
//         inputs.insert(inputs.end(), all_actions_mask.begin(), all_actions_mask.end());

//         std::vector<torch::Tensor> all_dfg_pad_mask;
//         for (auto indice : indices) {
//         auto& state = states[indice];
//         auto dfg_pad_mask = state->get_dfg_pad_masks();
//         all_dfg_pad_mask.push_back(dfg_pad_mask);
//         }
//         inputs.insert(inputs.end(), all_dfg_pad_mask.begin(), all_dfg_pad_mask.end());

//         std::vector<torch::Tensor> all_cgra_pad_mask;
//         for (auto indice : indices) {
//         auto& state = states[indice];
//         auto cgra_pad_mask = state->get_cgra_pad_masks();
//         all_cgra_pad_mask.push_back(cgra_pad_mask);
//         }
//         inputs.insert(inputs.end(), all_cgra_pad_mask.begin(), all_cgra_pad_mask.end());

//         inputs.push_back(torch::tensor(static_cast<int>(indices.size())));
//         inputs.push_back(torch::tensor(max_dfg_size));
//         inputs.push_back(torch::tensor(max_cgra_size));
//         inputs.push_back(torch::tensor(max_actions_size));

//         return inputs;
// }
