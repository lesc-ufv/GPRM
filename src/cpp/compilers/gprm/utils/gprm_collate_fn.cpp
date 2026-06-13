#include "src/hpp/compilers/gprm/utils/gprm_collate_fn.hpp"
std::vector<c10::IValue> gprm_collate_fn(const std::vector<std::shared_ptr<GPRMState>> &states, const std::vector<int> &indices)
    {    
        
    int batch_size = indices.size();
    std::vector<torch::Tensor> batch_dfg_node_features;
    std::vector<torch::Tensor> batch_dfg_edge_indices;
    // std::vector<torch::Tensor> batch_dfg_placement_type_ids;
    // std::vector<torch::Tensor> batch_dfg_routing_type_ids;
    std::vector<torch::Tensor> batch_dfg_placement_order_ids;

    std::vector<torch::Tensor> batch_dfg_scheduling_type_ids;
    std::vector<torch::Tensor> batch_dfg_op_ids;

    std::vector<torch::Tensor> batch_adg_node_features;
    std::vector<torch::Tensor> batch_adg_edge_indices;
    // std::vector<torch::Tensor> batch_adg_placement_type_ids;
    // std::vector<torch::Tensor> batch_adg_routing_type_ids;
    std::vector<torch::Tensor> batch_adg_op_ids;
    std::vector<torch::Tensor> batch_adg_pos;

    std::vector<torch::Tensor> batch_mapping_edge_indices;
    std::vector<torch::Tensor> batch_placement_idx;
    std::vector<torch::Tensor> batch_routing_idx;
    std::vector<torch::Tensor> batch_batch_idx;
    std::vector<torch::Tensor> batch_seq_idx;

    std::vector<torch::Tensor> batch_contextual_action_seq_idx;
    std::vector<torch::Tensor> batch_action_batch_idx;
    std::vector<torch::Tensor> action_seq_id;
    std::vector<torch::Tensor> batch_state_seq_idx;

    batch_dfg_node_features.reserve(batch_size);
    batch_dfg_edge_indices.reserve(batch_size);
    // batch_dfg_placement_type_ids.reserve(batch_size);
    batch_dfg_placement_order_ids.reserve(batch_size);

    // batch_dfg_routing_type_ids.reserve(batch_size);
    batch_dfg_scheduling_type_ids.reserve(batch_size);
    batch_dfg_op_ids.reserve(batch_size);

    batch_adg_node_features.reserve(batch_size);
    batch_adg_edge_indices.reserve(batch_size);
    // batch_adg_placement_type_ids.reserve(batch_size);
    // batch_adg_routing_type_ids.reserve(batch_size);
    batch_adg_op_ids.reserve(batch_size);
    batch_adg_op_ids.reserve(batch_size);

    batch_mapping_edge_indices.reserve(batch_size);
    batch_placement_idx.reserve(batch_size);
    batch_routing_idx.reserve(batch_size);
    batch_batch_idx.reserve(batch_size);
    batch_seq_idx.reserve(batch_size);
    
    batch_contextual_action_seq_idx.reserve(batch_size);
    batch_action_batch_idx.reserve(batch_size);
    action_seq_id.reserve(batch_size);
    batch_state_seq_idx.reserve(batch_size);

    int max_seq_len = 0;

   std::vector<int> batch_dfg_node_sizes = {};
   batch_dfg_node_sizes.reserve(batch_size);
   std::vector<int> batch_adg_node_sizes = {};
   batch_adg_node_sizes.reserve(batch_size);
   std::vector<int> batch_dfg_edge_sizes = {};
   batch_dfg_edge_sizes.reserve(batch_size);
   std::vector<int> batch_adg_edge_sizes = {};
   batch_adg_edge_sizes.reserve(batch_size);
   std::vector<int> batch_mapping_node_sizes = {};
   batch_mapping_node_sizes.reserve(batch_size);
   std::vector<int> batch_mapping_edge_sizes = {};
   batch_mapping_edge_sizes.reserve(batch_size);
   std::vector<int> batch_actions_sizes = {};
   batch_actions_sizes.reserve(batch_size);

 
    int max_actions_size = 0;

    #ifdef DEBUG
        int sum_batch_size = 0;
    #endif
    for (auto state: states) {
        max_actions_size = std::max(max_actions_size,state->get_legal_actions_size());
    }
    for (auto indice: indices) {
        auto state = states[indice];
        batch_actions_sizes.emplace_back(state->get_legal_actions_size());
        #ifdef DEBUG
            sum_batch_size += state->get_max_seq_len();
        #endif
    }
    // std::vector<int> values(7,0);

    for (auto indice : indices) {
        auto state = states[indice];
    
    #ifdef DEBUG
        std::cout << "Processing index: " << indice << std::endl;
    #endif
    
        auto dfg_node_feat = state->getLpeDfg();
        auto adg_node_feat = state->getLpeAdg();
    
        batch_dfg_node_features.emplace_back(dfg_node_feat);
        batch_adg_node_features.emplace_back(adg_node_feat);
    
    #ifdef DEBUG
        std::cout << "DFG Node Feature Shape: " << dfg_node_feat.sizes() << std::endl;
        std::cout << "ADG Node Feature Shape: " << adg_node_feat.sizes() << std::endl;
    #endif
    
        // auto dfg_place_ids = state->getDfgPlacementTypeIds();
        // auto dfg_route_ids = state->getDfgRoutingTypeIds();
        auto dfg_sched_ids = state->getDfgNodeSchedulingIds();
        auto dfg_plac_order_ids = state->getDfgPlacementOrderIds();

        auto dfg_op_ids = state->getDfgOperationIds();
    
        // batch_dfg_placement_type_ids.emplace_back(dfg_place_ids);
        // batch_dfg_routing_type_ids.emplace_back(dfg_route_ids);
        batch_dfg_scheduling_type_ids.emplace_back(dfg_sched_ids);

        batch_dfg_placement_order_ids.emplace_back(dfg_plac_order_ids);
        batch_dfg_op_ids.emplace_back(dfg_op_ids);
        // #ifdef DEBUG
        // std::cout << "DFG Placement IDs Shape: " << dfg_place_ids.sizes() << std::endl;
        // std::cout << "DFG Routing IDs Shape: " << dfg_route_ids.sizes() << std::endl;
        // std::cout << "DFG Scheduling IDs Shape: " << dfg_sched_ids.sizes() << std::endl;
        // std::cout << "DFG Op IDs Shape: " << dfg_op_ids.sizes() << std::endl;
        // #endif
        
        // auto adg_place_ids = state->getAdgPlacementTypeIds();
        // auto adg_route_ids = state->getAdgRoutingTypeIds();
        auto adg_op_ids = state->getAdgOperationIds();
        
        // batch_adg_placement_type_ids.emplace_back(adg_place_ids);
        // batch_adg_routing_type_ids.emplace_back(adg_route_ids);
        batch_adg_op_ids.emplace_back(adg_op_ids);
        
        // #ifdef DEBUG
        // std::cout << "ADG Placement IDs Shape: " << adg_place_ids.sizes() << std::endl;
        // std::cout << "ADG Routing IDs Shape: " << adg_route_ids.sizes() << std::endl;
        // std::cout << "ADG Op IDs Shape: " << adg_op_ids.sizes() << std::endl;
        // #endif
        
        int dfg_nodes = state->get_num_dfg_nodes();
        int dfg_edges = state->get_num_dfg_edges();
        int adg_nodes = state->get_num_cgra_nodes();
        int adg_edges = state->get_num_cgra_edges();
        int mapping_nodes = state->get_num_mapping_nodes();
        int mapping_edges = state->get_num_mapping_edges();
        
        // assert(dfg_nodes + dfg_edges + adg_nodes + adg_edges + mapping_nodes + mapping_edges + 1 == state->calc_max_seq_len());
        // values[0] += dfg_nodes;
        // values[1] += dfg_edges;
        // values[2] += adg_nodes;
        // values[3] += adg_edges;
        // values[4] += mapping_nodes;
        // values[5] += mapping_edges;
        // values[6] += 1;

        batch_dfg_node_sizes.emplace_back(dfg_nodes);
        batch_dfg_edge_sizes.emplace_back(dfg_edges);
        batch_adg_node_sizes.emplace_back(adg_nodes);
        batch_adg_edge_sizes.emplace_back(adg_edges);
        batch_mapping_node_sizes.emplace_back(mapping_nodes);
        batch_mapping_edge_sizes.emplace_back(mapping_edges);
    #ifdef DEBUG
        std::cout << "DFG Nodes: " << dfg_nodes << ", Edges: " << dfg_edges << std::endl;
        std::cout << "ADG Nodes: " << adg_nodes << ", Edges: " << adg_edges << std::endl;
        std::cout << "Mapping Nodes: " << mapping_nodes << ", Edges: " << mapping_edges << std::endl;
    #endif
    
        auto dfg_edges_tensor = state->getDfgEdgeIndices(); 
        batch_dfg_edge_indices.push_back(dfg_edges_tensor);
    
        auto adg_edges_tensor = state->getAdgEdgeIndices();
        batch_adg_edge_indices.push_back(adg_edges_tensor);
    
        auto mapping_edges_tensor = state->getMappingEdgeIndices();
        batch_mapping_edge_indices.push_back(mapping_edges_tensor);
    
    #ifdef DEBUG
        std::cout << "DFG Edge Indices Shape: " << dfg_edges_tensor.sizes() << std::endl;
        std::cout << "ADG Edge Indices Shape: " << adg_edges_tensor.sizes() << std::endl;
        std::cout << "Mapping Edge Indices Shape: " << mapping_edges_tensor.sizes() << std::endl;
    #endif
    
        int seq_len = state->get_max_seq_len();
        max_seq_len = std::max(max_seq_len, seq_len);
    
        auto contextual_action_seq = state->getContextualActionSeqIdx();
        batch_contextual_action_seq_idx.emplace_back(contextual_action_seq);
    
        auto adg_pos = state->getADGPos();
        batch_adg_pos.push_back(adg_pos);
    
        #ifdef DEBUG
            std::cout << "Max sequence len so far: " << max_seq_len << std::endl;
            std::cout << "Action Seq Shape: " << contextual_action_seq.sizes() << std::endl;
            std::cout << "ADG Pos Shape: " << adg_pos.sizes() << std::endl;
        #endif
        batch_placement_idx.emplace_back(state->getPlacements());
        batch_routing_idx.emplace_back(state->getRoutings());

        auto cur_state_seq_idx = state->getStateSeqIdx();
        batch_state_seq_idx.emplace_back(cur_state_seq_idx);
    }
    
    
    auto batch_placement_idx_tensor = create_batch_placement_routing(batch_placement_idx,batch_dfg_node_sizes,batch_adg_node_sizes);
    auto batch_routing_idx_tensor = create_batch_placement_routing(batch_routing_idx, batch_dfg_edge_sizes, batch_adg_edge_sizes);
    std::vector<std::vector<int>> all_sizes = {std::ref(batch_dfg_node_sizes),std::ref(batch_dfg_edge_sizes),
    std::ref(batch_adg_node_sizes),std::ref(batch_adg_edge_sizes),
    std::ref(batch_mapping_node_sizes), std::ref(batch_mapping_edge_sizes),std::vector<int>(batch_size, 1)};
    auto previous_sizes = std::vector<int>(batch_size, 0);
    // int cum_sum = 0;
    // int  idx = 0;

    for(auto& vector: all_sizes) {
        add_batch_idx(batch_batch_idx ,vector);
        // auto cur_sum = 0;
        // for(torch::Tensor tensor: batch_batch_idx){
        //     cur_sum += tensor.numel();
        // }
        // cum_sum += values[idx];
        // assert(cur_sum == cum_sum);
        // idx += 1;
        add_seq_idx(batch_seq_idx,vector,previous_sizes);
        }
    add_batch_idx(batch_action_batch_idx,batch_actions_sizes);
    auto prev_sizes_actions = std::vector<int>(batch_size, 0);
    add_seq_idx(action_seq_id,batch_actions_sizes,prev_sizes_actions);
    //uasr actions_sizes para action_seq_idx
    auto dfg_edge_indice_tensor = create_batch_edge_indices(batch_dfg_node_sizes, batch_dfg_edge_indices);
    auto cgra_edge_indice_tensor = create_batch_edge_indices(batch_adg_node_sizes, batch_adg_edge_indices);
    auto mapping_edge_indice_tensor = create_batch_edge_indices(batch_mapping_node_sizes, batch_mapping_edge_indices);

    auto batch_adg_pos_tensor = torch::cat(batch_adg_pos,0);
    std::vector<c10::IValue> inputs;
    #ifdef DEBUG
    std::cout << "\n\nDFG Node Features Shape: " << torch::cat(batch_dfg_node_features, 0).sizes() << std::endl;
    std::cout << "DFG Edge Indices Size: " << batch_dfg_edge_indices.size() << std::endl;
    // std::cout << "DFG Placement Type IDs Shape: " << torch::cat(batch_dfg_placement_type_ids, 1).sizes() << std::endl;
    // std::cout << "DFG Routing Type IDs Shape: " << torch::cat(batch_dfg_routing_type_ids, 1).sizes() << std::endl;
    std::cout << "DFG Scheduling Type IDs Shape: " << torch::cat(batch_dfg_scheduling_type_ids, 1).sizes() << std::endl;
    std::cout << "DFG Op IDs Shape: " << torch::cat(batch_dfg_op_ids, 0).sizes() << std::endl;

    std::cout << "ADG Node Features Shape: " << torch::cat(batch_adg_node_features, 0).sizes() << std::endl;
    std::cout << "ADG Edge Indices Size: " << batch_adg_edge_indices.size() << std::endl;
    // std::cout << "ADG Placement Type IDs Shape: " << torch::cat(batch_adg_placement_type_ids,1).sizes() << std::endl;
    // std::cout << "ADG Routing Type IDs Shape: " << torch::cat(batch_adg_routing_type_ids,1).sizes() << std::endl;
    std::cout << "ADG Op IDs Shape: " << torch::cat(batch_adg_op_ids).sizes() << std::endl;
    std::cout << "ADG Pos Shape: " << batch_adg_pos_tensor.sizes() << std::endl;

    std::cout << "Mapping Edge Indices Size: " << mapping_edge_indice_tensor.sizes() << std::endl;
    std::cout << "Placement Idx Tensor Shape: " << batch_placement_idx_tensor.sizes() << std::endl;
    std::cout << "Routing Idx Tensor Shape: " << batch_routing_idx_tensor.sizes() << std::endl;
    auto temp = torch::cat(batch_batch_idx, 0);
    auto temp1 = torch::cat(batch_seq_idx, 0);

    std::cout << "Batch Idx Shape: " << temp.sizes() << std::endl;
    std::cout << "Seq Idx Shape: " << temp1.sizes() << std::endl;
    
    std::cout << "Contextual Action Seq Idx Shape: " << torch::cat(batch_contextual_action_seq_idx, 0).sizes() << std::endl;
    std::cout << "Action Batch Idx Shape: " << torch::cat(batch_action_batch_idx, 0).sizes() << std::endl;
    std::cout << "Action Seq Id Shape: " << torch::cat(action_seq_id, 0).sizes() << std::endl;
    std::cout << "State seq idx: " << torch::cat(batch_state_seq_idx, 0) << std::endl;
    std::cout << "Max Seq Len: " << max_seq_len << std::endl;
    std::cout << "Max Actions Size: " << max_actions_size << std::endl;
    // assert((sum_batch_size == temp.size(0)) && (sum_batch_size == temp1.size(0) ));
    #endif
    inputs.emplace_back(torch::cat(batch_dfg_node_features,0));
    inputs.emplace_back(dfg_edge_indice_tensor);
    inputs.emplace_back(torch::cat(batch_dfg_placement_order_ids,0));

    // inputs.emplace_back(torch::cat(batch_dfg_placement_type_ids,-1));
    // inputs.emplace_back(torch::cat(batch_dfg_routing_type_ids,-1));
    inputs.emplace_back(torch::cat(batch_dfg_scheduling_type_ids,1));
    inputs.emplace_back(torch::cat(batch_dfg_op_ids,0));

    inputs.emplace_back(torch::cat(batch_adg_node_features,0));
    inputs.emplace_back(cgra_edge_indice_tensor);
    // inputs.emplace_back(torch::cat(batch_adg_placement_type_ids,-1));
    // inputs.emplace_back(torch::cat(batch_adg_routing_type_ids,-1));
    inputs.emplace_back(torch::cat(batch_adg_op_ids));
    inputs.emplace_back(torch::cat(batch_adg_pos_tensor));

    inputs.emplace_back(mapping_edge_indice_tensor);
    inputs.emplace_back(batch_placement_idx_tensor);
    inputs.emplace_back(batch_routing_idx_tensor);
    
    inputs.emplace_back(torch::cat(batch_batch_idx,0));
    inputs.emplace_back(torch::cat(batch_seq_idx,0));
    inputs.emplace_back(torch::cat(batch_contextual_action_seq_idx,0));
    inputs.emplace_back(torch::cat(batch_action_batch_idx,0));
    inputs.emplace_back(torch::cat(action_seq_id,0));

    inputs.emplace_back(torch::cat(batch_state_seq_idx,0).squeeze(-1));
    inputs.emplace_back(torch::tensor(max_seq_len));
    inputs.emplace_back(torch::tensor(max_actions_size));

    return inputs;

}


torch::Tensor create_batch_placement_routing(
    const std::vector<torch::Tensor>& tensor,
    const std::vector<int>& sizes_idx_0,
    const std::vector<int>& sizes_idx_1) 
{
    int total_items = 0;

    for (const auto& edges : tensor) {
        total_items += edges.numel() > 0  ? edges.size(1) : 0; 
    }

    torch::Tensor batched_tensor = torch::empty({2, total_items}, torch::kInt32);

    int current_offset = 0; 
    int cum_sum_idx_0 = 0;
    int cum_sum_idx_1 = 0;

    for (size_t i = 0; i < tensor.size(); ++i) {
        const auto& items = tensor[i];
        int num_items = 0;
        if (items.numel() > 0){
            num_items = items.size(1);
            #ifdef DEBUG
            std::cout << "Item " << i << ": num_items = " << num_items << std::endl;
            std::cout << "  Cumulative idx0 offset: " << cum_sum_idx_0 << std::endl;
            std::cout << "  Cumulative idx1 offset: " << cum_sum_idx_1 << std::endl;
            #endif
            
            auto adjusted_0 = items[0] + cum_sum_idx_0;
            auto adjusted_1 = items[1] + cum_sum_idx_1;
            
            #ifdef DEBUG
            std::cout << "  Adjusted items[0]: " << adjusted_0 << std::endl;
            std::cout << "  Adjusted items[1]: " << adjusted_1 << std::endl;
            #endif
            batched_tensor[0].slice(0, current_offset, current_offset + num_items) = adjusted_0;
            batched_tensor[1].slice(0, current_offset, current_offset + num_items) = adjusted_1;
        }

        current_offset += num_items;
        cum_sum_idx_0 += sizes_idx_0[i];
        cum_sum_idx_1 += sizes_idx_1[i];
    }
    #ifdef DEBUG
        std::cout << "Final batched_tensor shape: " << batched_tensor.sizes() << std::endl;
    #endif
    
    return batched_tensor;
}

void add_batch_idx(std::vector<torch::Tensor>& batch_batch_idx, const std::vector<int>& sizes){

    
    int count = 0;
    for(auto& size: sizes){
        if (size > 0)
        {
            batch_batch_idx.emplace_back(torch::full({size},count, torch::kInt32));

        }
        count += 1;
    }
}

void add_seq_idx(std::vector<torch::Tensor>& batch_seq_idx, const std::vector<int>& sizes, std::vector<int>& previous_sizes){
    int count = 0;
    for(int i = 0; i < static_cast<int>(sizes.size()); i++){
        if(sizes[i] > 0){
            batch_seq_idx.emplace_back(torch::arange(0, sizes[i], torch::kInt32) + previous_sizes[i]);
            previous_sizes[i] += sizes[i];
        }
        count += 1;
    }
}