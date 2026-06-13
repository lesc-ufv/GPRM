template <typename State>
double compute_smartmap_reward(const State& state, const int& action) {
    const int bad_reward = -100;
    int reward = 0;
    int previous_mapped_node = state.get_prev_mapped_node();
    auto node_was_mapped = state.node_was_mapped(previous_mapped_node);

    const auto& PEs_to_routing = state.get_PEs_to_routing_();
    const auto& in_vertices_cur_node = state.get_in_vertices_dfg_by_node_id(previous_mapped_node);
    const auto& out_vertices_cur_node = state.get_out_vertices_dfg_by_node_id(previous_mapped_node);

    #ifdef DEBUG
        std::cout << "[DEBUG] Computing SmartMap Reward" << std::endl;
        std::cout << "[DEBUG] Previous mapped node: " << previous_mapped_node << std::endl;
        std::cout << "[DEBUG] Node was mapped? " << (node_was_mapped ? "Yes" : "No") << std::endl;
    #endif

    if (node_was_mapped) {
        #ifdef DEBUG
            std::cout << "[DEBUG] Processing input dependencies for node: " << previous_mapped_node << std::endl;
        #endif
    
        for (const auto& father : in_vertices_cur_node) {
            if (state.node_was_mapped(father)) {
                const auto& father_PE = state.get_assigned_PE_by_node_id(father);
                auto pe_routing_pair = std::make_pair(father_PE, action);
                const auto& cur_PEs_to_routing = PEs_to_routing.at(pe_routing_pair);
                int cur_PEs_to_routing_size = cur_PEs_to_routing.size();
    
                #ifdef DEBUG
                    std::cout << "    [DEBUG] Father node: " << father << " -> PE: " << father_PE << std::endl;
                    std::cout << "    [DEBUG] Routing path size: " << cur_PEs_to_routing_size << std::endl;
                #endif
    
                if (cur_PEs_to_routing_size == 0) {
                    reward += bad_reward;
                    #ifdef DEBUG
                        std::cout << "    [DEBUG] No valid routing found! Applying bad reward: " << bad_reward << std::endl;
                    #endif
                } else {
                    int penalty = -(cur_PEs_to_routing_size - 1);
                    reward += penalty;
                    #ifdef DEBUG
                        std::cout << "    [DEBUG] Applied routing penalty: " << penalty << std::endl;
                    #endif
                }
            }
        }
    
        #ifdef DEBUG
            std::cout << "[DEBUG] Processing output dependencies for node: " << previous_mapped_node << std::endl;
        #endif
    
        for (const auto& child : out_vertices_cur_node) {
            if (state.node_was_mapped(child)) {
                const auto& child_PE = state.get_assigned_PE_by_node_id(child);
                auto pe_routing_pair = std::make_pair(action, child_PE);
                const auto& cur_PEs_to_routing = PEs_to_routing.at(pe_routing_pair);
                int cur_PEs_to_routing_size = cur_PEs_to_routing.size();
    
                #ifdef DEBUG
                    std::cout << "    [DEBUG] Child node: " << child << " -> PE: " << child_PE << std::endl;
                    std::cout << "    [DEBUG] Routing path size: " << cur_PEs_to_routing_size << std::endl;
                #endif
    
                if (cur_PEs_to_routing_size == 0) {
                    reward += bad_reward;
                    #ifdef DEBUG
                        std::cout << "    [DEBUG] No valid routing found! Applying bad reward: " << bad_reward << std::endl;
                    #endif
                } else {
                    int penalty = -(cur_PEs_to_routing_size - 1);
                    reward += penalty;
                    #ifdef DEBUG
                        std::cout << "    [DEBUG] Applied routing penalty: " << penalty << std::endl;
                    #endif
                }
            }
        }
    }


    if (state.get_is_end_state()) {
        
        if(!state.all_nodes_were_mapped() || !Router::routing_is_valid(PEs_to_routing)) {
            auto& edges = state.get_dfg_edges();
            auto& node_to_PE = state.get_node_to_pe_();

            #ifdef DEBUG
                std::cout << "[DEBUG] Checking for unrouted edges..." << std::endl;
            #endif

            for (auto& pair : edges) {
                if (node_to_PE.find(pair.first) == node_to_PE.end()) {
                    reward += bad_reward;
                    #ifdef DEBUG
                        std::cout << "    [DEBUG] Edge source node " << pair.first << " is not mapped. Applying bad reward." << std::endl;
                    #endif
                    continue;
                }

                if (node_to_PE.find(pair.second) == node_to_PE.end()) {
                    reward += bad_reward;
                    #ifdef DEBUG
                        std::cout << "    [DEBUG] Edge destination node " << pair.second << " is not mapped. Applying bad reward." << std::endl;
                    #endif
                    continue;
                }

                auto father_PE = node_to_PE.at(pair.first);
                auto child_PE = node_to_PE.at(pair.second);
                if (PEs_to_routing.find({father_PE, child_PE}) == PEs_to_routing.end()) {
                    reward += bad_reward;
                    #ifdef DEBUG
                        std::cout << "    [DEBUG] No valid routing found between PEs " << father_PE << " and " << child_PE << ". Applying bad reward." << std::endl;
                    #endif
                }
            }
        }

        if (state.get_is_end_state() && !state.get_mapping_is_valid_()) {
            reward += bad_reward;
            #ifdef DEBUG
                std::cout << "[DEBUG] Mapping is invalid! Applying bad reward: " << bad_reward << std::endl;
            #endif
        }
    }

    #ifdef DEBUG
        std::cout << "[DEBUG] Final computed reward: " << reward / 100.0 << std::endl;
    #endif

    return reward / 100.0;
}
