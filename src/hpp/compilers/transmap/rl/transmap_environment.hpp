#ifndef TRANSMAP_ENVIRONMENT_HPP
#define TRANSMAP_ENVIRONMENT_HPP

template <typename State>
double compute_transmap_reward(const State& state, const int& action) {
    auto previous_mapped_node = state.get_prev_mapped_node();
    auto node_was_mapped = state.node_was_mapped(previous_mapped_node);

    const auto& in_vertices_cur_node = state.get_in_vertices_dfg_by_node_id(state.get_prev_mapped_node());

    double reward = 0;

    #ifdef DEBUG
        std::cout << "[DEBUG] Computing reward for action: " << action << std::endl;
        std::cout << "[DEBUG] Previous mapped node: " << previous_mapped_node 
                << ", Was mapped: " << node_was_mapped << std::endl;
    #endif

    if (node_was_mapped) {
        for (const auto& father : in_vertices_cur_node) {
            if (state.node_was_mapped(father)) {
                auto father_PE = state.get_assigned_PE_by_node_id(father);
                double distance_penalty = -manhattan_distance(state.get_PE_pos_by_PE(father_PE), state.get_PE_pos_by_PE(action));
                reward += distance_penalty;

#ifdef DEBUG
                std::cout << "[DEBUG] Distance penalty from father: " << distance_penalty << std::endl;
#endif
            }
        }
    
        const auto& out_vertices_cur_node = state.get_out_vertices_dfg_by_node_id(state.get_prev_mapped_node());

        for (const auto& child : out_vertices_cur_node) {
            if (state.node_was_mapped(child)) {
                auto child_PE = state.get_assigned_PE_by_node_id(child);
  
                double distance_penalty = -manhattan_distance(state.get_PE_pos_by_PE(action), state.get_PE_pos_by_PE(child_PE));
                reward += distance_penalty;

#ifdef DEBUG
                std::cout << "[DEBUG] Distance penalty to child: " << distance_penalty << std::endl;
#endif
            }
        }
    }

    if (state.get_is_end_state() && state.get_mapping_is_valid_()) {
        reward += state.get_II();
#ifdef DEBUG
        std::cout << "[DEBUG] Final valid state reached. II: " << state.get_II() 
                  << ", Total reward: " << reward << std::endl;
#endif
    }

#ifdef DEBUG
    std::cout << "[DEBUG] Final reward: " << reward << std::endl;
#endif

    return reward;
}

#endif
