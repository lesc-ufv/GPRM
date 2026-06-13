#ifndef E2EMAP_ENVIRONMENT_HPP
#define E2EMAP_ENVIRONMENT_HPP

#include <cmath>
template <typename State>
double compute_e2emap_reward(const State& state, const int& action) {

    double manhattan_reward = 0;
    double unmapped_nodes = 0;
    double unmapped_edges = 0;
    double penalty = 0;
    const auto& PEs_to_routing = state.get_PEs_to_routing_();
    if (state.get_is_end_state()) {
        unmapped_nodes = state.get_num_dfg_nodes() - state.get_num_mapped_nodes();
        unmapped_edges = state.get_num_dfg_edges() - state.get_num_mapped_edges();

        for(auto& dfg_edges : state.get_dfg_edges()) {
            auto src = dfg_edges.first;
            auto dst = dfg_edges.second;
            if(!state.node_was_mapped(src) || !state.node_was_mapped(dst)) {
                continue;
            }
            auto src_pe = state.get_assigned_PE_by_node_id(src);
            auto dst_pe = state.get_assigned_PE_by_node_id(dst);
            if(!PEs_to_routing.at({src_pe, dst_pe}).empty()) {
                manhattan_reward += manhattan_distance(state.get_PE_pos_by_PE(src_pe), state.get_PE_pos_by_PE(dst_pe));
            } 
            }
            penalty = manhattan_reward + 4*unmapped_edges + 8*unmapped_nodes;
        }
    
        #ifdef DEBUG
            std::cout << "[DEBUG] Final reward: " << penalty << std::endl;
        #endif
        penalty = -cbrt(penalty);

    return penalty;
}

#endif
