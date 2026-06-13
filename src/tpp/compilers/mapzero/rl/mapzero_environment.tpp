#include "src/hpp/compilers/mapzero/rl/mapzero_environment.hpp"

template <typename State>
double compute_mapzero_reward(const State& state, const int& action) {
    auto previous_mapped_node = state.get_prev_mapped_node();
    auto node_was_mapped = state.node_was_mapped(previous_mapped_node);
    double bad_reward = -100;

    #ifdef DEBUG
        std::cout << "[DEBUG] Computing MapZero Reward" << std::endl;
        std::cout << "[DEBUG] Previous mapped node: " << previous_mapped_node << std::endl;
        std::cout << "[DEBUG] Was node mapped? " << (node_was_mapped ? "Yes" : "No") << std::endl;
    #endif



    const auto& PEs_to_routing = state.get_PEs_to_routing_();
    const auto& in_vertices_cur_node = state.get_in_vertices_dfg_by_node_id(state.get_prev_mapped_node());
    double reward = 0;

    #ifdef DEBUG
        std::cout << "[DEBUG] Processing input dependencies for node: " << previous_mapped_node << std::endl;
    #endif
    if (node_was_mapped) {

        for (const auto& father : in_vertices_cur_node) {
            if (state.node_was_mapped(father)) {
                auto father_PE = state.get_assigned_PE_by_node_id(father);
    
                auto pe_routing_pair = std::make_pair(father_PE, action);
                auto& cur_routing = PEs_to_routing.at(pe_routing_pair);
                int cur_routing_size = cur_routing.size();
    
                #ifdef DEBUG
                    std::cout << "    [DEBUG] Father node: " << father << " -> PE: " << father_PE << std::endl;
                    std::cout << "    [DEBUG] Current routing size: " << cur_routing_size << std::endl;
                #endif
    
                if (cur_routing_size == 0) {
                    reward += bad_reward;
                    #ifdef DEBUG
                        std::cout << "    [DEBUG] No valid routing found! Applying bad reward: " << bad_reward << std::endl;
                    #endif
                } else {
                    double distance_penalty = -manhattan_distance(state.get_PE_pos_by_PE(father_PE), state.get_PE_pos_by_PE(action));
                    reward += distance_penalty;
                    #ifdef DEBUG
                        std::cout << "    [DEBUG] Manhattan distance penalty applied: " << distance_penalty << std::endl;
                    #endif
                }
            }
        }
    
        auto out_vertices_cur_node = state.get_out_vertices_dfg_by_node_id(state.get_prev_mapped_node());
    
        #ifdef DEBUG
            std::cout << "[DEBUG] Processing output dependencies for node: " << previous_mapped_node << std::endl;
        #endif
    
        for (const auto& child : out_vertices_cur_node) {
            if (state.node_was_mapped(child)) {
                auto occupied_PEs = state.get_occupied_PEs_();
                auto child_PE = state.get_assigned_PE_by_node_id(child);
    
                auto pe_routing_pair = std::make_pair(action, child_PE);
                auto& cur_routing = PEs_to_routing.at(pe_routing_pair);
                int cur_routing_size = cur_routing.size();
    
                #ifdef DEBUG
                    std::cout << "    [DEBUG] Child node: " << child << " -> PE: " << child_PE << std::endl;
                    std::cout << "    [DEBUG] Current routing size: " << cur_routing_size << std::endl;
                #endif
    
                if (cur_routing_size == 0) {
                    reward += bad_reward;
                    #ifdef DEBUG
                        std::cout << "    [DEBUG] No valid routing found! Applying bad reward: " << bad_reward << std::endl;
                    #endif
                } else {
                    double distance_penalty = -manhattan_distance(state.get_PE_pos_by_PE(action), state.get_PE_pos_by_PE(child_PE));
                    reward += distance_penalty;
                    #ifdef DEBUG
                        std::cout << "    [DEBUG] Manhattan distance penalty applied: " << distance_penalty << std::endl;
                    #endif
                }
            }
        }
    }

    if (state.get_is_end_state() && !state.get_mapping_is_valid_()) {
        reward += bad_reward;
        #ifdef DEBUG
            std::cout << "[DEBUG] Mapping is invalid! Applying bad reward: " << bad_reward << std::endl;
        #endif
    }

    if (reward < bad_reward) reward = bad_reward;
    #ifdef DEBUG
        std::cout << "[DEBUG] Final computed reward: " << reward << std::endl;
    #endif

    return reward;
}
