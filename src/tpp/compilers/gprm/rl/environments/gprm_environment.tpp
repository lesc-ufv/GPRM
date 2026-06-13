#include "src/hpp/compilers/gprm/rl/environments/gprm_environment.hpp"

template <typename State>
double compute_gprm_reward(const State& state, const int& action) {
    const auto& previous_mapped_node = state.get_prev_mapped_node();
    // const auto& in_vertices_cur_node = state.get_in_vertices_dfg_by_node_id(previous_mapped_node);
    // const auto& out_vertices_cur_node = state.get_out_vertices_dfg_by_node_id(previous_mapped_node);
    auto node_was_mapped = state.node_was_mapped(previous_mapped_node);

    double reward = 0;

    #ifdef DEBUG
    std::cout << "[DEBUG] === compute_gprm_reward ===" << std::endl;
    std::cout << "[DEBUG] Previous mapped node: " << previous_mapped_node << std::endl;
    std::cout << "[DEBUG] Node was mapped: " << node_was_mapped << std::endl;
    std::cout << "[DEBUG] Action: " << action << std::endl;
    #endif

    if (node_was_mapped) {
        reward += 1 * routing_node_reward(state, previous_mapped_node, action);

        #ifdef DEBUG
        std::cout << "[DEBUG] Routing reward (non-terminal): " << reward << std::endl;
        #endif
    }

    if (state.get_is_end_state()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] State is an end state." << std::endl;
        std::cout << "[DEBUG] Mapping is valid: " << state.get_mapping_is_valid_() << std::endl;
        #endif

        if (!state.get_mapping_is_valid_())
        {
            // int unmapped_nodes = state.g et_num_dfg_nodes() - state.get_num_mapped_nodes();
            int unmapped_edges = state.get_num_dfg_edges() - state.get_num_mapped_edges();
            int invalid_convergent_nodes = state.get_num_convergent_nodes() - state.get_num_valid_convergent_nodes();
            int invalid_convergent_edges = state.get_num_convergent_edges() - state.get_num_valid_convergent_edges();

            double penalty = -(state.get_num_dfg_nodes() + unmapped_edges + invalid_convergent_nodes + invalid_convergent_edges);

            reward += penalty;

            #ifdef DEBUG
            // std::cout << "[DEBUG] Invalid mapping. Computing detailed penalties:" << std::endl;
            // std::cout << "  Unmapped nodes: " << unmapped_nodes << std::endl;
            std::cout << "  Unmapped edges: " << unmapped_edges << std::endl;
            std::cout << "  Invalid convergent nodes: " << invalid_convergent_nodes << std::endl;
            std::cout << "  Invalid convergent edges: " << invalid_convergent_edges << std::endl;
            
            // std::cout << "  Num convergent nodes: " << state.get_num_convergent_nodes() << std::endl;
            // std::cout << "  Num valid convergent nodes: " << state.get_num_valid_convergent_nodes() << std::endl;
            // std::cout << "  Num convergent edges: " << state.get_num_convergent_edges() << std::endl;
            // std::cout << "  Num valid convergent nodes: " << state.get_num_valid_convergent_edges() << std::endl;

            std::cout << "  Total penalty: " << penalty << std::endl;
            #endif
        }
    }

    
    

    #ifdef DEBUG
    std::cout << "[DEBUG] Final computed reward: " << reward << std::endl;
    std::cout << "===============================" << std::endl;
    #endif


    return reward;
}



template <typename State>
double compute_gprm_reward_v2(const State& state, const int& action) {
    // const auto& previous_mapped_node = state.get_prev_mapped_node();
    // const auto& in_vertices_cur_node = state.get_in_vertices_dfg_by_node_id(previous_mapped_node);
    // const auto& out_vertices_cur_node = state.get_out_vertices_dfg_by_node_id(previous_mapped_node);
    // auto node_was_mapped = state.node_was_mapped(previous_mapped_node);

    double reward = 0;

    #ifdef DEBUG
    std::cout << "[DEBUG] === compute_gprm_reward ===" << std::endl;
    std::cout << "[DEBUG] Previous mapped node: " << previous_mapped_node << std::endl;
    std::cout << "[DEBUG] Node was mapped: " << node_was_mapped << std::endl;
    std::cout << "[DEBUG] Action: " << action << std::endl;
    #endif


    if (state.get_is_end_state()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] State is an end state." << std::endl;
        std::cout << "[DEBUG] Mapping is valid: " << state.get_mapping_is_valid_() << std::endl;
        #endif

        if (state.get_mapping_is_valid_())
        {
            auto rout_nodes = state.calc_num_rout_nodes();
            reward += 1./std::sqrt(rout_nodes+1);
        }
    }

    #ifdef DEBUG
    std::cout << "[DEBUG] Final computed reward: " << reward << std::endl;
    std::cout << "===============================" << std::endl;
    #endif


    return reward;
}

template <typename State>
double compute_gprm_reward_v3(const State& state, const int& action) {
    const auto& previous_mapped_node = state.get_prev_mapped_node();
    // const auto& in_vertices_cur_node = state.get_in_vertices_dfg_by_node_id(previous_mapped_node);
    // const auto& out_vertices_cur_node = state.get_out_vertices_dfg_by_node_id(previous_mapped_node);
    auto node_was_mapped = state.node_was_mapped(previous_mapped_node);

    double reward = 0;

    #ifdef DEBUG
    std::cout << "[DEBUG] === compute_gprm_reward ===" << std::endl;
    std::cout << "[DEBUG] Previous mapped node: " << previous_mapped_node << std::endl;
    std::cout << "[DEBUG] Node was mapped: " << node_was_mapped << std::endl;
    std::cout << "[DEBUG] Action: " << action << std::endl;
    #endif

    if (node_was_mapped) {
        reward += 1 * routing_node_reward(state, previous_mapped_node, action);

        #ifdef DEBUG
        std::cout << "[DEBUG] Routing reward (non-terminal): " << reward << std::endl;
        #endif
    }
     
    #ifdef DEBUG
    std::cout << "[DEBUG] Final computed reward: " << reward << std::endl;
    std::cout << "===============================" << std::endl;
    #endif


    return reward;
}
