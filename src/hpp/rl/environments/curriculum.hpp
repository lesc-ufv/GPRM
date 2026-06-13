#ifndef CURRICULUM_ENVIRONMENT_HPP
#define CURRICULUM_ENVIRONMENT_HPP

template <typename State>
double compute_placement_reward(const State& state, const int& action){
    if (state.get_is_end_state()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] State is an end state." << std::endl;
        std::cout << "[DEBUG] Mapping is valid: " << state.get_mapping_is_valid_() << std::endl;
        #endif

        if (!state.get_mapping_is_valid_())
        {
            return unmapped_nodes = state.get_num_dfg_nodes() - state.get_num_mapped_nodes();
        }

        return 0.0;
    }else{
        return 0.0;
    }
};

template <typename State>
double compute_placement_reward_v2(const State& state, const int& action){
    if (state.get_is_end_state()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] State is an end state." << std::endl;
        std::cout << "[DEBUG] Mapping is valid: " << state.get_mapping_is_valid_() << std::endl;
        #endif

        if (!state.get_mapping_is_valid_())
        {
            return unmapped_nodes = state.get_num_dfg_nodes() - state.get_num_mapped_nodes();
        }

        return 0.0;
    }else{
        return 0.0;
    }
};


template <typename State>
double compute_routing_reward(const State& state, const int& action);

template <typename State>
double compute_scheduling_reward(const State& state, const int& action);

#endif