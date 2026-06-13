#include "src/hpp/rl/buffers/PER_buffer.hpp"

template <typename StateT, typename ModelConfig>
void PER_add_mapping(BaseBuffer<StateT, ModelConfig>& buffer, 
                     const MappingHistory<StateT>& mapping_history) {   
    auto values = torch::tensor(mapping_history.get_values());
    auto idx = buffer.get_number_of_mappings() % buffer.get_buffer_size();

    buffer.save_mapping(mapping_history);

    const auto& target_values = buffer.get_target_values_by_idx(idx);
    auto sliced_values = values.slice(0, 0, values.size(0) - 1);
    
    auto cur_priorities = torch::abs(target_values - sliced_values);
    auto mapping_priority = cur_priorities.mean().template item<double>();    

    buffer.set_mapping_priority_by_idx(mapping_priority, idx);
    buffer.set_states_priorities_by_idx(cur_priorities, idx);
    
    #ifdef DEBUG
        std::cout << "[DEBUG] Mapping index: " << idx << std::endl;
        std::cout << "[DEBUG] Target values: " << target_values << std::endl;
        std::cout << "[DEBUG] Sliced values: " << sliced_values << std::endl;
        std::cout << "[DEBUG] Current priorities: " << cur_priorities << std::endl;
        std::cout << "[DEBUG] Mapping priority: " << mapping_priority << std::endl;
    #endif
}

template <typename StateT, typename ModelConfig>
void PER_update_priorities(BaseBuffer<StateT, ModelConfig>& base_buffer, 
                           torch::Tensor mapping_indices,
                           torch::Tensor states_indices,
                           torch::Tensor new_priorities) {
    auto map_indices = std::get<0>(at::_unique2(mapping_indices, false));

    for (int i = 0; i < map_indices.size(0); i++) {
        int cur_map_idx = map_indices[i].item<int>();
        auto mask = (mapping_indices == cur_map_idx);

        auto cur_priorities = new_priorities.index({mask});
        auto cur_states_indices = states_indices.index({mask});

        base_buffer.set_states_priorities_by_idxs(cur_priorities, cur_map_idx, cur_states_indices);
        torch::Tensor updated_priorities = base_buffer.get_state_priorities_by_idx(cur_map_idx);
        float mapping_priority = updated_priorities.mean().item<float>();

        base_buffer.set_mapping_priority_by_idx(mapping_priority, cur_map_idx);

        #ifdef DEBUG
            std::cout << "[DEBUG] Current map index: " << cur_map_idx << std::endl;
            std::cout << "[DEBUG] Mask: " << mask << std::endl;
            std::cout << "[DEBUG] Current priorities: " << cur_priorities << std::endl;
            std::cout << "[DEBUG] updated priorities : " << updated_priorities << std::endl;
            std::cout << "[DEBUG] Current state indices: " << cur_states_indices << std::endl;
            std::cout << "[DEBUG] Mapping priority: " << mapping_priority << std::endl;
        #endif
    }
}
