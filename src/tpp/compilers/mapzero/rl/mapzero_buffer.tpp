#include "src/hpp/compilers/mapzero/rl/mapzero_buffer.hpp"

template<typename StateT, typename ModelConfig>
void mapzero_add_mapping(BaseBuffer<StateT, ModelConfig>& buffer, const MappingHistory<StateT>& mapping_history) {   

    float max_priority;
    auto idx = buffer.get_number_of_mappings() % buffer.get_buffer_size();

    #ifdef DEBUG
    std::cout << "\nNumber of mappings: " << buffer.get_number_of_mappings() << std::endl;
    #endif

    if (buffer.get_number_of_mappings() == 0){
        max_priority = 100000; 
        #ifdef DEBUG
        std::cout << "Buffer is empty. Setting max_priority to: " << max_priority << std::endl;
        #endif
    } else {
        auto max_result = torch::max(buffer.get_mapping_priorities(), 0);
        max_priority = std::get<0>(max_result).item().toFloat();
        std::max(max_priority,static_cast<float>(1e-4));
        #ifdef DEBUG
        std::cout << "Max priority from buffer: " << max_priority << std::endl;
        #endif
    }

    auto cur_priorities = torch::full(mapping_history.size() - 1, max_priority);
    buffer.set_mapping_priority_by_idx(max_priority, idx);
    buffer.set_states_priorities_by_idx(cur_priorities, idx);

    #ifdef DEBUG
    std::cout << "Set mapping priority at index " << idx << " with value " << max_priority << std::endl;
    std::cout << "Set states priorities at index " << idx << " with value " << cur_priorities << std::endl;
    #endif

    buffer.save_mapping(mapping_history);
}

template<typename StateT, typename ModelConfig>
void mapzero_update_priorities(BaseBuffer<StateT, ModelConfig>& buffer, 
                    torch::Tensor mapping_indices,
                    torch::Tensor states_indices,
                    torch::Tensor new_priorities) {

    #ifdef DEBUG
    std::cout << "\nUpdating priorities with " << mapping_indices.size(0) << " mappings" << std::endl;
    #endif

    buffer.reduce_mapping_priorities_by_alpha(mapping_indices);
    buffer.reduce_states_priorities_by_alpha(mapping_indices, states_indices);

    #ifdef DEBUG
    std::cout << "Updated mapping and state priorities using alpha." << std::endl;
    #endif
}


