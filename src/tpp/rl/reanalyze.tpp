#include "src/hpp/rl/reanalyze.hpp"

template<typename StateT, typename ModelConfig>
Reanalyze<StateT, ModelConfig>::Reanalyze(std::shared_ptr<Buffer<StateT, ModelConfig>> buffer, SelfMapping<StateT,ModelConfig>& self_mapping) 
: buffer(buffer), self_mapping(self_mapping){
        
}

template<typename StateT, typename ModelConfig>
Reanalyze<StateT, ModelConfig>::Reanalyze(){

}

template<typename StateT, typename ModelConfig>
void Reanalyze<StateT, ModelConfig>::reanalyze(){

    if (buffer->get_num_data_in_buffer() == 0){
        #ifdef DEBUG
            std::cout << "[Reanalyze] Buffer is empty. Cannot reanalyze." << std::endl;
        #endif
        return;
    }
    int map_idx = this->buffer->get_random_mapping_idx();
    this->buffer->set_cur_reanalyzed_map_idx(map_idx);
    std::cout << "[Reanalyze] Selected mapping index: " << map_idx << std::endl;

    const auto states = this->buffer->get_states_by_idx(map_idx);
    std::cout << "[Reanalyze] Retrieved " << states.size() << " states for mapping." << std::endl;

    if (states.empty()) {
        std::cerr << "[Reanalyze] Warning: No states retrieved for index " << map_idx << std::endl;
        return;
    }
    MappingHistory<StateT> mapping_history;
    mapping_history.init_values(states[0]);
    // std::cout << "[Reanalyze] Initialized mapping history with first state." << std::endl;

    for (int i = 0; i < static_cast<int>(states.size()) - 1; ++i) {
        try {
            std::cout << "[Reanalyze] Reanalyzing state " << i << std::endl;
            this->self_mapping.reanalyze(states[i], mapping_history, this->gen);
            this->reanalyzed_states++;
        }
        catch (const std::exception& e) {
            throw std::runtime_error(
                "[Reanalyze] Failure while reanalyzing state " +
                std::to_string(i) + ": " + e.what()
            );
        }
    }
    // mapping_history.append_state(states[states.size() - 1]);
    mapping_history.append_value(0.0);

    std::cout << "[Reanalyze] Updating buffer with reanalyzed mapping." << std::endl;
    // mapping_history.print();
    this->buffer->update_with_renalyzed_mapping(mapping_history, map_idx);
    std::cout << "[Reanalyze] Done." << std::endl;
    this->reanalyzed_mappings++;
    this->buffer->set_cur_reanalyzed_map_idx(-1);

}
