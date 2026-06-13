#include "src/hpp/rl/buffers/buffer.hpp"

template<typename StateT, typename ModelConfig>
Buffer<StateT,ModelConfig>::Buffer( GlobalConfig<ModelConfig>& global_config,
        std::shared_ptr<zmq::socket_t> socket,
        const ServerConstants& server_k,
        std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester,
        std::string func_name)
    : socket(socket), server_k(server_k), batch_requester(batch_requester), func_name(func_name) {
    this->buffer = BaseBuffer<StateT, ModelConfig>(global_config);
    this->global_config = global_config;
    this->gen = torch::make_generator<torch::CPUGeneratorImpl>(this->global_config.getTrainConfig().getSeed());
    this->build_functions();
} 

template<typename StateT, typename ModelConfig>
Buffer<StateT,ModelConfig>::Buffer() {}

template<typename StateT, typename ModelConfig>
void Buffer<StateT,ModelConfig>::add_mapping(const MappingHistory<StateT>& mapping_history){
    // auto aug_data = global_config.getBufferConfig().getAugmentData();
    this->save(this->buffer, mapping_history);
    // if(aug_data){
    //     this->augment_data(mapping_history);
    // }
}


template<typename StateT, typename ModelConfig>
void Buffer<StateT, ModelConfig>::update_priorities( torch::Tensor mapping_indices,
                        torch::Tensor states_indices,
                        torch::Tensor new_priorities){
    return this->update_buffer_priorities(this->buffer, mapping_indices, states_indices, new_priorities);

};

template<typename StateT, typename ModelConfig>
MappingDataset<StateT> Buffer<StateT, ModelConfig>::get_batch(
    std::optional<EnumReplayBuffer> enum_replay_buffer_opt){
    EnumReplayBuffer enum_replay_buffer;
    if(enum_replay_buffer_opt.has_value()){
        enum_replay_buffer = enum_replay_buffer_opt.value();
    }else{
        enum_replay_buffer = global_config.getLauncherConfig().getReplayBuffer();
    }

    if(enum_replay_buffer == EnumReplayBuffer::MAPZERO_BUFFER ||  enum_replay_buffer == EnumReplayBuffer::PER_BUFFER){
        return this->buffer.get_batch();
    }else if (enum_replay_buffer == EnumReplayBuffer::SMARTMAP_PRE_TRAINING_BUFFER){
        return pretraining_smartmap_get_batch<StateT,ModelConfig>(this->buffer);
    }else if (enum_replay_buffer == EnumReplayBuffer::GPRM_BUFFER){
        return gprm_get_batch(this->buffer);
    }else if(enum_replay_buffer == EnumReplayBuffer::SUPERVISED_BUFFER){
        return supervised_buffer_get_batch<StateT,ModelConfig>(this->buffer);
    }else if(enum_replay_buffer == EnumReplayBuffer::DTB){
        return get_batch_dtb<StateT,ModelConfig>(this->buffer, this->socket, this->server_k, this->batch_requester, this->func_name);
    }else{
        throw std::runtime_error("Invalid buffer");
    }
}

template<typename StateT, typename ModelConfig>
int Buffer<StateT, ModelConfig>::get_random_mapping_idx(){
    int max_samples = this->buffer.get_num_data_in_buffer();
    auto rand_idx = torch::randint(0, max_samples, {1},this->gen).item<int>();
    std::cout << "[INFO] Random mapping idx: " << rand_idx << std::endl;
    return rand_idx;
}

template<typename StateT, typename ModelConfig>
const std::vector<std::shared_ptr<StateT>> Buffer<StateT, ModelConfig>::get_states_by_idx(const int& map_idx) const{
    return this->buffer.get_mapping_by_idx(map_idx);
}

template<typename StateT, typename ModelConfig>
int Buffer<StateT, ModelConfig>::get_num_data_in_buffer() const{
    return this->buffer.get_num_data_in_buffer();
}

template<typename StateT, typename ModelConfig>
void Buffer<StateT, ModelConfig>::update_with_renalyzed_mapping(MappingHistory<StateT>& mapping_history, const int& idx){
    this->buffer.update_with_renalyzed_mapping(mapping_history, idx);
}


template <typename StateT, typename ModelConfig>
template <typename T>
typename std::enable_if<std::is_same<T, GPRMConfig>::value>::type
Buffer<StateT, ModelConfig>::augment_data(const MappingHistory<StateT>& mapping_history) {
    std::runtime_error("Data augmentation for GPRM is not supported yet.");
}

template <typename StateT, typename ModelConfig>
template <typename T>
typename std::enable_if<std::is_same<T, SmartMapConfig>::value>::type
Buffer<StateT, ModelConfig>::augment_data(const MappingHistory<StateT>& mapping_history) {
    std::runtime_error("Data augmentation for SmartMap is not supported yet.");
}

template <typename StateT, typename ModelConfig>
template <typename T>
typename std::enable_if<std::is_same<T, TransMapConfig>::value>::type
Buffer<StateT, ModelConfig>::augment_data(const MappingHistory<StateT>& mapping_history) {
    std::runtime_error("Data augmentation for TransMap is not supported yet.");
}

template <typename StateT, typename ModelConfig>
template <typename T>
typename std::enable_if<std::is_same<T, MapZeroConfig>::value>::type
Buffer<StateT, ModelConfig>::augment_data(const MappingHistory<StateT>& mapping_history) {
        // auto is_mcts_data = this->global_config.getLauncherConfig().getTraining() == EnumTraining::MCTS;
        // assert(is_mcts_data && "Data augmentation with MCTS data.");

        // #ifdef DEBUG
        // std::cout << "[DEBUG] Starting augment_data" << std::endl;
        // std::cout << "Original mapping history size: " << mapping_history.get_states().size() << std::endl;
        // #endif
    
        // std::vector<std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>>> node_maps;
     
        // auto states = mapping_history.get_states();
        // auto last_state = states.back();
        // auto PE_to_node = last_state->get_PE_to_node();
        // auto PE_to_coord = last_state->get_PE_to_coord_();
        // auto coord_to_PE = last_state->get_coord_to_PE_();
        // auto dims = last_state->get_cgra_dimensions_();
        // auto number_of_spatial_PEs = last_state->get_number_of_spatial_PEs();
        
        // auto buffer_config = this->global_config.getBufferConfig();
        // auto degrees = buffer_config.getDegrees();
        // auto shifts = buffer_config.getShifts();
        // auto flips = buffer_config.getFlips();
        
        // std::vector<torch::Tensor> indices_per_node_maps;
        // Environment<MapZeroState> env = Environment<MapZeroState>(EnumEnvironment::MAPZERO);
        
        // torch::Tensor actions = torch::tensor(mapping_history.get_actions());
    
        // #ifdef DEBUG
        // std::cout << "[DEBUG] Actions tensor shape: " << actions.sizes() << std::endl;
        // std::cout << "First 5 actions: ";
        // for (int i = 0; i < std::min(5, (int)actions.size(0)); i++) {
        //     std::cout << actions[i].item<int>() << " ";
        // }
        // std::cout << std::endl;
        // #endif
    
        // #ifdef DEBUG
        // std::cout << "[DEBUG] Applying rotations..." << std::endl;
        // std::cout << "Number of rotations: " << degrees.size() << std::endl;
        // #endif
        // for(auto& degrees: degrees){
        //     std::vector<int> indices;
        //     node_maps.emplace_back(
        //         rotate(PE_to_node, PE_to_coord, coord_to_PE, dims.first,dims.second, degrees)
        //         );
        //     for(int i = 0 ; i < number_of_spatial_PEs; i++){
        //         indices.emplace_back(rotate(i,PE_to_coord, coord_to_PE, dims.first,dims.second, degrees));   
        //     }
        //     indices_per_node_maps.emplace_back(torch::tensor(indices,torch::kInt32));
        // }
    
        // for(auto& shift_values: shifts){
    
        //     auto temp_dicts_pair = shift(PE_to_node, PE_to_coord, coord_to_PE, dims.first,dims.second, shift_values,false);
        //     if (!temp_dicts_pair.first.empty()){
        //         std::vector<int> indices;
    
        //         node_maps.emplace_back(
        //             temp_dicts_pair
        //             );
        //         for(int i = 0 ; i < number_of_spatial_PEs; i++){
        //             indices.emplace_back(shift(i,PE_to_coord, coord_to_PE, dims.first,dims.second, shift_values, false));   
        //             }
        //         indices_per_node_maps.emplace_back(torch::tensor(indices,torch::kInt32));
        //         }
        // }
    
        // for(auto& flip_dims: flips){
        //     std::vector<int> indices;
    
        //     node_maps.emplace_back(
        //         flip(PE_to_node, PE_to_coord, coord_to_PE,dims.first,dims.second, flip_dims)
        //         );
        //     for(int i = 0 ; i < number_of_spatial_PEs; i++){
        //             indices.emplace_back(flip(i,PE_to_coord, coord_to_PE, dims.first,dims.second, flip_dims));   
        //            }
        //     indices_per_node_maps.emplace_back(torch::tensor(indices,torch::kInt32));
        
        // }
    
        // torch::Tensor policies = matrix_to_tensor<float, float>(mapping_history.get_child_visits());
        // int idx = 0;
        
        // #ifdef DEBUG
        // std::cout << "[DEBUG] Policies tensor shape: " << policies.sizes() << std::endl;
        // std::cout << "Number of node maps: " << node_maps.size() << std::endl;
        // std::cout << "[DEBUG] Starting state mapping..." << std::endl;
        // #endif
        // for (auto& dict_pair: node_maps){
        //     MappingHistory<StateT> new_mapping_history;
        //     new_mapping_history.set_rewards(mapping_history.get_rewards());
        //     new_mapping_history.set_values(mapping_history.get_values());
        //     torch::Tensor cur_rewards = torch::tensor(mapping_history.get_rewards()); 
        //     std::vector<int> new_actions = {actions[0].item<int>()};
        //     std::vector<int> new_actions_idxs = {-1};
        //     new_actions.reserve(actions.size(0));
        //     std::vector<std::shared_ptr<MapZeroState>> new_states = {states[0]};
        //     new_states.reserve(states.size());
        //     auto cur_state = states[0];
        //     bool aug_is_valid = true;
        //     torch::Tensor new_policies = torch::zeros({policies.size(0),policies.size(1)}, policies.options());
        //     #ifdef DEBUG
        //     std::cout << "[DEBUG] Indices node map " <<  indices_per_node_maps[idx] << std::endl;
    
        //     #endif
        //     new_policies[0].index_put_({indices_per_node_maps[idx]}, policies[0]);
        
        //     #ifdef DEBUG
        //     std::cout << "[DEBUG] Processing node map " << idx << std::endl;
        //     #endif
        
        //     for (int i = 0; i  < (static_cast<int>(states.size()) - 1); i++){
        //         int new_action = dict_pair.first.at(
        //             PE_to_node.at(
        //                 actions[i + 1].item<int>()
        //             )
        //         );
                
        //         #ifdef DEBUG
        //         std::cout << "Original action: " << actions[i+1].item<int>() 
        //                  << ", Mapped action: " << new_action << std::endl;
        //         #endif
                
        //         if (cur_state->pe_is_occupied(new_action)){
        //             #ifdef DEBUG
        //             std::cout << "PE " << new_action << " is occupied - augmentation invalid" << std::endl;
        //             #endif
        //             aug_is_valid = false;
        //             break;
        //         }
        //         new_actions_idxs.push_back(cur_state->get_action_idx_by_action(new_action));
        //         auto [next_state, reward,__ ] = env.step(*cur_state,new_action);
                
        //         #ifdef DEBUG
        //         std::cout << "Step " << i << " reward: " << reward 
        //                  << " (expected: " << cur_rewards[i + 1].item<double>() << ")" << std::endl;
        //         #endif
                
        //         // if(reward != cur_rewards[i + 1].item<double>()) {
        //         //     #ifdef DEBUG
        //         //     std::cout << "Reward mismatch - augmentation invalid" << std::endl;
        //         //     #endif
        //         //     aug_is_valid = false;
        //         //     break;
        //         // }
    
        //         if(i != static_cast<int>(states.size()) - 1){
        //             new_policies[i+1].index_put_({indices_per_node_maps[idx]}, policies[i+1]);
        //             auto pe_mask = next_state->get_mask();
        //             auto masked_values = new_policies[i+1].masked_select(pe_mask == 0);
        //             bool all_zeros = torch::all(masked_values == 0).item<bool>();
                    
        //             #ifdef DEBUG
        //             std::cout << "Policy validity check: " << (all_zeros ? "valid" : "invalid") << std::endl;
        //             #endif
                    
        //             if (!all_zeros){
        //                 aug_is_valid = false;
        //                 break;
        //             }
        //         }
    
        //         new_states.push_back(next_state);
        //         new_actions.push_back(new_action);
        //         cur_state = next_state;
        //     }
    
        //     if(aug_is_valid) {
        //         #ifdef DEBUG
        //         std::cout << "Augmentation valid - saving new mapping" << std::endl;
        //         std::cout << "New states size: " << new_states.size() << std::endl;
        //         std::cout << "New actions size: " << new_actions.size() << std::endl;
        //         #endif
                
        //         new_mapping_history.set_states(new_states);
        //         new_mapping_history.set_actions_indices(new_actions_idxs);
        //         new_mapping_history.set_actions(new_actions);
    
        //         #ifdef DEBUG
        //         std::cout << "[DEBUG] Appending child visits..." << std::endl;
        //         std::cout << "New policies shape: " << new_policies.sizes() << std::endl;
        //         #endif
        //         for (int i = 0; i < new_policies.size(0); ++i) {
        //             torch::Tensor row = new_policies[i];
                
        //             std::vector<float> child_visits(row.data_ptr<float>(), row.data_ptr<float>() + row.numel());
                    
        //             new_mapping_history.append_child_visit(std::move(child_visits));
        //         }
                
        //        this->save(this->buffer,new_mapping_history);
    
        //         #ifdef DEBUG
        //         std::cout << "[DEBUG] New mapping saved successfully" << std::endl;
        //         std::cout << "Buffer size: " << this->buffer.get_number_of_mappings() << std::endl;
        //         #endif
        //     } else {
        //         #ifdef DEBUG
        //         std::cout << "Augmentation invalid - skipping" << std::endl;
        //         #endif
        //     }
        //     idx++;
        // }
        
        // #ifdef DEBUG
        // std::cout << "[DEBUG] augment_data completed" << std::endl;
        // #endif
}

