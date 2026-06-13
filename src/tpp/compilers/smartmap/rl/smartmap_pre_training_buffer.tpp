#include "src/hpp/compilers/smartmap/rl/smartmap_pre_training_buffer.hpp"
template<typename StateT, typename ModelConfig>
void pretraining_smartmap_add_mapping(BaseBuffer<StateT, ModelConfig>& buffer, const MappingHistory<StateT>& mapping_history)
{
    // mapping_history.print();

    buffer.increment_total_samples(mapping_history.get_states().size() - 1);
    
    const auto& rewards_vec = mapping_history.get_rewards();
    
    #ifdef DEBUG
    std::cout << "\n==== DEBUG: Saving Mapping History ====\n" << std::endl;
    mapping_history.print();
    #endif

    auto sliced_rewards = torch::tensor(rewards_vec, torch::kFloat32).slice(0, 1, rewards_vec.size());
    // auto norm_rewards = buffer.get_norm_rewards(sliced_rewards);
    
    #ifdef DEBUG
    std::cout << "\n==== DEBUG: Sliced Rewards ====\n" << sliced_rewards << std::endl;
    #endif

    const auto& actions_indices = mapping_history.get_actions_indices();
    auto cur_actions_indices_tensor = torch::tensor(actions_indices).slice(0, 1, actions_indices.size());

    torch::Tensor tensor_policies;
    // auto is_mcts_data = buffer.get_is_mcts_data(); 

    const auto& policies = mapping_history.get_child_visits();
    tensor_policies = pad_matrix_to_tensor<float, float>(policies, 0.);

    #ifdef DEBUG
    std::cout << "\n==== DEBUG: Processed Policies ====\n" << tensor_policies << std::endl;
    #endif
    

    const auto& states = mapping_history.get_states();
    const auto& values_vec = mapping_history.get_values();
    auto values = torch::tensor(values_vec);
    // if(is_mcts_data){
    //     values = buffer.get_norm_values(values);
    // }

    #ifdef DEBUG
    std::cout << "\n==== DEBUG: Appending Data to Buffer ====\n" << std::endl;
    std::cout << "[DEBUG] States size: " << states.size() << std::endl;
    std::cout << "[DEBUG] Values: " << values << std::endl;
    std::cout << "[DEBUG] Actions Indices Tensor: " << cur_actions_indices_tensor << std::endl;
    assert(tensor_policies.slice(0, 0, tensor_policies.size(0) - 1).sizes().size() == 2);
    #endif
    if(buffer.buffer_is_full()){
        auto idx = buffer.get_number_of_mappings() % buffer.get_buffer_size();
        buffer.increment_total_samples(-(buffer.get_mapping_size_by_idx(idx) -  1));
    }
    buffer.circular_append_states(states);
    buffer.circular_append_rewards(sliced_rewards);
    buffer.circular_append_values(values);
    // if (!is_mcts_data) {
        buffer.circular_append_actions_indices(cur_actions_indices_tensor);
    // }

    buffer.circular_append_policies(tensor_policies.slice(0, 0, tensor_policies.size(0) - 1));

    buffer.increment_number_of_mappings(1);
}


template<typename StateT, typename ModelConfig>
MappingDataset<StateT> pretraining_smartmap_get_batch(BaseBuffer<StateT, ModelConfig>& buffer)
{   
    torch::Tensor target_values = torch::tensor({});
    torch::Tensor advantages = torch::tensor({});
    torch::Tensor tensor_actions_indices = torch::tensor({}, torch::kInt32);
    torch::Tensor tensor_policies = torch::tensor({});
    std::vector< torch::Tensor> tensor_values;

    std::vector<std::shared_ptr<StateT>> all_states;
    auto first_reward = torch::tensor({-1});
    auto rewards_tensor = torch::cat(buffer.get_rewards(),0);
    int total_size = rewards_tensor.size(0);

    #ifdef DEBUG
    std::cout << "[DEBUG] get_train_dataset:" << std::endl;
    std::cout << "  Total reward indices size: " << total_size << std::endl;
    #endif

    all_states.reserve(total_size);
    torch::Tensor norm_rewards;
    if (buffer.getBufferConfig().getNormRewardsPTbuffer()) {
        norm_rewards = norm_by_enum(rewards_tensor, EnumNormalization::MEAN_STD, 0, 1);
    }else{
        norm_rewards = rewards_tensor;
    }
    #ifdef DEBUG
        std::cout << "  Norm rewards \n" << std::endl;
        std::cout  << norm_rewards << std::endl;
        #endif
    int cur_init_indice = 0;
    const auto& states = buffer.get_states();
    const auto& values = buffer.get_values_vec();
    const auto& actions_indices = buffer.get_actions_indices();
    const auto& policies = buffer.get_policies();
    auto is_mcts_data = buffer.get_is_mcts_data(); 
    auto is_ppo_train = buffer.get_is_ppo();

    torch::Tensor sum_values = torch::tensor(0.0);
    torch::Tensor sum_squared_values = torch::tensor(0.0);
    double total_values = 0;

    for(int i = 0; i < static_cast<int>(states.size()); i++){
        auto final_indice = cur_init_indice + states[i].size() - 1;
        auto cur_rewards = norm_rewards.slice(0,cur_init_indice, final_indice);
        
        #ifdef DEBUG
        std::cout << "  Processing Mapping " << i << ":" << std::endl;
        std::cout << "    Final indice: " << final_indice << std::endl;
        std::cout << "    Initial indice: " << cur_init_indice << std::endl;
        #endif

        auto cur_values = values[i];
        cur_init_indice = final_indice;
        total_values += cur_values.size(0) - 1;
        sum_values += cur_values.sum();
        sum_squared_values += cur_values.pow(2).sum();


        if (is_ppo_train) tensor_values.push_back(cur_values.slice(0, 0, cur_values.size(0) - 1));

        #ifdef DEBUG
        std::cout << "    Current Rewards: " << cur_rewards << std::endl;
        std::cout << "    Current Values: " << cur_values << std::endl;
        #endif
        
        for (auto it = states[i].begin(); it != states[i].end() - 1; ++it) {
             all_states.push_back(std::move(*it));
        }
        auto temp_rewards = torch::cat({first_reward,cur_rewards}, 0);

        #ifdef DEBUG
        std::cout << "    Temp Rewards (with first reward): " << temp_rewards << std::endl;
        #endif

        auto cur_target_values = buffer.get_target_values(temp_rewards, cur_values);
        target_values = torch::cat({target_values,cur_target_values},0);
        #ifdef DEBUG
        std::cout << "    Target Values: " << cur_target_values << std::endl;
        #endif
        if(!is_mcts_data){
            auto cur_advantages = cur_target_values - cur_values.slice(0, 0, cur_values.size(0) - 1);
            advantages = torch::cat({advantages,cur_advantages},0);
    
            #ifdef DEBUG
            std::cout << "   Advantages: " << cur_advantages << std::endl;
            #endif
    
            auto cur_actions_indices = actions_indices[i];
            tensor_actions_indices = torch::cat({tensor_actions_indices, cur_actions_indices}, 0);

        }

        
    }
    double raw_mean_advantages = advantages.mean().item<double>();
    double raw_std_advantages = advantages.std().item<double>();
    double raw_mean_values = target_values.mean().item<double>();
    double raw_std_values = target_values.std().item<double>();

    if(!is_mcts_data && buffer.getNormAdvantages()) advantages = (advantages - raw_mean_advantages) / raw_std_advantages;
    if(buffer.getNormReturns()) target_values = (target_values - raw_mean_values) / raw_std_values;

    tensor_policies = tensor_vector_to_tensor_with_padding<float>(policies,0.);
    #ifdef DEBUG
    std::cout << "[DEBUG] Final Actions Indices: " << tensor_actions_indices << std::endl;
    std::cout << "[DEBUG] Returning MappingDataset<StateT>." << std::endl;
    #endif
    auto mapping_dataset = MappingDataset<StateT>(
        std::move(all_states), 
        std::move(target_values), 
        is_ppo_train ? std::make_optional(torch::cat(tensor_values)) :  std::nullopt ,
        std::make_optional(std::move(tensor_policies)), 
        is_mcts_data ? std::nullopt : std::make_optional(std::move(tensor_actions_indices)), 
        is_mcts_data ?  std::nullopt :std::make_optional(std::move(advantages)),
        std::nullopt, 
        std::nullopt, 
        std::nullopt);
        
    mapping_dataset.set_raw_mean_advantages(raw_mean_advantages);
    mapping_dataset.set_raw_std_advantages(raw_std_advantages);
    mapping_dataset.set_mean_values(raw_mean_values);
    mapping_dataset.set_std_values(raw_std_values);

    return mapping_dataset;
}



