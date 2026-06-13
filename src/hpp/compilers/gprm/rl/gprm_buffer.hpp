#ifndef GPRM_BUFFER_HPP
#define GPRM_BUFFER_HPP

#include "vector"
#include "torch/torch.h"
#include "src/hpp/enums/enum_nomalization.hpp"
#include "src/hpp/compilers/smartmap/datasets/smartmap_dataset.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"
#include "src/hpp/utils/util_train.hpp"
#include "src/hpp/rl/mapping_history.hpp"
#include "src/hpp/rl/buffers/base_buffer.hpp"
#include "src/hpp/utils/util_torch.hpp"
// #define DEBUG
template<typename StateT, typename ModelConfig>
void gprm_add_mapping(BaseBuffer<StateT, ModelConfig>& buffer, const MappingHistory<StateT>& mapping_history){

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
    auto is_mcts_data = buffer.get_is_mcts_data(); 

    const auto& policies = mapping_history.get_child_visits();
    tensor_policies = pad_matrix_to_tensor<float, float>(policies, -std::numeric_limits<float>::infinity());

    #ifdef DEBUG
    std::cout << "\n==== DEBUG: Processed Policies ====\n" << tensor_policies << std::endl;
    #endif
    

    const auto& states = mapping_history.get_states();
    const auto& values_vec = mapping_history.get_values();
    auto values = torch::tensor(values_vec);


    #ifdef DEBUG
    std::cout << "\n==== DEBUG: Appending Data to Buffer ====\n" << std::endl;
    std::cout << "[DEBUG] States size: " << states.size() << std::endl;
    std::cout << "[DEBUG] Values: " << values << std::endl;
    std::cout << "[DEBUG] Actions Indices Tensor: " << cur_actions_indices_tensor << std::endl;
    assert(tensor_policies.slice(0, 0, tensor_policies.size(0) - 1).sizes().size() == 2);
    #endif

    auto idx = buffer.get_number_of_mappings() % buffer.get_buffer_size();
    if(buffer.buffer_is_full()){
        buffer.increment_total_samples(-(buffer.get_mapping_size_by_idx(idx) -  1));
    }
    buffer.circular_append_states(states);
    buffer.circular_append_rewards(sliced_rewards);
    buffer.circular_append_values(values);
    if (!is_mcts_data) {
        buffer.circular_append_actions_indices(cur_actions_indices_tensor);
    }

    buffer.circular_append_policies(tensor_policies.slice(0, 0, tensor_policies.size(0) - 1));

    buffer.increment_number_of_mappings(1);

}
template<typename StateT, typename ModelConfig>
MappingDataset<StateT> gprm_get_batch(BaseBuffer<StateT, ModelConfig>& buffer){
    auto num_to_batch = buffer.get_num_to_batch();
    torch::Tensor mapping_indices = torch::randint(0, buffer.get_num_data_in_buffer(), {num_to_batch}, torch::kInt);
#ifdef DEBUG
    std::cout << "Num to batch: " << num_to_batch << std::endl;
    std::cout << "Mapping indices: " << mapping_indices << std::endl;
#endif

    auto rewards_vec = std::vector<torch::Tensor>();
    int* mapping_data_ptr = mapping_indices.data_ptr<int>();
    auto unroll_steps = buffer.get_num_unroll_steps();

    std::vector<std::shared_ptr<StateT>> cur_states;
    std::vector<torch::Tensor> all_target_values;
    std::vector<torch::Tensor> all_policies;
    std::vector<torch::Tensor> all_advantages;
    std::vector<torch::Tensor> all_values;
    std::vector<torch::Tensor> all_actions;

    for (int i = 0; i < mapping_indices.size(0); i++) {
        auto& mapping_idx = mapping_data_ptr[i];
#ifdef DEBUG
        std::cout << "Fetching rewards for index: " << mapping_idx << std::endl;
#endif
        rewards_vec.push_back(buffer.get_rewards_by_idx(mapping_idx));
    }

    int cur_init_indice = 0;
    auto norm_rewards = buffer.get_norm_rewards(torch::cat(rewards_vec, 0));

    for (int i = 0; i < mapping_indices.size(0); i++) {
        auto& mapping_idx = mapping_data_ptr[i];
        auto& cur_mapping = buffer.get_mapping_by_idx(mapping_idx);

        auto final_indice = cur_init_indice + cur_mapping.size() - 1;
        auto cur_map_rewards = norm_rewards.slice(0, cur_init_indice, final_indice);
        cur_init_indice = final_indice;

        auto cur_map_size = cur_mapping.size();
        auto initial_state_idx = torch::randint(0, cur_map_size - 1, {1}, torch::kInt32).template item<int>();

        bool clip = (unroll_steps == -1) ? true : (static_cast<int>(initial_state_idx) + unroll_steps >= static_cast<int>(cur_mapping.size()) - 1);
        auto final_idx = clip ? cur_mapping.size() - 1 : (static_cast<int>(initial_state_idx) + unroll_steps);

#ifdef DEBUG
        std::cout << "Mapping index: " << mapping_idx 
                  << ", Map size: " << cur_map_size 
                  << ", Initial idx: " << initial_state_idx 
                  << ", Final idx: " << final_idx 
                  << ", Clip: " << clip << std::endl;
#endif

        for (int k = initial_state_idx; k < static_cast<int>(final_idx); k++) {
            cur_states.push_back(cur_mapping[k]);
        }

        auto cur_rewards = cur_map_rewards.slice(0, initial_state_idx, final_idx);
        auto temp_rewards = torch::cat({torch::tensor({-1.}, torch::TensorOptions().dtype(cur_rewards.dtype())), cur_rewards}, 0);

        auto cur_values = buffer.get_values_by_idx(mapping_idx);
        auto sliced_cur_values = cur_values.slice(0, initial_state_idx, final_idx + 1);
        auto cur_target_values = buffer.get_target_values(temp_rewards, sliced_cur_values);

        all_target_values.push_back(cur_target_values);

        if (buffer.get_is_ppo()) {
            all_values.push_back(cur_values.slice(0, initial_state_idx, final_idx));
        }

        auto cur_policies = buffer.get_policies_by_idx(mapping_idx);
        all_policies.push_back(cur_policies.slice(0, initial_state_idx, final_idx));

        if (!buffer.get_is_mcts_data()) {
            auto cur_advantages = cur_target_values - sliced_cur_values.slice(0, 0, sliced_cur_values.size(0) - 1);
            auto cur_actions = buffer.get_actions_indices_by_idx(mapping_idx).slice(0, initial_state_idx, final_idx);
            all_advantages.push_back(cur_advantages);
            all_actions.push_back(cur_actions);

#ifdef DEBUG
            std::cout << "Advantages size: " << cur_advantages.sizes() 
                      << ", Actions size: " << cur_actions.sizes() << std::endl;
#endif
        }
    }

    torch::Tensor all_advantages_tensor;
    auto buffer_config = buffer.getBufferConfig();

    if (!buffer.get_is_mcts_data()) {
        all_advantages_tensor = torch::cat(all_advantages, 0);
        if (buffer_config.getNormAdvantages()) {
#ifdef DEBUG
            std::cout << "Normalizing advantages..." << std::endl;
#endif
            all_advantages_tensor = norm_by_enum(all_advantages_tensor, EnumNormalization::MEAN_STD, 0, 1);
        }
    }

#ifdef DEBUG
    std::cout << "Final target values shape: " << torch::cat(all_target_values, 0) << std::endl;
    if (!all_advantages_tensor.numel() == 0)
        std::cout << "Final advantages shape: " << all_advantages_tensor << std::endl;
#endif

    return MappingDataset<StateT>(
        cur_states,
        torch::cat(all_target_values, 0),
        buffer.get_is_ppo() ? std::make_optional(torch::cat(all_values, 0)) : std::nullopt,
        all_policies.empty() ? std::nullopt : std::make_optional(tensor_vector_to_tensor_with_padding<float>(all_policies, -std::numeric_limits<float>::infinity())),
        all_actions.empty() ? std::nullopt : std::make_optional(torch::cat(all_actions, 0)),
        all_advantages_tensor.numel() == 0 ? std::nullopt : std::make_optional(all_advantages_tensor),
        std::nullopt,
        std::nullopt,
        std::nullopt
    );
}

#endif