#ifndef SUPERVISED_BUFFER_HPP
#define SUPERVISED_BUFFER_HPP

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
#include "src/hpp/utils/util_train.hpp"

template<typename StateT, typename ModelConfig>
void supervised_buffer_add_mapping(BaseBuffer<StateT, ModelConfig>& buffer, const MappingHistory<StateT>& mapping_history){

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

    
    const auto& states = mapping_history.get_states();

    #ifdef DEBUG
    std::cout << "\n==== DEBUG: Appending Data to Buffer ====\n" << std::endl;
    std::cout << "[DEBUG] States size: " << states.size() << std::endl;
    // std::cout << "[DEBUG] Values: " << values << std::endl;
    std::cout << "[DEBUG] Actions Indices Tensor: " << cur_actions_indices_tensor << std::endl;
    // assert(tensor_policies.slice(0, 0, tensor_policies.size(0) - 1).sizes().size() == 2);
    #endif

    buffer.append_states(states);
    buffer.append_rewards(sliced_rewards);
    buffer.append_actions_indices(cur_actions_indices_tensor);
    buffer.increment_number_of_mappings(1);
}
template<typename StateT, typename ModelConfig>
MappingDataset<StateT> supervised_buffer_get_batch(BaseBuffer<StateT, ModelConfig>& buffer){

    const auto& rewards_vec   = buffer.get_rewards();
    const auto& all_mappings  = buffer.get_states();
    std::vector<std::shared_ptr<StateT>> all_states; 
    const auto& all_actions   = buffer.get_actions_indices();

    auto norm_rewards = buffer.get_norm_rewards(torch::cat(rewards_vec, 0));
    auto values = std::vector<torch::Tensor>();

    int initial_state_idx = 0;
    auto first_reward = torch::tensor({-1});
    int total_states = 0;

#ifdef DEBUG
    std::cout << "[DEBUG] rewards_vec size: " << rewards_vec.size() << std::endl;
    std::cout << "[DEBUG] all_mappings size: " << all_mappings.size() << std::endl;
    std::cout << "[DEBUG] all_actions size: " << all_actions.size() << std::endl;
    std::cout << "[DEBUG] norm_rewards shape: " << norm_rewards.sizes() << std::endl;
#endif

    for (size_t map_idx = 0; map_idx < all_mappings.size(); ++map_idx) {
        const auto& mapping = all_mappings[map_idx];

        for(int i = 0; i < static_cast<int>(mapping.size()) - 1; i++){
            all_states.push_back(mapping[i]);
            total_states++;
        }
        auto map_len  = mapping.size();
        auto final_idx = initial_state_idx + map_len - 1;
        auto cur_rewards = norm_rewards.slice(0, initial_state_idx, final_idx);
        auto temp_rewards = torch::cat({first_reward, cur_rewards}, 0);
        initial_state_idx = final_idx;

#ifdef DEBUG
        std::cout << "[DEBUG] map_idx: " << map_idx 
                  << " | map_len: " << map_len 
                  << " | cur_rewards shape: " << cur_rewards.sizes() 
                  << " | temp_rewards shape: " << temp_rewards.sizes() 
                  << std::endl;
#endif

        auto target_values = compute_n_step_target_values(temp_rewards, torch::tensor({}), -1, 1);
        values.push_back(target_values);
    }
    
    torch::Tensor values_tensor = torch::cat(values, 0);

#ifdef DEBUG
    std::cout << "[DEBUG] values_tensor shape: " << values_tensor.sizes() << std::endl;
    std::cout << "[DEBUG] total_states: " << total_states << std::endl;
#endif

    assert(values_tensor.numel() == total_states);

    return MappingDataset<StateT>(
        all_states,
        values_tensor,
        std::nullopt,
        std::nullopt,
        std::make_optional(torch::cat(all_actions, 0)),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt
    );
}


#endif