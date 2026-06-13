#include "src/hpp/rl/buffers/base_buffer.hpp"
// #define DEBUG    

template <typename StateT, typename ModelConfig>
BaseBuffer<StateT,ModelConfig>::BaseBuffer(GlobalConfig<ModelConfig>& global_config)  {
    this->total_samples = 0;
    this->number_of_mappings = 0;
    this->buffer_size = global_config.getBufferConfig().getBufferSize();
    this->global_config = global_config;
    if(buffer_size != -1){
        states.reserve(buffer_size);
        target_values.reserve(buffer_size);
        actions_indices.reserve(buffer_size);
        target_policies.reserve(buffer_size);
        advantages.reserve(buffer_size);
        rewards.reserve(buffer_size);
        states_priorities =std::vector<torch::Tensor>(buffer_size, torch::tensor({}));
    }
    auto training = global_config.getLauncherConfig().getTraining();
    this->is_mcts_data = training == EnumTraining::MCTS;
    this->batch_size = global_config.getTrainConfig().getBatchSizeMappings();
    this->mapping_priorities = torch::tensor({});
    this->is_ppo = training == EnumTraining::PPO || training == EnumTraining::HYBRID;  
    auto seed = global_config.getTrainConfig().getSeed();
    gen = torch::make_generator<torch::CPUGeneratorImpl>(seed);  
}



template <typename StateT, typename ModelConfig>
MappingDataset<StateT> BaseBuffer<StateT, ModelConfig>::get_dataset(torch::Tensor& mapping_indices) {
   

}


template<typename StateT, typename ModelConfig>
const std::vector<torch::Tensor>&  BaseBuffer<StateT,ModelConfig>::get_rewards() const{
    return this->rewards;
}

template<typename StateT, typename ModelConfig>
const std::vector<torch::Tensor>&  BaseBuffer<StateT,ModelConfig>::get_values() const{
    return this->target_values;
}

template<typename StateT, typename ModelConfig>
const std::vector<torch::Tensor>&  BaseBuffer<StateT,ModelConfig>::get_actions_indices() const{
    return this->actions_indices;
}

template<typename StateT, typename ModelConfig>
const std::vector<torch::Tensor>&  BaseBuffer<StateT,ModelConfig>::get_policies() const{
    return this->target_policies;
}

template<typename StateT, typename ModelConfig>
const std::vector<std::vector<std::shared_ptr<StateT>>>& BaseBuffer<StateT,ModelConfig>::get_states() const{
    return this->states;
}

template<typename StateT, typename ModelConfig>
torch::Tensor BaseBuffer<StateT,ModelConfig>::get_target_values(const torch::Tensor& rewards,const torch::Tensor& values){
    torch::Tensor target_values;
    auto rl_config = this->global_config.getRLConfig();
    auto use_GAE = rl_config.getUseGAE();
    auto td_steps = rl_config.getTDSteps();
    auto lambda = rl_config.getLambda();
    auto discount = rl_config.getDiscount();
    
    #ifdef DEBUG
        std::cout << "\n===== DEBUG: Computing Target Values =====" << std::endl;
        std::cout << "Use GAE: " << use_GAE << std::endl;
        std::cout << "TD Steps: " << td_steps << std::endl;
        std::cout << "Lambda: " << lambda << std::endl;
        std::cout << "Discount: " << discount << std::endl;
        std::cout << "Rewards: " << rewards << std::endl;
        std::cout << "Values: " << values << std::endl;
    #endif
    
    if (use_GAE) {
        target_values = compute_target_values_with_gae(rewards, values, lambda, discount);
    #ifdef DEBUG
        std::cout << "Computed Target Values (GAE): " << target_values << std::endl;
    #endif
    } else {
        target_values = compute_n_step_target_values(rewards, values, td_steps, discount);
    #ifdef DEBUG
        std::cout << "Computed Target Values (N-Step): " << target_values << std::endl;
    #endif
    }
    
    #ifdef DEBUG
        std::cout << "===== END DEBUG =====\n" << std::endl;
    #endif
    
    return target_values;
    
    
}
template<typename StateT, typename ModelConfig>
torch::Tensor BaseBuffer<StateT,ModelConfig>::get_norm_rewards(const torch::Tensor& rewards) {
    #ifdef DEBUG
    std::cout << "[DEBUG] Starting get_norm_rewards" << std::endl;
    std::cout << "[DEBUG] Original rewards: " << rewards << std::endl;
    #endif

    auto sliced_rewards = rewards.slice(0, 1, rewards.size(0));
    auto rl_config = this->global_config.getRLConfig();
    torch::Tensor norm_rewards;

    if (rl_config.getUseMeanNorm()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] Applying MEAN_STD normalization" << std::endl;
        #endif

        auto norm_sliced_rewards = norm_by_enum(sliced_rewards, EnumNormalization::MEAN_STD, 0, 1);
        auto first_reward = rewards.index({0});
        norm_rewards = torch::cat({first_reward.unsqueeze(0), norm_sliced_rewards}, 0);
    } 
    else if (rl_config.getUseMin0Max1Norm()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] Applying MIN_MAX normalization (0 to 1)" << std::endl;
        #endif

        auto norm_sliced_rewards = norm_by_enum(sliced_rewards, EnumNormalization::MIN_MAX, 0, 1);
        auto first_reward = rewards.index({0});
        norm_rewards = torch::cat({first_reward.unsqueeze(0), norm_sliced_rewards}, 0);
    }else if (rl_config.getUseMinMinus1Max1Norm()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] Applying MIN_MAX normalization (-1 to 1)" << std::endl;
        #endif

        auto norm_sliced_rewards = norm_by_enum(sliced_rewards, EnumNormalization::MIN_MAX, -1, 1);
        auto first_reward = rewards.index({0});
        norm_rewards = torch::cat({first_reward.unsqueeze(0), norm_sliced_rewards}, 0);
    } else {
        #ifdef DEBUG
        std::cout << "[DEBUG] No normalization applied." << std::endl;
        #endif

        norm_rewards = rewards;
    }

    #ifdef DEBUG
    std::cout << "[DEBUG] Normalized rewards: " << norm_rewards << std::endl;
    #endif

    return norm_rewards;
}

template<typename StateT, typename ModelConfig>
torch::Tensor BaseBuffer<StateT,ModelConfig>::get_norm_values(const torch::Tensor& values) {
    #ifdef DEBUG
    std::cout << "[DEBUG] Starting get_norm_values" << std::endl;
    std::cout << "[DEBUG] Original values: " << values << std::endl;
    #endif

    auto sliced_values = values.slice(0, 0, values.size(0) - 1);
    auto rl_config = this->global_config.getRLConfig();
    torch::Tensor norm_values;

    auto last_value = values.index({ values.size(0) - 1});
    if (rl_config.getUseMeanNorm()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] Applying MEAN_STD normalization" << std::endl;
        #endif
        auto norm_sliced_values = norm_by_enum(sliced_values, EnumNormalization::MEAN_STD, 0, 1);
        norm_values = torch::cat({norm_sliced_values, last_value.unsqueeze(0)}, 0);
    } 
    else if (rl_config.getUseMin0Max1Norm()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] Applying MIN_MAX normalization (0 to 1)" << std::endl;
        #endif

        auto norm_sliced_values = norm_by_enum(sliced_values, EnumNormalization::MIN_MAX, 0, 1);
        norm_values = torch::cat({norm_sliced_values, last_value.unsqueeze(0)}, 0);
    }else if (rl_config.getUseMinMinus1Max1Norm()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] Applying MIN_MAX normalization (-1 to 1)" << std::endl;
        #endif

        auto norm_sliced_values = norm_by_enum(sliced_values, EnumNormalization::MIN_MAX, -1, 1);
        norm_values = torch::cat({norm_sliced_values, last_value.unsqueeze(0)}, 0);
    } else {
        #ifdef DEBUG
        std::cout << "[DEBUG] No normalization applied." << std::endl;
        #endif

        norm_values = values;
    }

    #ifdef DEBUG
    std::cout << "[DEBUG] Normalized values: " << norm_values << std::endl;
    #endif

    return norm_values;
}



template <typename StateT, typename ModelConfig>
void BaseBuffer<StateT, ModelConfig>::save_mapping(const MappingHistory<StateT> &mapping_history)
{
    total_samples += states.size() - 1;

#ifdef DEBUG
    std::cout << "\n==== DEBUG: Saving Mapping History ====" << std::endl;
    mapping_history.print();
#endif

    auto reward_tensor = torch::tensor(mapping_history.get_rewards());
    auto norm_rewards = this->get_norm_rewards(reward_tensor);
    torch::Tensor values = torch::tensor(mapping_history.get_values());

    if (this->is_mcts_data) {
        values = this->get_norm_values(values);
    }

#ifdef DEBUG
    std::cout << "\n==== DEBUG: Normalized Rewards ====" << std::endl;
    std::cout << norm_rewards << std::endl;
#endif

    auto target_values = this->get_target_values(norm_rewards, values);

#ifdef DEBUG
    std::cout << "\n==== DEBUG: Target Values ====" << std::endl;
    std::cout << target_values << std::endl;
#endif

    auto idx = this->number_of_mappings % this->buffer_size;

    while(this->cur_reanalyzed_map_idx == idx){
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    const auto& actions_indices = mapping_history.get_actions_indices();
    auto cur_actions_indices_tensor = torch::tensor(actions_indices).slice(0, 1, actions_indices.size());

    torch::Tensor tensor_policies;
    torch::Tensor advantages;

    if (!this->is_mcts_data) {
        advantages = target_values - values.slice(0, 0, values.size(0) - 1);
    }

    const auto& probs = mapping_history.get_probabilities();
    tensor_policies = pad_matrix_to_tensor<float, float>(probs, -std::numeric_limits<float>::infinity());

    const auto& states = mapping_history.get_states();

    if (this->number_of_mappings < this->buffer_size) {
        this->states.push_back(states);
        this->target_values.push_back(target_values);
        this->rewards.push_back(norm_rewards);

        if (!this->is_mcts_data) {
            this->actions_indices.push_back(cur_actions_indices_tensor);
            this->advantages.push_back(advantages);
        } 
        this->target_policies.push_back(tensor_policies.slice(0, 0, tensor_policies.size(0) - 1));
        if(this->is_ppo){
            values_vec.push_back(values.slice(0, 0, values.size(0) - 1));
        }

#ifdef DEBUG
        std::cout << "\n==== DEBUG: Added Data to Buffer ====" << std::endl;
        std::cout << "Buffer size: " << this->number_of_mappings + 1 << std::endl;
#endif

    } else {
        this->total_samples -= (this->states[idx].size() - 1);
        this->states[idx] = states;
        this->target_values[idx] = target_values;
        this->rewards[idx] = norm_rewards;

        if (!this->is_mcts_data) {
            this->actions_indices[idx] = cur_actions_indices_tensor;
            this->advantages[idx] = advantages;
        }            
            this->target_policies[idx] = tensor_policies.slice(0, 0, tensor_policies.size(0) - 1);
        if(this->is_ppo){
            this->values_vec[idx] = values.slice(0, 0, values.size(0) - 1);
        }
#ifdef DEBUG
        std::cout << "\n==== DEBUG: Overwrote Data in Buffer ====" << std::endl;
        std::cout << "Index: " << idx << std::endl;
#endif
    }

    this->number_of_mappings += 1;

#ifdef DEBUG
    std::cout << "\n==== DEBUG: Total Number of Mappings (Max: " << this->buffer_size << ")====" << std::endl << std::flush;
    std::cout << this->number_of_mappings << std::endl;
#endif
}


template <typename StateT, typename ModelConfig>
MappingDataset<StateT> BaseBuffer<StateT, ModelConfig>::get_batch() {
    #ifdef DEBUG
    std::cout << "\n===== DEBUG: get_batch =====" << std::endl;
    std::cout << "Entering get_batch with batch_size: " << batch_size << std::endl;
    #endif
    int num_to_batch = get_num_to_batch();
    
    
    auto buffer_config = this->global_config.getBufferConfig();
    auto train_config = this->global_config.getTrainConfig();
    auto PER = buffer_config.getPER();
    auto PER_alpha = buffer_config.getPERAlpha();
    auto beta = buffer_config.getISWBeta();

    #ifdef DEBUG
    std::cout << "Number of mappings available: " << this->states.size() << std::endl;
    std::cout << "Number of mappings to batch: " << num_to_batch << std::endl;
    std::cout << "PER Enabled: " << (PER ? "Yes" : "No") << std::endl;
    if (PER) {
        std::cout << "PER Alpha: " << PER_alpha << std::endl;
    }
    #endif

torch::Tensor mapping_indices, mapping_probs;
num_to_batch = std::min(num_to_batch, static_cast<int>(this->states.size()));
if (PER) {
    
    #ifdef DEBUG
        std::cout << "[DEBUG] Using PER sampling for batch selection." << std::endl;
        std::cout << "[DEBUG] Mapping priorities tensor: " << this->mapping_priorities << std::endl;
    #endif
    auto  priorities = this->mapping_priorities.pow(PER_alpha);
    mapping_probs =  priorities / priorities.sum();
    #ifdef DEBUG
        std::cout << "[DEBUG] Generated mapping probabilities: " << mapping_probs << " " << num_to_batch << std::endl;
    #endif

    mapping_indices = torch::multinomial(mapping_probs, num_to_batch, false).to(torch::kInt);
    
    #ifdef DEBUG
        std::cout << "[DEBUG] Selected mapping indices: " << mapping_indices << std::endl;
    #endif
} else {
    mapping_indices = torch::randint(0, this->states.size(), {num_to_batch}, torch::kInt);

    #ifdef DEBUG
        std::cout << "Using random sampling for batch selection." << std::endl;
        std::cout << "Selected mapping indices: " << mapping_indices << std::endl;
    #endif
}

   #ifdef DEBUG
    std::cout << "(get_batch)\n";
    std::cout << "Initializing batch processing...\n";
    #endif

    std::vector<std::shared_ptr<StateT>> cur_states;
    std::vector<torch::Tensor> all_target_values;
    std::vector<torch::Tensor> all_policies;
    std::vector<torch::Tensor> all_priorities;
    std::vector<torch::Tensor> all_advantages;
    std::vector<torch::Tensor> all_values;
    std::vector<torch::Tensor> all_isw;
    std::vector<torch::Tensor> all_actions;
    std::vector<torch::Tensor> mapping_indices_tensor;
    std::vector<torch::Tensor> state_indices_tensor;

    int total_size = 0;
    int* mapping_data_ptr = mapping_indices.data_ptr<int>();

    for (int i = 0; i < mapping_indices.size(0); i++) {
        total_size += this->states[mapping_data_ptr[i]].size();
    }

    cur_states.reserve(total_size);
    all_target_values.reserve(total_size);
    all_values.reserve(total_size);
    all_policies.reserve(total_size);
    all_advantages.reserve(total_size);
    all_actions.reserve(total_size);
    all_priorities.reserve(total_size);
    all_isw.reserve(total_size);


    auto rl_config = this->global_config.getRLConfig();
    auto unroll_steps = rl_config.getUnrollSteps();
    auto use_ISW = buffer_config.getUseISW();

    #ifdef DEBUG
        std::cout << "Max Total states to process: " << total_size << "\n";
        std::cout << "PER enabled: " << PER << "\n";
        std::cout << "Unroll steps: " << unroll_steps << "\n";
    #endif

        for (int i = 0; i < mapping_indices.size(0); i++) {
            auto& mapping_idx = mapping_data_ptr[i];
            auto& cur_mapping = this->states[mapping_idx];
          
            torch::Tensor cur_state_probs = this->states_priorities[mapping_idx].pow(PER_alpha);
            cur_state_probs /= cur_state_probs.sum();
        
        int initial_state_idx = PER ? 
            torch::multinomial(cur_state_probs, 1, false, this->gen).item<int>() : 
             torch::randint(
                0,
                states[mapping_idx].size() - 1,
                {1},
                gen
            ).template item<int>();
        bool clip = (unroll_steps == -1) ? true : (static_cast<int>(initial_state_idx) + unroll_steps >= static_cast<int>(cur_mapping.size()) - 1);
        auto final_idx = clip ? cur_mapping.size() - 1 : (static_cast<int>(initial_state_idx) + unroll_steps);
        
        #ifdef DEBUG
            std::cout << "[DEBUG] Mapping Selection Details" << std::endl;
            std::cout << "Selected Mapping Index: " << mapping_idx << std::endl;
            std::cout << "Mapping Size: " << this->states[mapping_idx].size() << std::endl;
            std::cout << "State Priorities (Before Normalization): " << this->states_priorities[mapping_idx] << std::endl;
            std::cout << "State Probabilities (After Normalization): " << cur_state_probs << std::endl;
            std::cout << "Initial State Index Selected: " << initial_state_idx << std::endl;
        #endif

    #ifdef DEBUG
        std::cout << "Processing mapping index: " << mapping_idx << ", initial state index: " << initial_state_idx << "\n";
    #endif
        for(int k = initial_state_idx; k < static_cast<int>(final_idx); k++){
            cur_states.push_back(cur_mapping[k]);
        }

        
        auto cur_target_values = this->target_values[mapping_idx];
        all_target_values.push_back(cur_target_values.slice(0, initial_state_idx, final_idx));

        if(this->is_ppo){
            auto cur_values = this->values_vec[mapping_idx];
            all_values.push_back(cur_values.slice(0, initial_state_idx, final_idx));
        }
        auto cur_policies = this->target_policies[mapping_idx];
        all_policies.push_back(cur_policies.slice(0, initial_state_idx, final_idx));
        if (!this->is_mcts_data) {
            auto cur_advantages = this->advantages[mapping_idx];
            auto cur_actions = this->actions_indices[mapping_idx];
            all_advantages.push_back(cur_advantages.slice(0, initial_state_idx, final_idx));
            all_actions.push_back(cur_actions.slice(0, initial_state_idx, final_idx));
        }

        if (PER) {
            auto cur_priorities = this->states_priorities[mapping_idx].slice(0, initial_state_idx, final_idx);
            auto num_states = final_idx - initial_state_idx;
            auto map_idx_tensor = torch::full({static_cast<int>(num_states)}, mapping_idx, torch::dtype(torch::kInt32));
            mapping_indices_tensor.push_back(map_idx_tensor);
            state_indices_tensor.push_back(torch::arange(initial_state_idx, final_idx));

            all_priorities.push_back(cur_priorities);
            if (use_ISW) {
                auto N = std::min(this->number_of_mappings, this->buffer_size);
                auto indices = torch::arange(initial_state_idx, final_idx);
                auto selected_probs = cur_state_probs.index_select(0, indices);
                auto scaled_probs = selected_probs * mapping_probs[i] * N;
                auto cur_isw = torch::pow(scaled_probs, -beta);
                
                #ifdef DEBUG
                    std::cout << "[DEBUG] Importance Sampling Weight Calculation" << std::endl;
                    std::cout << "N (Min of number_of_mappings and buffer_size): " << N << std::endl;
                    std::cout << "Initial state index: " << initial_state_idx << ", Final index: " << final_idx << std::endl;
                    std::cout << "Selected state probabilities: " << selected_probs << std::endl;
                    std::cout << "Mapping probability for trajectory " << i << ": " << mapping_probs[i].item<double>() << std::endl;
                    std::cout << "Scaled probabilities (selected_probs * mapping_probs[i] * N): " << scaled_probs << std::endl;
                    std::cout << "Importance Sampling Weights (ISW): " << cur_isw << std::endl;
                #endif         
                all_isw.push_back(cur_isw);
            }
            
        }
    }
    torch::Tensor all_advantages_tensor;
    if(!is_mcts_data){
        all_advantages_tensor = torch::cat(all_advantages, 0);
        if(buffer_config.getNormAdvantages()){
           all_advantages_tensor = norm_by_enum(all_advantages_tensor,EnumNormalization::MEAN_STD, 0, 1); 
       }
    }
    MappingDataset<StateT> dataset;
    if (PER) {
        auto all_priorities_tensor = torch::cat(all_priorities, 0);
        auto mapping_indices_tensor_cat = torch::cat(mapping_indices_tensor, 0);
        auto state_indices_tensor_cat = torch::cat(state_indices_tensor, 0);
        
        torch::Tensor isw_tensor;
        if (use_ISW) {
            isw_tensor = torch::cat(all_isw, 0);
            if(buffer_config.getNormISWWithMax()){
                isw_tensor /= isw_tensor.max();
            }
            #ifdef DEBUG
                std::cout << "Returning dataset with Importance Sampling Weights.\n";
                std::cout << "Importance Sampling Weights: " << isw_tensor << std::endl;

            #endif
        }
        #ifdef DEBUG
        std::cout << "Returning dataset with Importance Sampling Weights.\n";
        std::cout << "Mapping indices \n " << mapping_indices_tensor_cat << std::endl;
        std::cout << "state indices :.\n" << state_indices_tensor_cat << std::endl;
        std::cout << "Importance Sampling Weights: " << isw_tensor << std::endl;

        #endif
        
        dataset =  MappingDataset<StateT>(
            cur_states, 
            torch::cat(all_target_values, 0),
            this->is_ppo ? std::make_optional(torch::cat(all_values, 0)) : std::nullopt,
            all_policies.empty() ? std::nullopt : std::make_optional(tensor_vector_to_tensor_with_padding<float>(all_policies, -std::numeric_limits<float>::infinity())),
            all_actions.empty() ? std::nullopt : std::make_optional(torch::cat(all_actions, 0)),
            all_advantages_tensor.numel() == 0 ? std::nullopt : std::make_optional(all_advantages_tensor),
            mapping_indices_tensor_cat,
            state_indices_tensor_cat,
            use_ISW ?  std::make_optional(isw_tensor) : std::nullopt
        );

    } else {

    #ifdef DEBUG
        std::cout << "Returning dataset without Importance Sampling Weights.\n";
    #endif

        dataset =  MappingDataset<StateT>(
            cur_states, 
            torch::cat(all_target_values, 0),
            this->is_ppo ? std::make_optional(torch::cat(all_values, 0)) : std::nullopt,
            std::make_optional(tensor_vector_to_tensor_with_padding<float>(all_policies, 0)),
            all_actions.empty() ? std::nullopt : std::make_optional(torch::cat(all_actions, 0)),
            all_advantages_tensor.numel() == 0 ? std::nullopt : std::make_optional(all_advantages_tensor),
            std::nullopt,
            std::nullopt,
            std::nullopt
        );
    }
    torch::Tensor target_values = dataset.get_target_values_();
    double raw_mean_values = target_values.mean().item<double>();
    double raw_std_values = target_values.std().item<double>();
    dataset.set_mean_values(raw_mean_values);
    dataset.set_std_values(raw_std_values);
    return dataset;

}




template <typename StateT, typename ModelConfig>
int BaseBuffer<StateT,ModelConfig>::get_number_of_mappings() const {
    if(this->key_to_dtb.size() > 0){
        return static_cast<int>(this->key_to_dtb.size());
    }
    return this->number_of_mappings;
}

template <typename StateT, typename ModelConfig>
int BaseBuffer<StateT,ModelConfig>::get_num_data_in_buffer() const{
    if(this->key_to_dtb.size() > 0){
        return static_cast<int>(this->key_to_dtb.size());
    }
    return static_cast<int>(this->states.size());
}

template <typename StateT, typename ModelConfig>
void BaseBuffer<StateT,ModelConfig>::set_states_priorities_by_idx(torch::Tensor states_priorities,const int& idx){
    this->states_priorities[idx] = states_priorities;
}

template <typename StateT, typename ModelConfig>
void BaseBuffer<StateT,ModelConfig>::set_states_priorities_by_idxs(torch::Tensor states_priorities,const int& map_idx, 
    torch::Tensor states_indices){
        if (states_priorities.isnan().any().item<bool>() || states_priorities.isinf().any().item<bool>()) {
            throw std::runtime_error("States priorities contain NaN or inf values.");
        }
         
        this->states_priorities[map_idx].index_put_({states_indices}, states_priorities);
    }

template <typename StateT, typename ModelConfig>
void BaseBuffer<StateT, ModelConfig>::set_mapping_priority_by_idx(
    double mapping_priority, const int& idx)
{
    if (!this->global_config.getBufferConfig().getPER()) return;

    TORCH_CHECK(this->mapping_priorities.defined(),
                "mapping_priorities is not initialized");

    if (idx > this->mapping_priorities.size(0) - 1) {

        auto new_tensor = torch::tensor(
            {mapping_priority},
            torch::TensorOptions()
                .dtype(this->mapping_priorities.dtype())
                .device(this->mapping_priorities.device())
        );

        this->mapping_priorities = torch::cat(
            {this->mapping_priorities, new_tensor}, 0);

    } else {
        this->mapping_priorities[idx] = mapping_priority;
    }
}



template <typename StateT, typename ModelConfig>
torch::Tensor BaseBuffer<StateT,ModelConfig>::get_target_values_by_idx(const int& idx) const {
    return this->target_values[idx];
}


template <typename StateT, typename ModelConfig>
torch::Tensor BaseBuffer<StateT,ModelConfig>::get_state_priorities_by_idx(const int& idx) const {
    return this->states_priorities[idx];
}


template <typename StateT, typename ModelConfig>
torch::Tensor BaseBuffer<StateT,ModelConfig>::get_mapping_priorities() const {
    return this->mapping_priorities;
}

template <typename StateT, typename ModelConfig>
void BaseBuffer<StateT,ModelConfig>::increment_total_samples(int samples){
    this->total_samples += samples;
}

template <typename StateT, typename ModelConfig>
void BaseBuffer<StateT,ModelConfig>::increment_number_of_mappings(int value){
    this->number_of_mappings += value;

}

template <typename StateT, typename ModelConfig>
void BaseBuffer<StateT,ModelConfig>::reduce_mapping_priorities_by_alpha(torch::Tensor mapping_indices){
    auto buffer_config = global_config.getBufferConfig();
    auto alpha_reduce = buffer_config.getAlphaReduce();

    auto unique_mapping_indices = std::get<0>(at::_unique2(mapping_indices));
    
    #ifdef DEBUG
        std::cout << "\n===== DEBUG: Updating Mapping Priorities =====" << std::endl;
        std::cout << "PER Alpha: " << alpha_reduce << std::endl;
        std::cout << "Original Mapping Indices: " << mapping_indices << std::endl;
        std::cout << "Unique Mapping Indices: " << unique_mapping_indices << std::endl;
        std::cout << "Original Mapping Priorities (before update): " 
                  << this->mapping_priorities.index({unique_mapping_indices}) << std::endl;
    #endif
    
    this->mapping_priorities.index_put_({unique_mapping_indices}, 
        this->mapping_priorities.index({unique_mapping_indices}) * alpha_reduce);
    
    #ifdef DEBUG
        std::cout << "Updated Mapping Priorities: " 
                  << this->mapping_priorities.index({unique_mapping_indices}) << std::endl;
        std::cout << "===== END DEBUG =====\n" << std::endl;
    #endif
    
    
}

template <typename StateT, typename ModelConfig>
void BaseBuffer<StateT,ModelConfig>::reduce_states_priorities_by_alpha(torch::Tensor mapping_indices, torch::Tensor states_indices){
    auto unique_mapping_indices = std::get<0>(at::_unique2(mapping_indices));
    auto buffer_config = global_config.getBufferConfig();
    auto alpha_reduce = buffer_config.getAlphaReduce();
    
    #ifdef DEBUG
        std::cout << "\n===== DEBUG: Updating State Priorities =====" << std::endl;
        std::cout << "PER Alpha: " << alpha_reduce << std::endl;
        std::cout << "Unique Mapping Indices: " << unique_mapping_indices << std::endl;
    #endif
    
    for (int i = 0; i < unique_mapping_indices.size(0); i++) {
        auto mapping_idx = unique_mapping_indices[i].item<int>();
        auto cur_states_indices = states_indices.index({mapping_indices == mapping_idx});
    
    #ifdef DEBUG
        std::cout << "\nProcessing Mapping Index: " << mapping_idx << std::endl;
        std::cout << "Current States Indices: " << cur_states_indices << std::endl;
        std::cout << "Original State Priorities: " 
                  << this->states_priorities[mapping_idx].index({cur_states_indices}) << std::endl;
    #endif
    
        this->states_priorities[mapping_idx].index_put_({cur_states_indices}, 
            this->states_priorities[mapping_idx].index({cur_states_indices}) * alpha_reduce);
    
    #ifdef DEBUG
        std::cout << "Updated State Priorities: " 
                  << this->states_priorities[mapping_idx].index({cur_states_indices}) << std::endl;
    #endif
    }
    
    #ifdef DEBUG
        std::cout << "===== END DEBUG =====\n" << std::endl;
    #endif
    
}

template <typename StateT, typename ModelConfig>
const std::vector<std::shared_ptr<StateT>>& BaseBuffer<StateT,ModelConfig>::get_mapping_by_idx(const int& idx) const {
    return this->states[idx];
}

template <typename StateT, typename ModelConfig>
void BaseBuffer<StateT,ModelConfig>::update_with_renalyzed_mapping(MappingHistory<StateT>& mapping_history, const int& idx){
    auto norm_rewards = this->rewards[idx];

    // #ifdef DEBUG
    //     std::cout << "\n===== DEBUG: Updating Target Values and Policies =====" << std::endl;
    //     std::cout << "Index: " << idx << std::endl;
    //     std::cout << "Normalized Rewards: " << norm_rewards << std::endl;
    // #endif
    
    auto values = torch::tensor(mapping_history.get_values(), torch::kFloat32);
    
    // #ifdef DEBUG
    //     std::cout << "Original Values: " << values << std::endl;
    // #endif
    
    auto cur_target_values = this->get_target_values(norm_rewards, values);
    this->target_values[idx] = cur_target_values;
    
    // #ifdef DEBUG
    //     std::cout << "Updated Target Values: " << cur_target_values << std::endl;
    // #endif
    
    if (!this->target_policies.empty()) {
        auto policies = pad_matrix_to_tensor<float, float>(mapping_history.get_child_visits(),-std::numeric_limits<float>::infinity());
    
    // #ifdef DEBUG
    //     std::cout << "Target Policies before update: " << this->target_policies[idx] << std::endl;
    //     std::cout << "New Policies: " << policies << std::endl;
    // #endif
    
        this->target_policies[idx] = policies;
    
    // #ifdef DEBUG
    //     std::cout << "Updated Target Policies: " << this->target_policies[idx] << std::endl;
    // #endif
    }
    
    // #ifdef DEBUG
    //     std::cout << "===== END DEBUG =====\n" << std::endl;
    // #endif
    const auto& target_values = this->get_target_values_by_idx(idx);
    auto sliced_values = values.slice(0, 0, values.size(0) - 1);
    
    auto cur_priorities = torch::abs(target_values - sliced_values);
    auto mapping_priority = cur_priorities.mean().template item<double>();    
    if(this->mapping_priorities.numel() > 0){
        this->set_mapping_priority_by_idx(mapping_priority, idx);
        this->set_states_priorities_by_idx(cur_priorities, idx);
    }
    
}

template <typename StateT, typename ModelConfig>
const int& BaseBuffer<StateT,ModelConfig>::get_buffer_size() const {
    return this->buffer_size;
}

template <typename StateT, typename ModelConfig>
void BaseBuffer<StateT,ModelConfig>::clear(){
    target_values.clear();
    actions_indices.clear();
    target_policies.clear();
    advantages.clear();
    rewards.clear();
    values_vec.clear();
    states_priorities.clear();
    this->states.clear();
    if(this->buffer_size != -1){
        states.reserve(buffer_size);
        target_values.reserve(buffer_size);
        actions_indices.reserve(buffer_size);
        target_policies.reserve(buffer_size);
        advantages.reserve(buffer_size);
        rewards.reserve(buffer_size);
        states_priorities =std::vector<torch::Tensor>(buffer_size, torch::tensor({}));
        this->mapping_priorities = torch::tensor({});
    }
        
    total_samples = 0;
    number_of_mappings = 0;
}
template<typename StateT, typename ModelConfig>
bool BaseBuffer<StateT, ModelConfig>::get_is_mcts_data() const {
    return this->is_mcts_data;
}

template<typename StateT, typename ModelConfig>
bool BaseBuffer<StateT, ModelConfig>::get_is_ppo() const {
    return this->is_ppo ;
}
template<typename StateT, typename ModelConfig>
void BaseBuffer<StateT, ModelConfig>::append_states(const std::vector<std::shared_ptr<StateT>>& states){
    this->states.push_back(states);
}

template<typename StateT, typename ModelConfig>
void BaseBuffer<StateT, ModelConfig>::append_rewards(torch::Tensor rewards){
    this->rewards.push_back(rewards);
}



template<typename StateT, typename ModelConfig>
void BaseBuffer<StateT, ModelConfig>::append_values(torch::Tensor values){

    this->values_vec.push_back(values);
}


template<typename StateT, typename ModelConfig>
void BaseBuffer<StateT, ModelConfig>::append_policies(torch::Tensor policies){
    this->target_policies.push_back(policies);
}

template<typename StateT, typename ModelConfig>
void BaseBuffer<StateT, ModelConfig>::append_actions_indices(torch::Tensor actions_indices){
    this->actions_indices.push_back(actions_indices);
}
