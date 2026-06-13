#include "src/hpp/datasets/mapping_dataset.hpp"

template <typename StateT>
MappingDataset<StateT>::MappingDataset(const std::vector<std::shared_ptr<StateT>>& states, 
    const torch::Tensor& target_values,
    const std::optional<torch::Tensor>& values,
    const std::optional<torch::Tensor>& target_policies,
    const std::optional<torch::Tensor>& action_indices,
    const std::optional<torch::Tensor>& advantages,
    const std::optional<torch::Tensor>& mapping_indices,
    const std::optional<torch::Tensor>& state_indices,
    const std::optional<torch::Tensor>& IS_weight)
{
    #ifdef DEBUG
    std::cout << "\n=== Initializing MappingDataset ===" << std::endl;
    std::cout << "Number of states: " << states.size() << std::endl;
    std::cout << "Target values shape: " << target_values.sizes() << std::endl;
    #endif

    if (target_values.size(0) != static_cast<int>(states.size())) {
        std::stringstream ss;
        ss << "Mismatch between states (" << states.size() 
           << ") and target_values (" << target_values.size(0) << ")";
        throw std::invalid_argument(ss.str());
    }

    this->states = states;
    this->target_values = target_values;

    auto initialize_optional = [](auto& member, const auto& optional_val, const std::string& name) {
        if (optional_val.has_value()) {
            member = optional_val.value();
            #ifdef DEBUG
            std::cout << name << " shape: " << member.sizes() << std::endl;
            #endif
        } else {
            member = torch::tensor({});
            #ifdef DEBUG
            std::cout << name << " not provided, using empty tensor" << std::endl;
            #endif
        }
    };

    initialize_optional(this->target_policies, target_policies, "Target policies");
    initialize_optional(this->actions, action_indices, "Action indices");
    initialize_optional(this->advantages, advantages, "Advantages");
    initialize_optional(this->mapping_indices, mapping_indices, "Mapping indices");
    initialize_optional(this->state_indices, state_indices, "State indices");
    initialize_optional(this->IS_weight, IS_weight, "IS weights");
    initialize_optional(this->values, values, "Values");
    // std::cout << "Target policies shape: " << this->target_policies.sizes() << std::endl;
    // std::cout << "Values shape: " << this->values.sizes() << std::endl;
    if (target_policies.has_value() && static_cast<int>(target_policies.value().size(0)) != static_cast<int>(states.size())) {
        throw std::invalid_argument("Target policies size mismatch with states");
    }

    if (action_indices.has_value() && static_cast<int>(action_indices.value().size(0) )!= static_cast<int>(states.size())) {
        throw std::invalid_argument("Action indices size mismatch with states");
    }
    
    if (values.has_value() && static_cast<int>(values.value().size(0) )!= static_cast<int>(states.size())) {
        throw std::invalid_argument("Values indices size mismatch with states");
    }
    

    #ifdef DEBUG
    std::cout << "=== MappingDataset initialized successfully ===" << std::endl;
    #endif
}

template <typename StateT>
const std::vector<std::shared_ptr<StateT>>& MappingDataset<StateT>::get_states_() const {
    return this->states;
}

template <typename StateT>
const torch::Tensor& MappingDataset<StateT>::get_target_values_() const{
    return this->target_values;
}

template <typename StateT>
const torch::Tensor& MappingDataset<StateT>::get_target_policies_() const{
    return this->target_policies;
}

template <typename StateT>
const torch::Tensor& MappingDataset<StateT>::get_actions_() const{
    return this->actions;
}

template <typename StateT>
const torch::Tensor& MappingDataset<StateT>::get_advantages_() const {
    return this->advantages;
}

template <typename StateT>
const bool& MappingDataset<StateT>::is_mcts_data_() const{
    return this->advantages.numel() > 0;  
}

template <typename StateT>
int MappingDataset<StateT>::size() const {
    return this->states.size();
}

template <typename StateT>
void MappingDataset<StateT>::add_IS_weigths(std::optional<torch::Tensor>& IS_weigth) {
    if(IS_weigth.has_value()){
        this->IS_weight = IS_weigth.value();
    }else{
        this->IS_weigth = IS_weigth;
    }
}

template <typename StateT>
std::vector<c10::IValue> MappingDataset<StateT>::get_train_data(
    const Collate<StateT>& collate,
    const std::vector<int>& indices
) {
#ifdef DEBUG
    std::cout << "[DEBUG] Starting to collect training data..." << std::endl;
#endif

    auto train_data = collate.collate(this->states, indices);
    auto tensor_indices = torch::tensor(indices, torch::kInt32);

    if (target_values.numel() > 0) {
#ifdef DEBUG
        std::cout << "[DEBUG] Adding target values to the training data." << std::endl;
#endif
        auto cur_target_values = target_values.index({tensor_indices});
        train_data.emplace_back(cur_target_values);
    }

    if (values.numel() > 0) {
#ifdef DEBUG
        std::cout << "[DEBUG] Adding values to the training data." << std::endl;
#endif
        auto cur_values = values.index({tensor_indices});
        train_data.emplace_back(cur_values);
    }

    if (target_policies.numel() > 0) {
#ifdef DEBUG
        std::cout << "[DEBUG] Adding target policies to the training data." << std::endl;
#endif
        auto cur_policies = target_policies.index({tensor_indices});
        train_data.emplace_back(cur_policies);
    }

    if (actions.numel() > 0) {
#ifdef DEBUG
        std::cout << "[DEBUG] Adding actions ato the training data." << std::endl;
#endif
        auto cur_actions = actions.index({tensor_indices});
        train_data.emplace_back(cur_actions);
    }

    if (advantages.numel() > 0) {
        #ifdef DEBUG
                std::cout << "[DEBUG] Adding advantages to the training data." << std::endl;
        #endif
            auto cur_advantages = advantages.index({tensor_indices});
            train_data.emplace_back(cur_advantages);
            }
        
    if (IS_weight.numel() > 0) {
        #ifdef DEBUG
                std::cout << "[DEBUG] Adding IS weight to the training data." << std::endl;
        #endif
        auto cur_IS_weight = IS_weight.index({tensor_indices});
        train_data.emplace_back(cur_IS_weight);
    }

#ifdef DEBUG
    std::cout << "[DEBUG] Finished collecting training data." << std::endl;
#endif

    return train_data;
}


template <typename StateT>
torch::Tensor MappingDataset<StateT>::get_mapping_idxs_by_idxs(torch::Tensor idxs){
    return this->mapping_indices.index_select(0,{idxs});
}

template <typename StateT>
torch::Tensor MappingDataset<StateT>::get_state_idxs_by_idxs(torch::Tensor idxs){
    return this->state_indices.index_select(0,{idxs});
}
template <typename StateT>
const torch::Tensor& MappingDataset<StateT>::get_mapping_indices_() const{
    return this->mapping_indices;
}
template <typename StateT>

const torch::Tensor& MappingDataset<StateT>::get_state_indices_() const{
    return this->state_indices;
}
template <typename StateT>

const torch::Tensor& MappingDataset<StateT>::get_IS_weigth_() const{
    return this->IS_weight;
}