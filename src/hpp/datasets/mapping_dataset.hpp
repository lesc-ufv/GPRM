#ifndef MAPPING_DATASET_HPP
#define MAPPING_DATASET_HPP

#include <iostream>
#include <torch/torch.h>
#include "src/hpp/rl/collate.hpp"
template <typename StateT>
class MappingDataset {
private:
    std::vector<std::shared_ptr<StateT>> states;
    torch::Tensor target_values, values;
    torch::Tensor target_policies, actions, advantages;
    torch::Tensor mapping_indices, state_indices, IS_weight;
    double raw_mean_advantages, raw_std_advantages, mean_values, std_values;
public:
    MappingDataset() {}
    MappingDataset(const std::vector<std::shared_ptr<StateT>>& states, 
                        const torch::Tensor& target_values,
                        const std::optional<torch::Tensor>& values,
                        const std::optional<torch::Tensor>& target_policies,
                        const std::optional<torch::Tensor>& action_indices,
                        const std::optional<torch::Tensor>& advantages,
                        const std::optional<torch::Tensor>& mapping_indices,
                        const std::optional<torch::Tensor>& state_indices,
                        const std::optional<torch::Tensor>& IS_weigth
                    );

    void set_raw_mean_advantages(double raw_mean_advantages){
        this->raw_mean_advantages = raw_mean_advantages;
    };

    void set_raw_std_advantages(double raw_std_advantages){
        this->raw_std_advantages = raw_std_advantages;
    };

    void set_mean_values(double mean_values){
        this->mean_values = mean_values;
    };

    void set_std_values(double std_values){
        this->std_values = std_values;
    };

    double get_mean_values() const {
        return mean_values;
    };

    double get_std_values() const {
        return std_values;
    };
    
    double get_raw_mean_advantages() const {
        return raw_mean_advantages;
    };
    double get_raw_std_advantages() const {
        return raw_std_advantages;
    };
    const std::vector<std::shared_ptr<StateT>>& get_states_() const;
    const torch::Tensor& get_target_values_() const;
    const torch::Tensor& get_target_policies_() const;
    const torch::Tensor& get_actions_() const;
    const torch::Tensor& get_advantages_() const;
    const torch::Tensor& get_mapping_indices_() const;
    const torch::Tensor& get_state_indices_() const;
    const torch::Tensor& get_IS_weigth_() const;


    const bool& is_mcts_data_() const;
    int size() const;
    void add_IS_weigths(std::optional<torch::Tensor>& IS_weigth);
    torch::Tensor get_mapping_idxs_by_idxs(torch::Tensor idxs);
    torch::Tensor get_state_idxs_by_idxs(torch::Tensor idxs);

    std::vector<c10::IValue> get_train_data(const Collate<StateT>& collate,const std::vector<int>& indices);

};

#include "src/tpp/datasets/mapping_dataset.tpp"

#endif