#ifndef GPRM_COLLATE_FN_HPP
#define GPRM_COLLATE_FN_HPP

#include <iostream>
#include <torch/torch.h>
#include "src/hpp/utils/util_train.hpp"
#include <string>
#include <unordered_map>
#include "src/hpp/compilers/gprm/rl/states/gprm_state.hpp"

std::vector<c10::IValue> gprm_collate_fn(const std::vector<std::shared_ptr<GPRMState>>& states, 
                                                const std::vector<int>& indices);

void add_batch_idx(std::vector<torch::Tensor>& batch_batch_idx, const std::vector<int>& sizes);

void add_seq_idx(std::vector<torch::Tensor>& batch_seq_idx, const std::vector<int>& sizes, std::vector<int>& previous_sizes);

torch::Tensor create_batch_placement_routing( const std::vector<torch::Tensor>& tensor,const std::vector<int>& sizes_idx_0, const std::vector<int>& sizes_idx_1);

#endif