#ifndef MAPZERO_COLLATE_FN_HPP
#define MAPZERO_COLLATE_FN_HPP

#include <iostream>
#include <torch/torch.h>
#include "src/hpp/compilers/mapzero/rl/mapzero_state.hpp"
#include "src/hpp/utils/util_train.hpp"
#include "torch/torch.h"
#include <string>
#include <unordered_map>

std::vector<c10::IValue> mapzero_collate_fn(const std::vector<std::shared_ptr<MapZeroState>>& states, const std::vector<int>& indices);

#endif