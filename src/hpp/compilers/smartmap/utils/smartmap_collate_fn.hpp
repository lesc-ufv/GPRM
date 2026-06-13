#ifndef SMARTMAP_COLLATE_HPP
#define SMARTMAP_COLLATE_HPP

#include <vector>
#include "torch/torch.h"
#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"
#include "src/hpp/utils/util_train.hpp"
#include "src/hpp/utils/util_torch.hpp"

std::vector<c10::IValue> smartmap_collate_fn(const std::vector<std::shared_ptr<SmartMapState>>& states, const std::vector<int>& indices);


#endif