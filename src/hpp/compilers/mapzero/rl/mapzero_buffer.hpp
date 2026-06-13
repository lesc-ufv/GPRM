#ifndef MAPZERO_BUFFER_DATA_HPP
#define MAPZERO_BUFFER_DATA_HPP

#include <iostream>
#include <torch/torch.h>
#include "src/hpp/compilers/mapzero/rl/mapzero_state.hpp"
#include "src/hpp/utils/util_train.hpp"
#include <cmath>
#include "src/hpp/utils/mapping_augmentation.hpp"
#include "src/hpp/utils/util_print.hpp"
#include "src/hpp/compilers/mapzero/rl/mapzero_environment.hpp"
#include "src/hpp/enums/enum_nomalization.hpp"
#include "src/hpp/rl/buffers/base_buffer.hpp"
#include "src/hpp/rl/mapping_history.hpp"

template<typename StateT,typename ModelConfig>
void mapzero_add_mapping(BaseBuffer<StateT,ModelConfig>& buffer, const MappingHistory<StateT>& mapping_history);

template<typename StateT,typename ModelConfig>
void mapzero_update_priorities(BaseBuffer<StateT, ModelConfig>& buffer, 
        torch::Tensor mapping_indices,
        torch::Tensor states_indices,
        torch::Tensor new_priorities);

#include "src/tpp/compilers/mapzero/rl/mapzero_buffer.tpp"
#endif