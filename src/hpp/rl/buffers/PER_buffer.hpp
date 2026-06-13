#ifndef PER_BUFFER_DATA_HPP
#define PER_BUFFER_DATA_HPP

#include <iostream>
#include <torch/torch.h>
#include "src/hpp/utils/util_train.hpp"
#include <cmath>
#include "src/hpp/utils/mapping_augmentation.hpp"
#include "src/hpp/utils/util_print.hpp"
#include "src/hpp/enums/enum_nomalization.hpp"
#include "src/hpp/rl/mapping_history.hpp"
#include "src/hpp/rl/buffers/base_buffer.hpp"

template <typename StateT,typename ModelConfig>
void PER_add_mapping(BaseBuffer<StateT,ModelConfig>& buffer, const MappingHistory<StateT>& mapping_history);

template <typename StateT,typename ModelConfig>
void PER_update_priorities(BaseBuffer<StateT, ModelConfig>& base_buffer, 
            torch::Tensor mapping_indices,
            torch::Tensor states_indices,
            torch::Tensor new_priorities) ;


#include "src/tpp/compilers/smartmap/rl/PER_buffer.tpp"

#endif