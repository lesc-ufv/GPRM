#ifndef SMARTMAP_PRE_TRAINING_BUFFER_HPP
#define SMARTMAP_PRE_TRAINING_BUFFER_HPP

#include "vector"
#include "torch/torch.h"
#include "src/hpp/enums/enum_nomalization.hpp"
#include "src/hpp/compilers/smartmap/datasets/smartmap_dataset.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"
#include "src/hpp/utils/util_train.hpp"
#include "src/hpp/rl/mapping_history.hpp"
#include "src/hpp/rl/buffers/base_buffer.hpp"
#include "src/hpp/utils/util_torch.hpp"

template<typename StateT, typename ModelConfig>
void pretraining_smartmap_add_mapping(BaseBuffer<StateT, ModelConfig>& buffer, const MappingHistory<StateT>& mapping_history);

template<typename StateT, typename ModelConfig>
MappingDataset<StateT> pretraining_smartmap_get_batch(BaseBuffer<StateT, ModelConfig>& buffer);

#include "src/tpp/compilers/smartmap/rl/smartmap_pre_training_buffer.tpp"


#endif