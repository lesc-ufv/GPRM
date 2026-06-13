#ifndef MAPZERO_SELF_MAPPING_HPP
#define MAPZERO_SELF_MAPPING_HPP

#include <vector>
#include <algorithm>
#include "src/hpp/compilers/mapzero/rl/mapzero_state.hpp"
#include "src/hpp/compilers/mapzero/rl/mapzero_mcts.hpp"
#include "src/hpp/compilers/mapzero/rl/mapzero_environment.hpp"
#include "zmq.h"
#include <cmath>
#include "src/hpp/utils/util_mcts.hpp"
#include "src/hpp/rl/mapping_history.hpp"
#include "src/hpp/enums/enum_task_type.hpp"
#include "src/hpp/rl/mapping_history.hpp"
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/rl/node.hpp"

template<typename StateT, typename ModelConfig>
MappingHistory<StateT>  map_with_mapzero_mcts( GlobalConfig<ModelConfig>& global_config,
                               std::shared_ptr<zmq::socket_t> socket,
                                const ServerConstants& server_k, const std::string& func_name,
                                            const std::shared_ptr<StateT> initial_state, Environment<StateT>& environment,
                                            bool train,  double train_ratio, int seed,
                                            std::optional<std::unordered_map<std::string,int>> node_to_action,
                                            std::optional<std::shared_ptr<BatchRequester<StateT>>> opt_batch_requester);
template<typename StateT>
int select_action(const std::shared_ptr<Node<StateT>>& node);

template<typename StateT, typename ModelConfig>    
void reanalyze_with_mapzero_mcts( GlobalConfig<ModelConfig>& config,
    MappingHistory<StateT>& mapping_history, std::shared_ptr<zmq::socket_t> socket,
    const ServerConstants& server_k, const std::string& func_name, std::shared_ptr<StateT> state, 
    Environment<StateT>& environment,  std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester);

#include "src/tpp/compilers/mapzero/rl/mapzero_self_mapping.tpp"

#endif 
