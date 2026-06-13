#ifndef SMARTMAP_SELF_MAPPING_HPP
#define SMARTMAP_SELF_MAPPING_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <iostream>
#include <zmq.hpp>
#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_mcts.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_environment.hpp"
#include "src/hpp/rl/mapping_history.hpp"
#include "src/hpp/enums/enum_task_type.hpp"
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/utils/constants/server_constants.hpp"
#include "src/hpp/rl/node.hpp"
#include "src/hpp/rl/environment.hpp"
#include "src/hpp/utils/util_mcts.hpp"
#include "src/hpp/utils/util_infer.hpp"

template <typename StateT, typename ModelConfig>
MappingHistory<StateT> map_with_smartmap_mcts( GlobalConfig<ModelConfig>& global_config,
                                            std::shared_ptr<zmq::socket_t> socket,
                                            const ServerConstants& server_k, const std::string& func_name,
                                            const std::shared_ptr<StateT> initial_state, Environment<StateT>& environment,
                                            bool train, double train_ratio, int seed, 
                                            std::optional<std::unordered_map<std::string,int>> node_to_action,
                                            std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester
                                            );
                                        

template<typename StateT>    
int select_action(std::shared_ptr<Node<StateT>> node, const double& temperature, std::mt19937& gen, torch::Generator& torch_gen);

template<typename StateT, typename ModelConfig>    
void reanalyze_with_smartmap_mcts( GlobalConfig<ModelConfig>& config,
    MappingHistory<StateT>& mapping_history, std::shared_ptr<zmq::socket_t> socket,
    const ServerConstants& server_k, const std::string& func_name, std::shared_ptr<StateT> state,
    Environment<StateT>& environment,
    std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester,std::mt19937& gen);

#include "src/tpp/compilers/smartmap/rl/smartmap_self_mapping.tpp"

#endif 
