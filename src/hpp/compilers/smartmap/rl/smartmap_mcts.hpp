#ifndef MCTS_SMARTMAP_HPP
#define MCTS_SMARTMAP_HPP

#include <iostream>
#include <torch/torch.h>
#include <limits>
#include <string>
#include "src/hpp/compilers/smartmap/rl/smartmap_environment.hpp"
#include "src/hpp/utils/min_max_stats.hpp"
#include "src/hpp/utils/mcts.hpp"
#include "src/hpp/utils/constants/server_constants.hpp"
#include <nlohmann/json.hpp>
#include "src/hpp/utils/min_max_stats.hpp"
#include "src/hpp/configs/mcts_config.hpp"
#include "src/hpp/rl/environment.hpp"
#include "src/hpp/rl/node.hpp"
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/utils/util_server.hpp"
#include "src/hpp/utils/util_infer.hpp"
#include "src/hpp/utils/batch_requester.hpp"
template<typename StateT, typename ModelConfig>
class SmartMapMCTS{
private:
    GlobalConfig<ModelConfig> global_config;
    std::shared_ptr<zmq::socket_t> socket;
    std::string func_name;
    ServerConstants server_k;
    double discount;
public:
    SmartMapMCTS(GlobalConfig<ModelConfig>& global_config,  std::shared_ptr<zmq::socket_t> socket,  const std::string& func_name);

    std::tuple<std::shared_ptr<Node<StateT>>, int, float, std::vector<std::shared_ptr<Node<StateT>>>>
    run(const std::shared_ptr<StateT> root_state, std::optional<std::shared_ptr<Node<StateT>>> root_to_start,
    Environment<StateT>& environment, bool get_first_valid_solution, bool add_exploration_noise,
    std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester, std::mt19937& gen, bool return_optimal_solution);
                                                
    std::pair<int, std::shared_ptr<Node<StateT>>> select_child(const Node<StateT>& node, MinMaxStats& min_max_stats);
    void backpropagate(std::vector<std::shared_ptr<Node<StateT>>>& traversed_path, double last_value, 
        const double& discount, MinMaxStats& min_max_stats);    
};

#include "src/tpp/compilers/smartmap/rl/smartmap_mcts.tpp"

#endif