#ifndef MCTS_MAPZERO_HPP
#define MCTS_MAPZERO_HPP

#include <iostream>
#include <torch/torch.h>
#include <limits>
#include <string>
#include "src/hpp/compilers/mapzero/rl/mapzero_environment.hpp"
#include "src/hpp/utils/min_max_stats.hpp"
#include "src/hpp/utils/constants/server_constants.hpp"
#include <nlohmann/json.hpp>
#include "src/hpp/utils/util_server.hpp"
#include "src/hpp/utils/mcts.hpp"
#include "src/hpp/configs/mcts_config.hpp"
#include "src/hpp/rl/environment.hpp"
#include <tuple>
#include "src/hpp/enums/enum_task_type.hpp"
#include "src/hpp/rl/node.hpp"
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/utils/batch_requester.hpp"
using json = nlohmann::json;
template <typename StateT, typename ModelConfig>
class MapZeroMCTS{

private:
       GlobalConfig<ModelConfig> global_config;
       const std::string func_name;
    std::shared_ptr<zmq::socket_t> socket;
       ServerConstants server_k;
public:
    MapZeroMCTS( GlobalConfig<ModelConfig>& global_config,
       const std::string& func_name,
    std::shared_ptr<zmq::socket_t> socket);

    std::tuple<std::shared_ptr<Node<StateT>>, int , double, std::vector<std::shared_ptr<Node<StateT>>>> 
    run(std::shared_ptr<StateT> root_state, std::optional<std::shared_ptr<Node<StateT>>> root_to_start,
    Environment<StateT>& environment, bool get_first_valid_solution, 
    std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester);
    std::pair<int, std::shared_ptr<Node<StateT>>> select_child(const Node<StateT>& node);
    void backpropagate(std::vector<std::shared_ptr<Node<StateT>>>& traversed_path, double last_value, const double& discount);
};

#include "src/tpp/compilers/mapzero/rl/mapzero_mcts.tpp"

#endif