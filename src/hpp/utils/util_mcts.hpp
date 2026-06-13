#ifndef UTIL_MCTS_HPP
#define UTIL_MCTS_HPP
#include <iostream>
#include <vector>
#include "src/hpp/rl/node.hpp"
#include "src/hpp/rl/environment.hpp"
template<typename StateT>
int select_action_with_backtracking(const std::shared_ptr<Node<StateT>> root, Environment<StateT>& environment,
    std::shared_ptr<StateT>& cur_state,int& action, int& action_indice, double& reward, bool& done);

#include "src/tpp/utils/util_mcts.tpp"

#endif