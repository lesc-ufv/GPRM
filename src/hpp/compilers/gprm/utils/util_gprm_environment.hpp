#ifndef UTIL_GPRM_ENVIRONMENT
#define UTIL GPRM_ENVIRONMENT

#include "src/hpp/rl/base_state.hpp"
#include <unordered_set>
#include <vector>

double num_mapped_edges_by_neigh(const BaseState& state, const int& node, const int& action, bool in_vertices);
double num_mapped_edges(const BaseState& state, const int& node, const int& action, bool in_vertices);

std::tuple<double, double> calc_routing_node_reward_by_neigh(const BaseState& state, const int& node, const int& action, bool in_vertices);
double routing_node_reward(const BaseState& state, const int& node, const int& action);

double num_placed_neigh_nodes_by_dir(const BaseState& state, const int& node,bool in_vertices);
double num_placed_neigh_nodes(const BaseState& state, const int& node, const int& action);

#endif