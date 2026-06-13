#ifndef UTIL_DFG_H
#define UTIL_DFG_H

#include <vector>
#include <tuple>
#include <map>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <stdexcept>
#include <unordered_set>

template <typename T>
std::tuple<std::vector<int>, std::map<T, int>, std::map<int, T>> reset_vertices_labels(const std::vector<T>& vertices);

template <typename T>
std::vector<std::tuple<int,int>> reset_edges_by_dict(std::vector<std::tuple<T, T>>& real_edges,
                                                     std::map<T, int>& real_to_new_vertices);

std::unordered_map<int,int> get_node_to_planned_modulo_time_slice(std::unordered_map <int,int>& node_to_level, const unsigned int& II);

std::unordered_map<int, std::unordered_set<int>> generate_in_vertices_dict(const std::vector<std::pair<int,int>>& edges);

std::unordered_map<int, std::unordered_set<int>> generate_out_vertices_dict(const std::vector<std::pair<int,int>>& edges);

std::vector<int> calc_scheduling(std::vector<int> topological_sort, const std::unordered_map<int, std::unordered_set<int>>& in_vertices,
const std::unordered_map<int, std::unordered_set<int>>& out_vertices, const std::vector<int>& root_nodes, bool asap);

#include "src/tpp/utils/util_dfg.tpp"

#endif