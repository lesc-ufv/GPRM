#ifndef UTIL_GRAPH_HPP
#define UTIL_GRAPH_HPP

#include <iostream>
#include <vector>
#include <tuple>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include "src/hpp/utils/hashes.hpp"
std::vector<std::vector<int>> create_edges_indices(const std::vector<std::pair<int,int>>& edges);
std::unordered_map<std::pair<int,int>,int> create_edge_to_id( const std::vector<std::vector<int>>& edges_indices);

template <typename BoostGraph>
std::unordered_map<int, std::unordered_set<int>> generate_cgra_in_vertices_dict(const BoostGraph& graph);

template <typename BoostGraph>
std::unordered_map<int, std::unordered_set<int>> generate_cgra_out_vertices_dict(const BoostGraph& graph);

#include "src/tpp/utils/util_graph.tpp"

#endif