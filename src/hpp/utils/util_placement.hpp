#ifndef UTIL_PLACEMENT_HPP
#define UTIL_PLACEMENT_HPP

#include <iostream>
#include <vector>
#include <tuple>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "src/hpp/utils/hashes.hpp"
#include <deque>
#include "src/hpp/utils/util_print.hpp"

using Annotation = std::pair<int, std::vector<std::pair<int,int>>>;

int get_dir_to_follow(const int& dir, const int& node, bool is_zizag, const std::unordered_map<int, std::unordered_set<int>>& fan_in, 
const std::unordered_map<int, std::unordered_set<int>>& fan_out);

std::unordered_set<int> get_neighbors(const int& dir, const int& node, 
                            const std::unordered_map<int, std::unordered_set<int>>& fan_in, 
                            const std::unordered_map<int, std::unordered_set<int>>& fan_out);

void note(
  std::unordered_map<std::pair<int,int>, std::vector<Annotation>>& edge_to_annotations,  
  const std::unordered_map<int, int>& dst_to_src, const int& src, const int& dst, const int& conv_node, Annotation& annotation,
  const std::unordered_map<int, std::unordered_set<int>>& fan_in, int cur_depth, int max_depth);

                          

std::pair<std::vector<std::tuple<int, int, int>>, std::unordered_map<std::pair<int,int>, std::vector<Annotation>>> do_traversal( 
    const std::vector<int>& nodes, 
    const std::vector<std::pair<int,int>>& edges,
    const std::unordered_map<int, std::unordered_set<int>>& fan_in, 
    const std::unordered_map<int, std::unordered_set<int>>& fan_out,
    int max_depth, bool use_stack, bool start_from_outputs, bool get_annotations, bool make_shuffle, bool is_zigzag);

std::unordered_map<int,std::tuple<int, int, int>> gen_node_to_traversal_edge(std::vector<std::tuple<int,int, int>>& tarversal_edges);

std::vector<int> gen_placement_order(const std::vector<std::tuple<int,int, int>>& traversal_edges);

#endif