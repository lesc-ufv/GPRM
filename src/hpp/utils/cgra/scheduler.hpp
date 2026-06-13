#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP

#include <queue>
#include "src/hpp/utils/hashes.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <algorithm>
#include <tuple>
#include <cassert>
#include <utility> 

class Scheduler{
public:
static bool scheduling_is_valid(const std::vector<int>& vertices,
                                 const std::unordered_map<int, int>& node_to_sched_time_slice,
                                 const std::unordered_map<int, std::unordered_set<int>>& node_to_in_vertices,
                                 const std::unordered_map<int, int>& node_to_pe,
                                 const std::unordered_map<std::pair<int, int>, std::vector<int>>& pes_to_routing);

static std::tuple<std::unordered_map<int,int>,bool> schedule(
                                            const std::vector<int>& root_dfg_nodes,
                                            const std::unordered_map<int, int> &node_to_pe, 
                                            const std::unordered_map<int, std::unordered_set<int>> &in_vertices,
                                            const std::unordered_map<int, std::unordered_set<int>> &out_vertices,
                                            const std::unordered_map<std::pair<int, int>,std::vector<int>> &pes_to_routing);                                
// private: 
// static bool down_top_aux_schedule(const int& node, int time_to_sched_node,
//  std::unordered_map<int,int>& node_to_sched_time, const std::unordered_map<int, std::unordered_set<int>> &in_vertices, 
//   const std::unordered_map<std::pair<int, int>,std::vector<int>> &pes_to_routing, std::queue<int>& fifo,
//   const std::unordered_map<int, int> &node_to_pe);
};

#endif