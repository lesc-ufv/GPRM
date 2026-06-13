#ifndef ROUTER_HPP
#define ROUTER_HPP

#include <iostream>
#include <vector>
#include <map>
#include <unordered_set>
#include <queue>
#include <stdexcept>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <map>
#include "unordered_map"
#include "src/hpp/utils/util_print.hpp"

class Router {
public:
    static std::tuple<
       std::vector<int>, 
        int
    > route(
    int source_pe, std::pair<int, int> arch_dims, int target_pe,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<int, std::unordered_set<int>>& occupied_pes_per_module, 
    const std::unordered_map<std::pair<int, int>, std::vector<int>>& pes_to_routing,
    const std::unordered_map<int, std::unordered_set<int>>& free_interconnections, 
    const std::unordered_map<int, std::unordered_set<int>>& out_vertices);
    
    static std::tuple<std::vector<int>, int> route(
    int source_pe, std::pair<int, int> arch_dims, int target_pe,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_set<int>& occupied_pes, 
    const std::unordered_map<std::pair<int, int>, std::vector<int>>& pes_to_routing,
    const std::unordered_map<int, std::unordered_set<int>>& free_interconnections, 
    const std::unordered_map<int, std::unordered_set<int>>& out_vertices);
    
    static bool routing_is_valid(const std::unordered_map<std::pair<int, int>, std::vector<int>>& pes_to_routing);
    static bool dfs_exact_distance(int current, int target, int depth, int max_depth,
        const std::unordered_set<int>& occupied_pes,
        const std::unordered_map<int, std::unordered_set<int>>& out_vertices,
        const std::unordered_map<int, std::unordered_set<int>>& free_interconnections,
        std::unordered_set<int>& visited,
        std::vector<int>& path) {
        path.push_back(current);
        visited.insert(current);

        #ifdef DEBUG
        std::cout << "[DEBUG] DFS depth " << depth << ", current PE: " << current << "\n";
        #endif

        if (depth == max_depth) {
        if (current == target) return true;
        path.pop_back();
        visited.erase(current);
        return false;
        }

        for (int neighbor : out_vertices.at(current)) {
        if ((occupied_pes.count(neighbor) && neighbor != target) || !free_interconnections.at(current).count(neighbor))
        continue;
        if (visited.count(neighbor)) continue;

        if (dfs_exact_distance(neighbor, target, depth + 1, max_depth,
                    occupied_pes, out_vertices, free_interconnections,
                    visited, path))
        return true;
        }

        path.pop_back();
        visited.erase(current);
        return false;
        }

        static std::vector<int> find_path_exact_distance(int source, int target, int distance,
                                const std::unordered_set<int>& occupied_pes,
                                const std::unordered_map<int, std::unordered_set<int>>& out_vertices,
                                const std::unordered_map<int, std::unordered_set<int>>& free_interconnections) {
        std::vector<int> path;
        std::unordered_set<int> visited;
        if (Router::dfs_exact_distance(source, target, 0, distance, occupied_pes, out_vertices, free_interconnections, visited, path))
        return path;
        return {};
        }

private:
    
    static std::vector<int> a_star_routing(
    int source_pe, std::pair<int, int> arch_dims, int target_pe, 
    const std::unordered_map<int, std::unordered_set<int>>& occupied_pes_per_module, 
    const std::unordered_map<int, std::unordered_set<int>>& out_vertices, 
    const std::unordered_map<int, std::unordered_set<int>>& free_interconnections);

    static std::vector<int> a_star_routing(
        int source_pe, std::pair<int, int> arch_dims, int target_pe, 
        const std::unordered_set<int>& occupied_pes, 
        const std::unordered_map<int, std::unordered_set<int>>& out_vertices, 
        const std::unordered_map<int, std::unordered_set<int>>& free_interconnections);
    
    
    static std::vector<int> reconstruct_routing_path(const std::unordered_map<int, int>& parent, int target_pe, int source_pe);

    

};

#endif