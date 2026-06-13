#include "src/hpp/utils/util_dfg.hpp"


std::unordered_map<int,int> get_node_to_planned_modulo_time_slice(std::unordered_map <int,int>& node_to_level, const unsigned int& II){
    std::unordered_map<int,int> planned_mod_time_slice;
    for (const auto pair: node_to_level) planned_mod_time_slice[pair.first] = pair.second % II;
    return planned_mod_time_slice;
}

std::unordered_map<int, std::unordered_set<int>> generate_in_vertices_dict(const std::vector<std::pair<int,int>>& edges) {
    std::unordered_map<int, std::unordered_set<int>> in_vertices_dict;
    for (auto& pair: edges){
        in_vertices_dict[pair.second].insert(pair.first);
    }
   
    return in_vertices_dict;

}

std::unordered_map<int, std::unordered_set<int>> generate_out_vertices_dict(const std::vector<std::pair<int,int>>& edges) {
    std::unordered_map<int, std::unordered_set<int>> out_vertices_dict;
    for (auto& pair: edges){
        out_vertices_dict[pair.first].insert(pair.second);
    }

    return out_vertices_dict;
}


std::vector<int> calc_scheduling(std::vector<int> topological_sort, const std::unordered_map<int, std::unordered_set<int>>& in_vertices,
const std::unordered_map<int, std::unordered_set<int>>& out_vertices, const std::vector<int>& root_nodes, bool asap) {
    int V = topological_sort.size();
    int last_node = V + 1;

    std::vector<int> dist(V + 2, std::numeric_limits<int>::min());  
    dist[asap ? V : last_node] = 0;  

    std::unordered_set<int> roots(root_nodes.begin(), root_nodes.end());

    std::unordered_set<int> leaves;
    std::vector<std::vector<int>> topo_order_list;

    for (const auto& pair : out_vertices) {
        if (pair.second.empty()) {
            leaves.insert(pair.first);
        }
    }

    if (!asap) std::reverse(topological_sort.begin(), topological_sort.end());

    topo_order_list.push_back({ asap ? V : last_node });
    topo_order_list.push_back(std::ref(topological_sort));
    topo_order_list.push_back({ asap ? last_node : V });

#ifdef DEBUG
    std::cout << "V: " << V << ", last_node: " << last_node << std::endl;
    std::cout << "Root Nodes: ";
    for (int r : roots) std::cout << r << " ";
    std::cout << std::endl;

    std::cout << "Leaves: ";
    for (int l : leaves) std::cout << l << " ";
    std::cout << std::endl;

    std::cout << "Topological Order: ";
    for (int n : topological_sort) std::cout << n << " ";
    std::cout << std::endl;
#endif

    std::unordered_map<int, std::unordered_set<int>> aux;

    for (const auto& topo_list : topo_order_list) {
        for (const int& u : topo_list) {
            aux.clear();  
            if (u == V) {
                aux[V] = asap ? roots : std::unordered_set<int>();
            } else if (asap ? leaves.find(u) != leaves.end() : roots.find(u) != roots.end()) {
                aux[u] = {asap ? last_node : V};
            } else if (u == last_node) {
                aux[last_node] = asap ? std::unordered_set<int>() : leaves;
            } else {
                aux[u] = asap ? out_vertices.at(u) : in_vertices.at(u);
            }

#ifdef DEBUG
            std::cout << "Processing node: " << u << std::endl;
            std::cout << "Edges: ";
            for (const auto& pair : aux) {
                std::cout << pair.first << " -> { ";
                for (auto v : pair.second) std::cout << v << " ";
                std::cout << "} ";
            }
            std::cout << std::endl;
#endif

            for (const auto& pair : aux) {
                for (auto v : pair.second) {
                    if (dist[u] != std::numeric_limits<int>::min()) {
                        dist[v] = std::max(dist[v], dist[u] + 1);

#ifdef DEBUG
                        std::cout << "Updating dist[" << v << "] = max(" << dist[v] << ", " << dist[u] << " + 1)" << std::endl;
#endif
                    }
                }
            }
        }
    }

    auto max_dist = dist.at(V); 
    dist.erase(dist.begin() + last_node); 
    dist.erase(dist.begin() + V);

#ifdef DEBUG
    std::cout << "Max Distance: " << max_dist << std::endl;
#endif

    for (int& d : dist) {
        if (d != std::numeric_limits<int>::min()) {
            d = asap ? d - 1 : max_dist - d - 1; 
        }
    }

#ifdef DEBUG
    std::cout << "Final Distances: ";
    for (int d : dist) std::cout << d << " ";
    std::cout << std::endl;
#endif

    return dist;
}
