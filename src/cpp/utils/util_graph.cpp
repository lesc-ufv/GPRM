#include "src/hpp/utils/util_graph.hpp"


std::vector<std::vector<int>> create_edges_indices(const std::vector<std::pair<int,int>>& edges){
    std::vector<std::vector<int>> edges_indices(2);

    int src, dst;
    for (const auto& edge : edges) {
        std::tie(src, dst) = edge;
    
        edges_indices[0].emplace_back(src);
        edges_indices[1].emplace_back(dst);
    }

    return edges_indices;
}

std::unordered_map<std::pair<int,int>, int> create_edge_to_id( const std::vector<std::vector<int>>& edges_indices){
    std::unordered_map<std::pair<int,int>,int> edge_to_id;

    for(int i = 0; i < static_cast<int>(edges_indices[0].size()); i++ ){
        auto src = edges_indices[0][i];
        auto dst = edges_indices[1][i];
        edge_to_id[std::make_pair(src,dst)] = i; 
    }
    return  edge_to_id;
}