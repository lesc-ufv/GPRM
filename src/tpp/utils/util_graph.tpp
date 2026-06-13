#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

template <typename BoostGraph>
std::unordered_map<int, std::unordered_set<int>> generate_cgra_in_vertices_dict(const BoostGraph& graph){
    std::unordered_map<int, std::unordered_set<int>> in_vertices_dict;

    for (auto vp = vertices(graph).first; vp != vertices(graph).second; ++vp) {
        int vertex = *vp; 
        std::unordered_set<int> in_vertices;
        for (auto it = vertices(graph).first; it != vertices(graph).second; ++it) {
            int other_vertex = *it;
            if (other_vertex != vertex && boost::edge(other_vertex, vertex, graph).second) {
                in_vertices.insert(other_vertex);
            }
        }

        in_vertices_dict[vertex] = in_vertices;
    }

    return in_vertices_dict;
}

template <typename BoostGraph>
std::unordered_map<int, std::unordered_set<int>> generate_cgra_out_vertices_dict(const BoostGraph& graph)
{
    std::unordered_map<int, std::unordered_set<int>> out_vertices_dict;

    for (auto vp = vertices(graph).first; vp != vertices(graph).second; ++vp) {
        int vertex = *vp; 
        
        std::unordered_set<int> out_vertices;
        auto adj_vp = boost::adjacent_vertices(vertex, graph);
        
        for (auto it = adj_vp.first; it != adj_vp.second; ++it) {
            out_vertices.insert(*it);
        }

        out_vertices_dict[vertex] = out_vertices;
    }

    return out_vertices_dict;
}
