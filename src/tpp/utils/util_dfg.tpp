#include <vector>
#include <map>
#include <iostream>
template <typename T>
std::tuple<std::vector<int>, std::map<T, int>, std::map<int, T>> reset_vertices_labels(const std::vector<T>& vertices){
    std::map<T,int> real_to_new_vertices;
    std::map<int, T> new_to_real_vertices;
    std::vector<int> new_vertices;
    int idx = 0;
    for (const T& vertice: vertices){
        real_to_new_vertices[vertice] = idx;
        new_to_real_vertices[idx] = vertice;
        new_vertices.emplace_back(idx);
        idx += 1;
    }

    return std::make_tuple(new_vertices,real_to_new_vertices,new_to_real_vertices);
}

template <typename T>
std::vector<std::tuple<int, int>> reset_edges_by_dict(
    const std::vector<std::tuple<T,T>>& real_edges,
    const std::map<T, int>& real_to_new_vertices) {

    std::vector<std::tuple<int, int>> new_edges;

    T u, v;
    for (const auto& edge : real_edges) {
        std::tie(u, v) = edge;
        new_edges.emplace_back(std::make_tuple(real_to_new_vertices.at(u), real_to_new_vertices.at(v)));
    }

    return new_edges;
}