#ifndef UTIL_INTERCONNECTIONS_HPP
#define UTIL_INTERCONNECTIONS_HPP

#include <vector>
#include <tuple>
#include <iostream>
#include "src/hpp/utils/hashes.hpp"
#include "src/hpp/utils/cgra/util_cgra.hpp"


std::vector<int> generate_interconnections(
    std::vector<std::pair<int, int>>& adj_positions,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection
);

std::vector<int> generate_mesh_interconnection_by_pe_pos(
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection
);

std::vector<int> generate_diagonal_interconnection_by_pe_pos(
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection
);
std::vector<int> generate_one_hop_interconnection_by_pe_pos(
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection
);


std::vector<int> generate_toroidal_interconnection_by_pe_pos(
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection
);

std::vector<int> generate_connections_by_enum(
    EnumInterconnectionStyles interconnection_style,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection);

#endif
