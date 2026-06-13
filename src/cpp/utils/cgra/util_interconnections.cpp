#include "src/hpp/utils/cgra/util_interconnections.hpp"


std::vector<int> generate_interconnections(
    std::vector<std::pair<int, int>>& adj_positions,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection)
    {
    std::pair<int,int> self_position = {0,0};
    if (add_self_connection) adj_positions.emplace_back(self_position);

    std::vector<int> interconnections;
    std::tuple<int,int,int> pe_coord = pe_to_coord.at(PE);

    for (const auto& adj_position : adj_positions) {

        int new_modulo_coord = (std::get<0>(pe_coord) + 1) % II;
        int new_row_coord = std::get<1>(pe_coord) + adj_position.first;
        int new_col_coord = std::get<2>(pe_coord) + adj_position.second;
        if (!pe_is_out_of_border({new_row_coord, new_col_coord}, cgra_dims)) {
            interconnections.emplace_back(coord_to_pe.at({new_modulo_coord,new_row_coord, new_col_coord}));
        }
    }

    return interconnections;
};

std::vector<int> generate_mesh_interconnection_by_pe_pos(
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection
) {
    std::vector<std::pair<int, int>> adj_positions = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
    return generate_interconnections(adj_positions, pe_to_coord, coord_to_pe,cgra_dims,PE,II,add_self_connection);
};

std::vector<int> generate_diagonal_interconnection_by_pe_pos(
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection
) {
    std::vector<std::pair<int, int>> adj_positions = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    return generate_interconnections(adj_positions, pe_to_coord, coord_to_pe,cgra_dims,PE,II,add_self_connection);
};

std::vector<int> generate_one_hop_interconnection_by_pe_pos(
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection
){
    std::vector<std::pair<int, int>> adj_positions = {{-2, 0}, {2, 0}, {0, 2}, {0, -2}};

    return generate_interconnections(adj_positions, pe_to_coord, coord_to_pe,cgra_dims,PE,II,add_self_connection);

};

std::vector<int> generate_toroidal_interconnection_by_pe_pos(
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection
) {
    auto coord = pe_to_coord.at(PE);
    auto two_d_coord = three_d_to_2_d_coord(coord); 
    if (pe_is_border(two_d_coord, cgra_dims)) {
        std::vector<std::pair<int, int>> adj_positions = {
            {0, std::get<1>(cgra_dims) - 1},
            {0, -(std::get<1>(cgra_dims) - 1)},
            {std::get<0>(cgra_dims) - 1, 0},
            {-(std::get<0>(cgra_dims) - 1), 0}
        };
        return generate_interconnections(adj_positions, pe_to_coord, coord_to_pe,cgra_dims,PE,II,add_self_connection);
    }
    return {};
};

std::vector<int> generate_connections_by_enum(
    EnumInterconnectionStyles interconnection_style,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    const std::pair<int, int>& cgra_dims,
    const int& PE,
    const unsigned int& II,
    bool& add_self_connection){
    switch(interconnection_style){
        case EnumInterconnectionStyles::MESH:
            return generate_mesh_interconnection_by_pe_pos(pe_to_coord,coord_to_pe,cgra_dims,PE,II,add_self_connection);
            break;
        case EnumInterconnectionStyles::ONE_HOP:
            return generate_one_hop_interconnection_by_pe_pos(pe_to_coord,coord_to_pe,cgra_dims,PE,II,add_self_connection);
            break;
        case EnumInterconnectionStyles::DIAGONAL:
            return generate_diagonal_interconnection_by_pe_pos(pe_to_coord,coord_to_pe,cgra_dims,PE,II,add_self_connection);
            break;
        case EnumInterconnectionStyles::TOROIDAL:
            return generate_toroidal_interconnection_by_pe_pos(pe_to_coord,coord_to_pe,cgra_dims,PE,II,add_self_connection);
            break;
        default:
            std::runtime_error("Unknown interconnection style.");
            return {};
    }   

}
