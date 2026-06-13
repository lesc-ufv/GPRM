#include "src/hpp/utils/mapping_augmentation.hpp"
std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>> rotate(
    std::unordered_map<int, int> PE_to_node,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols, int degrees) {

    std::unordered_map<int, int> rotated_PE_to_node;
    std::unordered_map<int, int> rotated_node_to_pe;

    for (const auto& [pe, node] : PE_to_node) {
        int new_pe = rotate(pe, pe_to_coord, coord_to_pe,rows,cols,degrees);
        if (node != -2 && node != -1) {
            rotated_node_to_pe[node] = new_pe;
        }
        rotated_PE_to_node[new_pe] = node;
        
    }

    return {rotated_node_to_pe, rotated_PE_to_node};
}

int rotate(
    const int& PE,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols, int degrees){
        auto coord = pe_to_coord.at(PE);
        int cycle = std::get<0>(coord);
        int x = std::get<1>(coord);
        int y = std::get<2>(coord);

        if (degrees == 90) {
            std::tie(x, y) = std::make_pair(y, rows - 1 - x);
        } else if (degrees == 180) {
            x = rows - 1 - x;
            y = cols - 1 - y;
        } else if (degrees == 270) {
            std::tie(x, y) = std::make_pair(cols - 1 - y, x);
        }
    auto new_coord = std::make_tuple(cycle, x, y);

    return coord_to_pe.at(new_coord);
}

std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>> shift(
    std::unordered_map<int, int> PE_to_node,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols, std::pair<int, int> shifts, bool allow_border_shift) {

    std::unordered_map<int, int> shifted_PE_to_node;
    std::unordered_map<int, int> shifted_node_to_pe;

    for (const auto& [pe, node] : PE_to_node) {
        auto coord = pe_to_coord.at(pe);
        int cycle = std::get<0>(coord);
        int y = std::get<1>(coord);
        int x = std::get<2>(coord);

        x = (x + shifts.first);
        y = (y + shifts.second);
        if (!allow_border_shift){
            if(x >= rows || x < 0 || y >=cols || y < 0){
                return std::make_pair<std::unordered_map<int, int>, std::unordered_map<int, int>>({},{});
            }
        }

        x = x % cols;
        y = y % rows;

        auto new_coord = std::make_tuple(cycle, y, x);
        if (coord_to_pe.count(new_coord) > 0) {
            int new_pe = coord_to_pe.at(new_coord);
            if (node != -2 && node != -1) {
                shifted_node_to_pe[node] = new_pe;
            }
            shifted_PE_to_node[new_pe] = node;
        }
    }

    return {shifted_node_to_pe, shifted_PE_to_node};
}

int shift(
    const int& PE,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols,  std::pair<int, int> shifts, bool allow_border_shift){
        auto coord = pe_to_coord.at(PE);
        int cycle = std::get<0>(coord);
        int y = std::get<1>(coord);
        int x = std::get<2>(coord);

        x = (x + shifts.first);
        y = (y + shifts.second);

        // x = x % cols;
        // y = y % rows;

        auto new_coord = std::make_tuple(cycle, y, x);
        if (coord_to_pe.count(new_coord) > 0) {
            return coord_to_pe.at(new_coord);
        }else{
          return -1;
        }
}


std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>> flip(
    std::unordered_map<int, int> PE_to_node,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols, std::pair<int, int> dims) {

    std::unordered_map<int, int> flipped_PE_to_node;
    std::unordered_map<int, int> flipped_node_to_pe;

    for (const auto& [pe, node] : PE_to_node) {
        auto new_pe =  flip(pe,pe_to_coord, coord_to_pe, rows,cols,dims);

        if (node != -2 && node != -1) {
            flipped_node_to_pe[node] = new_pe;
        }

        flipped_PE_to_node[new_pe] = node;
        
    }
    return {flipped_node_to_pe, flipped_PE_to_node};
}


int flip(
    const int& PE,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols, std::pair<int, int> dims){

    auto coord = pe_to_coord.at(PE);
    int cycle = std::get<0>(coord);
    int y = std::get<1>(coord);
    int x = std::get<2>(coord);

    if (dims.first == 1) {
        y = rows - 1 - y;
    }
    if (dims.second == 1) {
        x = cols - 1 - x;
    }

    auto new_coord = std::make_tuple(cycle, y, x);

    return coord_to_pe.at(new_coord);
}