#ifndef MAPPING_AUGMENTATION_HPP
#define MAPPING_AUGMENTATION_HPP
#include <unordered_map>
#include <tuple>
#include <utility>
#include <tuple>
#include <string>
#include <iostream>
#include "src/hpp/utils/hashes.hpp"

std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>> rotate(
    std::unordered_map<int, int> PE_to_node,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols, int degrees);

int rotate(
    const int& PE,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols, int degrees);

std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>> shift(
    std::unordered_map<int, int> PE_to_node,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols, std::pair<int, int> shifts, bool allow_border_shift);


int shift(
    const int& PE,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols,  std::pair<int, int> shifts, bool allow_border_shift);

std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>> flip(
    std::unordered_map<int, int> PE_to_node,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols, std::pair<int, int> dims);

int flip(
    const int& PE,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_pe,
    int rows, int cols, std::pair<int, int> dims);
#endif  