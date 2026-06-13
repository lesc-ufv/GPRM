#ifndef UTIL_CGRA_HPP
#define UTIL_CGRA_HPP
#include <iostream>
#include <map>
#include <vector>
#include <cmath>  
#include <utility>
#include <tuple>
#include "src/hpp/enums/enum_functionalities.hpp"
#include "src/hpp/data_structures/cgras/cgra.hpp"
#include "src/hpp/enums/enum_interconnection_styles.hpp"
#include "src/hpp/enums/enum_invalid_mapping_reason.hpp"
#include <string>
#include "src/hpp/utils/hashes.hpp"
#include "src/hpp/utils/util_print.hpp"
 #include "src/hpp/configs/cgra_config.hpp"
std::pair<int, int> three_d_to_2_d_coord(const std::tuple<int,int,int>& coord);

bool pe_is_out_of_border(const std::pair<int, int>& row_col_coord, const std::pair<int, int>& dim_cgra);

bool pe_is_border(const std::pair<int, int>& row_col_coord, const std::pair<int, int>& dim_cgra);

bool pe_is_corner(const std::pair<int, int>& row_col_coord, const std::pair<int, int>& dim_cgra);

std::vector<int> generate_PEs(const std::pair<int,int>& arch_dimensions, const unsigned int& II);

std::tuple<std::unordered_map<int, std::tuple<int, int, int>>, std::unordered_map<std::tuple<int, int, int>, int>> generate_PEs_coord_maps(
                                                                                                            const std::pair<int, int> &arch_dimensions, 
                                                                                                            const unsigned int& II);
std::tuple<int,int,int> calc_pe_to_coordinate(const int &PE, const int& number_of_spatial_PEs, const int& number_of_columns);



std::unordered_map<int,std::unordered_set<EnumFunctionalities>> init_homogeneous_pe_to_functionalities(const int& total_PEs, 
                                                                const std::unordered_set<EnumFunctionalities>& functionalities);


int manhattan_distance(const std::pair<int, int>& pe_pos1, const std::pair<int, int>& pe_pos2);

std::string get_invalid_mapping_reason(EnumInvalidMappingReason reason);

std::string bit_pos_to_connections_style(int pos);

int calc_II(const int number_of_nodes, int number_of_spatial_PEs);

std::string get_cgra_connection_style_str(const std::unordered_set<EnumInterconnectionStyles>& connection_styles);

std::string get_cgra_homogeneous_functionalities_str(const std::unordered_set<EnumFunctionalities>& functionalities);

template <typename CGRAT, typename... Args>
std::shared_ptr<CGRAT> initialize_homogeneous_cgra(const std::unordered_set<EnumInterconnectionStyles>& interconnection_styles, const int& II,
                                                                 const std::pair<int,int>& arch_dimensions, 
                                                                 const std::unordered_set<EnumFunctionalities>& functionalities,const Args&... args );

template <typename CGRAT, typename... Args>
std::vector<std::shared_ptr<CGRAT>> initialize_homogeneous_cgras(const std::vector<std::unordered_set<EnumInterconnectionStyles>>& interconnection_styles_list, 
                                                    const int& II, const std::pair<int,int>& arch_dimensions,
                                            const std::vector<std::unordered_set<EnumFunctionalities>>& functionalities_list, const Args&... args );



#include "src/tpp/utils/util_cgra.tpp"

#endif 