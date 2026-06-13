#include "src/hpp/utils/cgra/util_cgra.hpp"

std::pair<int,int> three_d_to_2_d_coord(const std::tuple<int,int,int>& coord){
    return std::pair<int,int>(std::get<1>(coord),std::get<2>(coord)); 
}

bool pe_is_out_of_border(const std::pair<int, int>& row_col_coord, const std::pair<int, int>& dim_cgra) {
    int pe_i = row_col_coord.first;
    int pe_j = row_col_coord.second;
    int rows = dim_cgra.first;
    int cols = dim_cgra.second;
    return pe_i < 0 || pe_i >= rows || pe_j < 0 || pe_j >= cols;
}

bool pe_is_border(const std::pair<int, int>& row_col_coord, const std::pair<int, int>& dim_cgra) {
    return row_col_coord.first == 0 || row_col_coord.first == (dim_cgra.first - 1) || 
           row_col_coord.second == 0 || row_col_coord.second == (dim_cgra.second - 1);
}

bool pe_is_corner(const std::pair<int, int>& row_col_coord, const std::pair<int, int>& dim_cgra) {
    int i = row_col_coord.first;
    int j = row_col_coord.second;
    int rows = dim_cgra.first;
    int cols = dim_cgra.second;

    return (i == 0 && j == 0) || (i == 0 && j == cols - 1) ||
           (i == rows - 1 && j == 0) || (i == rows - 1 && j == cols - 1);
}

std::vector<int> generate_PEs(const std::pair<int, int>& arch_dimensions, const unsigned int& II)
{
    std::vector<int> PEs;
    int num_PEs = arch_dimensions.first*arch_dimensions.second*II; 
    for (int i = 0; i < num_PEs; i += 1){
        PEs.emplace_back(i);
    }
    return PEs;
}

std::tuple<std::unordered_map<int, std::tuple<int, int, int>>, std::unordered_map<std::tuple<int, int, int>, int>> generate_PEs_coord_maps(const std::pair<int, int> &arch_dimensions, 
                                                                                                             const unsigned int& II)
{
    std::unordered_map<int, std::tuple<int, int, int>> pe_to_coord;

    std::unordered_map<std::tuple<int, int, int>, int> coord_to_pe;

    int number_of_spatial_PEs = arch_dimensions.first*arch_dimensions.second; 
    int num_PEs = number_of_spatial_PEs*II;

    for (int PE = 0; PE < num_PEs ; PE++){
        std::tuple<int,int,int> coordinate = calc_pe_to_coordinate(PE,number_of_spatial_PEs,arch_dimensions.second);
        pe_to_coord[PE] = coordinate;
        coord_to_pe[coordinate] = PE;
    }
    return std::make_tuple(pe_to_coord,coord_to_pe);
}

    std::tuple<int, int, int> calc_pe_to_coordinate(const int &PE, const int& number_of_spatial_PEs, const int& number_of_columns)
    {
        int modulo, row, col;

        modulo = PE/number_of_spatial_PEs;
        row = (PE % number_of_spatial_PEs) / number_of_columns;
        col = (PE % number_of_spatial_PEs) % number_of_columns;
        return std::make_tuple(modulo,row,col);
    }



std::unordered_map<int,std::unordered_set<EnumFunctionalities>> init_homogeneous_pe_to_functionalities(const int& total_PEs, const std::unordered_set<EnumFunctionalities>& functionalities){
    std::unordered_map<int,std::unordered_set<EnumFunctionalities>> pe_to_functionalities;
    for (int i = 0; i < total_PEs; i++) pe_to_functionalities[i] = functionalities;
    return pe_to_functionalities;
}   

int manhattan_distance(const std::pair<int, int>& pe_pos1, const std::pair<int, int>& pe_pos2){
    return std::abs(pe_pos1.first - pe_pos2.first) + std::abs(pe_pos1.second - pe_pos2.second);
}

std::string get_invalid_mapping_reason(EnumInvalidMappingReason reason)
{   
    switch(reason) {
        case EnumInvalidMappingReason::INVALID_PLACEMENT:
            return "Invalid placement. Not all nodes were mapped.";
        
        case EnumInvalidMappingReason::INVALID_ROUTING:
            return "Invalid routing. At least one routing was not performed.";
        
        case EnumInvalidMappingReason::INVALID_SCHEDULING:
            return "Invalid scheduling. The mapping is unbalanced.";
        default:
            return "None -> Mapping is valid.";
    }
}

std::string bit_pos_to_connections_style(int pos){
    switch(pos){
        case 0:
            return "Mesh";
        case 1:
            return "One-Hop";
        case 2:
            return "Diagonal";
        case 3:
            return "Toroidal";
        default:
            std::runtime_error("Invalid connection.");
            return "";
    }
}
int calc_II(const int number_of_nodes, int number_of_spatial_PEs)
{
    return  static_cast<int>(std::ceil(static_cast<float>(number_of_nodes) / static_cast<float>(number_of_spatial_PEs)));
}

std::string get_cgra_connection_style_str(const std::unordered_set<EnumInterconnectionStyles>& connection_styles) {
    std::string connection_style_str = "";
    
    for (const auto& style : connection_styles) {
        connection_style_str += enum_connection_style_to_string(style) + " + ";
    }

    if (!connection_style_str.empty()) {
        connection_style_str = connection_style_str.substr(0, connection_style_str.size() - 3);
    }

    return connection_style_str;
}

std::string get_cgra_homogeneous_functionalities_str(const std::unordered_set<EnumFunctionalities>& functionalities) {
    std::string functionality_str = "";
    
    for (const auto& functionality : functionalities) {
        functionality_str += enum_functionality_to_string(functionality) + " + ";
    }

    if (!functionality_str.empty()) {
        functionality_str = functionality_str.substr(0, functionality_str.size() - 3);
    }

    return functionality_str;
}
