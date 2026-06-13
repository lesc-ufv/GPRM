#ifndef CGRA_CONFIG_HPP
#define CGRA_CONFIG_HPP

#include <vector>
#include <unordered_set>
#include <iostream>
#include <sstream>
#include <utility>
#include "src/hpp/enums/enum_functionalities.hpp"
#include "src/hpp/enums/enum_interconnection_styles.hpp"

class CGRAConfig {
private:
    std::vector<std::pair<int, int>> dimensions_list;
    std::vector<std::unordered_set<EnumInterconnectionStyles>> connections_list;
    std::vector<std::unordered_set<EnumFunctionalities>> functionalities_list;
    int unroll_II = 0;

public:
    CGRAConfig() = default;
    int getUnrollII() const { return unroll_II; }
    void setUnrollII(int value)  {  unroll_II = value; }
  
    std::vector<std::pair<int, int>> getDimensionsList() const { return dimensions_list; }
    std::vector<std::unordered_set<EnumInterconnectionStyles>> getConnectionsList() const { return connections_list; }
    std::vector<std::unordered_set<EnumFunctionalities>> getFunctionalitiesList() const { return functionalities_list; }

    void setDimensionsList(const std::vector<std::pair<int, int>> value) { dimensions_list = value; }
    void setConnectionsList(const std::vector<std::unordered_set<EnumInterconnectionStyles>> value) { connections_list = value; }
    void setFunctionalitiesList(const std::vector<std::unordered_set<EnumFunctionalities>> value) { functionalities_list = value; }

    void print_ss(std::stringstream& ss) const {
        ss << "\nCGRA Configuration:\n";
        ss << "\tDimensions:\n";
        for (const auto& dim : dimensions_list) {
            ss << "\t\t  (" << dim.first << ", " << dim.second << ")\n";
        }

        ss << "Interconnection Styles:\n";
        for (const auto& conn_set : connections_list) {
            ss << "\t\t" << get_str_styles(conn_set) << "\n";
        }

        ss << "Functionalities:\n";
        for (const auto& func_set : functionalities_list) {
            ss << "\t\t" << get_str_functionalities(func_set) << "\n";
        }

        ss << "\t Unroll II: " << unroll_II << "\n";

    }

    void print() const {
        std::stringstream ss;
        print_ss(ss);
        std::cout << ss.str();
    }
};

#endif
