#ifndef ENUM_FUNCTIONALITIES_H
#define ENUM_FUNCTIONALITIES_H
#include <iostream>
#include <string>
#include <cereal/archives/binary.hpp>

enum class EnumFunctionalities{
    LOGICAL = 0,
    ARITHMETIC = 1,
    MEMORY_ACCESS = 2,
    ADD_AND_SUB = 3,
    MULT_AND_DIV = 4,
    LOGIC = 5,
    SHIFT = 6,
    COMPARISON = 7
};

inline std::string enum_functionality_to_string(EnumFunctionalities functionality) {
    switch (functionality) {
        case EnumFunctionalities::LOGICAL:
            return "LOGICAL";
        case EnumFunctionalities::ARITHMETIC:
            return "ARITHMETIC";
        case EnumFunctionalities::MEMORY_ACCESS:
            return "MEMORY_ACCESS";
        default:
            return "Unknown";
    }
}

inline std::string get_str_functionalities(std::unordered_set<EnumFunctionalities> functionalities){
    std::string str = "";
    for (const auto& functionality : functionalities) {
        str += enum_functionality_to_string(functionality) + " + ";
    }
    str.pop_back();
    return str;
}

inline std::vector<EnumFunctionalities> get_all_functionality_values() {
    return {
        EnumFunctionalities::LOGICAL,
        EnumFunctionalities::ARITHMETIC,
        EnumFunctionalities::MEMORY_ACCESS
    };
}

#endif