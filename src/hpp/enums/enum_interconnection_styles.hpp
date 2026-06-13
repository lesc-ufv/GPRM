#ifndef ENUM_INTERCONNECTION_STYLE_H
#define ENUM_INTERCONNECTION_STYLE_H
#include <iostream>
#include <unordered_set>
#include <cereal/archives/binary.hpp>

enum class EnumInterconnectionStyles{
    MESH = 0,
    ONE_HOP = 1,
    TOROIDAL = 2,
    DIAGONAL = 3 
};


inline unsigned char get_bin_by_style(EnumInterconnectionStyles style){
    switch(style){
        case EnumInterconnectionStyles::MESH:
            return 0b0001;
        case EnumInterconnectionStyles::TOROIDAL:
            return 0b0010;
        case EnumInterconnectionStyles::DIAGONAL:
            return 0b0100;
        case EnumInterconnectionStyles::ONE_HOP:
            return 0b1000;
        default:
            throw std::runtime_error("Unknown style;");
    }

};

inline unsigned char gen_bin_by_styles(std::unordered_set<EnumInterconnectionStyles> styles){
    unsigned char bin = 0b0000;
    for (const auto& style : styles) {
        bin |= get_bin_by_style(style);
    }
    return bin;
};


inline std::string enum_connection_style_to_string(EnumInterconnectionStyles style) {
    switch (style) {
        case EnumInterconnectionStyles::MESH:
            return "MESH";
        case EnumInterconnectionStyles::ONE_HOP:
            return "ONE_HOP";
        case EnumInterconnectionStyles::TOROIDAL:
            return "TOROIDAL";
        case EnumInterconnectionStyles::DIAGONAL:
            return "DIAGONAL";
        default:
            throw std::runtime_error("Unknown style;");
            return "Unknown";
    }
};

inline std::string get_str_styles(std::unordered_set<EnumInterconnectionStyles> styles){
    std::string str = "";
    for (const auto& style : styles) {
        str += enum_connection_style_to_string(style) + " + ";
    }
    str.pop_back();
    return str;
};

inline std::vector<EnumInterconnectionStyles> get_all_enum_connect_styles_values() {
    return {
        EnumInterconnectionStyles::MESH,
        EnumInterconnectionStyles::ONE_HOP,
        EnumInterconnectionStyles::TOROIDAL,
        EnumInterconnectionStyles::DIAGONAL
    };
};



#endif