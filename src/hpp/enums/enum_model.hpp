#ifndef ENUM_MODEL_H
#define ENUM_MODEL_H

#include <string>
#include <stdexcept>

enum class EnumModel {
    MAPZERO = 0,
    SMARTMAP = 1,
    GPRM = 2,
    TRANSMAP = 3,
    DEFAULT = 4
};

inline std::string get_model_name_by_enum(const EnumModel& enum_model) {
    switch (enum_model) {
        case EnumModel::MAPZERO: 
            return "MapZero";
    
        case EnumModel::SMARTMAP:
            return "SmartMap";
    
        case EnumModel::GPRM:
            return "GPRM";
    
        case EnumModel::TRANSMAP:
            return "TransMap";

        case EnumModel::DEFAULT:
            return "None";
        default:
            throw std::runtime_error("Unknown Model with id " + std::to_string(static_cast<int>(enum_model))); 
    }
};

#endif
