#ifndef ENUM_END_STATE_H
#define ENUM_END_STATE_H

#include <string>
#include <stdexcept>

enum class EnumEndState {
    GPRM = 0,
    SMARTMAP = 1,
    MAPZERO = 2,
    DEFAULT = 3
};

inline std::string enumEndStateToString(EnumEndState end_state) {
    switch (end_state) {
        case EnumEndState::GPRM:     return "GPRM";
        case EnumEndState::SMARTMAP: return "SmartMap";
        case EnumEndState::MAPZERO:  return "MapZero";
        case EnumEndState::DEFAULT:  return "DEFAULT";
        default:                     return "Unknown";
    }
}
#endif
