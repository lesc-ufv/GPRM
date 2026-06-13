#ifndef ENUM_ENVIRONMENT_H
#define ENUM_ENVIRONMENT_H

#include <string>
#include <stdexcept>

enum class EnumEnvironment {
    GPRM = 0,
    SMARTMAP = 1,
    MAPZERO = 2,
    TRANSMAP = 3,
    E2EMAP = 4,
    GPRM_V2 = 5,
    GPRM_V3 = 6,
    DEFAULT = 7
};

inline std::string enumEnvironmentToString(EnumEnvironment env) {
    switch (env) {
        case EnumEnvironment::GPRM:     return "GPRM";
        case EnumEnvironment::GPRM_V2:     return "GPRM_V2";
        case EnumEnvironment::GPRM_V3:     return "GPRM_V3";
        case EnumEnvironment::SMARTMAP: return "SmartMap";
        case EnumEnvironment::MAPZERO:  return "MapZero";
        case EnumEnvironment::TRANSMAP:  return "TransMap";
        case EnumEnvironment::E2EMAP:  return "E2EMap";
        case EnumEnvironment::DEFAULT:  return "DEFAULT";
        default:                        return "Unknown";
    }
}
#endif
