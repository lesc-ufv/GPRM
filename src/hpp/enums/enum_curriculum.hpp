#ifndef ENUM_CURRICULUM_HPP
#define ENUM_CURRICULUM_HPP
#include <string>

enum class EnumCurriculum{
    DFG_SIZE = 0,
    HUMAN = 1,
    HUMANV2 = 2,
    HUMANV2_OCCUPANCY = 3,
    MAP_COMPLEXITY = 4,
    MAP_COMPLEXITYV2 = 5,
    DFG_COMPLEXITY = 6


};

inline std::string enum_curriculum_to_string(EnumCurriculum enum_curriculum) {
    switch (enum_curriculum) {
        case EnumCurriculum::DFG_SIZE:
            return "DFG_SIZE";
        case EnumCurriculum::HUMAN:
            return "HUMAN";
        case EnumCurriculum::HUMANV2:
            return "HUMANV2";
        case EnumCurriculum::HUMANV2_OCCUPANCY:
            return "HUMANV2_OCCUPANCY";
        case EnumCurriculum::MAP_COMPLEXITY:
            return "MAP_COMPLEXITY";
        case EnumCurriculum::MAP_COMPLEXITYV2:
            return "MAP_COMPLEXITYV2";
        default:
            return "UNKNOWN";
    }
}
#endif