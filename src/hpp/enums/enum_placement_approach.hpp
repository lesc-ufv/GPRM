#ifndef ENUM_PLACEMENT_APPROACH_H
#define ENUM_PLACEMENT_APPROACH_H

#include <string>
#include <stdexcept>

enum class  EnumPlacementApproach {
    RANDOM = 0,
    TOPOLOGICAL_ORDER = 1,
    ZIGZAG = 2,
    DFS = 3,
    BFS = 4,
    DEFAULT = 5
};

inline std::string enumPlacementToString(EnumPlacementApproach approach) {
    switch (approach) {
        case EnumPlacementApproach::RANDOM:
            return "Random";
        case EnumPlacementApproach::TOPOLOGICAL_ORDER:
            return "Topological";
        case EnumPlacementApproach::ZIGZAG:
            return "ZigZag";
        case EnumPlacementApproach::DFS:
            return "DFS";
        case EnumPlacementApproach::BFS:
            return "BFS";
        case EnumPlacementApproach::DEFAULT:
            return "DEFAULT";
        default:
            throw std::invalid_argument("Unkwnown");
    }
}

#endif
