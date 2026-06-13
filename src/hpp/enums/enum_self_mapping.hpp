#ifndef ENUM_SELF_MAPPING_H
#define ENUM_SELF_MAPPING_H

#include <string>
#include <stdexcept>

enum class EnumSelfMapping {
    POLICY_LOGITS = 0,
    SMARTMAP_MCTS = 1,
    MAPZERO_MCTS = 2,
    BEAM_SEARCH = 3,
    DEFAULT = 4
};

inline std::string enumSelfMappingToString(EnumSelfMapping mapping) {
    switch (mapping) {
        case EnumSelfMapping::POLICY_LOGITS:
            return "Policy logits";
        case EnumSelfMapping::SMARTMAP_MCTS:
            return "MuZero MCTS";
        case EnumSelfMapping::MAPZERO_MCTS:
            return "MapZero MCTS";
        case EnumSelfMapping::BEAM_SEARCH:
            return "Beam Search";
        case EnumSelfMapping::DEFAULT:
            return "DEFAULT";
        default:
            throw std::invalid_argument("Unknown");
    }
}

#endif
