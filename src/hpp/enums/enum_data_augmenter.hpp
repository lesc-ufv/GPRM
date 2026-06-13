#pragma once

#include <string>
#include <stdexcept>

enum class EnumDataAugmenter {
    CGRA_MAPPING_INCREMENTAL = 0,
    CURRICULUM_AWARE = 1,
    CGRA_MAPPING_COMBINATORIAL = 2
};

inline std::string enum_data_augmenter_to_string(EnumDataAugmenter aug) {
    switch (aug) {
        case EnumDataAugmenter::CGRA_MAPPING_INCREMENTAL:
            return "CGRA_MAPPING_INCREMENTAL";
        case EnumDataAugmenter::CURRICULUM_AWARE:
            return "CURRICULUM_AWARE";
        case EnumDataAugmenter::CGRA_MAPPING_COMBINATORIAL:
            return "CGRA_MAPPING_COMBINATORIAL:";
    }
    
    throw std::runtime_error("Unknown EnumDataAugmenter");
}
