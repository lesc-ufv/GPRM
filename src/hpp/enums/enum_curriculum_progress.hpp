#ifndef ENUM_CURRICULUM_PROGRESS_HPP
#define ENUM_CURRICULUM_PROGRESS_HPP
#include <string>

enum class EnumCurriculumProgress{
    FIXED = 0,
    ADAPTIVE = 1,
    DYN_ADAPTIVE = 2,
};

inline std::string enum_curriculum_progress_to_string(EnumCurriculumProgress enum_curriculum) {
    switch (enum_curriculum) {
        case EnumCurriculumProgress::FIXED:
            return "DFG_SIZE";
        case EnumCurriculumProgress::ADAPTIVE:
            return "HUMAN";
        case EnumCurriculumProgress::DYN_ADAPTIVE:
            return "DYNAMIC_ADAPTIVE";
        default:
            return "UNKNOWN";
    }
}
#endif