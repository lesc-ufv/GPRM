#ifndef ENUM_SCHEDULING_HPP
#define ENUM_SCHEDULING_HPP

enum class EnumScheduling{
    ASAP = 0,
    ALAP = 1,
    MOBILITY = 2,
    TIGHT = 3,
    NONE = 4
};

inline std::string enumSchedulingToString(EnumScheduling sched) {
    switch (sched) {
        case EnumScheduling::ASAP:
            return "ASAP";
        case EnumScheduling::ALAP:
            return "ALAP";
        case EnumScheduling::MOBILITY:
            return "Mobility";
        case EnumScheduling::TIGHT:
            return "TIGHT";
        case EnumScheduling::NONE:
            return "NONE";
        default:
            throw std::invalid_argument("Unkwnown");
    }
}

#endif