#ifndef ENUM_GREEDY_H
#define ENUM_GREEDY_H

#include <string>
#include <stdexcept>

enum class EnumGreedyType {
    NONE = 0,
    GREEDY = 1,
    YOTO = 2,
    YOTT = 3,
    DEFAULT = 4
};

inline std::string enumToString(EnumGreedyType type) {
    switch (type) {
        case EnumGreedyType::NONE:
            return "none";
        case EnumGreedyType::YOTO:
            return "yoto";
        case EnumGreedyType::GREEDY:
            return "greedy";
        case EnumGreedyType::YOTT:
            return "yott";
        case EnumGreedyType::DEFAULT:
            return "DEFAULT";
        default:
            throw std::invalid_argument("Unkwown");
    }
}

#endif
