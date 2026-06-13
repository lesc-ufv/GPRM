#ifndef ENUM_OPTIMIZER_HPP
#define ENUM_OPTIMIZER_HPP

enum class EnumOptimizer{
    SGD = 0,
    ADAMW = 1,
    DEFAULT = 2
};

inline std::string enum_optimizer_to_string(EnumOptimizer optimizer) {
    switch (optimizer) {
        case EnumOptimizer::SGD:
            return "SGD";
        case EnumOptimizer::ADAMW:
            return "ADAMW";
        case EnumOptimizer::DEFAULT:
            return "DEFAULT";
        default:
            throw std::invalid_argument("Unknown");
    }
}
#endif