#ifndef ENUM_NORMALIZATION_HPP
#define ENUM_NORMALIZATION_HPP
#include "src/hpp/utils/util_torch.hpp"

enum class EnumNormalization{
    MIN_MAX = 0,
    MEAN_STD = 1,
    NONE = 2
};

inline torch::Tensor norm_by_enum(torch::Tensor tensor, EnumNormalization norm, double min_or_mean, double max_or_std){
    switch (norm)
    {
    case EnumNormalization::MIN_MAX:
        return min_max_norm(tensor, min_or_mean,max_or_std);
    case EnumNormalization::MEAN_STD:
        return mean_std_norm(tensor, min_or_mean,max_or_std);
    case EnumNormalization::NONE:
        return tensor;
    default:
        std::runtime_error("EnumNormalization doesn't exist.");
        return torch::tensor({});
    }
};

inline std::string enum_norm_to_string(EnumNormalization norm){
    switch (norm)
    {
    case EnumNormalization::MIN_MAX:
        return "MIN_MAX";
    case EnumNormalization::MEAN_STD:
        return "MEAN_STD";
    case EnumNormalization::NONE:
        return "NONE";
    default:
        std::runtime_error("EnumNormalization doesn't exist.");
        return "";
    }
    // return "";
};



#endif