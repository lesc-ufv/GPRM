#ifndef ENUM_ENV_SAMPLER_HPP
#define ENUM_ENV_SAMPLER_HPP

enum class EnumEnvSampler{
    DOUBLE_BUFFER = 0,
    UCB = 1,
    RANDOM = 2,
    PLR = 3,
    SEQUENTIAL = 4,
    DEFAULT = 5
};

inline std::string enum_env_sampler_to_string(EnumEnvSampler sampler){
    switch(sampler){
        case EnumEnvSampler::DOUBLE_BUFFER:
            return "DOUBLE_BUFFER";
        case EnumEnvSampler::UCB:
            return "UCB";
        case EnumEnvSampler::RANDOM:
            return "RANDOM";
        case EnumEnvSampler::PLR:
            return "PLR";
        case EnumEnvSampler::SEQUENTIAL:
            return "SEQUENTIAL";
        default:
            return "UNKNOWN";
    }
}
#endif