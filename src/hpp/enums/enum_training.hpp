#ifndef ENUM_TRAINING_HPP
#define ENUM_TRAINING_HPP

enum class EnumTraining{
    PPO = 0,
    POLICY_GRADIENT = 1,
    MCTS = 2,
    SUPERVISED = 3,
    HYBRID = 4,
    PPO_TRANSMAP = 5,
    DEFAULT = 6
};


inline std::string enumToString(EnumTraining trainingType) {
    switch (trainingType) {
        case EnumTraining::PPO:
            return "PPO";
        case EnumTraining::POLICY_GRADIENT:
            return "Policy Gradient";
        case EnumTraining::MCTS:
            return "MCTS";
        case EnumTraining::DEFAULT:
            return "DEFAULT";
        case EnumTraining::SUPERVISED:
            return "Supervised";
        case EnumTraining::HYBRID:
            return "Hybrid";
        case EnumTraining::PPO_TRANSMAP:
            return "PPO_TRANSMAP";
        default:
            throw std::invalid_argument("Unknown");
    }
}
#endif