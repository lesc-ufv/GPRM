#ifndef ENUM_TASK_TYPE_HPP
#define ENUM_TASK_TYPE_HPP


#include <string>
#include <stdexcept>

enum class EnumTaskType {
    TRAIN = 0,
    ZERO_SHOT_MAP = 1,
    FINETUNE_MAP = 2,
    DEFAULT = 3
};

inline std::string enumToString(EnumTaskType taskType) {
    switch (taskType) {
        case EnumTaskType::TRAIN:
            return "Train";
        case EnumTaskType::ZERO_SHOT_MAP:
            return "Zero-shot mapping";
        case EnumTaskType::FINETUNE_MAP:
            return "Finetune mapping";
        case EnumTaskType::DEFAULT:
            return "DEFAULT";
        default:
            throw std::invalid_argument("Unknown EnumTaskType");
    }
}



#endif
