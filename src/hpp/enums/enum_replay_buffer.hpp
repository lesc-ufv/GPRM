#ifndef ENUM_REPLAY_BUFFER_HPP
#define ENUM_REPLAY_BUFFER_HPP

#include <string>
#include <stdexcept>

enum class EnumReplayBuffer {
    MAPZERO_BUFFER = 0,
    SMARTMAP_PRE_TRAINING_BUFFER = 1,
    PER_BUFFER = 2,
    GPRM_BUFFER = 3,
    SUPERVISED_BUFFER = 4,
    DTB = 5,
    DEFAULT = 6
};

inline std::string enum_replay_buffer_to_string(const EnumReplayBuffer& buffer) {
    switch (buffer) {
        case EnumReplayBuffer::MAPZERO_BUFFER:
            return "MapZero buffer";
        case EnumReplayBuffer::SMARTMAP_PRE_TRAINING_BUFFER:
            return "SmartMap pre-training buffer";
        case EnumReplayBuffer::PER_BUFFER:
            return "PER Buffer";
        case EnumReplayBuffer::GPRM_BUFFER:
            return "GPRM Buffer";
        case EnumReplayBuffer::SUPERVISED_BUFFER:
            return "Suipervised Buffer";
        case EnumReplayBuffer::DTB:
            return "DTB";
        case EnumReplayBuffer::DEFAULT:
            return "DEFAULT";
        default:
            return "UNKNOWN"; 
    }
}
#endif
