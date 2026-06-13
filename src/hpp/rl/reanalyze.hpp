#ifndef REANALYZE_HPP
#define REANALYZE_HPP

#include "src/hpp/rl/self_mapping.hpp"
#include "src/hpp/rl/buffers/buffer.hpp"
#include <string>
#include "src/hpp/rl/mapping_history.hpp"

template <typename StateT, typename ModelConfig>
class Reanalyze {
private:
    std::shared_ptr<Buffer<StateT, ModelConfig>> buffer;
    SelfMapping<StateT,ModelConfig> self_mapping;
    std::string value_k = "value";
    std::string target_policy_k = "target_policy";
    int reanalyzed_mappings = 0;
    int reanalyzed_states = 0;
    std::mt19937 gen = RNG::get();
public:
    Reanalyze(std::shared_ptr<Buffer<StateT, ModelConfig>> buffer, SelfMapping<StateT,ModelConfig>& self_mapping);
    Reanalyze();
    void reanalyze();

    inline int get_reanalyzed_mappings() const { return reanalyzed_mappings; }
    inline int get_reanalyzed_states() const { return reanalyzed_states; }

    
    template<typename Archive>
    void serialize(Archive& ar) {
        ar(reanalyzed_mappings, reanalyzed_states);

        std::stringstream ss;

        if constexpr (Archive::is_saving::value) {
            ss << gen;
            std::string gen_state = ss.str();
            ar(gen_state);
        } 
        else {
            std::string gen_state;
            ar(gen_state);
            ss.str(gen_state);
            ss >> gen;
        }
}

};

#include "src/tpp/rl/reanalyze.tpp"

#endif