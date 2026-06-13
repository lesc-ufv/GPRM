#include "src/hpp/rl/environment.hpp"

template<typename State>
Environment<State>::Environment(const EnumEnvironment& enum_environment)
{
    if (enum_environment == EnumEnvironment::GPRM) {
        this->reward_function = &compute_gprm_reward<State>;
    } else if(enum_environment == EnumEnvironment::GPRM_V2){
        this->reward_function = &compute_gprm_reward_v2<State>;
    } else if(enum_environment == EnumEnvironment::GPRM_V3){
        this->reward_function = &compute_gprm_reward_v3<State>;
    } else if(enum_environment == EnumEnvironment::SMARTMAP){
        this->reward_function = &compute_smartmap_reward<State>;
    } else if (enum_environment == EnumEnvironment::MAPZERO){
        this->reward_function = &compute_mapzero_reward<State>;
    } else if (enum_environment == EnumEnvironment::TRANSMAP){
        this->reward_function = &compute_transmap_reward<State>;
    }else if(enum_environment == EnumEnvironment::E2EMAP){
        this->reward_function = &compute_e2emap_reward<State>;
    } else {
        throw std::invalid_argument("Invalid EnumEnvironment type");
    }
}

template<typename State>
Environment<State>::Environment(){};

template <typename State>
std::tuple<std::shared_ptr<State>, double, bool> Environment<State>::step(State state, int action)
{
    if (!state.get_is_end_state()) {
        // compute next state
        state.step(action);
    
        // compute_reward
        auto reward = this->reward_function(state, action);
    
        return std::make_tuple(std::make_shared<State>(std::move(state)), reward, state.get_is_end_state());
    }
    return std::make_tuple(std::make_shared<State>(std::move(state)), 0.0, true);

}
