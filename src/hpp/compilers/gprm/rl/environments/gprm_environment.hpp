
#ifndef GPRM_ENVIRONMENT_HPP
#define GPRM_ENVIRONMENT_HPP
// #define DEBUG

#include "src/hpp/compilers/gprm/utils/util_gprm_environment.hpp"
#include "src/hpp/utils/util_print.hpp"

template <typename State>
double compute_gprm_reward(const State& state, const int& action);

template <typename State>
double compute_gprm_reward_v2(const State& state, const int& action);

template <typename State>
double compute_gprm_reward_v3(const State& state, const int& action);

#include "src/tpp/compilers/gprm/rl/environments/gprm_environment.tpp"

#endif