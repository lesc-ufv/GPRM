#ifndef SMARTMAP_ENVIRONMENT_HPP
#define SMARTMAP_ENVIRONMENT_HPP

#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"
#include "src/hpp/utils/cgra/router.hpp"
#include <iostream>
#include <tuple>
#include "src/hpp/utils/cgra/util_cgra.hpp"
#include "src/hpp/utils/cgra/scheduler.hpp"
#include "src/hpp/enums/enum_invalid_mapping_reason.hpp"


template <typename State>
double compute_smartmap_reward(const State& state, const int& action);

#include "src/tpp/compilers/smartmap/rl/smartmap_environment.tpp"
#endif