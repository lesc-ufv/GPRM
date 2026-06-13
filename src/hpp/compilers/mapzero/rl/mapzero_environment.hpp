#ifndef MAPZERO_ENVIRONMENT_HPP
#define MAPZERO_ENVIRONMENT_HPP

#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"
#include "src/hpp/utils/cgra/router.hpp"
#include <iostream>
#include <tuple>
#include "src/hpp/utils/cgra/util_cgra.hpp"
#include "src/hpp/utils/cgra/scheduler.hpp"


template <typename State>
double compute_mapzero_reward(const State& state, const int& action);

#include "src/tpp/compilers/mapzero/rl/mapzero_environment.tpp"

#endif