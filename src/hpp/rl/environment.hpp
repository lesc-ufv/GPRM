#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

#include <iostream>
#include <type_traits>
#include "src/hpp/compilers/gprm/rl/environments/gprm_environment.hpp"
#include "src/hpp/enums/enum_environment.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_environment.hpp"
#include "src/hpp/compilers/mapzero/rl/mapzero_environment.hpp"
#include "src/hpp/compilers/transmap/rl/transmap_environment.hpp"
#include "src/hpp/compilers/e2emap/rl/e2emap_environment.hpp"

template <typename State>
class Environment{
private:
    double (*reward_function)(const State&, const int&) = nullptr;
public:
    Environment(const EnumEnvironment& enum_environment);
    Environment();
    std::tuple<std::shared_ptr<State>, double, bool> step(State state, int action);
};

#include "src/tpp/rl/environment.tpp"
#endif

