#ifndef COLLATE_HPP
#define COLLATE_HPP

#include <iostream>
#include <type_traits>
#include "src/hpp/enums/enum_model.hpp"
#include "src/hpp/compilers/mapzero/utils/mapzero_collate_fn.hpp"
#include "src/hpp/compilers/smartmap/utils/smartmap_collate_fn.hpp"
#include "src/hpp/compilers/gprm/utils/gprm_collate_fn.hpp"
#include "src/hpp/compilers/transmap/utils/transmap_collate_fn.hpp"

template <typename StateT>
class Collate{
private:
    std::function<std::vector<c10::IValue>(const std::vector<std::shared_ptr<StateT>>&, const std::vector<int>&)> collate_function;
public:
    Collate(const EnumModel& model);
    Collate();

    std::vector<c10::IValue> collate(const std::vector<std::shared_ptr<StateT>>& states, const std::vector<int>& indices) const;
};

#include "src/tpp/utils/collate.tpp"

#endif

