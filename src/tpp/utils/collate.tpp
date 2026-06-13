#include "src/hpp/rl/collate.hpp"

template <typename StateT>
Collate<StateT>::Collate() {};

template <typename StateT>
Collate<StateT>::Collate(const EnumModel &model) {
    if (model == EnumModel::GPRM) {
        this->collate_function = [](const std::vector<std::shared_ptr<StateT>>& states, const std::vector<int>& indices) {
            return gprm_collate_fn(reinterpret_cast<const std::vector<std::shared_ptr<GPRMState>>&>(states), indices);
        };
    } else if (model == EnumModel::SMARTMAP) {
        this->collate_function = [](const std::vector<std::shared_ptr<StateT>>& states, const std::vector<int>& indices) {
            return smartmap_collate_fn(reinterpret_cast<const std::vector<std::shared_ptr<SmartMapState>>&>(states), indices);
        };
    } else if (model == EnumModel::MAPZERO) {
        this->collate_function = [](const std::vector<std::shared_ptr<StateT>>& states, const std::vector<int>& indices) {
            return mapzero_collate_fn(reinterpret_cast<const std::vector<std::shared_ptr<MapZeroState>>&>(states), indices);
        };
    }
    else if (model == EnumModel::TRANSMAP) {
        this->collate_function = [](const std::vector<std::shared_ptr<StateT>>& states, const std::vector<int>& indices) {
            return transmap_collate_fn(reinterpret_cast<const std::vector<std::shared_ptr<TransMapState>>&>(states), indices);
        };
    }else{
        throw std::runtime_error("Invalid Model. (Collate)");
    }
}

template <typename StateT>
std::vector<c10::IValue> Collate<StateT>::collate(
    const std::vector<std::shared_ptr<StateT>>& states, const std::vector<int>& indices) const
{
    return this->collate_function(states, indices);
}
