
#ifndef GLOBAL_CONFIG_VARIANT_HPP
#define GLOBAL_CONFIG_VARIANT_HPP

#include <variant>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include "src/hpp/enums/enum_model.hpp"
#include "src/hpp/configs/global_config.hpp"

using GlobalConfigVariantType = std::variant<
    GlobalConfig<MapZeroConfig>,
    GlobalConfig<SmartMapConfig>,
    GlobalConfig<GPRMConfig>,
    GlobalConfig<TransMapConfig>

>;

class GlobalConfigVariant {
private:
    GlobalConfigVariantType config_variant;
    EnumModel model;

public:
    GlobalConfigVariant(
        PPOConfig& ppo_config,
        TrainConfig& train_config,
        RLConfig& rl_config,
        PathConfig& path_config,
        CGRAConfig& cgra_config,
        DFGConfig& dfg_config,
        MCTSConfig& mcts_config,
        BufferConfig& buffer_config,
        LauncherConfig& launcher_config,
        MappingConfig& mapping_config,
        StateConfig& state_config
    ) {
        model = launcher_config.getModel();
        switch (model) {
            case EnumModel::MAPZERO:{
                auto mapzero_config = MapZeroConfig();
                config_variant = GlobalConfig<MapZeroConfig>(
                    ppo_config, train_config, rl_config, path_config, cgra_config,
                    dfg_config, mcts_config, buffer_config, launcher_config,
                    mapzero_config, mapping_config, state_config
                );
                break;}
            case EnumModel::SMARTMAP:{
                auto smartmap_config = SmartMapConfig();
                config_variant = GlobalConfig<SmartMapConfig>(
                    ppo_config, train_config, rl_config, path_config, cgra_config,
                    dfg_config, mcts_config, buffer_config, launcher_config,
                    smartmap_config, mapping_config, state_config
                );
                break;}
            case EnumModel::GPRM:{
                auto gprm_config = GPRMConfig();
                config_variant = GlobalConfig<GPRMConfig>(
                    ppo_config, train_config, rl_config, path_config, cgra_config,
                    dfg_config, mcts_config, buffer_config, launcher_config,
                    gprm_config, mapping_config, state_config
                );
                break;
            }
            case EnumModel::TRANSMAP:{
                auto transmap_config = TransMapConfig();
                config_variant = GlobalConfig<TransMapConfig>(
                    ppo_config, train_config, rl_config, path_config, cgra_config,
                    dfg_config, mcts_config, buffer_config, launcher_config,
                    transmap_config, mapping_config, state_config
                );
                break;
            }
            default:
                throw std::runtime_error("Unknown model");
        }
    }
    void set_model_config_from_json(const json& js) {
        std::visit([&js](auto& config) {
            config.set_model_config_from_json(js);
        }, config_variant);
    }

    template<typename ModelConfig>
    GlobalConfig<ModelConfig>& getGlobalConfigAs() {
        if (std::holds_alternative<GlobalConfig<ModelConfig>>(config_variant)) {
            return std::get<GlobalConfig<ModelConfig>>(config_variant);
        } else {
            throw std::runtime_error("Invalid model type");
        }
    }

    EnumModel getModel() const {
        return model;
    }
};

#endif
