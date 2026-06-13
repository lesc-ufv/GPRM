
#include "src/hpp/configs/global_config.hpp"

template<typename ModelConfig>
GlobalConfig<ModelConfig>::GlobalConfig(
    PPOConfig& ppo_config,
    TrainConfig& train_config,
    RLConfig& rl_config,
    PathConfig& path_config,
    CGRAConfig& cgra_config,
    DFGConfig& dfg_config,
    MCTSConfig& mcts_config,
    BufferConfig& buffer_config,
    LauncherConfig& launcher_config,
    ModelConfig& model_config,
    MappingConfig& mapping_config,
    StateConfig& state_config
)
    : ppo_config(ppo_config),
      train_config(train_config),
      rl_config(rl_config),
      path_config(path_config),
      cgra_config(cgra_config),
      dfg_config(dfg_config),
      mcts_config(mcts_config),
      buffer_config(buffer_config),
      launcher_config(launcher_config),
      model_config(model_config),
      mapping_config(mapping_config),
      state_config(state_config) {
      }

template<typename ModelConfig>
void GlobalConfig<ModelConfig>::print_ss(std::stringstream& ss) {
    ppo_config.print_ss(ss);
    train_config.print_ss(ss);
    rl_config.print_ss(ss);
    path_config.print_ss(ss);
    cgra_config.print_ss(ss);
    dfg_config.print_ss(ss);
    mcts_config.print_ss(ss);
    buffer_config.print_ss(ss);
    launcher_config.print_ss(ss);
    model_config.print_ss(ss);
    mapping_config.print_ss(ss);
    state_config.print_ss(ss);
}
