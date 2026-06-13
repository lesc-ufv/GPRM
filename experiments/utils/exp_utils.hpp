#ifndef EXP_UTILS_HPP
#define EXP_UTILS_HPP
#include "src/hpp/configs/path_config.hpp"
#include "src/hpp/configs/train_config.hpp"
#include <string>
#include "src/hpp/configs/ppo_config.hpp"
#include <filesystem>
#include "experiments/configs/configs.hpp"
#include "src/hpp/configs/state_config.hpp"

using json = nlohmann::json;

using AllConfigsSM = std::tuple<PPOConfig,TrainConfig, PathConfig, CGRAConfig, RLConfig, 
    DFGConfig, MCTSConfig, StateConfig,MappingConfig,
    SmartMapConfig, BufferConfig, LauncherConfig>;
template<typename StateT,typename ModelConfig>
std::vector<std::shared_ptr<StateT>> save_data_by_config( std::string supervised_data_path,  std::string test_data_path ,std::string path_to_save_data, std::string port,
    GlobalConfig<ModelConfig>& global_config){
    
    auto launcher = Launcher<StateT,ModelConfig>(global_config);
    auto eval_initial_states = launcher.get_eval_states();
    if(!std::filesystem::exists(path_to_save_data)){
        std::filesystem::create_directories(path_to_save_data);
        auto train_config = global_config.getTrainConfig();
        auto mappings = launcher.gen_supervised_data(eval_initial_states);
        auto batches = launcher.generate_batches(mappings, train_config.getBatchSizeStates());
        launcher.send_data_to_server(batches);
        launcher.save_supervised_data_server(path_to_save_data);
        mappings.clear();
        batches.clear();
    }
    launcher.finish_socket();
    return eval_initial_states;
}
    
bool load_checkpoint_if_exists(std::string load_path, PathConfig& path_config, TrainConfig& train_config, PPOConfig& ppo_config){
    if((fs::exists(load_path) && fs::is_directory(load_path))) {
        std::cout << "Tensorboard log path already exists: " << load_path << "\n";
        if(fs::exists(load_path + "train_info.json")) {
            std::cout << "Checkpoint already exists, starting from it \n";
            json js;
            std::ifstream file(load_path + "train_info.json");
            if (file.is_open()) {
                file >> js;
                bool train_finished = js["train_finished"];
                if (train_finished){
                    file.close();
                    return true;
                    }
                else{
                    std::cout << "[INFO] Loading configs \n";
                    train_config.setInitialIter(static_cast<int>(js["final_iter"]) + 1);
                    path_config.setLoadCheckpointPath(load_path);
                    ppo_config.setInitialEntropyCoef(js["entropy_coef"]);
                }
            } else {
                std::cerr << "[Error] Error opening file: " << load_path + "train_info.json" << "\n";
                throw;
            }
            
            path_config.setLoadCheckpointPath(load_path);
            file.close();
        }
    }
    return false;
}

void reset_configs(TrainConfig& train_config, PathConfig& path_config){
    train_config.setInitialIter(0);
    path_config.setLoadCheckpointPath("");
}

void finish_exp(AllConfigsSM& configs){
    auto [ppo_config, train_config, path_config, cgra_config,
        rl_config, dfg_config, mcts_config, state_config, mapping_config,
        model_config, buffer_config, launcher_config] = configs;
    
    auto global_config = GlobalConfig<SmartMapConfig>(std::ref(ppo_config),std::ref(train_config), std::ref(rl_config), std::ref(path_config),
            std::ref( cgra_config),std::ref(dfg_config), std::ref(mcts_config), std::ref(buffer_config), std::ref(launcher_config), 
            std::ref(model_config), std::ref(mapping_config),std::ref(state_config));
    
    auto launcher = Launcher<SmartMapState,SmartMapConfig>(std::ref(global_config));
    launcher.finish();
}
void run_placement_mask_exp(AllConfigsSM& configs,std::vector<EnumPlacementApproach>& placement_approach,
                            std::string& base_tensorboard_dir){
    auto [ppo_config, train_config, path_config, cgra_config,
        rl_config, dfg_config, mcts_config, state_config, mapping_config,
        model_config, buffer_config, launcher_config] = configs;
    std::stringstream ss;
    std::vector<std::shared_ptr<SmartMapState>> train_initial_states = {};
    std::vector<std::shared_ptr<SmartMapState>> eval_initial_states = {};
    
    for(int i =0 ; i < 3; i++){
        train_config.setSeed(train_config.getSeed() + i*100 );
        for(auto placement: placement_approach){
            dfg_config.setPlacementApproach(placement);
    
            reset_configs(train_config, path_config);
            std::string tensorboard_log = base_tensorboard_dir + std::format("{}/seed_{}/",enumPlacementToString(placement),train_config.getSeed());
                                        
            bool train_finished = load_checkpoint_if_exists(tensorboard_log,path_config,train_config,ppo_config);
            
            if (train_finished) {
                std::cout << "[INFO] Training already finished, skipping this configuration \n\n";
                continue;
            }

            path_config.setTensorboardLogPath(tensorboard_log);
            path_config.setSaveCheckpointPath(tensorboard_log);
    
            auto global_config = GlobalConfig<SmartMapConfig>(std::ref(ppo_config),std::ref(train_config), std::ref(rl_config), std::ref(path_config),
            std::ref( cgra_config),std::ref(dfg_config), std::ref(mcts_config), std::ref(buffer_config), std::ref(launcher_config), 
            std::ref(model_config), std::ref(mapping_config),std::ref(state_config));
    
            auto launcher = Launcher<SmartMapState,SmartMapConfig>(std::ref(global_config));
            if(train_initial_states.empty()){
                std::tie(train_initial_states,eval_initial_states) = launcher.get_initial_states();
                std::cout << "[INFO] Number of train states " << train_initial_states.size() <<  "\n[INFO] Number of eval states: " << eval_initial_states.size()  << "\n";
            }

            launcher.run_config_server();
            auto _ = launcher.ppo_train(train_initial_states, eval_initial_states);
            launcher.finish_socket();
            
        }
    }
    
}

AllConfigsSM get_configs(std::string& train_path, std::string& eval_path,
                                                                     std::string& port, int num_threads){
            PPOConfig ppo_config = get_ppo_config(0.1, 0.05);
            TrainConfig train_config = get_train_config(EnumModel::SMARTMAP);
            PathConfig path_config;
            path_config.setTrainDfgsDotPath(train_path);
            path_config.setEvalDfgsDotPath(eval_path);
            CGRAConfig cgra_config = get_cgra_config();
            RLConfig rl_config = get_rl_config();
            MCTSConfig mcts_config;
            StateConfig state_config = get_state_config();
            MappingConfig mapping_config = get_mapping_config();
            SmartMapConfig model_config = get_smartmap_config();
            BufferConfig buffer_config = get_buffer_config(train_config);
            LauncherConfig launcher_config = get_launcher_config(EnumEnvironment::SMARTMAP,EnumModel::SMARTMAP, port, num_threads);
            DFGConfig dfg_config = get_dfg_config();
            return std::make_tuple(ppo_config, train_config, path_config, cgra_config,
                 rl_config, dfg_config, mcts_config, state_config, mapping_config,
                  model_config, buffer_config, launcher_config);
}


PPOConfig   get_ppo_config_mrl(){
    return get_ppo_config(0.1, 0.05);
}
#endif