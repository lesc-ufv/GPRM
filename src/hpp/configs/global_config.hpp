#ifndef GLOBAL_CONFIG_HPP
#define GLOBAL_CONFIG_HPP

#include "src/hpp/configs/ppo_config.hpp"
#include "src/hpp/configs/train_config.hpp"
#include "src/hpp/configs/rl_config.hpp"
#include "src/hpp/configs/path_config.hpp"
#include "src/hpp/configs/cgra_config.hpp"
#include "src/hpp/configs/dfg_config.hpp"
#include "src/hpp/configs/mcts_config.hpp"
#include "src/hpp/configs/buffer_config.hpp"
#include "src/hpp/configs/launcher_config.hpp"
#include "src/hpp/configs/mapping_config.hpp"
#include "src/hpp/configs/state_config.hpp"
#include <sstream>
#include <iostream>
#include "nlohmann/json.hpp"
#include "src/hpp/compilers/gprm/configs/gprm_config.hpp"
#include "src/hpp/compilers/mapzero/configs/mapzero_config.hpp"
#include "src/hpp/compilers/smartmap/configs/smartmap_config.hpp"
#include "src/hpp/compilers/transmap/rl/configs/transmap_config.hpp"
#include <zmq.hpp>
#include "src/hpp/utils/constants/server_constants.hpp"

using json = nlohmann::json;


template<typename ModelConfig>
class GlobalConfig {
public:    
    PPOConfig ppo_config;
    TrainConfig train_config;
    RLConfig rl_config;
    PathConfig path_config;
    CGRAConfig cgra_config;
    DFGConfig dfg_config;
    MCTSConfig mcts_config;
    BufferConfig buffer_config;
    LauncherConfig launcher_config;
    ModelConfig model_config;
    MappingConfig mapping_config;
    StateConfig state_config;

    GlobalConfig() {}

    GlobalConfig(
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
    ) : ppo_config(ppo_config),
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
        state_config(state_config) {}

    
    template <typename T = ModelConfig>
    typename std::enable_if<
        std::is_same<T, GPRMConfig>::value || std::is_same<T, TransMapConfig>::value,
        std::tuple<
            std::shared_ptr<zmq::socket_t>,
            ServerConstants,
            ModelConfig,
            StateConfig
        >
    >::type
    get_state_args(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k) {
        return std::make_tuple(
            socket,
            server_k,
            this->getModelConfig(),
            this->getStateConfig()
        );
    }


    template <typename T = ModelConfig>
    typename std::enable_if<
        !std::is_same<T, GPRMConfig>::value && !std::is_same<T, TransMapConfig>::value,
        std::tuple<StateConfig>
    >::type
    get_state_args(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k) {
        return std::make_tuple(
            this->getStateConfig()
        );
    }

    bool should_calc_node_dists(){
        auto greedy_type = this->getDFGConfig().getGreedyType();
        return greedy_type == EnumGreedyType::YOTT;

    }

    PPOConfig& getPPOConfig() { return ppo_config; }
    TrainConfig& getTrainConfig() { return train_config; }
    RLConfig& getRLConfig() { return rl_config; }
    PathConfig& getPathConfig() { return path_config; }
    CGRAConfig& getCGRAConfig() { return cgra_config; }
    DFGConfig& getDFGConfig() { return dfg_config; }
    MCTSConfig& getMCTSConfig() { return mcts_config; }
    BufferConfig& getBufferConfig() { return buffer_config; }
    LauncherConfig& getLauncherConfig() { return launcher_config; }
    MappingConfig& getMappingConfig() { return mapping_config; }
    ModelConfig& getModelConfig() { return model_config; }
    StateConfig& getStateConfig() { return state_config; }
    void set_model_config_from_json(const json& js) {
        model_config.set_config_from_json(js);
    }

    void set_buffer_config(BufferConfig buffer_config){
        this->buffer_config = buffer_config;
    }

    void set_dfg_config(DFGConfig dfg_config){
        this->dfg_config = dfg_config;
    }

    void set_cgra_config(CGRAConfig cgra_config){
        this->cgra_config = cgra_config;
    }

    void set_train_config(TrainConfig train_config){
        this->train_config = train_config;
    }

    std::string get_model_name() {
        return get_model_name_by_enum(launcher_config.getModel());
    }
    void print_ss(std::stringstream& ss) {
        ss << "=== Global Configuration ===\n";
        {
            std::stringstream tmp;
            ppo_config.print_ss(tmp);
            ss << tmp.str();
        }
        {
            std::stringstream tmp;
            train_config.print_ss(tmp);
            ss << tmp.str();
        }
        {
            std::stringstream tmp;
            rl_config.print_ss(tmp);
            ss << tmp.str();
        }
        {
            std::stringstream tmp;
            path_config.print_ss(tmp);
            ss << tmp.str();
        }
        {
            std::stringstream tmp;
            cgra_config.print_ss(tmp);
            ss << tmp.str();
        }
        {
            std::stringstream tmp;
            dfg_config.print_ss(tmp);
            ss << tmp.str();
        }
        {
            std::stringstream tmp;
            mcts_config.print_ss(tmp);
            ss << tmp.str();
        }
        {
            std::stringstream tmp;
            buffer_config.print_ss(tmp);
            ss << tmp.str();
        }
        {
            std::stringstream tmp;
            launcher_config.print_ss(tmp);
            ss << tmp.str();
        }
        {
            std::stringstream tmp;
            mapping_config.print_ss(tmp);
            ss << tmp.str();
        }
        {
            std::stringstream tmp;
            model_config.print_ss(tmp);
            ss << tmp.str();
        }
        {
            std::stringstream tmp;
            state_config.print_ss(tmp);
            ss << tmp.str();
        }
    }

    void print() {
        std::stringstream ss;
        print_ss(ss);
        std::cout << ss.str();
    }
};

#endif
