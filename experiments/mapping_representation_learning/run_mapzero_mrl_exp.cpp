
#include <iostream>
#include <vector>
#include <unordered_set>
#include <filesystem>
#include "src/hpp/enums/enum_functionalities.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_self_mapping.hpp"
#include "src/hpp/utils/constants/server_constants.hpp"
#include <zmq.hpp>
#include "src/hpp/enums/enum_model.hpp"
#include <msgpack.hpp> 
#include "src/hpp/utils/constants/path_constants.hpp"
#include "src/hpp/utils/util_dfg.hpp"
#include "src/hpp/utils/util_server.hpp"
#include "src/hpp/utils/util_torch.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_pre_training_buffer.hpp"
#include "src/hpp/utils/util_args.hpp"
#include "src/hpp/utils/util_train.hpp"
#include "src/hpp/data_structures/dfgs/ordered_dfg.hpp"
#include "src/hpp/data_structures/cgras/cgra.hpp"
#include "src/hpp/compilers/smartmap/configs/smartmap_config.hpp"
#include "src/hpp/compilers/gprm/configs/gprm_config.hpp"
#include "src/hpp/configs/ppo_config.hpp"
#include "src/hpp/configs/train_config.hpp"
#include "src/hpp/configs/rl_config.hpp"
#include "src/hpp/configs/path_config.hpp"
#include "src/hpp/configs/cgra_config.hpp"
#include "src/hpp/compilers/smartmap/configs/smartmap_config.hpp"
#include "src/hpp/compilers/mapzero/configs/mapzero_config.hpp"
#include "src/hpp/configs/dfg_config.hpp"
#include "src/hpp/configs/mcts_config.hpp"
#include "src/hpp/configs/buffer_config.hpp"
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/configs/mapping_config.hpp"
#include "src/hpp/launcher.hpp"
#include "src/hpp/rl/buffers/buffer.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_self_mapping.hpp"
#include "experiments/mapping_representation_learning/mrl_configs.hpp"
#include <filesystem>
#include <tbb/global_control.h>
#include "experiments/utils/exp_utils.hpp"

using json = nlohmann::json;

int main(int argc, char* argv[]){
    std::string tb_path = "results/mapping_representation_learning/MapZero/wo_aug/";    
    int out_dim = std::stoi(argv[6]);
    int n_heads = std::stoi(argv[7]);
    tb_path += "outdim_" + std::to_string(out_dim) + "_nheads_" + std::to_string(n_heads) + "/";

    MapZeroConfig model_config;
    model_config.setNegativeSlope(0.02);
    model_config.setNHeads(n_heads);
    model_config.setOutDim(out_dim);
    model_config.setLenActionSpace(16);
    run_mrl_experiment<MapZeroState>(argv,
        tb_path, 
        false,
        EnumModel::MAPZERO,
       model_config);

}