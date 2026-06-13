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
    std::string tb_path = "results/mapping_representation_learning/GPRM/wo_aug/";
    int out_dim = std::stoi(argv[6]);
    int feature_out_dim = std::stoi(argv[7]);
    int n_heads = std::stoi(argv[8]);
    int transformer_layers = std::stoi(argv[9]);
    tb_path += "out_dim_" +  std::to_string(out_dim) + "_featdim_" + std::to_string(feature_out_dim) + "_nheads_" + std::to_string(n_heads) + "_TL_" + std::to_string(transformer_layers) + "/";
    
    GPRMConfig model_config;
    model_config.setOutDim(out_dim);
    model_config.setFeatureOutDim(feature_out_dim);
    model_config.setNHeads(n_heads);
    model_config.setNGraphEmbedLayers(3);
    model_config.setNTransformerLayers(transformer_layers);
    model_config.setLpeDim(16);
    model_config.setFfDim(2*out_dim);
    model_config.setGinLayersList({1,2,3});
    model_config.setActivationFunc("gelu");
    model_config.setUseMlp(true);
    model_config.setDropout(0.1);
    model_config.setUseRmsNorm(true);
    model_config.setUseNormBeforePred(true);
    model_config.setNegativeSlope(0.02);
    model_config.setMaxLen(5000);
    model_config.setNormInitialEmb(true);
    model_config.setUseGinEmb(true);
    model_config.setUseMobilityInfo(true);
    model_config.setUseCoordInfo(true);
    model_config.setUseGnn(true);
    model_config.setUseTransformer(true);
    model_config.setUsePlacementOrder(true);
    model_config.setIgnoreLpe(false);
    model_config.setUseScale(true);


    run_mrl_experiment<GPRMState>( argv,
        tb_path, 
        true,
        EnumModel::GPRM,
       model_config);

}