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
    std::string to_remove = argv[6];
    int gin_layers = std::stoi(argv[7]);
    
    // assert(false && "Add the best config from mrl experiments here");
    int out_dim = 64;
    int feature_out_dim = 64;
    int n_heads = 8;
    int transformer_layers = 12;
    tb_path += "out_dim_" +  std::to_string(out_dim) + "_featdim_" + std::to_string(feature_out_dim) + "_nheads_" + std::to_string(n_heads) + "_TL_" + std::to_string(transformer_layers) + "/";

    GPRMConfig model_config;
    model_config.setOutDim(out_dim);
    model_config.setFeatureOutDim(feature_out_dim);
    model_config.setNHeads(n_heads);
    model_config.setNGraphEmbedLayers(3);
    model_config.setNTransformerLayers(transformer_layers);
    model_config.setLpeDim(16);
    model_config.setFfDim(2*out_dim);
    if(gin_layers == 0){
        model_config.setGinLayersList({});
    } else if (gin_layers == 1){
        model_config.setGinLayersList({1});
    } else if (gin_layers == 2){
        model_config.setGinLayersList({1,2});
    } else if (gin_layers == 3){
        model_config.setGinLayersList({1,2,3});
    }
    tb_path += "ginlayers_" + std::to_string(gin_layers)  + "/";
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

    if (to_remove == "MLP"){
        model_config.setUseMlp(false);
    }else if (to_remove == "RMSNorm"){
        model_config.setUseRmsNorm(false);
    }else if (to_remove == "NormBeforePred"){
        model_config.setUseNormBeforePred(false);
    }else if (to_remove == "GinEmb"){
        model_config.setUseGinEmb(false);
    }else if (to_remove == "Mobility"){
        model_config.setUseMobilityInfo(false);
    }else if (to_remove == "Coord"){
        model_config.setUseCoordInfo(false);
    }else if (to_remove == "GNN"){
        model_config.setUseGnn(false);
    }else if (to_remove == "Transformer"){
        model_config.setUseTransformer(false);
    }else if (to_remove == "PlacementOrder"){
        model_config.setUsePlacementOrder(false);
    }else if (to_remove == "IgnoreLPE"){
        model_config.setIgnoreLpe(true);
        tb_path += "ablation_ignore_lpe/";
    }else if ("NormInitialEmb" == to_remove){
        model_config.setNormInitialEmb(false);

    } else if (to_remove == "UseScale"){
        model_config.setUseScale(false);
    }
    else if(to_remove == ""){
        // No ablation
    }else{
        assert(false && "Unknown ablation option");
    }
    if(to_remove != "IgnoreLPE"){
        tb_path += "ablation_wo_" + to_remove + "/";
    }

    run_mrl_experiment<GPRMState>( argv,
        tb_path, 
        true,
        EnumModel::GPRM,
       model_config);

}