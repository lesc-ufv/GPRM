#include "src/hpp/configs/ppo_config.hpp"
#include "src/hpp/configs/train_config.hpp"
#include "src/hpp/configs/cgra_config.hpp"
#include "src/hpp/configs/rl_config.hpp"
#include "src/hpp/configs/dfg_config.hpp"
#include "src/hpp/configs/launcher_config.hpp"
#include "src/hpp/configs/mapping_config.hpp"
#include "src/hpp/configs/buffer_config.hpp"
#include "src/hpp/configs/path_config.hpp"
#include "src/hpp/configs/mcts_config.hpp"
#include "src/hpp/configs/state_config.hpp"
#include "src/hpp/compilers/smartmap/configs/smartmap_config.hpp"
#include "src/hpp/launcher.hpp"
#include <iostream>
#include <string>
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/utils/util_args.hpp"
#include "src/hpp/rl/data_augmenter.hpp"

int main(int argc, char* argv[]){
    auto source_path = argv[1];
    auto path_to_save_mappings = argv[2];
    bool del_old_file = argv[3] == std::string("true") ? true : false;
    int num_threads = std::stoi(argv[4]);

    PPOConfig ppo_config;

    TrainConfig train_config;
    train_config.addAugmenterEnum(EnumDataAugmenter::CGRA_MAPPING_INCREMENTAL);
    // train_config.addAugmenterEnum(EnumDataAugmenter::CURRICULUM_AWARE);     
    // train_config.setMinDFGSizeToAugment(8);  
    

    BufferConfig buffer_config;
    buffer_config.setBufferSize(1);
    buffer_config.setDegrees({});
    buffer_config.setShifts({});
    buffer_config.setFlips({});
    buffer_config.setAugmentData(true);

    CGRAConfig cgra_config;
    // cgra_config.setDimensionsList({
    //     {4,4}
    //     });

    // cgra_config.setConnectionsList({
    //     {EnumInterconnectionStyles::MESH}
    // });

    cgra_config.setFunctionalitiesList(process_functionalities("111"));
    cgra_config.setUnrollII(1);

    PathConfig path_config;
    RLConfig rl_config;
    MCTSConfig mcts_config;
    StateConfig cur_state_config;
    DFGConfig dfg_config;
    dfg_config.setPlacementApproach(EnumPlacementApproach::TOPOLOGICAL_ORDER);
    dfg_config.setGreedyType(EnumGreedyType::NONE);
    dfg_config.setScheduling(EnumScheduling::NONE);
    dfg_config.setStartTraversalFromOutput(true);
    SmartMapConfig model_config;
    MappingConfig mapping_config;
        
   
    LauncherConfig launcher_config;
    launcher_config.setEnumTaskType(EnumTaskType::TRAIN);
    launcher_config.setServerPort("0000");
    launcher_config.setSaveModel(true);
    launcher_config.setEnvironment(EnumEnvironment::SMARTMAP);
    launcher_config.setModel(EnumModel::SMARTMAP);
    launcher_config.setReplayBuffer(EnumReplayBuffer::SUPERVISED_BUFFER);
    launcher_config.setTraining(EnumTraining::SUPERVISED);
    launcher_config.setSelfMapping(EnumSelfMapping::POLICY_LOGITS);
    launcher_config.setNumThreads(num_threads);
    launcher_config.setUseBatchRequester(true);
    launcher_config.setBatchSizeRequester(8192);
    auto global_config = GlobalConfig<SmartMapConfig>(
        std::ref(ppo_config),
        std::ref(train_config),
        std::ref(rl_config),
        std::ref(path_config),
        std::ref(cgra_config),
        std::ref(dfg_config),
        std::ref(mcts_config),
        std::ref(buffer_config),
        std::ref(launcher_config),
        std::ref(model_config),
        std::ref(mapping_config),
        std::ref(cur_state_config)
    );


    std::filesystem::create_directories(path_to_save_mappings);
    auto launcher = Launcher<SmartMapState,SmartMapConfig>(std::ref(global_config));
    launcher.aug_and_save_supervised_data(source_path, path_to_save_mappings, del_old_file);    
    // launcher.finish();    
}