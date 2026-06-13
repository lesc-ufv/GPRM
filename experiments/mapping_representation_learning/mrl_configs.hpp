#ifndef MRL_CONFIGS_HPP
#define MRL_CONFIGS_HPP
#include "src/hpp/configs/ppo_config.hpp"
#include "src/hpp/configs/train_config.hpp"
#include "src/hpp/configs/cgra_config.hpp"
#include "src/hpp/configs/rl_config.hpp"
#include "src/hpp/configs/dfg_config.hpp"
#include "src/hpp/configs/launcher_config.hpp"
#include "src/hpp/configs/mapping_config.hpp"
#include "src/hpp/configs/state_config.hpp"

using json = nlohmann::json;

GPRMConfig get_gprm_mrl_config(){
    GPRMConfig model_config;

    int out_dim = 64;
    int feature_out_dim = 64;
    int n_heads = 8;
    int transformer_layers = 12;

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
    return model_config;
}
inline CGRAConfig get_4x4_cgra_mrl_config(){
    CGRAConfig cgra_config;
    
    cgra_config.setUnrollII(0);

    cgra_config.setDimensionsList({
        {4,4}
        });

    cgra_config.setConnectionsList({
        {EnumInterconnectionStyles::MESH,EnumInterconnectionStyles::TOROIDAL,EnumInterconnectionStyles::ONE_HOP, EnumInterconnectionStyles::DIAGONAL},
        {EnumInterconnectionStyles::MESH}
    });

    cgra_config.setFunctionalitiesList(process_functionalities("111"));
    return cgra_config;
}

inline CGRAConfig get_4x4_cgra_mrl_config_ii_2(){
    CGRAConfig cgra_config = get_4x4_cgra_mrl_config();
    cgra_config.setUnrollII(1);
    return cgra_config;
}


inline CGRAConfig get_4x4_cgra_mrl_config_transmap(){
    CGRAConfig cgra_config = get_4x4_cgra_mrl_config();
    cgra_config.setFunctionalitiesList(process_functionalities("00111111"));
    return cgra_config;
}



inline TrainConfig get_train_mrl_config(EnumModel enum_model, std::string path_to_load_sup_data){
    TrainConfig train_config;
    int total_steps = 200000;
    train_config.setEpochs(1);
    train_config.setIterations(1);
    train_config.setSteps(total_steps);
    train_config.setBatchSizeStates(512);
    train_config.setBatchSizeMappings(1);
    train_config.setActorLearningRate(1e-3);
    train_config.setActorMinLr(1e-5);
    train_config.setCriticLearningRate(0.001);
    train_config.setCriticMinLr(0.001);
    train_config.setMaxGradNorm(1000);
    train_config.setLrDecay(0.95);
    train_config.setLrDecaySteps(0);
    train_config.setDevice("cuda");
    train_config.setSeed(1234);
    train_config.setEvalIntervalTrain(15000000);
    train_config.setEvalIntervalTest(10000);
    train_config.setSaveInterval(10000);
    train_config.setAddDataInterval(0);
    train_config.setUseCurriculumLearning(false);
    train_config.setShuffleTrainData(false);
    train_config.setWeightDecay(0.0001);
    train_config.setUseXavieruniform(true);
    train_config.setNumSamplesPerMapping(1);
    train_config.setInitialIter(0);
    train_config.setEnumModel(enum_model);
    train_config.setEnumOptimizer(EnumOptimizer::ADAMW);
    train_config.setPathToLoadSupervisedData(path_to_load_sup_data);
    train_config.setWarmupSteps(5000);
    train_config.setLRScheduler("cosine");

    // train_config.setEnumCurriculum(EnumCurriculum::DFG_SIZE);
    // train_config.setEnumCurriculumProgress(EnumCurriculumProgress::FIXED);
    train_config.setCurriculumStepSize(1);
    train_config.setNumEnvs(-1);
    train_config.setEnvPriorityDecayAlpha(1.0f);
    train_config.setStepToStartEval(0);
    train_config.setEvalFirstStep(true);
    train_config.setLambdaHumanComplexity(1);
    train_config.setLambdaComplexity(1);
    train_config.setEmaAlpha(1);
    train_config.setMaxStepsWOProgress(200);
    train_config.setMinDFGSizeToAugment(1);
    train_config.setUpdateModelInterval(1);

    // train_config.addAugmenterEnum(EnumDataAugmenter::CGRA_MAPPING_INCREMENTAL);
    // train_config.setUseSequenceCurriculum();

    return train_config;
}

inline RLConfig get_rl_mrl_config(){
    RLConfig rl_config;
    rl_config.setDiscount(1);
    rl_config.setUseMeanNorm(false);
    rl_config.setTDSteps(-1);
    rl_config.setUnrollSteps(-1);
    rl_config.setLambda(1);
    rl_config.setUseGAE(false);
    rl_config.setUseMinMinus1Max1Norm(false);
    rl_config.setUseMin0Max1Norm(false);
    rl_config.setOverrideBaseState(false);
    rl_config.setMaskInvalidPEs(false);
    return rl_config;
}


inline LauncherConfig get_launcher_mrl_config( EnumModel enum_model, std::string port, int num_threads){
    LauncherConfig launcher_config;
    launcher_config.setEnumTaskType(EnumTaskType::TRAIN);
    launcher_config.setServerPort(port);
    launcher_config.setSaveModel(true);
    launcher_config.setEnvironment(EnumEnvironment::SMARTMAP);
    launcher_config.setModel(enum_model);
    launcher_config.setReplayBuffer(EnumReplayBuffer::SUPERVISED_BUFFER);
    launcher_config.setTraining(EnumTraining::SUPERVISED);
    launcher_config.setSelfMapping(EnumSelfMapping::POLICY_LOGITS);
    launcher_config.setNumThreads(num_threads);
    launcher_config.setUseBatchRequester(true);
    launcher_config.setBatchSizeRequester(8192);
    return launcher_config;
}

inline BufferConfig get_buffer_mrl_config(TrainConfig train_config){
    BufferConfig buffer_config;
    buffer_config.setBufferSize(500000);
    buffer_config.setClearBufferAfterTraining(false);
    // buffer_config.setDegrees({90,180,270});
    // buffer_config.setShifts({{1,0},{0,1},{-1,0},{0,-1}});
    // buffer_config.setFlips({{1,0},{0,1}});
    buffer_config.setAugmentData(false);
    buffer_config.setPER(false);
    buffer_config.setUseISW(false);
    buffer_config.setPERAlpha(1);
    buffer_config.setAlphaReduce(1);
    buffer_config.setUseReanalyze(false);   
    buffer_config.setNormAdvantages(false);
    buffer_config.setNormRewardsPTbuffer(false);  
    buffer_config.setNormReturns(false);   
    return buffer_config;
}


inline MappingConfig get_mapping_mrl_config(){
    MappingConfig mapping_config;
    mapping_config.setFirstValidSolution(false);
    mapping_config.setUseBacktracking(false);
    return mapping_config;
    
}

inline StateConfig get_state_mrl_config(){
    StateConfig state_config;
    state_config.setOverrideBaseState(false);
    state_config.setUseCongestionEdgeMask(false);
    state_config.setUseSymmetricMask(false);
    state_config.setUseUsedPeMask(false);
    state_config.setUseRTScheduler(true);
    state_config.setEndState(EnumEndState::DEFAULT);
    state_config.setUseSpatioTemporalInput(true);
    state_config.setUseSyncRouting(false);
    return state_config;
}



inline DFGConfig get_dfg_mrl_config(){
    DFGConfig dfg_config;
    dfg_config.setGreedyType(EnumGreedyType::NONE);
    dfg_config.setPlacementApproach(EnumPlacementApproach::TOPOLOGICAL_ORDER);
    dfg_config.setScheduling(EnumScheduling::TIGHT);
    dfg_config.setMaxDepth(2);
    dfg_config.setStartTraversalFromOutput(true);
    dfg_config.setAlphaComplexity(1);
    dfg_config.setBetaComplexity(1);
    // dfg_config.setGammaComplexity(1);
    return dfg_config;
}

template<typename StateT, typename ModelConfig>
int run_mrl_experiment(char* argv[],
    std::string tb_path, 
    bool use_spatio_temporal,
    EnumModel model,
    ModelConfig model_config, 
    EnumPlacementApproach placement_approach = EnumPlacementApproach::TOPOLOGICAL_ORDER,
    std::optional<DFGConfig> dfg_config = std::nullopt,
    std::optional<StateConfig> state_config = std::nullopt,
    std::optional<bool> unroll_II_1 = std::make_optional(false),
    std::optional<std::string> aug_map = std::nullopt,
    std::optional<std::string> cut_dfg = std::nullopt

){

    auto global_config = get_configurations( argv, tb_path, use_spatio_temporal, model, model_config, placement_approach, dfg_config,state_config,unroll_II_1);
    auto& ppo_config = global_config.getPPOConfig();
    auto& train_config = global_config.getTrainConfig();
    auto& path_config = global_config.getPathConfig();

    if(aug_map.has_value()){
        if (aug_map.value() == "none"){
            train_config.setAugmenterEnums({});
        }else if(aug_map.value() == "map_comb"){
            train_config.setAugmenterEnums({EnumDataAugmenter::CGRA_MAPPING_COMBINATORIAL});
        }else{
            throw std::invalid_argument("Invalid augmenter type");
        }
    }

    if(cut_dfg.has_value()){
        assert(cut_dfg.value() == "dfg_cutting");
        train_config.setAugmenterEnums({EnumDataAugmenter::CURRICULUM_AWARE});
        train_config.setMinDFGSizeToAugment(8);
    }
    // auto& cgra_config = global_config.getCGRAConfig();

    auto train_finished = load_checkpoint_if_exists(tb_path,path_config,train_config,ppo_config);

    if(train_finished){
        std::cout << "[INFO] Training already finished, skipping this configuration \n\n";
        return 0;
    }
    path_config.setTensorboardLogPath(tb_path);
    path_config.setSaveCheckpointPath(tb_path);
    auto eval_path = path_config.getEvalDfgsDotPath();
    auto train_path = path_config.getTrainDfgsDotPath();
    auto port = global_config.getLauncherConfig().getServerPort();
    std::string path_to_load_sup_data = train_config.getPathToLoadSupervisedData();

  

    auto eval_initial_states = save_data_by_config<StateT>( train_path, eval_path, path_to_load_sup_data, port, global_config);
    auto launcher = Launcher<StateT,ModelConfig>(std::ref(global_config));

    launcher.run_config_server();
    launcher.train_supervised(eval_initial_states);
    launcher.finish();
    return 0;
}

template<typename ModelConfig>
GlobalConfig<ModelConfig> get_configurations(char* argv[], 
    std::string tb_path, 
    bool use_spatio_temporal,
    EnumModel model,
    ModelConfig model_config, 
    EnumPlacementApproach placement_approach,
    std::optional<DFGConfig> dfg_config = std::nullopt,
    std::optional<StateConfig> state_config = std::nullopt,
    std::optional<bool> unroll_II_1 = std::make_optional(false)
    ) {
    
    PPOConfig ppo_config;
    
    std::string train_path = argv[1];
    std::string eval_path = argv[2];
    std::string port = argv[3];
    int num_threads = std::stoi(argv[4]);
    std::string path_to_load_sup_data = argv[5];

    TrainConfig train_config = get_train_mrl_config(model, path_to_load_sup_data);

    PathConfig path_config;
    path_config.setTrainDfgsDotPath(train_path);
    path_config.setEvalDfgsDotPath(eval_path);
    CGRAConfig cgra_config;
    if(unroll_II_1.has_value() && unroll_II_1.value()){
        cgra_config = get_4x4_cgra_mrl_config_ii_2();
    } else if (EnumModel::TRANSMAP == model){
        cgra_config = get_4x4_cgra_mrl_config_transmap();
    }else{
        cgra_config = get_4x4_cgra_mrl_config();
    }

    RLConfig rl_config = get_rl_mrl_config();
    DFGConfig cur_dfg_config;
    if(dfg_config.has_value()){
        cur_dfg_config = dfg_config.value();
    }else{
        cur_dfg_config = get_dfg_mrl_config();
        cur_dfg_config.setPlacementApproach(placement_approach);
    }
    
    MCTSConfig mcts_config;
    StateConfig cur_state_config;

    if(state_config.has_value()){
        cur_state_config = state_config.value();
    }else{
        cur_state_config = get_state_mrl_config();
        cur_state_config.setUseSpatioTemporalInput(use_spatio_temporal);
    }    

    MappingConfig mapping_config = get_mapping_mrl_config();
    
    BufferConfig buffer_config = get_buffer_mrl_config(train_config);
   
    LauncherConfig launcher_config = get_launcher_mrl_config(model, port, num_threads);

    path_config.setTensorboardLogPath(tb_path);
    path_config.setSaveCheckpointPath(tb_path);

    return GlobalConfig<ModelConfig>(
        std::ref(ppo_config),
        std::ref(train_config),
        std::ref(rl_config),
        std::ref(path_config),
        std::ref(cgra_config),
        std::ref(cur_dfg_config),
        std::ref(mcts_config),
        std::ref(buffer_config),
        std::ref(launcher_config),
        std::ref(model_config),
        std::ref(mapping_config),
        std::ref(cur_state_config)
    );
}


#endif