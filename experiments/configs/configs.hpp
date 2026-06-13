#ifndef HYPERPARAM_CONFIGS_HPP
#define HYPERPARAM_CONFIGS_HPP
#include "src/hpp/configs/ppo_config.hpp"
#include "src/hpp/configs/train_config.hpp"
#include "src/hpp/configs/cgra_config.hpp"
#include "src/hpp/configs/rl_config.hpp"
#include "src/hpp/configs/dfg_config.hpp"
#include "src/hpp/configs/launcher_config.hpp"
#include "src/hpp/configs/mapping_config.hpp"
#include "src/hpp/configs/state_config.hpp"

inline PPOConfig get_ppo_config(double initial_entropy_coef, double target_entropy_coef){
    PPOConfig ppo_config;
    ppo_config.setClipParam(0.3);
    ppo_config.setInitialEntropyCoef(initial_entropy_coef);
    ppo_config.setKlTarget(0.05);
    ppo_config.setKlCoefficient(1);
    ppo_config.setValueLossCoef(1);
    ppo_config.setTargetEntropyCoef(target_entropy_coef);
    return ppo_config;
}

inline CGRAConfig get_cgra_config(){
    CGRAConfig cgra_config;
    cgra_config.setUnrollII(0);
    cgra_config.setDimensionsList({
        {4,4},
        {8,8}
    });

    cgra_config.setConnectionsList({
        {EnumInterconnectionStyles::MESH,EnumInterconnectionStyles::TOROIDAL,EnumInterconnectionStyles::ONE_HOP, EnumInterconnectionStyles::DIAGONAL},
        {EnumInterconnectionStyles::MESH}
    });

    cgra_config.setFunctionalitiesList(process_functionalities("111"));

    return cgra_config;
}

inline CGRAConfig get_4x4_cgra_config(){
    CGRAConfig cgra_config = get_cgra_config();
    cgra_config.setDimensionsList({
        {4,4}
    });
    return cgra_config;
}

inline CGRAConfig get_8x8_cgra_config(){
    CGRAConfig cgra_config = get_cgra_config();
    cgra_config.setDimensionsList({
        {8,8}
    });
    return cgra_config;
}



inline TrainConfig get_train_config(EnumModel enum_model){
    TrainConfig train_config;
    train_config.setAddDataInterval(1);
    train_config.setBatchSizeStates(1024);
    train_config.setDevice("cuda");
    train_config.setEnumModel(enum_model);
    train_config.setEnumOptimizer(EnumOptimizer::ADAMW);
    train_config.setEvalIntervalTest(5);
    train_config.setEvalIntervalTrain(1000);
    train_config.setShuffleTrainData(true);
    train_config.setUseCurriculumLearning(true);
    train_config.setSeed(1234);
    train_config.setNumSamplesPerMapping(1);
    train_config.setUseXavieruniform(true);
    train_config.setWeightDecay(1e-5);
    train_config.setMaxGradNorm(5);
    train_config.setLrDecay(0.9);
    train_config.setLrDecaySteps(3000);
    train_config.setIterations(200);
    train_config.setEpochs(5);
    train_config.setSteps(0);
    train_config.setSaveInterval(50);
    train_config.setActorMinLr(1e-4);
    train_config.setActorLearningRate(1e-3);
    train_config.setCriticLearningRate(1e-2);
    train_config.setCriticMinLr(1e-3);
    train_config.setNumEnvs(2048);
    train_config.setEnvPriorityDecayAlpha(1);
    return train_config;
}

inline RLConfig get_rl_config(){
    RLConfig rl_config;
    rl_config.setDiscount(0.997);
    rl_config.setUseMeanNorm(false);
    rl_config.setTDSteps(-1);
    rl_config.setUnrollSteps(-1);
    rl_config.setLambda(0.95);
    rl_config.setUseGAE(true);
    rl_config.setUseMinMinus1Max1Norm(false);
    rl_config.setUseMin0Max1Norm(false);
    return rl_config;
}


inline LauncherConfig get_launcher_config(EnumEnvironment enum_environment, EnumModel enum_model, std::string port, int num_threads){
    LauncherConfig launcher_config;
    launcher_config.setEnumTaskType(EnumTaskType::TRAIN);
    launcher_config.setServerPort(port);
    launcher_config.setSaveModel(true);
    launcher_config.setEnvironment(enum_environment);
    launcher_config.setModel(enum_model);
    launcher_config.setReplayBuffer(EnumReplayBuffer::SMARTMAP_PRE_TRAINING_BUFFER);
    launcher_config.setTraining(EnumTraining::PPO);
    launcher_config.setSelfMapping(EnumSelfMapping::POLICY_LOGITS);
    launcher_config.setNumThreads(num_threads);
    return launcher_config;
}

inline SmartMapConfig get_smartmap_config(){
    SmartMapConfig model_config;
    model_config.setOutDim(32);
    model_config.setNHeads(8);
    model_config.setNegativeSlope(0.02);
    return model_config;
}
inline BufferConfig get_buffer_config(TrainConfig train_config){
    BufferConfig buffer_config;
    buffer_config.setAugmentData(false);
    buffer_config.setBufferSize(2048*train_config.getNumSamplesPerMapping()*3);
    buffer_config.setDegrees({});
    buffer_config.setShifts({});
    buffer_config.setFlips({});
    buffer_config.setPER(false);
    buffer_config.setUseISW(false);
    buffer_config.setPERAlpha(1);
    buffer_config.setAlphaReduce(1);
    buffer_config.setUseReanalyze(false);   
    buffer_config.setNormAdvantages(true);
    buffer_config.setNormRewardsPTbuffer(false);  
    buffer_config.setNormReturns(false);   
    return buffer_config;
}



inline MappingConfig get_mapping_config(){
    MappingConfig mapping_config;
    mapping_config.setFirstValidSolution(false);
    mapping_config.setUseBacktracking(false);
    return mapping_config;
    
}

inline StateConfig get_state_config(){
    StateConfig state_config;
    state_config.setOverrideBaseState(false);
    state_config.setUseCongestionEdgeMask(true);
    state_config.setUseSymmetricMask(true);
    state_config.setUseUsedPeMask(true);
    state_config.setUseRTScheduler(true);
    state_config.setEndState(EnumEndState::DEFAULT);
    return state_config;
}



inline DFGConfig get_dfg_config(){
    DFGConfig dfg_config;
    dfg_config.setGreedyType(EnumGreedyType::NONE);
    dfg_config.setPlacementApproach(EnumPlacementApproach::TOPOLOGICAL_ORDER);
    dfg_config.setScheduling(EnumScheduling::ALAP);
    dfg_config.setMaxDepth(0);
    dfg_config.setStartTraversalFromOutput(true);
    return dfg_config;
}


#endif