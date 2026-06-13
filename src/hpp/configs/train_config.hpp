#ifndef TRAIN_CONFIG_HPP
#define TRAIN_CONFIG_HPP

#include <string>
#include <iostream>
#include <sstream>
#include "src/hpp/enums/enum_environment.hpp"
#include "src/hpp/enums/enum_model.hpp"
#include "src/hpp/enums/enum_self_mapping.hpp"
#include "src/hpp/enums/enum_replay_buffer.hpp"
#include "src/hpp/enums/enum_optimizer.hpp"
#include "src/hpp/enums/enum_curriculum.hpp"
#include "src/hpp/enums/enum_curriculum_progress.hpp"
#include "src/hpp/enums/enum_data_augmenter.hpp"
#include "src/hpp/enums/enum_env_sampler.hpp"
#include <unordered_set>
class TrainConfig {
private:
    int epochs;
    int iterations;
    int steps;
    int batch_size_states;
    int batch_size_mappings;
    double actor_learning_rate;
    double critic_learning_rate;
    double actor_min_lr;
    double critic_min_lr;
    double max_grad_norm;
    double lr_decay;
    double lr_decay_steps;
    std::string device;
    int seed;
    int eval_interval_train;
    int eval_interval_test;
    int save_interval;
    int add_data_interval;
    bool use_curriculum_learning;
    bool shuffle_train_data;
    double weight_decay;
    bool use_xavier_uniform;
    int num_samples_per_mapping;
    int initial_iter;
    EnumModel enum_model;
    EnumOptimizer enum_optimizer;
    int num_envs;
    float env_priority_decay_alpha;
    int step_to_start_eval;
    bool eval_first_step;
    EnumCurriculum enum_curriculum;
    EnumCurriculumProgress enum_curriculum_progress;
    int curriculum_step_size;
    double mean_routing_nodes_threshold;
    double valid_mapping_rate_threshold;
    int max_steps_wo_progress;
    double lambda_human_complexity;
    double lambda_complexity;
    double ema_alpha;
    std::unordered_set<EnumDataAugmenter> augmenter_enums;
    int min_dfg_size_to_augment;
    int update_model_interval;
    bool use_sequence_curriculum;
    int beam_width;
    int use_reward_beam_search;

    int supervised_train_interval;
    std::string path_to_load_supervised_data;
    int warmup_steps;
    double max_rout_nodes_threshold;
    double min_valid_mapping_rate_threshold;
    double decay_threshold_rate;
    bool use_efficient_sampling;
    double rout_node_threshold_sampling;
    int min_good_samples;
    double good_sample_rate;
    EnumEnvSampler env_sampler;
    double c_env_sampler;
    double alpha_return;
    std::string load_model_path;
    double plr_alpha, plr_beta;
    int buffer_size_to_start_training;
    double label_smoothing = 0;
    std::string lr_scheduler = "exponential";
    int num_dtbs = 1;
    int mcts_train_interval = 1;
    int ppo_train_interval = 1;
    int rl_train_start = 0;
    bool add_all_curriculum_states = false;
    bool use_chunked_supervised_dataset = false;
    bool skip_buffer_loading  = false;

public:
    TrainConfig()
        : epochs(0), iterations(0), steps(0), batch_size_states(0), batch_size_mappings(0),
          actor_learning_rate(0), critic_learning_rate(0), actor_min_lr(0), critic_min_lr(0),max_grad_norm(0), lr_decay(0),  lr_decay_steps(0),
          device("cpu"), seed(0), eval_interval_train(0), eval_interval_test(0), save_interval(0),
          add_data_interval(0), use_curriculum_learning(false), shuffle_train_data(false),
          weight_decay(0), use_xavier_uniform(false), num_samples_per_mapping(1),
          initial_iter(0), enum_model(EnumModel::DEFAULT), enum_optimizer(EnumOptimizer::DEFAULT), num_envs(-1), env_priority_decay_alpha(1),
          step_to_start_eval(0), eval_first_step(true), enum_curriculum(EnumCurriculum::DFG_SIZE), enum_curriculum_progress(EnumCurriculumProgress::FIXED), 
          curriculum_step_size(1), max_steps_wo_progress(0), lambda_human_complexity(0),lambda_complexity(0), ema_alpha(1),augmenter_enums(),
          min_dfg_size_to_augment(0), update_model_interval(1), use_sequence_curriculum(true), beam_width(1),
          supervised_train_interval(1), path_to_load_supervised_data(""), warmup_steps(0), max_rout_nodes_threshold(1),min_valid_mapping_rate_threshold(1),
          decay_threshold_rate(1), use_efficient_sampling(false), rout_node_threshold_sampling(1), min_good_samples(1), good_sample_rate(0.1),
          env_sampler(EnumEnvSampler::DEFAULT), c_env_sampler(0), alpha_return(1), plr_alpha(1), plr_beta(1), buffer_size_to_start_training(1){}
    
    std::string getLRScheduler() const {
        return lr_scheduler;
    }

    bool getUseRewardBeamSearch(){
        return use_reward_beam_search;
    }

    void setUseRewardBeamSearch(bool use_reward_beam_search){
        this->use_reward_beam_search = use_reward_beam_search;
    }

    bool getSkipLoadingBuffer() const {
        return skip_buffer_loading;
    }

    void setSkipLoadingBuffer(bool value) {
        skip_buffer_loading = value;
    }
        

    bool getUseChunkedSupervisedDataset() const {
        return use_chunked_supervised_dataset;
    }

    void setUseChunkedSupervisedDataset(bool value) {
        use_chunked_supervised_dataset = value;
    }
    bool getAddAllCurriculumStates() const {
        return add_all_curriculum_states;
    }

    void setAddAllCurriculumStates(bool value) {
        add_all_curriculum_states = value;
    }

    int getNumDTBsToSampleFrom() const {
        return num_dtbs;
    }

    void setNumDTBsToSampleFrom(int value) {
        num_dtbs = value;
    }
    
    void setLRScheduler(const std::string& value) {
        lr_scheduler = value;
    }
    double getLabelSmoothing() const {
        return label_smoothing;
    }
    void setLabelSmoothing(double value) {
        label_smoothing = value;
    }
    bool getUseSequenceCurriculum() const {
        return use_sequence_curriculum;
    }

    int getBufferSizeToStartTraining() const {
        return buffer_size_to_start_training;
    }

    void setBufferSizeToStartTraining(int value) {
        buffer_size_to_start_training = value;
    }

   double getPLRAlpha() const {
        return plr_alpha;
    }

    void setPLRAlpha(double value) {
        plr_alpha = value;
    }

    double getPLRBeta() const {
        return plr_beta;
    }

    void setPLRBeta(double value) {
        plr_beta = value;
    }

    double getAlphaReturn() const {
        return alpha_return;
    }

    void setAlphaReturn(double value) {
        alpha_return = value;
    }

    EnumEnvSampler getEnvSampler() const {
        return env_sampler;
    }
    void setEnvSampler(EnumEnvSampler value) {
        env_sampler = value;
    }
    double getCEnvSampler() const {
        return c_env_sampler;
    }
    void setCEnvSampler(double value) {
        c_env_sampler = value;
    }

    bool getUseEfficientSampling() const {
        return use_efficient_sampling;
    }
    void setUseEfficientSampling(bool value) {
        use_efficient_sampling = value;
    }

    double getRoutNodeThresholdSampling() const {
        return rout_node_threshold_sampling;
    }

    void setRoutNodeThresholdSampling(double value) {
        rout_node_threshold_sampling = value;
    }

    int getMinGoodSamples() const {
        return min_good_samples;
    }
    void setMinGoodSamples(int value) {
        min_good_samples = value;
    }
    double getGoodSampleRate() const {
        return good_sample_rate;
    }
    void setGoodSampleRate(double value) {
        good_sample_rate = value;
    }


    double getDecayThresholdRate() const {
        return decay_threshold_rate;
    }
    void setDecayThresholdRate(double value) {
        decay_threshold_rate = value;
    }

    int getWarmupSteps() const {
        return warmup_steps;
    }
    void setWarmupSteps(int value) {
        warmup_steps = value;
    }

    double getMaxRoutNodesThreshold() const {
        return max_rout_nodes_threshold;
    }
    void setMaxRoutNodesThreshold(double value) {
        max_rout_nodes_threshold = value;
    }
    double getMinValidMappingRateThreshold() const {
        return min_valid_mapping_rate_threshold;
    }
    void setMinValidMappingRateThreshold(double value) {
        min_valid_mapping_rate_threshold = value;
    }
    std::string getPathToLoadSupervisedData() const {
        return this->path_to_load_supervised_data;
    }

    void setPathToLoadSupervisedData(std::string path_to_load_supervised_data){
        this->path_to_load_supervised_data = path_to_load_supervised_data;
    }

    void setUseSequenceCurriculum(bool value){
        use_sequence_curriculum = value;
    }

    int getSupervisedTrainInterval(){
        return supervised_train_interval;
    }

    void setSupervisedTrainInterval(int value){
        supervised_train_interval = value;
    }

    int getMCTSTrainInterval(){
        return mcts_train_interval;
    }

    void setMCTSTrainInterval(int value){
        mcts_train_interval = value;
    }

    int getPPOTrainInterval(){
        return ppo_train_interval;
    }

    void setPPOTrainInterval(int value){
        ppo_train_interval = value;
    }


    int getRLTrainStart(){
        return rl_train_start;
    }

    void setRLTrainStart(int value){
        rl_train_start = value;
    }
    
    double getMeanRoutingNodesThreshold() const { return mean_routing_nodes_threshold; }
    double getValidMappingRateThreshold() const { return valid_mapping_rate_threshold; }
    void setMeanRoutingNodesThreshold(double value) { mean_routing_nodes_threshold = value; }
    void setValidMappingRateThreshold(double value) { valid_mapping_rate_threshold = value; }
    int getEpochs() const { return epochs; }
    int getBeamWidth() const { return beam_width; }
    void setBeamWidth(int value){beam_width = value;}
    int getUpdateModelInterval() const {return update_model_interval;}
    void setUpdateModelInterval(int value) { update_model_interval = value;} 
    int getIterations() const { return iterations; }
    int getSteps() const { return steps; }
    int getMinDFGSizeToAugment() const { return min_dfg_size_to_augment; }
    void setMinDFGSizeToAugment(int value) { min_dfg_size_to_augment = value; }
    double getEmaAlpha() const { return ema_alpha; }
    void setEmaAlpha(double value) { ema_alpha = value; }
    const EnumModel& getEnumModel() const { return enum_model; }
    const EnumOptimizer& getEnumOptimizer() const { return enum_optimizer; }
    const EnumCurriculum& getEnumCurriculum() const {return enum_curriculum;}
    const EnumCurriculumProgress& getEnumCurriculumProgress() const {return enum_curriculum_progress;}
    double getLambdaHumanComplexity() const { return lambda_human_complexity; }
    const std::unordered_set<EnumDataAugmenter>& getAugmenterEnums() const {
        return augmenter_enums;
    }

    bool augment_is_combinatorial(){
        return augmenter_enums.find(EnumDataAugmenter::CGRA_MAPPING_COMBINATORIAL) != augmenter_enums.end();
    }
    bool augment_is_incremental(){
        return augmenter_enums.find(EnumDataAugmenter::CGRA_MAPPING_INCREMENTAL) != augmenter_enums.end();
    }
    
    bool augment_mapping(){
        return augmenter_enums.find(EnumDataAugmenter::CGRA_MAPPING_INCREMENTAL) != augmenter_enums.end() 
            || augmenter_enums.find(EnumDataAugmenter::CGRA_MAPPING_COMBINATORIAL) != augmenter_enums.end();
    }

    bool augment_dfgs_guided_by_curriculum(){
        return augmenter_enums.find(EnumDataAugmenter::CURRICULUM_AWARE) != augmenter_enums.end();
    }
    
    void setAugmenterEnums(std::unordered_set<EnumDataAugmenter> value) {
        augmenter_enums = value;
    }
    
    void addAugmenterEnum(EnumDataAugmenter augmenter) {
        augmenter_enums.insert(augmenter);
    }
    void setLambdaHumanComplexity(double value) { lambda_human_complexity = value; }
    double getLambdaComplexity() const { return lambda_complexity; }
    void setLambdaComplexity(double value) { lambda_complexity = value; }
    int getBatchSizeStates() const { return batch_size_states; }
    int getCurriculumStepSize() const { return curriculum_step_size;}
    int getBatchSizeMappings() const { return batch_size_mappings; }
    double getActorLearningRate() const { return actor_learning_rate; }
    double getActorMinLr() const { return actor_min_lr; }
    double getCriticLearningRate() const { return critic_learning_rate; }
    double getCriticMinLr() const { return critic_min_lr; }
    double getMaxGradNorm() const { return max_grad_norm; }
    double getLrDecay() const { return lr_decay; }
    double getWeightDecay() const { return weight_decay; }
    double getLrDecaySteps() const { return lr_decay_steps; }
    double getEnvPriorityDecayAlpha() const { return env_priority_decay_alpha; }
    int getSeed() const { return seed; }
    int getMaxStepsWOProgress() const { return max_steps_wo_progress; }
    int getEvalIntervalTrain() const { return eval_interval_train; }
    int getSaveInterval() const { return save_interval; }
    int getEvalIntervalTest() const { return eval_interval_test; }
    int getAddDataInterval() const { return add_data_interval; }
    bool getUseCurriculumLearning() const { return use_curriculum_learning; }
    bool getUseXavieruniform() const { return use_xavier_uniform; }
    bool getShuffleTrainData() const { return shuffle_train_data; }
    std::string getDevice() const { return device; }
    int getNumSamplesPerMapping() const { return num_samples_per_mapping; }
    int getInitialIter() const { return initial_iter; }
    int getNumEnvs() const { return num_envs; }
    int getStepToStartEval() const { return step_to_start_eval; }
    bool getEvalFirstStep() const {return eval_first_step; }

    void setNumEnvs(int value) { num_envs = value; }
    void setMaxStepsWOProgress(int value) { max_steps_wo_progress = value; }
    void setCurriculumStepSize(int value) { curriculum_step_size = value; }
    void setEpochs(int value) { epochs = value; }
    void setIterations(int value) { iterations = value; }
    void setSteps(int value) { steps = value; }
    void setBatchSizeStates(int value) { batch_size_states = value; }
    void setBatchSizeMappings(int value) { batch_size_mappings = value; }
    void setActorLearningRate(double value) { actor_learning_rate = value; }
    void setActorMinLr(double value) { actor_min_lr = value; }
    void setCriticLearningRate(double value) { critic_learning_rate = value; }
    void setCriticMinLr(double value) { critic_min_lr = value; }

    void setMaxGradNorm(double value) { max_grad_norm = value; }
    void setEnvPriorityDecayAlpha(float value) { env_priority_decay_alpha = value; }
    void setStepToStartEval(int value) { step_to_start_eval = value; }

    void setLrDecay(double value) { lr_decay = value; }
    void setLrDecaySteps(double value) { lr_decay_steps = value; }
    void setSeed(int value) { seed = value; }
    void setEvalIntervalTrain(int value) { eval_interval_train = value; }
    void setEvalIntervalTest(int value) { eval_interval_test = value; }
    void setSaveInterval(int value) { save_interval = value; }
    void setAddDataInterval(int value) { add_data_interval = value; }
    void setUseCurriculumLearning(bool value) { use_curriculum_learning = value; }
    void setEnumModel(EnumModel model) { enum_model = model; }
    void setEnumOptimizer(EnumOptimizer value) { enum_optimizer = value; }
    void setShuffleTrainData(bool value) { shuffle_train_data = value; }
    void setEnumCurriculum(EnumCurriculum value) { enum_curriculum = value; }
    void setEnumCurriculumProgress(EnumCurriculumProgress value) { enum_curriculum_progress = value; }

    void setDevice(const std::string& value) { device = value; }
    void setWeightDecay(double value) { weight_decay = value; }
    void setUseXavieruniform(bool value) { use_xavier_uniform = value; }
    void setNumSamplesPerMapping(int value) { num_samples_per_mapping = value; }
    void setInitialIter(int value) { initial_iter = value; }
    void setEvalFirstStep(bool value) { eval_first_step = value; }

    void print_ss(std::stringstream& ss) const {
        ss << "\nTrain Configuration:\n"
           << "\tEpochs: " << epochs << "\n"
           << "\tIterations: " << iterations << "\n"
           << "\tSteps: " << steps << "\n"
           << "\tBatch Size (States): " << batch_size_states << "\n"
           << "\tBatch Size (Mappings): " << batch_size_mappings << "\n"
           << "\tActor Learning Rate: " << actor_learning_rate << "\n"
           << "\tActor Min LR: " << actor_min_lr << "\n"
           << "\tCritic Learning Rate: " << critic_learning_rate << "\n"
           << "\tCritic Min LR: " << critic_min_lr << "\n"
           << "\tMax Grad Norm: " << max_grad_norm << "\n"
           << "\tLR Decay: " << lr_decay << "\n"
           << "\tLR Decay Steps: " << lr_decay_steps << "\n"
           << "\tWeight Decay: " << weight_decay << "\n"
           << "\tDevice: " << device << "\n"
           << "\tModel: " << get_model_name_by_enum(enum_model) << "\n"
           << "\tOptimizer: " << enum_optimizer_to_string(enum_optimizer) << "\n"
           << "\tCurriculum: " << enum_curriculum_to_string(enum_curriculum) << "\n"
           << "\tCurriculum Progress: " << enum_curriculum_progress_to_string(enum_curriculum_progress) << "\n"
           << "\tCurriculum Step Size: " << curriculum_step_size << "\n"
           << "\tNum Envs: " << num_envs << "\n"
           << "\tSeed: " << seed << "\n"
           << "\tEval Interval Train: " << eval_interval_train << "\n"
           << "\tEval Interval Test: " << eval_interval_test << "\n"
           << "\tSave Interval: " << save_interval << "\n"
           << "\tAdd Data Interval: " << add_data_interval << "\n"
           << "\tCurriculum Learning: " << (use_curriculum_learning ? "true" : "false") << "\n"
           << "\tShuffle Train Data: " << (shuffle_train_data ? "true" : "false") << "\n"
           << "\tUse Xavier Uniform: " << (use_xavier_uniform ? "true" : "false") << "\n"
           << "\tSamples per Mapping: " << num_samples_per_mapping << "\n"
           << "\tInitial Iter: " << initial_iter << "\n"
           << "\tStep to Start Eval: " << step_to_start_eval << "\n"
           << "\tEval First Step: " << (eval_first_step ? "true" : "false") << "\n"
           << "\tMax Steps Without Progress: " << max_steps_wo_progress << "\n"
           << "\tMean Routing Nodes Threshold: " << mean_routing_nodes_threshold << "\n"
           << "\tValid Mapping Rate Threshold: " << valid_mapping_rate_threshold << "\n"
           << "\tMax Routing Nodes Threshold: " << max_rout_nodes_threshold << "\n"
           << "\tMin Valid Mapping Rate Threshold: " << min_valid_mapping_rate_threshold << "\n"
           << "\tDecay Threshold Rate: " << decay_threshold_rate << "\n"
           << "\tLambda Human Complexity: " << lambda_human_complexity << "\n"
           << "\tLambda Complexity: " << lambda_complexity << "\n"
           << "\tEMA alpha: " << ema_alpha << "\n"
           << "\tWarmup Steps: " << warmup_steps << "\n"
           << "\tMin DFG Size to Augment: " << min_dfg_size_to_augment << "\n"
           << "\tUpdate Model Interval: " << update_model_interval << "\n"
           << "\tUse Sequence Curriculum: " << (use_sequence_curriculum ? "true" : "false") << "\n"
           << "\tBeam Width: " << beam_width << "\n"
           << "\tSupervised Train Interval: " << supervised_train_interval << "\n"
           << "\tPath to Load Supervised Data: " << path_to_load_supervised_data << "\n"
           << "\tUse Efficient Sampling: " << (use_efficient_sampling ? "true" : "false") << "\n"
           << "\tRouting Node Threshold Sampling: " << rout_node_threshold_sampling << "\n"
           << "\tMin Good Samples: " << min_good_samples << "\n"
           << "\tGood Sample Rate: " << good_sample_rate << "\n"
           << "\tEnv Sampler: " << enum_env_sampler_to_string(env_sampler) << "\n"
           << "\t Alpha Return: " << alpha_return << "\n"
           << "\tC Env Sampler: " << c_env_sampler << "\n";

    
        ss << "\tData Augmenters: ";
        if (augmenter_enums.empty()) {
            ss << "None";
        } else {
            for (const auto& aug : augmenter_enums) {
                ss << enum_data_augmenter_to_string(aug) << " ";
            }
        }
        ss << "\n";
    }
    
};

#endif
