#ifndef CURRICULUM_PROGRESS_HPP
#define CURRICULUM_PROGRESS_HPP

#include "src/hpp/enums/enum_curriculum_progress.hpp"

template<typename ModelConfig>
class CurriculumProgress{
    private:
        EnumCurriculumProgress enum_curriculum_progress;
        int step_interval;
        double mean_rout_nodes_threshold, valid_map_rate_threshold, max_rout_nodes_threshold, min_valid_mapping_rate_threshold;
        int last_step;
        int total_curriculum_steps, cur_curriculum_step = 0;
        int max_steps_wo_progress;
        double decay_threshold_rate;
    public:
        CurriculumProgress(){}
        void set_total_curriculum_steps(int steps){
            this->total_curriculum_steps = steps;
        }
        CurriculumProgress(GlobalConfig<ModelConfig>& global_config){
            auto train_config = global_config.getTrainConfig();
            this->enum_curriculum_progress = train_config.getEnumCurriculumProgress();
            this->step_interval = train_config.getCurriculumStepSize();
            this->max_steps_wo_progress = train_config.getMaxStepsWOProgress();
            this->mean_rout_nodes_threshold = train_config.getMeanRoutingNodesThreshold();
            this->valid_map_rate_threshold = train_config.getValidMappingRateThreshold();
            this->max_rout_nodes_threshold = train_config.getMaxRoutNodesThreshold();
            this->min_valid_mapping_rate_threshold = train_config.getMinValidMappingRateThreshold();
            this->decay_threshold_rate = train_config.getDecayThresholdRate();
            this->last_step = 0;

        // #ifdef DEBUG
        // std::cout << "[DEBUG] CurriculumProgress initialized with:\n";
        // std::cout << "  enum_curriculum_progress = " << static_cast<int>(enum_curriculum_progress) << "\n";
        // std::cout << "  step_interval = " << step_interval << "\n";
        // std::cout << "  max_steps_wo_progress = " << max_steps_wo_progress << "\n";
        // std::cout << "  mean_rout_nodes_threshold = " << mean_rout_nodes_threshold << "\n";
        // std::cout << "  valid_map_rate_threshold = " << valid_map_rate_threshold << "\n";
        // std::cout << "  max_rout_nodes_threshold = " << max_rout_nodes_threshold << "\n";
        // std::cout << "  min_valid_mapping_rate_threshold = " << min_valid_mapping_rate_threshold << "\n";
        // std::cout << "  decay_threshold_rate = " << decay_threshold_rate << "\n";
        // #endif
        }

        // double update_threshold(double initital_value, double target_value, int max_steps, double cur_step){
        //     auto cosine_decay = 0.5 * (1 + cos(M_PI * static_cast<double>(cur_step) / static_cast<double>(max_steps)));
        //     return target_value + (initital_value - target_value) * cosine_decay;
        // }

        double update_threshold(double initial_value, double target_value, int cur_step, double decay_rate, bool increase){
            double threshold = 0;
            if(increase){
                threshold = std::min(initial_value*std::pow(2 - decay_rate, cur_step), target_value);
            }else{
                threshold = std::max(initial_value*std::pow(decay_rate, cur_step), target_value);
            }
            return threshold;

        }

        double get_mean_rout_nodes_threshold() const {
            return mean_rout_nodes_threshold;
        }
        double get_valid_map_rate_threshold() const {
            return valid_map_rate_threshold;
        }
        void update_all_thresholds(){
            if (this->enum_curriculum_progress == EnumCurriculumProgress::DYN_ADAPTIVE) {
                this->mean_rout_nodes_threshold = update_threshold(
                    this->mean_rout_nodes_threshold, this->max_rout_nodes_threshold, 
                    this->cur_curriculum_step, this->decay_threshold_rate, true
                );
                this->valid_map_rate_threshold = update_threshold(
                    this->valid_map_rate_threshold, this->min_valid_mapping_rate_threshold,
                    this->cur_curriculum_step, this->decay_threshold_rate, false
                );
        #ifdef DEBUG
                std::cout << "[DEBUG] step=" << cur_step 
                          << " mode=DYN_ADAPTIVE"
                          << " updated mean_rout_nodes_threshold=" << mean_rout_nodes_threshold
                          << " updated valid_map_rate_threshold=" << valid_map_rate_threshold
                          << " cur_curriculum_step=" << cur_curriculum_step << "\n";
        #endif
            }
        }
        bool should_increase_progress(int cur_step, double mean_rout_nodes, double valid_map_rate){
            // if (cur_step == 1) return false; 
        
            if (this->enum_curriculum_progress == EnumCurriculumProgress::FIXED) {
                bool should_increase_progress = ((cur_step + 1) % this->step_interval == 0);
        #ifdef DEBUG
                std::cout << "[DEBUG] step=" << cur_step 
                          << " mode=FIXED"
                          << " step_interval=" << step_interval
                          << " should_increase_progress=" << should_increase_progress
                          << " cur_curriculum_step=" << cur_curriculum_step << "\n";
        #endif
                if (should_increase_progress) {
                    this->cur_curriculum_step++;
                    this->update_all_thresholds();
                };
                return should_increase_progress;
            }
        
           
        
            bool condition1 = (mean_rout_nodes <= mean_rout_nodes_threshold) && (valid_map_rate >= valid_map_rate_threshold);
            bool condition2 = ((cur_step - this->last_step) >= this->max_steps_wo_progress);
        
        #ifdef DEBUG
            std::cout << "[DEBUG] step=" << cur_step 
                      << " mean_rout_nodes=" << mean_rout_nodes 
                      << " threshold=" << mean_rout_nodes_threshold
                      << " valid_map_rate=" << valid_map_rate 
                      << " threshold=" << valid_map_rate_threshold
                      << " condition1=" << condition1
                      << " condition2=" << condition2
                      << " last_step=" << last_step
                      << " cur_curriculum_step=" << cur_curriculum_step << "\n";
        #endif
        
            if (condition1 || condition2) {
                this->last_step = cur_step;
                this->cur_curriculum_step++;
        #ifdef DEBUG
                std::cout << "[DEBUG] >>> Progress increased! cur_curriculum_step=" 
                          << cur_curriculum_step << " at step=" << cur_step << "\n";
        #endif
                this->update_all_thresholds();

                return true;
            } 
        
            return false;
            
        }
        
    template<class Archive>
    void serialize(Archive& ar) {
        ar(last_step, enum_curriculum_progress, step_interval, mean_rout_nodes_threshold, 
            valid_map_rate_threshold, max_rout_nodes_threshold, min_valid_mapping_rate_threshold, max_steps_wo_progress,
        total_curriculum_steps, cur_curriculum_step,decay_threshold_rate);
    }
};

#endif