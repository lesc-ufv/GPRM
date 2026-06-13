#ifndef ENV_SAMPLER_HPP
#define ENV_SAMPLER_HPP

#include <vector>
#include <unordered_map>
#include <random>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <torch/torch.h>
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/enums/enum_env_sampler.hpp"
#include "src/hpp/utils/min_max_stats.hpp"
#include "src/hpp/rl/mapping_history.hpp"
#include "src/hpp/utils/util_train.hpp"

template<typename StateT, typename ModelConfig>
class EnvSampler {
private:
    struct StateInfo {
        int state_id;
        std::shared_ptr<StateT> state_ptr;
        bool is_good = false;            
        size_t index_in_current_vec = 0; 
    };

    std::unordered_map<int, StateInfo> state_map; 
    std::vector<int> good_state_vec;             
    std::vector<int> bad_state_vec;              

    double good_state_ratio = 0.0;
    int min_good_states = 0;
    std::unordered_map<int, double> state_id_ema_rout_nodes;
    std::unordered_map<int, double> state_id_valid_map_rate;
    torch::Tensor ema_return;     
    torch::Tensor visit_count;    
    torch::Tensor last_return; 
    torch::Tensor scores;
    int total_visit_count = 0;

    double ema_alpha = 0.99;
    double alpha_return;
    int seed = 0;
    double rout_nodes_threshold = 0.0;
    std::mt19937 gen;
    double c = 1.0;
    EnumEnvSampler env_sampler;
    std::unordered_map<int, MinMaxStats> state_return_stats;
    double lambda, discount;
    double plr_alpha = 0; 
    double plr_beta = 1;

public:

    std::vector<std::shared_ptr<StateT>> add_data_if_exists(std::vector<std::shared_ptr<StateT>>& initial_states){
        std::vector<std::shared_ptr<StateT>> cleaned_states;
        cleaned_states.reserve(initial_states.size());
        for(auto& state_ptr : initial_states){
            int state_id = state_ptr->get_id();
            if(state_map.find(state_id) != state_map.end()){
                this->add_old_sample(state_ptr);
            }else{
                cleaned_states.push_back(state_ptr);
            }
        }
        return cleaned_states;
    }
    EnvSampler() = default;
    const std::vector<double> get_visit_counts() const {
        return std::vector<double>(visit_count.data_ptr<double>(), visit_count.data_ptr<double>() + visit_count.numel());
    }
    void init_priorities(int size){
        if(env_sampler == EnumEnvSampler::PLR){
            scores = torch::full({size}, 0, torch::kDouble);
            ema_return = torch::tensor({});
            last_return = torch::tensor({});
        }else{
            scores = torch::tensor({});
            ema_return = torch::zeros({size}, torch::kDouble);
            last_return = torch::full({size}, 0, torch::kDouble);
        }
        visit_count = torch::zeros({size}, torch::kDouble);
    }
    
    void compute_score(const MappingHistory<StateT>& mapping){
        if(env_sampler == EnumEnvSampler::PLR){
            auto& states =  mapping.get_states();
            double normalized_score = get_abs_gae_score(mapping);
            int state_id = states[0]->get_id();
            scores[state_id] = normalized_score;
            #ifdef DEBUG
                std::cout << "[DEBUG][compute_score] state_id=" << state_id 
                        << ", normalized_score=" << normalized_score << std::endl;
            #endif
        }
    }
    double get_abs_gae_score(const MappingHistory<StateT>& mapping) {
        auto& states = mapping.get_states();
        auto& rewards = mapping.get_rewards();
        std::vector<float> values;
    
        if (mapping.is_mcts_data()) {
            values = mapping.get_predicted_values_mcts();
            std::cout << "[INFO] Using predicted_values_mcts()" << std::endl;
        } else {
            values = mapping.get_values();
            std::cout << "[INFO] Using values()" << std::endl;
        }
    
        // std::cout << "[DEBUG] states size: " << states.size() << std::endl;
        // std::cout << "[DEBUG] rewards size: " << rewards.size() << std::endl;
        // std::cout << "[DEBUG] values size: " << values.size() << std::endl;
    
        // std::cout << "[DEBUG] Rewards (up to 5): ";
        // for (size_t i = 0; i < std::min<size_t>(rewards.size(), 5); ++i)
        //     std::cout << rewards[i] << " ";
        // std::cout << std::endl;
    
        // std::cout << "[DEBUG] Values (up to 5): ";
        // for (size_t i = 0; i < std::min<size_t>(values.size(), 5); ++i)
        //     std::cout << values[i] << " ";
        // std::cout << std::endl;
    
        auto tensor_rewards = torch::tensor(rewards);
        auto tensor_values = torch::tensor(values);
    
        torch::Tensor gae = torch::abs(compute_gae(tensor_rewards, tensor_values, lambda, discount));
    
        // std::cout << "[DEBUG] GAE shape: " << gae.sizes() << std::endl;
        // std::cout << "[DEBUG] GAE sum: " << gae.sum().item<double>() << std::endl;
    
        double total_score = gae.sum().item<double>();
        double normalized_score = total_score / (states.size() - 1);
    
        // std::cout << "[RESULT] Total GAE score: " << total_score << std::endl;
        // std::cout << "[RESULT] Normalized score: " << normalized_score << std::endl;
        // int i;
        // std::cin >> i;
        return normalized_score;
    }
    // double get_gae_score(const MappingHistory<StateT>& mapping){
    //         auto& states =  mapping.get_states();
    //         auto& rewards = mapping.get_rewards();
    // //          if(mapping.is_mcts_data()){
    //         values = mapping.get_predicted_values_mcts();
    //     } else {
    //         values = mapping.get_values();
    //     }
    //         auto tensor_rewards = torch::tensor(rewards);
    //         auto tensor_values = torch::tensor(values);
    //         torch::Tensor gae = compute_gae(tensor_rewards, tensor_values, lambda, discount);
    //         double total_score = gae.sum().item<double>();
    //         return total_score / (states.size() - 1);    
    //     }
    torch::Tensor compute_priorities(const std::vector<int>& state_ids) {
        if (env_sampler == EnumEnvSampler::PLR) {
            torch::Tensor scores = this->scores.index({torch::tensor(state_ids, torch::kLong)}).pow(1./plr_beta);
            torch::Tensor cur_visit_count = visit_count.index({torch::tensor(state_ids, torch::kLong)});
        
            torch::Tensor staleness_score = (total_visit_count - cur_visit_count);
            staleness_score = staleness_score / (staleness_score.sum() + 1e-9);
            scores = scores / (scores.sum() + 1e-9);
        
            auto combined = ((1 - plr_alpha) * scores + (plr_alpha) * staleness_score);
        
            #ifdef DEBUG
            std::cout << "[PLR DEBUG] Computing priorities..." << std::endl;
            std::cout << "  State IDs: " << torch::tensor(state_ids, torch::kLong) << std::endl;
            std::cout << "  Raw scores: " << this->scores.index({torch::tensor(state_ids, torch::kLong)}) << std::endl;
            std::cout << "  Scores^(-beta): " << this->scores.index({torch::tensor(state_ids, torch::kLong)}).pow(1./plr_beta) << std::endl;
            std::cout << "  Visit counts: " << cur_visit_count << std::endl;
            std::cout << "  Total visits: " << total_visit_count << std::endl;
            std::cout << "  Staleness (normalized): " << staleness_score << std::endl;
            std::cout << "  Scores (normalized): " << scores << std::endl;
            std::cout << "  Combined priorities: " << combined << std::endl;
            #endif
        
            return combined;
        }
        
        torch::Tensor ids_tensor = torch::tensor(state_ids, torch::kLong);
        torch::Tensor cur_ema = ema_return.index({ids_tensor});
        torch::Tensor cur_last = last_return.index({ids_tensor});
        torch::Tensor cur_visits = visit_count.index({ids_tensor});
    
        std::vector<double> norm_last, norm_ema;
        norm_last.reserve(state_ids.size());
        norm_ema.reserve(state_ids.size());

        for (int sid : state_ids) {
            auto &stats = state_return_stats[sid];
            double last_val = last_return[sid].item<double>();
            double ema_val  = ema_return[sid].item<double>();

            norm_last.push_back(stats.normalize2(last_val));
            norm_ema.push_back(stats.normalize2(ema_val));
            //force infinity for unvisited states
            if(visit_count[sid].item<double>() <= 1){
                norm_last.back() = std::numeric_limits<double>::infinity();
                norm_ema.back() = 0;
            }
        }
        cur_last = torch::tensor(norm_last, torch::kDouble);
        cur_ema  = torch::tensor(norm_ema,  torch::kDouble);
    
        torch::Tensor priorities_tensor = torch::abs(cur_ema - cur_last) + 
                                            c * torch::sqrt(torch::log(torch::tensor(std::max(total_visit_count, 1))) / (cur_visits + 1e-9));
        #ifdef DEBUG
        std::cout << "[DEBUG][compute_priorities] state_ids=" << state_ids.size() 
                  << ", cur_ema=" << cur_ema 
                  << ", cur_last=" << cur_last 
                  << ", cur_visits=" << cur_visits 
                  << ", priorities=" << priorities_tensor << std::endl;
        #endif

        return priorities_tensor;
    }

    EnvSampler(const std::vector<std::shared_ptr<StateT>>& train_states, GlobalConfig<ModelConfig>& global_config) {
        for (auto& state : train_states) {
            int state_id = state->get_id();
            state_map[state_id] = {state_id, state, false, bad_state_vec.size()};
            bad_state_vec.push_back(state_id);
        }

        auto& train_config = global_config.getTrainConfig();
        good_state_ratio = train_config.getGoodSampleRate();
        min_good_states = train_config.getMinGoodSamples();
        ema_alpha = train_config.getEmaAlpha();
        rout_nodes_threshold = train_config.getRoutNodeThresholdSampling();
        seed = train_config.getSeed();
        // gen = std::mt19937(seed);
        env_sampler = train_config.getEnvSampler();
        c = train_config.getCEnvSampler();
        alpha_return = train_config.getAlphaReturn();
        lambda = global_config.getRLConfig().getLambda();
        discount = global_config.getRLConfig().getDiscount();
        plr_alpha= train_config.getPLRAlpha();
        plr_beta = train_config.getPLRBeta();
    }

    int get_num_good_states() const {
        return good_state_vec.size();
    }

    int get_num_bad_states() const {
        return bad_state_vec.size();
    }


    void add_sample(std::shared_ptr<StateT> state) {
        int state_id = state->get_id();
    
        if (state_map.find(state_id) != state_map.end()) {
            throw std::runtime_error("State with ID " + std::to_string(state_id) + " already exists in EnvSampler.");
        };
    
        StateInfo s_info;
        s_info.state_id = state_id;
        s_info.state_ptr = state;
        s_info.is_good = false;
        s_info.index_in_current_vec = bad_state_vec.size();
        if(env_sampler == EnumEnvSampler::SEQUENTIAL){
            visit_count[state_id] = 0;
        }else{
            visit_count[state_id] = state_map.size() == 0 ? 0 : total_visit_count/(state_map.size()) ;
        }

        state_map[state_id] = s_info;
        bad_state_vec.push_back(state_id);
    }

    void add_old_sample(std::shared_ptr<StateT> state) {
        int state_id = state->get_id();
    
        if (state_map.find(state_id) == state_map.end()) {
            throw std::runtime_error("State with ID " + std::to_string(state_id) + " is not in EnvSampler.");
        };
        
        auto& s_info = state_map[state_id];
        s_info.state_ptr = state;
    }

    void update_ema_rout_nodes(int state_id, double rout_nodes) {
        if (env_sampler == EnumEnvSampler::PLR) return;

        auto it = state_id_ema_rout_nodes.find(state_id);
        if (it == state_id_ema_rout_nodes.end())
            state_id_ema_rout_nodes[state_id] = rout_nodes;
        else
            state_id_ema_rout_nodes[state_id] = ema_alpha * rout_nodes + (1.0 - ema_alpha) *  it->second ;
    }

    void update_ema_valid_map_rate(int state_id, bool valid_map) {
        if (env_sampler == EnumEnvSampler::PLR) return;

        double val = valid_map ? 1.0 : 0.0;
        auto it = state_id_valid_map_rate.find(state_id);
        if (it == state_id_valid_map_rate.end())
            state_id_valid_map_rate[state_id] = val;
        else
            state_id_valid_map_rate[state_id] = ema_alpha * val  + (1.0 - ema_alpha) * it->second;
    }

    void update_last_return(int state_id, double R) {
        if (env_sampler == EnumEnvSampler::PLR) return;
        double cur_visits = visit_count[state_id].item<double>() - 1;
        if(cur_visits == 0){
            ema_return[state_id] = R;
            last_return[state_id] = R;
        }else {
            auto old_R = last_return[state_id].item<double>();
            double factor = (1 - std::pow(alpha_return, cur_visits))/(1 - alpha_return);
            double norm = (1 - alpha_return)/(1 - std::pow(alpha_return, cur_visits + 1));
            ema_return[state_id] = norm*(old_R + alpha_return*(ema_return[state_id].item<double>()*factor));
            last_return[state_id] = R;
        }   
        state_return_stats[state_id].update(R);
        // state_return_stats[state_id].update(ema_return[state_id].item<double>());

    #ifdef DEBUG
        std::cout << "[DEBUG][update_last_return] state_id=" << state_id 
                  << "\n last_return=" << R 
                  << "\n ema_return=" << ema_return[state_id].item<double>() << std::endl
                  << "\n visit_count=" << visit_count[state_id].item<double>() << std::endl
                  << "\n state_return_stats(min=" << state_return_stats[state_id].get_min()
                  << ", max=" << state_return_stats[state_id].get_max() << std::endl;
    #endif
    }

    
    void update_visit_count(int state_id) {
        visit_count[state_id] += 1.0;
        total_visit_count += 1;
    #ifdef DEBUG
        std::cout << "[DEBUG][update_visit_count] state_id=" << state_id 
                  << ", visit_count=" << visit_count[state_id].item<double>() 
                  << ", total_visit_count=" << total_visit_count << std::endl;
    #endif
    }

    void move_to_good(StateInfo& s) {
        if (!s.is_good) {
    #ifdef DEBUG
            std::cout << "[DEBUG][move_to_good] state_id=" << s.state_id << std::endl;
    #endif
            s.is_good = true;
    
            size_t idx = s.index_in_current_vec;
            std::swap(bad_state_vec[idx], bad_state_vec.back());
            state_map[bad_state_vec[idx]].index_in_current_vec = idx;
            bad_state_vec.pop_back();
    
            s.index_in_current_vec = good_state_vec.size();
            good_state_vec.push_back(s.state_id);
        }
    }
    
    void move_to_bad(StateInfo& s) {
        if (s.is_good) {
    #ifdef DEBUG
            std::cout << "[DEBUG][move_to_bad] state_id=" << s.state_id << std::endl;
    #endif
            s.is_good = false;
    
            size_t idx = s.index_in_current_vec;
            std::swap(good_state_vec[idx], good_state_vec.back());
            state_map[good_state_vec[idx]].index_in_current_vec = idx;
            good_state_vec.pop_back();
    
            s.index_in_current_vec = bad_state_vec.size();
            bad_state_vec.push_back(s.state_id);
        }
    }

    void update_states(const std::vector<std::shared_ptr<StateT>>& sampled_states_input) {
        if (env_sampler != EnumEnvSampler::DOUBLE_BUFFER) return;

        for (auto& state : sampled_states_input) {
            int state_id = state->get_id();
            auto& s_info = state_map[state_id];
            double ema_rout_nodes = state_id_ema_rout_nodes[state_id];
            double valid_map_rate = state_id_valid_map_rate[state_id];

            if (ema_rout_nodes <= rout_nodes_threshold && valid_map_rate >= 0.99)
                move_to_good(s_info);
            else
                move_to_bad(s_info);
        }
    }

    std::vector<std::shared_ptr<StateT>> sample_states(torch::Generator& generator) {
        std::vector<std::shared_ptr<StateT>> sampled_states;
    
        if (env_sampler == EnumEnvSampler::DOUBLE_BUFFER) {
    #ifdef DEBUG
            std::cout << "[DEBUG][sample_states] Using DOUBLE_BUFFER strategy" << std::endl;
    #endif
    
            int num_good = std::max(static_cast<int>(good_state_ratio * good_state_vec.size()), min_good_states);
            num_good = std::min(num_good, static_cast<int>(good_state_vec.size()));
    
            if (num_good > 0) {
                torch::Tensor priority_tensor = compute_priorities(good_state_vec);
                double sum_val = priority_tensor.sum().item<double>();
                torch::Tensor prob_tensor = (sum_val <= 0.0) ? torch::ones_like(priority_tensor) / priority_tensor.numel()
                                                             : priority_tensor / sum_val;
    
                auto indices = torch::multinomial(prob_tensor, num_good, false,generator);
                for (int i = 0; i < indices.size(0); ++i) {
                    int idx = indices[i].item<int>();
                    int state_id = good_state_vec[idx];
                    update_visit_count(state_id);
                    sampled_states.push_back(state_map[state_id].state_ptr);
    #ifdef DEBUG
                    std::cout << "[DEBUG][sample_states][DOUBLE_BUFFER] picked good_state_id=" << state_id << std::endl;
    #endif
                }
            }
    
            for (int state_id : bad_state_vec) {
                update_visit_count(state_id);
                sampled_states.push_back(state_map[state_id].state_ptr);
    #ifdef DEBUG
                std::cout << "[DEBUG][sample_states][DOUBLE_BUFFER] added bad_state_id=" << state_id << std::endl;
    #endif
            }
    
        } else if (env_sampler == EnumEnvSampler::UCB) {
    #ifdef DEBUG
            std::cout << "[DEBUG][sample_states] Using UCB strategy" << std::endl;
    #endif
    
            int sample_size = std::max(static_cast<int>(good_state_ratio * bad_state_vec.size()), min_good_states);
            sample_size = std::min(sample_size, static_cast<int>(bad_state_vec.size()));
    
            torch::Tensor priority_tensor = compute_priorities(bad_state_vec);
            //take the k largest priorities 
            auto indices = std::get<1>(priority_tensor.topk(sample_size, /*dim=*/0, /*largest=*/true, /*sorted=*/true));
            // double sum_val = priority_tensor.sum().item<double>();
            // torch::Tensor prob_tensor = priority_tensor / sum_val;
    
            // auto indices = torch::multinomial(prob_tensor, sample_size, false, generator);
            #ifdef DEBUG
                std::cout << "[DEBUG][priority_tensor] " << priority_tensor << std::endl;
                std::cout << "[DEBUG][sample_states] UCB sampled indices: " << indices << std::endl;
            #endif
            for (int i = 0; i < indices.size(0); ++i) {
                int idx = indices[i].item<int>();
                int state_id = bad_state_vec[idx];
                update_visit_count(state_id);
                sampled_states.push_back(state_map[state_id].state_ptr);
    #ifdef DEBUG
                std::cout << "[DEBUG][sample_states][UCB] picked bad_state_id=" << state_id 
                          << " priority=" << priority_tensor[idx].item<double>() << std::endl;
    #endif
            }
        } else if (env_sampler == EnumEnvSampler::RANDOM) {
            #ifdef DEBUG
                std::cout << "[DEBUG][sample_states] Using RANDOM strategy" << std::endl;
            #endif
            
                int sample_size = std::max(static_cast<int>(good_state_ratio * bad_state_vec.size()), min_good_states);
                sample_size = std::min(sample_size, static_cast<int>(bad_state_vec.size()));
            
                auto prob_tensor = torch::ones({(int64_t)bad_state_vec.size()}, torch::kDouble) / bad_state_vec.size();
            
                auto indices = torch::multinomial(prob_tensor, sample_size, false,generator);
                // std::cout << "[DEBUG][sample_states] UCB sampled indices: " << indices << std::endl;
                // int a;
                // std::cin >> a;
                for (int i = 0; i < indices.size(0); ++i) {
                    int idx = indices[i].item<int>();
                    int state_id = bad_state_vec[idx];
                    update_visit_count(state_id);
                    sampled_states.push_back(state_map[state_id].state_ptr);
            #ifdef DEBUG
                    std::cout << "[DEBUG][sample_states][RANDOM] picked bad_state_id=" 
                              << state_id << std::endl;
            #endif
                }
            }  else if (env_sampler == EnumEnvSampler::PLR) {
                torch::Tensor x = torch::rand({1},generator);
                double total_train_states = good_state_vec.size() + bad_state_vec.size();
            
            #ifdef DEBUG
                std::cout << "[DEBUG][PLR] sample_size=" << sample_size 
                          << ", rand_x=" << x.item<double>() 
                          << ", seen_ratio=" << (double)good_state_vec.size() / total_train_states
                          << ", good_states=" << good_state_vec.size()
                          << ", bad_states=" << bad_state_vec.size()
                          << ", total_train_states=" << total_train_states
                          << std::endl;
            #endif
            
                // seen/train
                if (x.item<double>() >= (double)good_state_vec.size() / (total_train_states)) {
                    int sample_size = std::max(static_cast<int>(good_state_ratio * total_train_states), min_good_states);

                    sample_size = std::min(sample_size, static_cast<int>(bad_state_vec.size()));
                    auto prob_tensor = torch::ones({(int64_t)bad_state_vec.size()}, torch::kDouble) / bad_state_vec.size();
                    auto indices = torch::multinomial(prob_tensor, sample_size, false, generator);
            
            #ifdef DEBUG
                    std::cout << "[DEBUG][PLR] Sampling from UNSEEN (bad_state_vec)" << std::endl;
            #endif
            
                    for (int i = 0; i < indices.size(0); ++i) {
                        int idx = indices[i].item<int>();
                        int state_id = bad_state_vec[idx];
                        update_visit_count(state_id);
                        sampled_states.push_back(state_map[state_id].state_ptr);
                        move_to_good(state_map[state_id]);
            
            #ifdef DEBUG
                        std::cout << "[DEBUG][PLR][UNSEEN] picked bad_state_id=" << state_id << std::endl;
            #endif
                    }
            
                } else {
                    int sample_size = std::max(static_cast<int>(good_state_ratio * total_train_states), min_good_states);

                    sample_size = std::min(sample_size, static_cast<int>(good_state_vec.size()));
                    torch::Tensor probs = compute_priorities(good_state_vec);
                    auto indices = torch::multinomial(probs, sample_size, false, generator);
            
            #ifdef DEBUG
                    std::cout << "[DEBUG][PLR] Sampling from SEEN (good_state_vec)" << std::endl;
                    std::cout << "[DEBUG][PLR] probs=" << probs << std::endl;
            #endif
            
                    for (int i = 0; i < indices.size(0); ++i) {
                        int idx = indices[i].item<int>();
                        int state_id = good_state_vec[idx];
                        update_visit_count(state_id);
                        sampled_states.push_back(state_map[state_id].state_ptr);
            
            #ifdef DEBUG
                        std::cout << "[DEBUG][PLR][SEEN] picked good_state_id=" << state_id << std::endl;
            #endif
                    }
                }
            }else if (env_sampler == EnumEnvSampler::SEQUENTIAL) {
                #ifdef DEBUG
                std::cout << "[DEBUG][sample_states] Using SEQUENTIAL strategy" << std::endl;
            #endif
            
                int sample_size = std::max(static_cast<int>(good_state_ratio * bad_state_vec.size()), min_good_states);
                sample_size = std::min(sample_size, static_cast<int>(bad_state_vec.size()));
                for (int i = 0; i < sample_size; ++i) {
                    auto idx = total_visit_count % bad_state_vec.size();
                    int state_id = bad_state_vec[idx];
                    update_visit_count(state_id);
                    sampled_states.push_back(state_map[state_id].state_ptr);
            #ifdef DEBUG
                    std::cout << "[DEBUG][sample_states][SEQUENTIAL] picked bad_state_id=" 
                              << state_id << std::endl;
                    std::cout << " total_visit_count=" << total_visit_count 
                              << " idx=" << idx << std::endl <<
                              " bad_state_vec.size()=" << bad_state_vec.size() << std::endl;
            #endif   
                }
            }
            
            
    
        return sampled_states;
    }

    template<typename Archive>
void serialize(Archive& ar) {
    ar(good_state_vec, bad_state_vec);
    ar(good_state_ratio, min_good_states, ema_alpha, seed, rout_nodes_threshold, c, env_sampler, total_visit_count, alpha_return);
    ar(state_id_ema_rout_nodes, state_id_valid_map_rate, state_return_stats, lambda, discount);

    serialize_tensor(ar, ema_return);
    serialize_tensor(ar, visit_count);
    serialize_tensor( ar, last_return);
    serialize_tensor(ar, scores);


    std::vector<int> state_ids;
    std::vector<bool> is_good_vec;
    std::vector<size_t> indices_vec;
    std::vector<std::shared_ptr<StateT>> state_ptrs;

    if (Archive::is_saving::value) {
        for (auto& [id, s_info] : state_map) {
            state_ids.push_back(id);
            is_good_vec.push_back(s_info.is_good);
            indices_vec.push_back(s_info.index_in_current_vec);
            // state_ptrs.push_back(s_info.state_ptr);
        }
        ar(state_ids, is_good_vec, indices_vec);
    } else {
        ar(state_ids, is_good_vec, indices_vec);
        state_map.clear();
        for (size_t i = 0; i < state_ids.size(); ++i) {
            state_map[state_ids[i]] = {state_ids[i], nullptr, is_good_vec[i], indices_vec[i]};
        }
    }
}

};

#endif
