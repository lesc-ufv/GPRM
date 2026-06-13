#ifndef BASE_BUFFER_DATA_HPP
#define BASE_BUFFER_DATA_HPP
#include <iostream>
#include <torch/torch.h>
#include "src/hpp/utils/util_train.hpp"
#include <cmath>
#include "src/hpp/utils/util_print.hpp"
#include "src/hpp/enums/enum_nomalization.hpp"
#include "src/hpp/datasets/mapping_dataset.hpp"
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/rl/environment.hpp"
#include <vector>
#include "src/hpp/rl/mapping_history.hpp"
#include "src/hpp/enums/enum_training.hpp"
#include "src/hpp/compilers/mapzero/rl/mapzero_environment.hpp"
#include "src/hpp/utils/batch_requester.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>



template<typename StateT>
struct Transition {
    std::shared_ptr<StateT> state = nullptr;
    double ret = -std::numeric_limits<double>::infinity();
    int action_index = -std::numeric_limits<int>::max();
    std::vector<float> old_policy;

    template <class Archive>
    void serialize(Archive& ar) {
        serialize_shared_ptr(ar, state);
        ar(ret, action_index, old_policy);
    }
};

template<typename StateT,typename ModelConfig>
struct DTB{
    std::vector<std::vector<Transition<StateT>>> transitions;
    int size;
    int curr_size = 0;

    void add_mapping_history(const MappingHistory<StateT>& mapping_history, GlobalConfig< ModelConfig>& global_config){
        auto discount = global_config.getRLConfig().getDiscount();
        auto states = mapping_history.get_states();
        auto rewards = mapping_history.get_rewards();
        auto action_indices = mapping_history.get_actions_indices();
        auto policies = mapping_history.get_child_visits();
        torch::Tensor returns = compute_n_step_target_values(torch::tensor(rewards), torch::tensor({}), -1, discount);
        auto map_idx = this->curr_size % this->size;
        if(transitions[map_idx].size() != states.size() - 1){
            transitions[map_idx].resize(states.size() - 1);
        }

        for(int i = 0; i < static_cast<int>(states.size()) - 1; i++){
            auto cur_state = states[i];
            auto cur_action_idx = action_indices[i + 1];
            auto ret = returns[i].item<double>();
            auto policy = policies[i];
            this->add_transition(cur_state, ret, cur_action_idx, std::move(policy),map_idx, i);
        }
            this->curr_size += 1;
    }

    void add_transition(std::shared_ptr<StateT> state, 
        double ret, 
        int action_idx, 
        std::vector<float> policy,
        int map_idx,
        int transition_idx){

        transitions[map_idx][transition_idx].state = state;
        transitions[map_idx][transition_idx].ret = ret;
        transitions[map_idx][transition_idx].action_index = action_idx;
        transitions[map_idx][transition_idx].old_policy = policy;
        // std::cout << "[DTB] Added transition to DTB at map_idx " << map_idx << " transition_idx " << transition_idx << "\n";
        // std::cout << "[DTB] State key: " << state->get_key() << " Ret: " << ret << " Action idx: " << action_idx << "\n";
        // std::cout << "[DTB] Trasitions size: " << transitions.size() << "\n";
        // std::cout << "[DTB] Current size: " << this->curr_size << " / " << this->size << "\n";
    }

    void init_DTB(int size){
        this->size = size;
        this->transitions.resize(size);
    }

    int sample_dtb_idx(){
        if(this->curr_size == 0) throw std::runtime_error("DTB is empty");
        torch::Tensor perms = torch::randperm(std::min(this->curr_size, this->size));
        auto idx = perms[0].item<int>();
        return idx;
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(size, curr_size);
        if constexpr (Archive::is_saving::value) {
            size_t outer = transitions.size();
            ar(outer);
            for (auto& inner : transitions) {
                size_t inner_sz = inner.size();
                ar(inner_sz);
                for (auto& tr : inner) {
                    ar(tr); 
                }
            }
        } else {
            size_t outer;
            ar(outer);
            transitions.resize(outer);
            for (size_t i = 0; i < outer; ++i) {
                size_t inner_sz;
                ar(inner_sz);
                transitions[i].resize(inner_sz);
                for (size_t j = 0; j < inner_sz; ++j) {
                    ar(transitions[i][j]);
                }
            }
        }
    }
};


template <typename StateT, typename ModelConfig>
class BaseBuffer{
private:
    std::vector<std::vector<std::shared_ptr<StateT>>> states;
    int total_samples;
    int number_of_mappings;
    int buffer_size;
    bool is_mcts_data, is_ppo;
    int batch_size;

    std::vector<torch::Tensor> target_values;
    std::vector<torch::Tensor> values_vec;
    std::vector<torch::Tensor> actions_indices;
    std::vector<torch::Tensor> target_policies;
    std::vector<torch::Tensor> advantages;
    std::vector<torch::Tensor> rewards;
    torch::Tensor mapping_priorities;
    std::vector<torch::Tensor> states_priorities;
    GlobalConfig<ModelConfig> global_config;
    std::unordered_map<std::string, DTB<StateT,ModelConfig>> key_to_dtb;
    std::vector<std::string> keys;
    int called_get_dtb_batch = 0;
    torch::Generator gen;
    int cur_reanalyzed_map_idx = -1;
public:
    void set_cur_reanalyzed_map_idx(int idx){
        this->cur_reanalyzed_map_idx = idx;
    }

    BaseBuffer(): 
        total_samples(0),
        number_of_mappings(0),
        buffer_size(-1),
        is_mcts_data(false),
        is_ppo(false),
        batch_size(-1), 
        target_values(),
        values_vec(),
        actions_indices(),
        target_policies(),
        advantages(),
        rewards(),
        mapping_priorities(torch::empty({})),
        states_priorities(),
        global_config(GlobalConfig<ModelConfig>())
        {}


        bool empty() const {
            return this->states.size() == 0;
        }

        torch::Generator& get_generator() {
            return this->gen;
        }

        void save_mapping_to_DTB(const MappingHistory<StateT>& mapping_history){
            auto& states = mapping_history.get_states();
            auto key = states[0]->get_key();
            auto dfg_size = states[0]->get_dfg()->get_num_nodes();
            auto dtb_size = dfg_size * 2;
            if(this->key_to_dtb.find(key) == this->key_to_dtb.end()){
                DTB<StateT, ModelConfig> dtb;
                dtb.init_DTB(dtb_size);
                this->key_to_dtb[key] = dtb;
                this->keys.push_back(key);
            }
            auto& dtb = this->key_to_dtb[key];
            dtb.add_mapping_history(mapping_history, this->global_config);
        }

        MappingDataset<StateT> get_batch_from_dtb(   std::shared_ptr<zmq::socket_t> socket,
            const ServerConstants& server_k,
            std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester,
            std::string func_name)
            {
                auto batch_size = this->global_config.getTrainConfig().getBatchSizeMappings();
                auto num_dtbs = this->global_config.getTrainConfig().getNumDTBsToSampleFrom();
                std::vector<std::shared_ptr<StateT>> cur_states;
                std::vector<double> all_target_values;
                std::vector<std::vector<float>> all_policies;
                std::vector<int> all_actions;
                std::vector<double> predicted_values;

                std::cout << "[DTB] Sampling batch from " << num_dtbs << " DTBs\n";
                for(int dtb_idx = 0; dtb_idx < num_dtbs; dtb_idx++){
                    auto& dtb = this->key_to_dtb.at(this->keys[(called_get_dtb_batch + dtb_idx) % this->key_to_dtb.size()]);
                    // std::cout << "[DTB] Called get batch " << called_get_dtb_batch << " times\n";
                    // std::cout << "[DTB] Number of keys in DTB: " << this->key_to_dtb.size() << "\n";
                    // std::cout << "[DTB] Current key: " << this->keys[called_get_dtb_batch % this->key_to_dtb.size()] << "\n";
                    // int a;
                    // std::cin >> a;
                    int num_episodes = std::min(dtb.curr_size, dtb.size);
                    int num_samples = std::min(batch_size, num_episodes);
                    std::vector<std::shared_ptr<StateT>> temp_states;
                    auto indices = torch::randperm(num_episodes, this->gen).slice(0, 0, num_samples);
                    for(int i = 0; i < static_cast<int>(indices.size(0)); i++){
                        auto indice = indices[i];
                        for(int j = 0; j < static_cast<int>(dtb.transitions[indice.item<int>()].size()); j++){
                            auto& transition = dtb.transitions[indice.item<int>()][j];
                            auto state = transition.state;
                            temp_states.push_back(state);
                            all_target_values.push_back(transition.ret);
                            all_policies.push_back(std::ref(transition.old_policy));
                            all_actions.push_back(transition.action_index);
                        }
                    }
                    torch::Tensor policy_logits, value;
                    std::vector<double> temp_predicted_values;
                    temp_predicted_values.resize( temp_states.size());
                    tbb::parallel_for(
                        tbb::blocked_range<int64_t>(0, temp_states.size()),
                        [&](const tbb::blocked_range<int64_t>& r) {
                            for (int64_t i = r.begin(); i != r.end(); ++i) {
                                torch::Tensor policy_logits, value;
                                std::tie(policy_logits, value) =
                                    infer(socket, server_k, batch_requester, func_name, temp_states[i]);
                                temp_predicted_values[i] = value.item<double>();
                            }
                        }
                    );
                    predicted_values.insert(predicted_values.end(), temp_predicted_values.begin(), temp_predicted_values.end());
                    cur_states.insert(cur_states.end(), temp_states.begin(), temp_states.end());
                }
                called_get_dtb_batch += num_dtbs;
                auto target_values_tensor = torch::tensor(all_target_values);
                auto predicted_values_tensor = torch::tensor(predicted_values);
                auto advantages_tensor = target_values_tensor - predicted_values_tensor;
                auto policies_tensor = matrix_to_tensor<float,float>(all_policies);
                auto actions_tensor = torch::tensor(all_actions, torch::kInt64);

                return MappingDataset<StateT>(
                    std::move(cur_states), 
                    std::move(target_values_tensor), 
                    std::nullopt ,
                    std::make_optional(policies_tensor), 
                    std::make_optional(actions_tensor), 
                    std::make_optional(advantages_tensor),
                    std::nullopt, 
                    std::nullopt, 
                    std::nullopt); 
        }


        void clear_from_to(int from, int to) {
        
            if (from == to) return; 
            
            if (from < 0 || to > this->number_of_mappings || from >= to) {
                throw std::runtime_error("Invalid range for clearing buffer");
            }
            
                for(int i = from; i < to; i++){
                    this->total_samples -= this->states[i].size() - 1;
                }
            
            if (!this->states.empty()) {
                this->states.erase(this->states.begin() + from, this->states.begin() + to);
            }
            if (!this->rewards.empty()) {
                this->rewards.erase(this->rewards.begin() + from, this->rewards.begin() + to);
            }
            if (!this->values_vec.empty()) {
                this->values_vec.erase(this->values_vec.begin() + from, this->values_vec.begin() + to);
            }
            if (!this->target_policies.empty()) {
                this->target_policies.erase(this->target_policies.begin() + from, this->target_policies.begin() + to);
            }
            if (!this->actions_indices.empty()) {
                this->actions_indices.erase(this->actions_indices.begin() + from, this->actions_indices.begin() + to);
            }
            if (!this->advantages.empty()) {
                this->advantages.erase(this->advantages.begin() + from, this->advantages.begin() + to);
            }
            if (!this->states_priorities.empty()) {
                this->states_priorities.erase(this->states_priorities.begin() + from, this->states_priorities.begin() + to);
            }
        
            this->mapping_priorities = this->mapping_priorities.slice(0, from, to);
        
            this->number_of_mappings -= (to - from);
        }

    BaseBuffer( GlobalConfig<ModelConfig>& global_config);
    int get_num_to_batch(){
        int num_to_batch;
        if(this->batch_size != -1){
            num_to_batch = std::min(this->batch_size, this->number_of_mappings);
        }else{
            num_to_batch = std::min(this->number_of_mappings, this->buffer_size);
        }
        return num_to_batch;
    }

    void set_config(GlobalConfig<ModelConfig> config){
        this->global_config = config;
    }

    int get_num_unroll_steps(){
        auto rl_config = this->global_config.getRLConfig();
        auto unroll_steps = rl_config.getUnrollSteps();
        return unroll_steps;
    }
    const BufferConfig& getBufferConfig()  { return this->global_config.getBufferConfig(); }
    bool getNormAdvantages(){ return this->global_config.getBufferConfig().getNormAdvantages();}
    bool getNormReturns(){ return this->global_config.getBufferConfig().getNormReturns();}
    MappingDataset<StateT> get_dataset(torch::Tensor& mapping_indices); 
    void save_mapping(const MappingHistory<StateT>& mapping_history);
    int get_number_of_mappings() const;
    int get_num_data_in_buffer() const;
    torch::Tensor get_norm_values(const torch::Tensor& values);
    void decrease_total_samples(int value) {
        this->total_samples -= value;

    }
    
    void circular_append_states(const std::vector<std::shared_ptr<StateT>>& data){
        this->circular_append(this->states,data);
    }

    void circular_append_rewards(const torch::Tensor& data){
        this->circular_append(this->rewards,data);
    }
    
    void circular_append_values(const torch::Tensor& data){
        this->circular_append(this->values_vec,data);
    }

    void circular_append_policies(const torch::Tensor& data){
        this->circular_append(this->target_policies,data);
    }

    void circular_append_actions_indices(const torch::Tensor& data){
        this->circular_append(this->actions_indices,data);
    }

    template<typename T>
    void circular_append(std::vector<T>& vec, const T& data){
        if (this->number_of_mappings < this->buffer_size) {
            vec.push_back(data);
        } else {
            auto idx = this->number_of_mappings % this->buffer_size;
            vec[idx] = data;
        }
    }
    int get_mapping_size_by_idx(int idx){
        return this->states[idx].size();
    }
    bool buffer_is_full(){
        return this->number_of_mappings >= this->buffer_size; 
    }

    // void increment_total_samples(int value){
    //     this->total_samples += states.size() - 1;

    // }
    const std::vector<torch::Tensor>& get_rewards() const;
    
    const torch::Tensor& get_rewards_by_idx(int idx) const {
        return this->rewards[idx];
    };

    const std::vector<torch::Tensor>& get_values() const;
    const std::vector<torch::Tensor>& get_values_vec() const { return values_vec;}

    const std::vector<std::vector<std::shared_ptr<StateT>>>& get_states() const;
    const std::vector<torch::Tensor>& get_actions_indices() const;
    const torch::Tensor& get_actions_indices_by_idx(int idx) const {
        return this->actions_indices[idx];
    }
    const std::vector<torch::Tensor>& get_policies() const;
    const torch::Tensor& get_policies_by_idx(int idx) const{
        return target_policies[idx];
    }

    void clear();
    torch::Tensor get_target_values(const torch::Tensor& rewards,const torch::Tensor& values);
    torch::Tensor get_norm_rewards(const torch::Tensor& rewards);
    MappingDataset<StateT> get_batch();
    const std::vector<std::shared_ptr<StateT>>& get_mapping_by_idx(const int& indx) const;
    const int& get_buffer_size() const ;
    void set_states_priorities_by_idx(torch::Tensor states_priorities,const int& idx);
    void set_mapping_priority_by_idx(double mapping_priority, const int& idx);
    torch::Tensor get_target_values_by_idx(const int& idx) const;
    torch::Tensor get_values_by_idx(const int& idx) const{
        return this->values_vec[idx];
    }
    void set_states_priorities_by_idxs(torch::Tensor states_priorities,const int& map_idx, torch::Tensor states_indices);
    void reduce_mapping_priorities_by_alpha(torch::Tensor mapping_indices);
    void reduce_states_priorities_by_alpha(torch::Tensor mapping_indices, torch::Tensor states_indices);
    void update_with_renalyzed_mapping(MappingHistory<StateT>& mapping_history, const int& idx);
    torch::Tensor get_mapping_priorities() const;

    void increment_total_samples(int samples);
    
    bool get_is_mcts_data() const;


    void append_states(const std::vector<std::shared_ptr<StateT>>& states);


    void append_rewards(torch::Tensor rewards);


    void append_policies(torch::Tensor policies);

    void append_actions_indices(torch::Tensor actions_indices);

    void append_values(torch::Tensor values);

    void increment_number_of_mappings(int value);
    torch::Tensor get_state_priorities_by_idx(const int& idx) const;
    void append_target_values(torch::Tensor target_values){
        this->target_values.push_back(target_values);
    }

    bool get_is_ppo() const;

    bool is_gae()  {
        return this->global_config.getRLConfig().getUseGAE();
    }

    int get_td_steps()  {
        return this->global_config.getRLConfig().getTDSteps();
    }

    void set_global_config(GlobalConfig<ModelConfig>& global_config) {
        this->global_config = global_config;
    }

    template <class Archive>
void serialize(Archive& ar) {
    serialize_vector_vector_shared_ptr(ar, states);

    ar(total_samples,
       number_of_mappings,
       buffer_size,
       is_mcts_data,
       is_ppo,
       batch_size,
       called_get_dtb_batch,
       keys);

    serialize_vector_tensor(ar, target_values);
    serialize_vector_tensor(ar, values_vec);
    serialize_vector_tensor(ar, actions_indices);
    serialize_vector_tensor(ar, target_policies);
    serialize_vector_tensor(ar, advantages);
    serialize_vector_tensor(ar, rewards);

    if (Archive::is_saving::value) {
        if (mapping_priorities.defined() && mapping_priorities.numel() > 0) {
            serialize_tensor(ar, mapping_priorities);
        } else {
            torch::Tensor dummy = torch::zeros({1});
            serialize_tensor(ar, dummy);
        }
    } else {
        torch::Tensor tmp;
        serialize_tensor(ar, tmp);
        mapping_priorities = tmp;
    }

    serialize_generator_state(ar, gen);
    
    serialize_vector_tensor(ar, states_priorities);

    ar(key_to_dtb);
}
};

#include "src/tpp/rl/base_buffer.tpp"
#endif