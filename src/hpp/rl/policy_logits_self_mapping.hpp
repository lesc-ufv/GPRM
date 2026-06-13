#ifndef SMARTMAP_PRE_TRAINING_SELF_MAPPING
#define SMARTMAP_PRE_TRAINING_SELF_MAPPING
#include "zmq.hpp"
#include <tuple>
#include <vector>
#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"
#include "src/hpp/utils/constants/server_constants.hpp"
#include "src/hpp/utils/util_server.hpp"
#include "src/hpp/enums/enum_task_type.hpp"
#include "src/hpp/rl/mapping_history.hpp"
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/rl/environment.hpp"
#include <thread>
#include "src/hpp/utils/util_infer.hpp"

template<typename StateT, typename ModelConfig>    
MappingHistory<StateT> map_with_policy_logits( GlobalConfig<ModelConfig>& config, std::shared_ptr<zmq::socket_t> socket,
    const ServerConstants& server_k, const std::string& func_name,const std::shared_ptr<StateT> initial_state, 
    Environment<StateT>& environment, bool train,  double train_ratio, int seed, 
    std::optional<std::unordered_map<std::string,int>> node_to_action, std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester);

template< typename StateT>    
std::tuple<int,int> get_action_and_action_indice(const std::shared_ptr<StateT> cur_state,
        torch::Tensor policy_probs, bool train, torch::Generator& gen);


template<typename StateT, typename ModelConfig>    
void reanalyze_with_policy_logits(GlobalConfig<ModelConfig>& config, MappingHistory<StateT>& mapping_history, std::shared_ptr<zmq::socket_t> socket,
    const ServerConstants& server_k, const std::string& func_name, std::shared_ptr<StateT> state, Environment<StateT>& environment,
    std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester);

#include "src/tpp/rl/policy_logits_self_mapping.tpp"

#endif