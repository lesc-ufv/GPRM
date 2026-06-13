#ifndef SELF_MAPPING_HPP
#define SELF_MAPPING_HPP

#include <iostream>
#include <type_traits>
#include "src/hpp/enums/enum_self_mapping.hpp"
#include "src/hpp/rl/policy_logits_self_mapping.hpp"
#include "src/hpp/compilers/mapzero/rl/mapzero_self_mapping.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_self_mapping.hpp"
#include "src/hpp/rl/environment.hpp"
#include "src/hpp/enums/enum_self_mapping.hpp"
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/enums/enum_task_type.hpp"
#include <zmq.hpp>
#include "src/hpp/utils/constants/server_constants.hpp"
#include "src/hpp/rl/mapping_history.hpp"
#include <optional>
#include "src/hpp/utils/batch_requester.hpp"
#include "src/hpp/rl/beam_search_self_mapping.hpp"
#include "src/hpp/context_holder.hpp"

template <typename State, typename ModelConfig>
class SelfMapping{
private:
    GlobalConfig<ModelConfig> global_config;
    ServerConstants server_k;
    std::string func_name;
    std::shared_ptr<State> state;
    Environment<State> environment;
    std::optional<std::shared_ptr<BatchRequester<State>>> batch_requester;

    MappingHistory<State> (*map_function)(  GlobalConfig<ModelConfig>& , std::shared_ptr<zmq::socket_t> ,
        const ServerConstants&, const std::string&,const std::shared_ptr<State>, 
        Environment<State>&, bool, double, int,
    std::optional<std::unordered_map<std::string,int>>,
    std::optional<std::shared_ptr<BatchRequester<State>>> ) = nullptr;

    void(*reanalyze_mapping)( GlobalConfig<ModelConfig>& ,
        MappingHistory<State>& , std::shared_ptr<zmq::socket_t>,
        const ServerConstants& , const std::string&, std::shared_ptr<State>, Environment<State>&,
        std::optional<std::shared_ptr<BatchRequester<State>>> batch_requester, std::mt19937& gen);
public:
    SelfMapping( GlobalConfig<ModelConfig>& global_config,
        const ServerConstants& server_k, const std::string& func_name, const Environment<State>& environment,
        std::optional<std::shared_ptr<BatchRequester<State>>> opt_batch_requester) ;
    SelfMapping() ;
    MappingHistory<State> map(std::shared_ptr<State> initial_state, bool train,  double train_ratio, int seed,
        std::optional<std::unordered_map<std::string,int>>);
    void reanalyze(std::shared_ptr<State> state, MappingHistory<State>& mapping_history, std::mt19937& gen);

};

#include "src/tpp/rl/self_mapping.tpp"

#endif

