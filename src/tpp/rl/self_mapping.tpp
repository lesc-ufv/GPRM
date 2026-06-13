#include "src/hpp/rl/self_mapping.hpp"

template <typename State, typename ModelConfig>
SelfMapping< State, ModelConfig>::SelfMapping(GlobalConfig<ModelConfig>& global_config,
                        const ServerConstants& server_k, const std::string& func_name, const Environment<State>& environment, 
                        std::optional<std::shared_ptr<BatchRequester<State>>> opt_batch_requester) :
                        global_config(global_config), server_k(server_k), func_name(func_name), environment(environment),
                        batch_requester(opt_batch_requester)
{     

    auto self_mapping_type = global_config.getLauncherConfig().getSelfMapping();
    if (self_mapping_type == EnumSelfMapping::POLICY_LOGITS) {
        this->map_function = map_with_policy_logits<State,ModelConfig>;
        this->reanalyze_mapping = nullptr;
    } else if (self_mapping_type == EnumSelfMapping::SMARTMAP_MCTS) {
        this->map_function = map_with_smartmap_mcts<State,ModelConfig>;
        this->reanalyze_mapping = reanalyze_with_smartmap_mcts<State,ModelConfig>;
    } else if (self_mapping_type == EnumSelfMapping::MAPZERO_MCTS){
        this->map_function = map_with_mapzero_mcts<State,ModelConfig>;
        this->reanalyze_mapping = nullptr;
    }else if (self_mapping_type == EnumSelfMapping::BEAM_SEARCH){
        this->map_function = map_with_beam_search<State,ModelConfig>;
        this->reanalyze_mapping = reanalyze_with_smartmap_mcts<State,ModelConfig>;
    }else{
        throw std::runtime_error("Invalid self mapping");
    }
}

template <typename State, typename ModelConfig>
SelfMapping< State, ModelConfig>::SelfMapping() { };

template <typename State, typename ModelConfig>
MappingHistory<State> SelfMapping<State,ModelConfig>::map(
    std::shared_ptr<State> initial_state, bool train, double train_ratio, int seed,
    std::optional<std::unordered_map<std::string,int>> node_to_action )
{
    auto& context = ZmqContextHolder::instance();

    auto sock = std::make_shared<zmq::socket_t>(context, ZMQ_DEALER);
    std::string port = this->global_config.getLauncherConfig().getServerPort();
    int timeout = -1;
    sock->set(zmq::sockopt::sndtimeo, timeout);
    sock->set(zmq::sockopt::rcvtimeo, timeout);
    sock->connect("tcp://localhost:" + port);


    auto ret = this->map_function(
        this->global_config, sock, this->server_k, this->func_name,
        initial_state, this->environment, train, train_ratio, seed,
        node_to_action, this->batch_requester
    );
    sock->close();
    // context.close();
    return ret;
}
template <typename State, typename ModelConfig>
void SelfMapping<State,ModelConfig>::reanalyze(std::shared_ptr<State> state, MappingHistory<State>& mapping_history, std::mt19937& gen)
{
    
    auto& context = ZmqContextHolder::instance();

    auto socket = std::make_shared<zmq::socket_t>(context, ZMQ_DEALER);
    std::string port = this->global_config.getLauncherConfig().getServerPort();
    int timeout = -1;
    socket->set(zmq::sockopt::sndtimeo, timeout);
    socket->set(zmq::sockopt::rcvtimeo, timeout);
    std::string unique_id = "reanalyze-" + std::to_string(reinterpret_cast<uintptr_t>(this));
   socket->set(zmq::sockopt::routing_id, zmq::buffer(unique_id));
    socket->connect("tcp://localhost:" + port);

    this->reanalyze_mapping(this->global_config, mapping_history, socket, this->server_k, this->func_name,
         state, this->environment, batch_requester, gen);

    socket->close();
    // context.close();
}

