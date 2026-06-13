#include "src/hpp/utils/util_server.hpp"
template <typename List>
msgpack::object get_vector_obj(const List& vector){
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, vector);

    msgpack::object_handle oh = msgpack::unpack(sbuf.data(), sbuf.size());
    return  oh.get();
}




template<typename ModelConfig, typename = void>
struct ModelConfigAdder {
    static void add(std::unordered_map<std::string, msgpack::object>&,
                   msgpack::zone&,
                   ServerConstants&,
                   GlobalConfig<ModelConfig>&) {
        static_assert(sizeof(ModelConfig) == 0, 
            "No specialization available for this ModelConfig type");
    }
};

template<typename ModelConfig>
struct ModelConfigAdder<ModelConfig, 
                       std::enable_if_t<std::is_same_v<ModelConfig, MapZeroConfig>>> {
    static void add(std::unordered_map<std::string, msgpack::object>& data_map,
                   msgpack::zone& zone,
                   ServerConstants& server_k,
                   GlobalConfig<ModelConfig>& config) {
        add_mapzero_config(data_map, zone, server_k, config);
    }
};

template<typename ModelConfig>
struct ModelConfigAdder<ModelConfig, 
                       std::enable_if_t<std::is_same_v<ModelConfig, TransMapConfig>>> {
    static void add(std::unordered_map<std::string, msgpack::object>& data_map,
                   msgpack::zone& zone,
                   ServerConstants& server_k,
                   GlobalConfig<ModelConfig>& config) {
        add_transmap_config(data_map, zone, server_k, config);
    }
};

template<typename ModelConfig>
struct ModelConfigAdder<ModelConfig, 
                       std::enable_if_t<std::is_same_v<ModelConfig, SmartMapConfig>>> {
    static void add(std::unordered_map<std::string, msgpack::object>& data_map,
                   msgpack::zone& zone,
                   ServerConstants& server_k,
                   GlobalConfig<ModelConfig>& config) {
        add_smartmap_config(data_map, zone, server_k, config);
    }
};

template<typename ModelConfig>
struct ModelConfigAdder<ModelConfig, 
                       std::enable_if_t<std::is_same_v<ModelConfig, GPRMConfig>>> {
    static void add(std::unordered_map<std::string, msgpack::object>& data_map,
                   msgpack::zone& zone,
                   ServerConstants& server_k,
                   GlobalConfig<ModelConfig>& config) {
        add_gprm_config(data_map, zone, server_k, config);
    }
};

template<typename ModelConfig>
void add_model_config(std::unordered_map<std::string, msgpack::object>& data_map,
                     msgpack::zone& zone,
                     ServerConstants& server_k,
                     GlobalConfig<ModelConfig>& config) {
    ModelConfigAdder<ModelConfig>::add(data_map, zone, server_k, config);
}


template<typename ModelConfig>
void config_server(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k,
    GlobalConfig<ModelConfig>& config
    ){
        std::unordered_map<std::string, msgpack::object> data_map;
        msgpack::zone zone;        
        data_map[server_k.func_k] = pack_str("config_server", zone);
        auto path_config = config.getPathConfig();
        auto train_config = config.getTrainConfig();
        auto launcher_config = config.getLauncherConfig();
        auto buffer_config = config.getBufferConfig();
        auto dfg_config = config.getDFGConfig();
        auto state_config = config.getStateConfig();

        auto enum_model = static_cast<int>(train_config.getEnumModel());
        data_map[server_k.model_id_k] = msgpack::object(static_cast<int>(enum_model));
        data_map["save_PER_last_prior"] = msgpack::object(launcher_config.getReplayBuffer() == EnumReplayBuffer::PER_BUFFER);
        data_map["use_xavier_uniform"] = msgpack::object(train_config.getUseXavieruniform());
        data_map["train_mode"] = msgpack::object(launcher_config.getEnumTaskType() == EnumTaskType::TRAIN);
        data_map["optimizer"] = msgpack::object(static_cast<int>(train_config.getEnumOptimizer()));
        data_map["use_ISW"] = msgpack::object(buffer_config.getUseISW());
        data_map["train_id"] = msgpack::object(static_cast<int>(launcher_config.getTraining()));
        data_map["buffer_id"] = msgpack::object(static_cast<int>(launcher_config.getReplayBuffer()));

        //dfg
        data_map["placement_id"] = msgpack::object(static_cast<int>(dfg_config.getPlacementApproach()));
        data_map["greedy_type_id"] = msgpack::object(static_cast<int>(dfg_config.getGreedyType()));
        data_map["scheduling_id"] = msgpack::object(static_cast<int>(dfg_config.getScheduling()));
        data_map["start_traversal_from_output"] = msgpack::object(dfg_config.getStartTraversalFromOutput());
        data_map["max_depth"] = msgpack::object(dfg_config.getMaxDepth());

        //state
        data_map["override_base_state"] = msgpack::object(state_config.getOverrideBaseState());
        data_map["use_edge_mask"] = msgpack::object(state_config.getUseCongestionEdgeMask());
        data_map["use_sym_mask"] = msgpack::object(state_config.getUseSymmetricMask());
        data_map["use_pe_mask"] = msgpack::object(state_config.getUseUsedPeMask());
        data_map["end_state_id"] = msgpack::object(static_cast<int>(state_config.getEndState()));
        data_map["use_rt_scheduler"] = msgpack::object(state_config.getUseRTScheduler());
        
        //launcher
        data_map["environment_id"] = msgpack::object(static_cast<int>(launcher_config.getEnvironment()));


        if(launcher_config.getTraining() == EnumTraining::PPO || launcher_config.getTraining() == EnumTraining::PPO_TRANSMAP || launcher_config.getTraining() == EnumTraining::HYBRID){
            add_ppo_config(data_map,config.getPPOConfig(),server_k,zone);
        }

        std::stringstream ss;
        config.print_ss(ss);
        data_map["config"] = msgpack::object(pack_str(ss.str(),zone));
        add_model_config<ModelConfig>(data_map,zone,server_k, config);
        add_path_config(data_map, path_config, server_k, zone);
        add_train_config(data_map, train_config, server_k, zone);
        send_data_map(socket, data_map);
        zmq::message_t reply;
        auto _ = socket->recv(reply, zmq::recv_flags::none);
        (void)_;

    }
    