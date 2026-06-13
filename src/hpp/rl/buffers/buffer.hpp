#ifndef BUFFER_HPP
#define BUFFER_HPP

#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_pre_training_buffer.hpp"
#include "src/hpp/datasets/mapping_dataset.hpp"
#include "src/hpp/enums/enum_replay_buffer.hpp"
#include "src/hpp/rl/buffers/PER_buffer.hpp"
#include "src/hpp/compilers/mapzero/rl/mapzero_buffer.hpp"
#include "src/hpp/rl/buffers/base_buffer.hpp"
#include <functional>
#include "src/hpp/utils/mapping_augmentation.hpp"
#include "src/hpp/rl/mapping_history.hpp"
#include "src/hpp/utils/mapping_augmentation.hpp"
#include "src/hpp/compilers/mapzero/configs/mapzero_config.hpp"
#include "src/hpp/compilers/smartmap/configs/smartmap_config.hpp"
#include "src/hpp/compilers/gprm/configs/gprm_config.hpp"
#include "src/hpp/compilers/transmap/rl/configs/transmap_config.hpp"
#include "src/hpp/compilers/gprm/rl/gprm_buffer.hpp"
#include "src/hpp/rl/buffers/supervised_buffer.hpp"
#include "src/hpp/rl/buffers/transmap_buffer.hpp"
template<typename StateT, typename ModelConfig>
class Buffer {
private:
    BaseBuffer<StateT, ModelConfig> buffer;
    GlobalConfig<ModelConfig> global_config;
    std::shared_ptr<zmq::socket_t> socket = nullptr;
    ServerConstants server_k;
    std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester = std::nullopt;
    std::string func_name;
    void (*save)(BaseBuffer<StateT,ModelConfig>& buffer, const MappingHistory<StateT>& mapping_history) = nullptr;

    void (*update_buffer_priorities)(BaseBuffer<StateT, ModelConfig>& base_buffer, 
                                torch::Tensor mapping_indices,
                                torch::Tensor states_indices,
                                torch::Tensor new_priorities) = nullptr;

    torch::Generator gen;

public:
    void set_cur_reanalyzed_map_idx(int idx){
        this->buffer.set_cur_reanalyzed_map_idx(idx);
    }
    Buffer(GlobalConfig<ModelConfig>& global_config,
        std::shared_ptr<zmq::socket_t> socket,
        const ServerConstants& server_k,
        std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester,
        std::string func_name);
    Buffer(); 
    void set_batch_requester(std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester){
        this->batch_requester = batch_requester;
    }
    std::shared_ptr<zmq::socket_t> get_socket()  {
        return this->socket;
    }
    void set_socket(std::shared_ptr<zmq::socket_t> socket){
        this->socket = socket;
    }

    void rebuild_from_config(GlobalConfig<ModelConfig>& global_config){
        this->global_config = global_config;
        this->buffer.set_config(global_config);
        this->build_functions();
    }

    void clear_buffer_from_to(int from, int to){
        this->buffer.clear_from_to(from, to);
    }
    void build_functions(){
        auto enum_replay_buffer = global_config.getLauncherConfig().getReplayBuffer();
        if(enum_replay_buffer == EnumReplayBuffer::PER_BUFFER){
            this->save = PER_add_mapping<StateT,ModelConfig>;
            this->update_buffer_priorities = PER_update_priorities<StateT,ModelConfig>;
        }else if(enum_replay_buffer == EnumReplayBuffer::SMARTMAP_PRE_TRAINING_BUFFER){
            this->save  = pretraining_smartmap_add_mapping<StateT,ModelConfig>;
            this->update_buffer_priorities = nullptr;
        }else if(enum_replay_buffer == EnumReplayBuffer::MAPZERO_BUFFER){
            this->save = mapzero_add_mapping<StateT,ModelConfig>;
            this->update_buffer_priorities = mapzero_update_priorities<StateT,ModelConfig>;
        }else if (enum_replay_buffer == EnumReplayBuffer::GPRM_BUFFER){
            this->save = gprm_add_mapping<StateT,ModelConfig>;
            this->update_buffer_priorities = nullptr;
        }else if( enum_replay_buffer == EnumReplayBuffer::SUPERVISED_BUFFER){
            this->save = supervised_buffer_add_mapping<StateT,ModelConfig>;
            this->update_buffer_priorities = nullptr;
        }else if( enum_replay_buffer == EnumReplayBuffer::DTB){
            this->save = add_mapping_dtb<StateT,ModelConfig>;
            this->update_buffer_priorities = nullptr;
        }
        else{
            std::runtime_error(std::format("Invalid ReplayBuffer '{}'", static_cast<int>(enum_replay_buffer)));
        }
    }

    void rebuild_functions(EnumReplayBuffer enum_replay_buffer){
        if(enum_replay_buffer == EnumReplayBuffer::PER_BUFFER){
            this->save = PER_add_mapping<StateT,ModelConfig>;
            this->update_buffer_priorities = PER_update_priorities<StateT,ModelConfig>;
        }else if(enum_replay_buffer == EnumReplayBuffer::SMARTMAP_PRE_TRAINING_BUFFER){
            this->save  = pretraining_smartmap_add_mapping<StateT,ModelConfig>;
            this->update_buffer_priorities = nullptr;
        }else if(enum_replay_buffer == EnumReplayBuffer::MAPZERO_BUFFER){
            this->save = mapzero_add_mapping<StateT,ModelConfig>;
            this->update_buffer_priorities = mapzero_update_priorities<StateT,ModelConfig>;
        }else if (enum_replay_buffer == EnumReplayBuffer::GPRM_BUFFER){
            this->save = gprm_add_mapping<StateT,ModelConfig>;
            this->update_buffer_priorities = nullptr;
        }else if( enum_replay_buffer == EnumReplayBuffer::SUPERVISED_BUFFER){
            this->save = supervised_buffer_add_mapping<StateT,ModelConfig>;
            this->update_buffer_priorities = nullptr;
        }
        else{
            std::runtime_error(std::format("Invalid ReplayBuffer '{}'", static_cast<int>(enum_replay_buffer)));
        }
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(this->buffer);
    }

    bool empty()const { return this->buffer.empty(); }


    int size(){ return this->buffer.get_number_of_mappings();}
    void add_mapping(const MappingHistory<StateT>& mapping_history);
    void update_priorities(torch::Tensor mapping_indices,
                           torch::Tensor states_indices,
                           torch::Tensor new_priorities);
    MappingDataset<StateT> get_batch(std::optional<EnumReplayBuffer> enum_replay_buffer_opt = std::nullopt);
    int get_random_mapping_idx();
    const std::vector<std::shared_ptr<StateT>> get_states_by_idx(const int& map_idx) const;
    int get_num_data_in_buffer() const;
    void update_with_renalyzed_mapping(MappingHistory<StateT>& mapping_history, const int& idx);

    template <typename T = ModelConfig>
    typename std::enable_if<std::is_same<T, GPRMConfig>::value>::type
    augment_data(const MappingHistory<StateT>& mapping_history);

    template <typename T = ModelConfig>
    typename std::enable_if<std::is_same<T, SmartMapConfig>::value>::type
    augment_data(const MappingHistory<StateT>& mapping_history);

    template <typename T= ModelConfig>
    typename std::enable_if<std::is_same<T, MapZeroConfig>::value>::type
    augment_data(const MappingHistory<StateT>& mapping_history);

    template <typename T = ModelConfig>
    typename std::enable_if<std::is_same<T, TransMapConfig>::value>::type
    augment_data(const MappingHistory<StateT>& mapping_history);

    void clear_buffer(){
        this->buffer.clear();
    }

};

#include "src/tpp/rl/buffer.tpp"

#endif