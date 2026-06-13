#ifndef UTIL_SERVER_HPP
#define UTIL_SERVER_HPP

#include <nlohmann/json.hpp>
#include <msgpack.hpp> 
#include <zmq_addon.hpp>
#include <vector>
#include <string>
#include <torch/torch.h>
#include "src/hpp/utils/constants/server_constants.hpp"
#include "src/hpp/configs/ppo_config.hpp"
#include "src/hpp/configs/train_config.hpp"
#include "src/hpp/configs/path_config.hpp"
#include "src/hpp/utils/util_torch.hpp"
#include "src/hpp/enums/enum_training.hpp"
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/enums/enum_task_type.hpp"
#include "src/hpp/compilers/gprm/configs/gprm_config.hpp"
#include "src/hpp/compilers/smartmap/configs/smartmap_config.hpp"
#include "src/hpp/compilers/mapzero/configs/mapzero_config.hpp"
#include "src/hpp/compilers/transmap/rl/configs/transmap_config.hpp"

using json = nlohmann::json;



std::vector<uint8_t> serialize_tensor(const torch::Tensor& tensor);

std::vector<torch::Tensor> get_last_priorities(std::shared_ptr<zmq::socket_t> socket, const ServerConstants& server_k);

void send_tensors(std::shared_ptr<zmq::socket_t> socket, const std::vector<c10::IValue>& values);

std::vector<torch::Tensor> receive_tensors(std::shared_ptr<zmq::socket_t> socket);


void send_data_map(std::shared_ptr<zmq::socket_t> socket, std::unordered_map<std::string, msgpack::object>& data_map ,zmq::send_flags flag = zmq::send_flags::none );


msgpack::object pack_str(const std::string& str,  msgpack::zone& zone);

std::tuple<std::string,std::string> format_checkpoint_strings(const int& n_rows, const int& n_cols, const std::string& checkpoint_path,const std::string& model_name);

void write_tensorboard(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k, 
    const std::unordered_map<std::string, double>& train_metrics,
    const std::unordered_map<std::string, double>& eval_metrics,
    const std::unordered_map<std::string, std::vector<double>>& train_hist_dict,
    const std::unordered_map<std::string, std::vector<double>>& test_hist_dict, const int& step);

void add_text_to_tensorboard(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k, const std::string& text);

void finish_server(std::shared_ptr<zmq::socket_t> socket, const ServerConstants& server_k);

void load_pre_trained_model(std::shared_ptr<zmq::socket_t> socket, const ServerConstants& server_k);

void save_model(std::shared_ptr<zmq::socket_t> socket, const ServerConstants& server_k, int cur_iter, bool train_finished ,
    const double entropy_coef, std::unordered_map<std::string, double>& extra_data);
    
zmq::multipart_t receive_msg(std::shared_ptr<zmq::socket_t> socket, int max_try);

std::vector<torch::Tensor> infer(std::shared_ptr<zmq::socket_t> socket, std::vector<c10::IValue>& model_inputs, const std::string& func_name, const ServerConstants& server_k);

torch::Tensor get_laplacian_embeddings(std::shared_ptr<zmq::socket_t> socket, std::string func_name, std::vector<std::vector<int>> edge_indices,
                              int D_dim, int num_nodes, bool is_undirected, const ServerConstants& server_k);


double compute_entropy_bonus(int step, int total_steps,PPOConfig& ppo_config);

void update_old_model(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k);

void add_path_config(std::unordered_map<std::string, msgpack::object>& data_map,const PathConfig& path_config, const ServerConstants& server_k,
msgpack::zone& zone);

void add_train_config(std::unordered_map<std::string, msgpack::object>& data_map,const TrainConfig& train_config, const ServerConstants& server_k,
msgpack::zone& zone);

void add_ppo_config(std::unordered_map<std::string, msgpack::object>& data_map,const PPOConfig& ppo_config, const ServerConstants& server_k,
msgpack::zone& zone);

inline void send_supervised_data(std::shared_ptr<zmq::socket_t> socket, 
                         std::vector<c10::IValue>& train_data,
                          ServerConstants& server_k, bool save_dataset){
    std::unordered_map<std::string, msgpack::object> data_map;
    msgpack::zone zone;

    data_map[server_k.func_k] = pack_str("receive_supervised_data", zone);
    data_map["save_dataset"] = msgpack::object(save_dataset, zone);
    send_data_map(socket, data_map, zmq::send_flags::sndmore);
    send_tensors(socket, train_data);

    #ifdef DEBUG
        std::cout << "[DEBUG] Waiting for training response from server..." << std::endl;
    #endif



    auto _ = receive_msg(socket, 10);
}

inline void send_supervised_data_parallel(std::shared_ptr<zmq::socket_t> socket, 
    std::vector<c10::IValue>& train_data,
     ServerConstants& server_k, int idx){
    std::unordered_map<std::string, msgpack::object> data_map;
    msgpack::zone zone;

    data_map[server_k.func_k] = pack_str("receive_supervised_data_parallel", zone);
    data_map["idx"] = msgpack::object(idx, zone);
    send_data_map(socket, data_map, zmq::send_flags::sndmore);
    send_tensors(socket, train_data);

    #ifdef DEBUG
    std::cout << "[DEBUG] Waiting for training response from server..." << std::endl;
    #endif

    auto _ = receive_msg(socket, 10);
}

inline void create_list_with_size(std::shared_ptr<zmq::socket_t> socket, 
     ServerConstants& server_k, int size){
    std::unordered_map<std::string, msgpack::object> data_map;
    msgpack::zone zone;

    data_map[server_k.func_k] = pack_str("create_dataset_with_size", zone);
    data_map["size"] = msgpack::object(size, zone);
    send_data_map(socket, data_map, zmq::send_flags::none);

    #ifdef DEBUG
    std::cout << "[DEBUG] Waiting for training response from server..." << std::endl;
    #endif

    auto _ = receive_msg(socket, 10);
}


inline void save_supervised_data(std::shared_ptr<zmq::socket_t> socket, 
    ServerConstants& server_k, std::string path){
   std::unordered_map<std::string, msgpack::object> data_map;
   msgpack::zone zone;

   data_map[server_k.func_k] = pack_str("save_dataset", zone);
   data_map["path"] = pack_str(path, zone);
   send_data_map(socket, data_map, zmq::send_flags::none);

   #ifdef DEBUG
   std::cout << "[DEBUG] Waiting for training response from server..." << std::endl;
   #endif

   auto _ = receive_msg(socket, 10);
}

inline void save_block_of_data(std::shared_ptr<zmq::socket_t> socket, 
    ServerConstants& server_k, std::string path){

   std::unordered_map<std::string, msgpack::object> data_map;
   msgpack::zone zone;

   data_map[server_k.func_k] = pack_str("save_block_of_supervised_data", zone);
   data_map["path_to_save_data"] = pack_str(path, zone);
   send_data_map(socket, data_map, zmq::send_flags::none);
 

   #ifdef DEBUG
   std::cout << "[DEBUG] Waiting for training response from server..." << std::endl;
   #endif

   auto _ = receive_msg(socket, 10);
}

inline void load_supervised_data(std::shared_ptr<zmq::socket_t> socket, 
    ServerConstants& server_k, std::string path){
   std::unordered_map<std::string, msgpack::object> data_map;
   msgpack::zone zone;

   data_map[server_k.func_k] = pack_str("load_dataset", zone);
   data_map["path"] = pack_str(path, zone);
   send_data_map(socket, data_map, zmq::send_flags::none);

   #ifdef DEBUG
   std::cout << "[DEBUG] Waiting for training response from server..." << std::endl;
   #endif

   auto _ = receive_msg(socket, 10);
}
inline void load_dataset_from_checkpoint(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k){
    std::unordered_map<std::string, msgpack::object> data_map;
    msgpack::zone zone;

    data_map[server_k.func_k] = pack_str("load_dataset_from_checkpoint", zone);

    send_data_map(socket, data_map, zmq::send_flags::none);
    
    auto _ = receive_msg(socket, 10);
    
}

std::unordered_map<std::string, double> server_train(std::shared_ptr<zmq::socket_t> socket, std::vector<c10::IValue> tensors, int cur_epoch, 
    const std::optional<double> w_entropy_loss,  
    ServerConstants& server_k,
    int train_id = -1
);
    
void add_mapzero_config( std::unordered_map<std::string, msgpack::object>& data_map, msgpack::zone& zone, ServerConstants& server_k,
    GlobalConfig<MapZeroConfig>& global_config);

void add_gprm_config(std::unordered_map<std::string, msgpack::object>& data_map, msgpack::zone& zone, ServerConstants& server_k,
GlobalConfig<GPRMConfig>& global_config);

void add_smartmap_config(std::unordered_map<std::string, msgpack::object>& data_map, msgpack::zone& zone, ServerConstants& server_k,
GlobalConfig<SmartMapConfig>& global_config);


inline void load_model_server(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k,
    std::string load_model 
    ){
        std::unordered_map<std::string, msgpack::object> data_map;
        msgpack::zone zone;        
        data_map[server_k.func_k] = pack_str("load_nn_only", zone);
        data_map["model_path"] = pack_str(load_model, zone);

        zmq::message_t reply;
        auto _ = socket->recv(reply, zmq::recv_flags::none);
        (void)_;
}

inline void add_transmap_config(std::unordered_map<std::string, msgpack::object>& data_map, msgpack::zone& zone, ServerConstants& server_k,
GlobalConfig<TransMapConfig>& global_config){
    auto transmap_config = global_config.getModelConfig();
    auto train_config = global_config.getTrainConfig();

    data_map["max_dfg_feat"] = msgpack::object(transmap_config.getMaxDfgFeat());  
    data_map["max_adg_feat"] = msgpack::object(transmap_config.getMaxAdgFeat());
    data_map["lpe_dim"] = msgpack::object(transmap_config.getLpeDim());
    data_map["E_dfg_node_dim"] = msgpack::object(transmap_config.getEDfgNodeDim());
    data_map["E_dfg_edge_dim"] = msgpack::object(transmap_config.getEDfgEdgeDim());
    data_map["E_adg_node_dim"] = msgpack::object(transmap_config.getEAdgNodeDim());
    data_map["E_adg_edge_dim"] = msgpack::object(transmap_config.getEAdgEdgeDim());
    data_map["out_dim_policy"] = msgpack::object(transmap_config.getOutDimPolicy());
    data_map["state_dim"] = msgpack::object(transmap_config.getStateDim());
    data_map["out_dim"] = msgpack::object(transmap_config.getOutDim());
    data_map["ff_dim"] = msgpack::object(transmap_config.getFfDim());
    data_map["n_heads"] = msgpack::object(transmap_config.getNHeads());
    data_map["hidden_dim"] = msgpack::object(transmap_config.getHiddenDim());
    data_map["ortho_scaling"] = msgpack::object(transmap_config.getOrthoScaling());
    data_map["dropout"] = msgpack::object(transmap_config.getDropout());
    data_map["activation_fn"] = pack_str(transmap_config.getActivationFn(), zone);
    data_map["nb_features"] = msgpack::object(transmap_config.getNbFeatures());
    data_map["num_mhd_layers"] = msgpack::object(transmap_config.getNumMhdLayers());

}

void send_data_with_tensor( std::shared_ptr<zmq::socket_t> socket,
    std::unordered_map<std::string, msgpack::object>& data_map,
    const std::vector<c10::IValue>& tensors
);

template<typename ModelConfig>
void config_server(std::shared_ptr<zmq::socket_t> socket,ServerConstants& server_k, GlobalConfig<ModelConfig>& config);

template <typename List>
msgpack::object get_vector_obj(const List& vector);

    

#include "src/tpp/utils/util_server.tpp"
#endif