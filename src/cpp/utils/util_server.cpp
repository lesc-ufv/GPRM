#include "src/hpp/utils/util_server.hpp"

std::vector<uint8_t> serialize_tensor(const torch::Tensor& tensor) {
    const auto* data_ptr = reinterpret_cast<const uint8_t*>(tensor.data_ptr());
    size_t data_size = tensor.nbytes();

    auto shape = tensor.sizes();
    auto dtype = tensor.dtype().name();

    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> packer(buffer);

    packer.pack_array(3);
    packer.pack_array(shape.size());
    for (auto dim : shape) packer.pack(dim);  
    packer.pack(std::string(dtype));        
    packer.pack_bin(data_size);             
    packer.pack_bin_body(reinterpret_cast<const char*>(data_ptr), data_size);  

    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

void send_tensors(std::shared_ptr<zmq::socket_t> socket, const std::vector<c10::IValue>& values) {
    zmq::multipart_t multipart_msg;

    for (const auto& val : values) {
        if (val.isTensor()) {
            torch::Tensor tensor = val.toTensor().contiguous();

            std::vector<uint8_t> serialized_data = serialize_tensor(tensor);

            zmq::message_t zmq_msg(serialized_data.size());
            std::memcpy(zmq_msg.data(), serialized_data.data(), serialized_data.size());
            multipart_msg.add(std::move(zmq_msg));
        }else if (val.isInt()){
            int64_t int_val = val.toInt();
            zmq::message_t int_msg(sizeof(int64_t));
            std::memcpy(int_msg.data(), &int_val, sizeof(int64_t));
            multipart_msg.add(std::move(int_msg));
        }else{
            throw std::runtime_error("Unsupported type for sending data with tensor.");
        }
        
    }

    multipart_msg.send(*socket);
}


torch::ScalarType parse_dtype(const std::string& dtype_str) {
    if (dtype_str == "torch.float32") {
        return torch::kFloat32;
    } else if (dtype_str == "torch.float64") {
        return torch::kFloat64;
    } else if (dtype_str == "torch.int64") {
        return torch::kInt64;
    } else {
        throw std::runtime_error("Unknown dtype: " + dtype_str);
    }
}
std::vector<torch::Tensor> receive_tensors(std::shared_ptr<zmq::socket_t> socket) {
    std::vector<torch::Tensor> tensors;
    ServerConstants server_k;
    int max_try = 10;
    zmq::multipart_t multipart;
    int count = 0;
    while (true) {
        try {
            multipart.recv(*socket);
            break;
        } catch (const zmq::error_t& e) {
            if (e.num() == EINTR) {
                count++;
                if (count >= max_try) throw;
                continue;
            } else {
                throw;
            }
        }
    }

#ifdef DEBUG
    std::cout << "[DEBUG] Received multipart with " << multipart.size() << " part(s)" << std::endl;
#endif

    if (multipart.empty()) {
        throw std::runtime_error("Received empty multipart message.");
    }

    size_t payload_index = multipart.size() == 1 ? 0 : 1;
    auto& reply = multipart.at(payload_index);

#ifdef DEBUG
    std::cout << "[DEBUG] First message part size: " << reply.size() << std::endl;
#endif

    msgpack::object_handle oh = msgpack::unpack(static_cast<const char*>(reply.data()), reply.size());
    msgpack::object obj = oh.get();

#ifdef DEBUG
    std::cout << "[DEBUG] Received msgpack type: " << static_cast<int>(obj.type) << std::endl;
    std::cout << "[DEBUG] Raw msgpack object: " << obj << std::endl << std::flush;
#endif

    if (obj.type != msgpack::type::MAP) {
        throw std::runtime_error("Error: server response is not a map.");
    }

    std::map<std::string, msgpack::object> response_data;
    obj.convert(response_data);

#ifdef DEBUG
    std::cout << "[DEBUG] Unpacked response map with keys: ";
    for (const auto& kv : response_data) {
        std::cout << kv.first << " ";
    }
    std::cout << std::endl;
#endif

    auto& tensor_specs_obj = response_data[server_k.tensors_k];
    auto& data_obj = response_data[server_k.data_k];

    std::vector<std::map<std::string, msgpack::object>> tensor_specs;
    tensor_specs_obj.convert(tensor_specs);

    std::vector<std::string> tensor_data;
    data_obj.convert(tensor_data);

    if (tensor_specs.size() != tensor_data.size()) {
        throw std::runtime_error("Mismatch between tensor specs and data.");
    }

#ifdef DEBUG
    std::cout << "[DEBUG] Number of tensors: " << tensor_specs.size() << std::endl;
#endif

    for (size_t i = 0; i < tensor_specs.size(); ++i) {
        auto& spec = tensor_specs[i];
        auto& data = tensor_data[i];

        auto shape = spec[server_k.shape_k].as<std::vector<int64_t>>();
        auto dtype_str = spec[server_k.dtype_k].as<std::string>();

        torch::ScalarType dtype = parse_dtype(dtype_str);
        auto buffer = std::vector<uint8_t>(data.begin(), data.end());
        torch::Tensor tensor = torch::from_blob(buffer.data(), shape, dtype).clone();

#ifdef DEBUG
        std::cout << "[DEBUG] Tensor " << i << ": shape = [";
        for (auto dim : shape) std::cout << dim << " ";
        std::cout << "], dtype = " << dtype_str << ", size = " << buffer.size() << std::endl;
#endif

        tensors.push_back(tensor);
    }

    return tensors;
}

void send_data_with_tensor(
    std::shared_ptr<zmq::socket_t> socket,
    std::unordered_map<std::string, msgpack::object>& data_map,
    const std::vector<c10::IValue>& tensors
) {
    zmq::multipart_t multipart_msg;
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, data_map);

#ifdef DEBUG
    std::cout << "[DEBUG] send_data_with_tensor: data_map size = " << data_map.size() << std::endl;
    for (const auto& kv : data_map) {
        std::cout << "  [DEBUG] key: " << kv.first << std::endl;
    }
#endif

    if (sbuf.size() == 0) {
#ifdef DEBUG
        std::cerr << "[ERROR] send_data_with_tensor: Packed data_map is empty!" << std::endl;
#endif
        throw std::runtime_error("send_data_with_tensor: Packed data_map is empty!");
    }

    zmq::message_t data_msg(sbuf.data(), sbuf.size());
    multipart_msg.add(std::move(data_msg));

    for (const auto& val : tensors) {
        if (val.isTensor()) {
            torch::Tensor tensor = val.toTensor().contiguous();
            std::vector<uint8_t> serialized_data = serialize_tensor(tensor);
            zmq::message_t tensor_msg(serialized_data.data(), serialized_data.size());
            multipart_msg.add(std::move(tensor_msg));
#ifdef DEBUG
            std::cout << "[DEBUG] Added tensor of size " << serialized_data.size() << " bytes to multipart message." << std::endl;
#endif
        } else if (val.isInt()) {
            int64_t int_val = val.toInt();
            zmq::message_t int_msg(sizeof(int64_t));
            std::memcpy(int_msg.data(), &int_val, sizeof(int64_t));
            multipart_msg.add(std::move(int_msg));
#ifdef DEBUG
            std::cout << "[DEBUG] Added int64 value to multipart message: " << int_val << std::endl;
#endif
        } else {
#ifdef DEBUG
            std::cerr << "[ERROR] Unsupported IValue type in send_data_with_tensor." << std::endl;
#endif
            throw std::runtime_error("Unsupported type for sending data with tensor.");
        }
    }

#ifdef DEBUG
    std::cout << "[DEBUG] Total multipart message parts: " << multipart_msg.size() << std::endl;
    for (size_t i = 0; i < multipart_msg.size(); ++i) {
        std::cout << "  [DEBUG] Part " << i << " size: " << multipart_msg.at(i).size() << " bytes" << std::endl;
    }
#endif

    multipart_msg.send(*socket);

#ifdef DEBUG
    std::cout << "[DEBUG] Multipart message sent." << std::endl;
#endif
}

void send_data_map(
    std::shared_ptr<zmq::socket_t> socket,
    std::unordered_map<std::string, msgpack::object>& data_map,
    zmq::send_flags flag
) {
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, data_map);

#ifdef DEBUG
    std::cout << "[DEBUG] Packing data map with " << data_map.size() << " keys: ";
    for (const auto& kv : data_map) {
        std::cout << kv.first << " ";
    }
    std::cout << "\n[DEBUG] Packed size: " << sbuf.size() << " bytes" << std::endl;
#endif
    if (sbuf.size() == 0) {
        throw std::runtime_error(" Packed data_map is empty!");
    }

    zmq::message_t request(sbuf.size());
    std::memcpy(request.data(), sbuf.data(), sbuf.size());

#ifdef DEBUG
    std::cout << "[DEBUG] Sending message with flag: " << static_cast<int>(flag) << std::endl;
#endif

    socket->send(request, flag);
}


msgpack::object pack_str(const std::string& str,msgpack::zone& zone){

    return msgpack::object(str, zone);
}




void save_model(std::shared_ptr<zmq::socket_t> socket, const ServerConstants& server_k, int cur_iter, bool train_finished ,
const double entropy_coef,std::unordered_map<std::string, double>& extra_data) {
    std::unordered_map<std::string, msgpack::object> data_map;
    msgpack::zone zone;

    data_map[server_k.func_k] = pack_str("save_model",zone);
    data_map["cur_iter"] = msgpack::object(cur_iter,zone);
    data_map["train_finished"] = msgpack::object(train_finished,zone);
    data_map["entropy_coef"]  = msgpack::object(entropy_coef);
    for (const auto& pair : extra_data) {
        data_map[pair.first] = msgpack::object(pair.second, zone);
    }
    send_data_map(socket, data_map);
    zmq::message_t reply;

    auto _ = receive_msg(socket,10);
    (void)_;

}
void finish_server(std::shared_ptr<zmq::socket_t> socket, const ServerConstants& server_k){
    std::unordered_map<std::string, msgpack::object> data_map;
    msgpack::zone zone;

    data_map[server_k.func_k] = pack_str("finish_server",zone);
    send_data_map(socket, data_map);
    zmq::message_t reply;
    auto _ = receive_msg(socket,10);
    (void)_;

}


std::vector<torch::Tensor> get_last_priorities(std::shared_ptr<zmq::socket_t> socket, const ServerConstants& server_k){
    std::unordered_map<std::string, msgpack::object> data_map;  
    msgpack::zone zone;

    data_map[server_k.func_k] = msgpack::object("get_last_priorities",zone);

    send_data_map(socket, data_map);

    zmq::message_t reply;
    return receive_tensors(socket);
}

torch::Tensor get_laplacian_embeddings(std::shared_ptr<zmq::socket_t> socket, std::string func_name, std::vector<std::vector<int>> edge_indices,
                              int D_dim, int num_nodes, bool is_undirected, const ServerConstants& server_k){
    std::unordered_map<std::string, msgpack::object> data_map;  
    msgpack::zone zone;

    data_map[server_k.func_k] = msgpack::object(func_name,zone);
    data_map["D_dim"] = msgpack::object(D_dim,zone);
    data_map["num_nodes"] = msgpack::object(num_nodes,zone);
    data_map["is_undirected"] = msgpack::object(is_undirected,zone);
    std::vector<c10::IValue> edge_idx = {matrix_to_tensor<int,int>(edge_indices)};                                
    send_data_with_tensor(socket, data_map, edge_idx);
    auto lpe = receive_tensors(socket);

    return torch::stack(lpe);
}

std::tuple<std::string,std::string> format_checkpoint_strings(const int& n_rows, const int& n_cols, const std::string& checkpoint_path,const std::string& model_name){
    std::string dims_str = std::to_string(n_rows) + "x" + std::to_string(n_cols);
    return std::make_tuple<std::string,std::string>(checkpoint_path + model_name + "/" + dims_str + "/",
                                                    model_name + dims_str + ".pth");
}
void write_tensorboard(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k, 
    const std::unordered_map<std::string, double>& train_metrics,
    const std::unordered_map<std::string, double>& eval_metrics,
    const std::unordered_map<std::string, std::vector<double>>& train_hist_dict,
    const std::unordered_map<std::string, std::vector<double>>& test_hist_dict, const int& step) {
    std::unordered_map<std::string, msgpack::object> data_map;

    msgpack::zone zone;

    std::stringstream buffer_train, buffer_eval, buffer_train_hist, buffer_test_hist;
    msgpack::pack(buffer_train, train_metrics);
    msgpack::pack(buffer_eval, eval_metrics);
    msgpack::pack(buffer_train_hist, train_hist_dict);
    msgpack::pack(buffer_test_hist, test_hist_dict);

    msgpack::object_handle train_handle     = msgpack::unpack(buffer_train.str().data(), buffer_train.str().size());
    msgpack::object_handle eval_handle      = msgpack::unpack(buffer_eval.str().data(), buffer_eval.str().size());
    msgpack::object_handle train_hist_handle= msgpack::unpack(buffer_train_hist.str().data(), buffer_train_hist.str().size());
    msgpack::object_handle test_hist_handle = msgpack::unpack(buffer_test_hist.str().data(), buffer_test_hist.str().size());

    data_map["Train"]       = train_handle.get();
    data_map["Eval"]        = eval_handle.get();
    data_map["TrainHist"]   = train_hist_handle.get();
    data_map["TestHist"]    = test_hist_handle.get();
    data_map["step"]        = msgpack::object(step);

    std::string func_name = "write_tensorboard";
    data_map[server_k.func_k] = pack_str(func_name, zone);

    send_data_map(socket, data_map, zmq::send_flags::none);

    zmq::multipart_t reply = receive_msg(socket, 10);
}


void add_text_to_tensorboard(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k, const std::string& text) {
    std::unordered_map<std::string, msgpack::object> data_map;

    msgpack::zone zone;

    data_map["text"] = pack_str(text, zone);  
    std::string func_name = "add_text_tensorboard";
    data_map[server_k.func_k] = pack_str(func_name,zone);
    send_data_map(socket, data_map, zmq::send_flags::none);

    zmq::multipart_t reply = receive_msg(socket,10);
}

zmq::multipart_t receive_msg(std::shared_ptr<zmq::socket_t> socket, int max_try) {
    int count = 0;

    while (true) {
        try {
            zmq::multipart_t multipart;
            multipart.recv(*socket);

            if (multipart.empty()) {
#ifdef DEBUG
                std::cerr << "[DEBUG] Received empty multipart message." << std::endl;
#endif
                throw std::runtime_error("Received empty multipart message.");
            }

#ifdef DEBUG
            std::cout << "[DEBUG] Received multipart message with "
                      << multipart.size() << " parts." << std::endl;
#endif

            return multipart;
        } catch (const zmq::error_t& e) {
            if (e.num() == EINTR) {
                count += 1;
#ifdef DEBUG
                std::cerr << "[DEBUG] EINTR detected, retrying " << count << "/" << max_try << std::endl;
#endif
                if (count >= 100000) {
                    throw std::runtime_error("Max retries exceeded in receive_msg.");
                }
                continue;
            } else {
                throw;
            }
        }
    }
}



std::vector<torch::Tensor> infer(
    std::shared_ptr<zmq::socket_t> socket, 
    std::vector<c10::IValue>& model_inputs, 
    const std::string& func_name, 
    const ServerConstants& server_k
) {
    try {
#ifdef DEBUG
        std::cout << "[DEBUG] Starting infer()" << std::endl;
        std::cout << "[DEBUG] Function name: " << func_name << std::endl;
        std::cout << "[DEBUG] Number of model inputs: " << model_inputs.size() << std::endl;
#endif

        std::unordered_map<std::string, msgpack::object> data_map;
        msgpack::zone zone;

        data_map[server_k.func_k] = pack_str(func_name, zone);

        msgpack::sbuffer sbuf;
        msgpack::pack(sbuf, data_map);

        zmq::multipart_t multipart;

        zmq::message_t data_msg(sbuf.size());
        std::memcpy(data_msg.data(), sbuf.data(), sbuf.size());
        multipart.add(std::move(data_msg));

#ifdef DEBUG
        std::cout << "[DEBUG] Added function metadata to multipart. Size: " << sbuf.size() << " bytes" << std::endl;
#endif

        for (size_t i = 0; i < model_inputs.size(); ++i) {
            const auto& val = model_inputs[i];
            if (val.isTensor()) {
                torch::Tensor tensor = val.toTensor().contiguous();
                std::vector<uint8_t> serialized = serialize_tensor(tensor);
                zmq::message_t tensor_msg(serialized.size());
                std::memcpy(tensor_msg.data(), serialized.data(), serialized.size());
                multipart.add(std::move(tensor_msg));

#ifdef DEBUG
                std::cout << "[DEBUG] Added tensor " << i
                          << " | shape: " << tensor.sizes()
                          << " | dtype: " << tensor.dtype()
                          << " | serialized size: " << serialized.size()
                          << std::endl;
#endif
            } else if (val.isInt()){
                int64_t int_val = val.toInt();
                zmq::message_t int_msg(sizeof(int64_t));
                std::memcpy(int_msg.data(), &int_val, sizeof(int64_t));
                std::cout << "[DEBUG] Added int " << int_val
                << " - iter: " << i << std::endl;
                multipart.add(std::move(int_msg));
            } else {
                throw std::runtime_error("[Error] Skipped non-tensor input at index ");

            }
        }

#ifdef DEBUG
        std::cout << "[DEBUG] Total multipart message parts: " << multipart.size() << std::endl;
        std::cout << "[DEBUG] Sending multipart message..." << std::endl;
#endif

        multipart.send(*socket);

#ifdef DEBUG
        std::cout << "[DEBUG] Message sent. Waiting for response..." << std::endl;
#endif

        return receive_tensors(socket);
    } catch (const std::exception& e) {
        std::cerr << "[DEBUG] Error in infer function: " << e.what() << std::endl;
         throw;
    }
}


void update_old_model(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k){
    std::unordered_map<std::string, msgpack::object> data_map;

    std::string  func_name = "update_old_model";
    msgpack::zone zone;

    data_map[server_k.func_k] = pack_str(func_name, zone);
        
    send_data_map(socket, data_map);

    zmq::multipart_t reply;
    reply = receive_msg(socket,10);
    
}

double compute_entropy_bonus(int step, int total_steps,PPOConfig& ppo_config) {
    double initial_entropy_weigth = ppo_config.getInitialEntropyCoef();   
    double final_entropy_weigth = ppo_config.getTargetEntropyCoef();   
    int step_to_start_decay_entropy = ppo_config.getStepToStartDecayEntropy();
    if (step < step_to_start_decay_entropy) {
        return initial_entropy_weigth;
    }

    step = step - step_to_start_decay_entropy;
    
    double weight_decay_rate = ppo_config.getWeightDecayEntropy();
    double progress = static_cast<double>(step) / total_steps;
    double entropy_weigth = std::pow(weight_decay_rate, step)*(initial_entropy_weigth + progress*(final_entropy_weigth - initial_entropy_weigth));
    return std::max(entropy_weigth, final_entropy_weigth);  
}

void add_path_config(std::unordered_map<std::string, msgpack::object>& data_map,const PathConfig& path_config, const ServerConstants& server_k,
msgpack::zone& zone){
    data_map[server_k.save_checkpoint_path_k] = pack_str(path_config.getSaveCheckpointPath(), zone);
    data_map[server_k.load_checkpoint_path_k] = pack_str(path_config.getLoadCheckpointPath(), zone);
    data_map[server_k.save_checkpoint_name_k] = pack_str(path_config.getCheckpointName(), zone);
    data_map[server_k.tensorboard_log_path_k] = pack_str(path_config.getTensorboardLogPath(),zone);
    data_map["load_model_path"] = pack_str(path_config.get_load_model_path(),zone);

}

void add_train_config(std::unordered_map<std::string, msgpack::object>& data_map,const TrainConfig& train_config, const ServerConstants& server_k,
msgpack::zone& zone){
    data_map[server_k.device_k] = pack_str(train_config.getDevice(), zone);
    data_map["actor_lr"] = msgpack::object(train_config.getActorLearningRate(),zone);
    data_map["critic_lr"] = msgpack::object(train_config.getCriticLearningRate(),zone);
    data_map["actor_min_lr"] = msgpack::object(train_config.getActorMinLr(),zone);
    data_map["critic_min_lr"] = msgpack::object(train_config.getCriticMinLr(),zone);
    data_map[server_k.lr_decay_k] = msgpack::object(train_config.getLrDecay(),zone);
    data_map[server_k.lr_decay_steps_k] = msgpack::object(train_config.getLrDecaySteps(),zone);
    data_map[server_k.SEED] = msgpack::object(train_config.getSeed(),zone);
    data_map[server_k.max_grad_norm_k] = msgpack::object(train_config.getMaxGradNorm(),zone);
    data_map["weight_decay"] = msgpack::object(train_config.getWeightDecay());
    data_map["warmup_steps"] = msgpack::object(train_config.getWarmupSteps());
    data_map["total_steps"] = msgpack::object(train_config.getSteps());
    data_map["lr_scheduler"] = pack_str(train_config.getLRScheduler(),zone);
    data_map["use_chunked_supervised_dataset"] = msgpack::object(train_config.getUseChunkedSupervisedDataset(),zone);
    data_map["chunked_dataset_path"] = pack_str(train_config.getPathToLoadSupervisedData(),zone);
}

void add_mapzero_config( std::unordered_map<std::string, msgpack::object>& data_map, msgpack::zone& zone, ServerConstants& server_k,
    GlobalConfig<MapZeroConfig>& global_config){

   auto mapzero_config = global_config.getModelConfig();
//    auto learning_rate = global_config.getTrainConfig().getLearningRate();
   auto dimensions = global_config.getCGRAConfig().getDimensionsList()[0];
   auto total_PEs = dimensions.first * dimensions.second;
   data_map[server_k.tensors_k] = pack_str(server_k.NONE,zone);  
   data_map[server_k.in_dim_feat_dfg_k] = msgpack::object(mapzero_config.dfg_feat_dim);
   data_map[server_k.in_dim_feat_cgra_k] = msgpack::object(mapzero_config.cgra_feat_dim);
   data_map["number_of_spatial_PEs"] = msgpack::object(total_PEs);

   data_map[server_k.out_dim_k] = msgpack::object(mapzero_config.getOutDim());
   data_map[server_k.n_heads_k] = msgpack::object(mapzero_config.getNHeads());
   data_map[server_k.negative_slope_k] = msgpack::object(mapzero_config.getNegativeSlope());
//    data_map[server_k.learning_rate_k] = msgpack::object(learning_rate);

}

void add_gprm_config(std::unordered_map<std::string, msgpack::object>& data_map, msgpack::zone& zone, ServerConstants& server_k,
    GlobalConfig<GPRMConfig>& global_config) {

    auto gprm_config = global_config.getModelConfig();
    data_map["lpe_dim"] = msgpack::object(gprm_config.getLpeDim());
    data_map["vocab_len"] = msgpack::object(gprm_config.getVocabLen());
    data_map["feat_out_dim"] = msgpack::object(gprm_config.getFeatureOutDim());
    data_map["out_dim"] = msgpack::object(gprm_config.getOutDim());
    data_map["ff_out_dim"] = msgpack::object(gprm_config.getFfDim());
    data_map["GIN_layers_list"] = msgpack::object(gprm_config.getGinLayersList(), zone);
    data_map["n_graph_embed_layers"] = msgpack::object(gprm_config.getNGraphEmbedLayers());
    data_map["n_transformer_layers"] = msgpack::object(gprm_config.getNTransformerLayers());
    data_map["n_heads"] = msgpack::object(gprm_config.getNHeads());
    data_map["activation_func"] = pack_str(gprm_config.getActivationFunc(),zone);
    data_map["use_mlp"] = msgpack::object(gprm_config.getUseMlp());
    data_map["dropout"] = msgpack::object(gprm_config.getDropout());
    data_map["use_rms_norm"] = msgpack::object(gprm_config.getUseRmsNorm());
    data_map["use_norm_before_pred"] = msgpack::object(gprm_config.getUseNormBeforePred());
    data_map["negative_slope"] = msgpack::object(gprm_config.getNegativeSlope());
    data_map["max_len"] = msgpack::object(gprm_config.getMaxLen());
    data_map["norm_initial_emb"] = msgpack::object(gprm_config.getNormInitialEmb());
    data_map["use_gin_emb"] = msgpack::object(gprm_config.getUseGinEmb());
    data_map["use_mobility_info"] = msgpack::object(gprm_config.getUseMobilityInfo());
    data_map["use_coord_info"] = msgpack::object(gprm_config.getUseCoordInfo());
    data_map["use_placement_order"] = msgpack::object(gprm_config.getUsePlacementOrder());
    data_map["use_gnn"] = msgpack::object(gprm_config.getUseGnn());
    data_map["use_transformer"] = msgpack::object(gprm_config.getUseTransformer());
    data_map["ignore_lpe"] = msgpack::object(gprm_config.getIgnoreLpe());
    data_map["use_scale"] = msgpack::object(gprm_config.getUseScale());
}
void add_smartmap_config(std::unordered_map<std::string, msgpack::object>& data_map, msgpack::zone& zone, ServerConstants& server_k,
    GlobalConfig<SmartMapConfig>& global_config) 
{

    auto smartmap_config = global_config.getModelConfig();
    auto train_config = global_config.getTrainConfig();

    // data_map[server_k.tensors_k] = pack_str(server_k.NONE,zone);  
    data_map[server_k.in_dim_feat_dfg_k] = msgpack::object(smartmap_config.dfg_in_feat_dim);
    data_map[server_k.in_dim_feat_cgra_k] = msgpack::object(smartmap_config.cgra_in_feat_dim);
    data_map[server_k.out_dim_k] = msgpack::object(smartmap_config.getOutDim());
    data_map[server_k.n_heads_k] = msgpack::object(smartmap_config.getNHeads());
    // data_map[server_k.learning_rate_k] = msgpack::object(train_config.getLearningRate());
    data_map["negative_slope"] = msgpack::object(smartmap_config.getNegativeSlope());

}


void add_ppo_config(std::unordered_map<std::string, msgpack::object>& data_map,const PPOConfig& ppo_config, const ServerConstants& server_k,
msgpack::zone& zone){
    data_map[server_k.clip_param_k] = msgpack::object(ppo_config.getClipParam(),zone);
    data_map[server_k.value_loss_coef_k] = msgpack::object(ppo_config.getValueLossCoef(),zone);
    data_map[server_k.kl_target_k] = msgpack::object(ppo_config.getKlTarget(),zone);
}


std::unordered_map<std::string, double> server_train(
    std::shared_ptr<zmq::socket_t> socket,
    std::vector<c10::IValue> tensors,
    int cur_step,
    const std::optional<double> w_entropy_loss,
    ServerConstants& server_k,
    int train_id
) {
    std::unordered_map<std::string, msgpack::object> data_map;
    msgpack::zone zone;

    data_map[server_k.func_k] = pack_str("train", zone);
    data_map["cur_step"] = msgpack::object(cur_step);
    data_map["train_id"] = msgpack::object(train_id, zone);

    if (w_entropy_loss.has_value()) {
        data_map[server_k.entropy_coef_k] = msgpack::object(w_entropy_loss.value());
    }

#ifdef DEBUG
    std::cout << "[DEBUG] Sending training request to server..." << std::endl;
#endif

    send_data_map(socket, data_map, zmq::send_flags::sndmore);
    send_tensors(socket, tensors);

#ifdef DEBUG
    std::cout << "[DEBUG] Waiting for training response from server..." << std::endl;
#endif

    zmq::multipart_t reply_parts = receive_msg(socket, 10);

    if (reply_parts.empty()) {
        throw std::runtime_error("Empty response received in server_train.");
    }

    auto& reply = reply_parts.at(0);  
    msgpack::object_handle oh = msgpack::unpack(static_cast<char*>(reply.data()), reply.size());
    msgpack::object deserialized = oh.get();

    std::unordered_map<std::string, double> losses;
    deserialized.convert(losses);

#ifdef DEBUG
    std::cout << "[DEBUG] Received losses:" << std::endl;
    for (const auto& [k, v] : losses) {
        std::cout << "  " << k << ": " << v << std::endl;
    }
#endif

    return losses;
}


