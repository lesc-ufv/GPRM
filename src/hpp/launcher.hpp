#ifndef LAUNCHER_HPP
#define LAUNCHER_HPP
#include <tbb/concurrent_unordered_set.h>
#include "src/hpp/rl/environment.hpp"
#include "src/hpp/rl/collate.hpp"
#include "src/hpp/rl/self_mapping.hpp"
#include "src/hpp/rl/buffers/buffer.hpp"
#include "src/hpp/utils/constants/path_constants.hpp"
#include "src/hpp/utils/constants/server_constants.hpp"
#include "src/hpp/compilers/gprm/configs/gprm_config.hpp"
#include "src/hpp/compilers/smartmap/configs/smartmap_config.hpp"
#include "src/hpp/compilers/mapzero/configs/mapzero_config.hpp"
#include "src/hpp/utils/util_server.hpp"
#include "src/hpp/enums/enum_task_type.hpp"
#include "src/hpp/enums/enum_training.hpp"
#include "src/hpp/enums/enum_self_mapping.hpp"
#include "src/hpp/rl/reanalyze.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <thread>
#include <zmq.hpp>
#include <unordered_map>
#include <optional>
#include <vector>
#include <utility>
#include <string>
#include "src/hpp/rl/mapping_history.hpp"
#include <functional>
#include <type_traits>
#include "src/hpp/compilers/mapzero/utils/util_mapzero_server.hpp"
#include "src/hpp/enums/enum_greedy_type.hpp"
#include <chrono>
#include "src/hpp/rng_singleton.hpp"
#include <thread>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <atomic> 
#include <tbb/enumerable_thread_specific.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include "src/hpp/utils/util_train.hpp"
#include "src/hpp/utils/csv_mapping_writer.hpp"
#include "src/hpp/utils/batch_requester.hpp"
#include "src/hpp/rl/map_curriculum_learning.hpp"
#include "src/hpp/rl/curriculum_progress.hpp"
#include "src/hpp/rl/data_augmenter.hpp"
#include "src/hpp/utils/util_serialize.hpp"
#include "src/hpp/utils/hashes.hpp"
#include <tbb/spin_mutex.h>
#include "src/hpp/rl/env_sampler.hpp"
// signal including
#include <csignal>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <tbb/combinable.h>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <bitset>
#include <omp.h>
#include <ATen/Parallel.h>

template<typename StateT>
struct TrainingData {
    std::vector<std::vector<c10::IValue>> batches;
    int total_mappings = 0;
    int total_samples = 0;
    std::unordered_set<int> skip_mapping_idx;

    template<typename Archive> 
    void serialize(Archive& ar){
        ar(total_mappings, total_samples, skip_mapping_idx);
        serialize_ivalue(ar, batches);
    }
};

template <typename StateT>
struct DFGGroupMapping {
    tbb::spin_mutex mutex;
    std::vector<MappingHistory<StateT>> mappings;
};

using json = nlohmann::json;

template<typename StateT, typename ModelConfig>
class Launcher {
private: 
    GlobalConfig<ModelConfig> global_config;
    GlobalConfig<ModelConfig> mcts_global_config;
    std::shared_ptr<zmq::socket_t> socket;
    zmq::context_t context;
    Reanalyze<StateT, ModelConfig> reanalyze;
    Environment<StateT> environment;
    Collate<StateT> collate;
    SelfMapping<StateT,ModelConfig> config_self_mapping;
    std::shared_ptr<Buffer<StateT,ModelConfig>> buffer;
    std::shared_ptr<Buffer<StateT,ModelConfig>> supervised_buffer;
    ServerConstants server_k;
    PathConstants path_k;
    bool debug_mode;
    bool mapping_mode;
    bool use_reanalyze;
    bool use_curriculum_learning;
    bool using_mcts;
    std::string train_func_name;
    tbb::global_control thread_control;
    tbb::task_arena arena;
    std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester;
    MapCurriculumLearning<StateT,ModelConfig> map_curriculum;
    CurriculumProgress<ModelConfig> curriculum_progress;
    double ema_rout_nodes = 0;
    double ema_valid_map_rate = 0;
    double ema_alpha;
    // tbb::task_scheduler_init init(nthread);
    TrainingData<StateT> training_data;
    EnvSampler<StateT,ModelConfig> env_sampler;
    SelfMapping<StateT,ModelConfig> mcts_self_mapping;
    Reanalyze<StateT, ModelConfig> mcts_reanalyze;
    std::shared_ptr<Buffer<StateT,ModelConfig>> mcts_buffer;
    std::chrono::system_clock::time_point last_time;
    double elapsed_time;

    public:
    double calc_ema(double prev_value, 
        double new_value, double alpha) {
        return alpha * new_value + (1 - alpha) * prev_value;
    }

    void initialize_timers() {
        last_time = std::chrono::system_clock::now();
        elapsed_time = 0.0;
    }

    void update_last_time() {
        elapsed_time += std::chrono::duration<double>(std::chrono::system_clock::now() - last_time).count();
        last_time =  std::chrono::system_clock::now();
    }

    void start_reanalyze_loop_mcts(bool& stop){
        std::cout << "[Thread] Reanalyze thread started\n" << std::flush;
        while(!stop){
            // std::cout << "[Thread] Calling reanalyze...\n" << std::flush;
            try {
                if(this->time_exceeded()) {
                    stop = true;
                    break;
                }
                this->mcts_reanalyze.reanalyze();
            } catch (const std::exception& e) {
                std::cerr << "[Thread] Exception during reanalyze: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[Thread] Unknown exception during reanalyze\n";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
        }
        std::cout << "[Thread] Reanalyze thread stopped\n" << std::flush;
    }
    void print_thread_info(const std::string& location) {
        std::cout << location << " - Thread ID: " 
                  << std::this_thread::get_id() << std::endl;
    }
    

    std::thread get_reanalyze_thread_mcts(bool& stop){
    return std::thread(&Launcher<StateT, ModelConfig>::start_reanalyze_loop_mcts, this, std::ref(stop));   
    }

    void save_timers(){
        auto path_config = this->global_config.getPathConfig();
        std::ofstream file(path_config.getSaveCheckpointPath() + "launcher_info.json");
        if (file.is_open()) {
            json js;
           
            js["elapsed_time"] = elapsed_time;

            file << js.dump(4);
            file.close();
        } else {
            throw std::runtime_error("Error opening file for writing: " + path_config.getSaveCheckpointPath() + "launcher_info.json");
        }
    }

    void load_timers(){
        auto path_config = this->global_config.getPathConfig();
        std::ifstream file(path_config.getLoadCheckpointPath() + "launcher_info.json");
        if (file.is_open()) {
            json js;
            file >> js;
            elapsed_time = js["elapsed_time"];
        } else {
            throw std::runtime_error("Error opening file: " + path_config.getLoadCheckpointPath() + "launcher_info.json");
        }
        last_time = std::chrono::system_clock::now();
    }

    Launcher(GlobalConfig<ModelConfig>& global_config, std::optional<GlobalConfig<ModelConfig>> opt_mcts_global_config = std::nullopt);

    std::vector<MappingHistory<StateT>> map_with_zero_shot(const std::vector<std::shared_ptr<StateT>>& states);
    std::vector<MappingHistory<StateT>> map_with_finetune(const std::vector<std::shared_ptr<StateT>>& states);
    Launcher();
    GlobalConfig<ModelConfig>& get_global_config_() const { return global_config;};

    void set_global_config(GlobalConfig<ModelConfig> global_config) { global_config = global_config;};

    std::tuple<std::unordered_map<std::string, double>, 
    std::unordered_map<std::string, double>, 
    std::unordered_map<std::string, std::vector<double>>, 
    std::unordered_map<std::string, std::vector<double>>, 

    std::optional<MappingHistory<StateT>>>  
    eval_step(const int& i, const int& steps, const std::vector<std::shared_ptr<StateT>>& test_initial_states, 
        const std::vector<std::shared_ptr<StateT>>& train_initial_states, bool force = false);


        std::tuple<std::optional<MappingHistory<StateT>>, std::unordered_map<std::string, double>> gen_train_data(
            SelfMapping<StateT,ModelConfig>& self_mapping,  std::vector<std::shared_ptr<StateT>>& train_initial_states, double train_ratio,
        std::unordered_map<std::pair<int,int>, std::vector<double>>& dims_to_action_count, int initial_seed); 

    std::vector<std::shared_ptr<StateT>> get_train_states(const std::vector<std::shared_ptr<StateT>>& all_initial_states);

    std::optional<MappingHistory<StateT>> ppo_train(const std::vector<std::shared_ptr<StateT>>& train_initial_states,const std::vector<std::shared_ptr<StateT>>& test_initial_states);

    std::tuple<std::unordered_map<std::string, double>, std::unordered_map<std::string, std::vector<double>>, std::optional<MappingHistory<StateT>>> eval(
        const std::vector<std::shared_ptr<StateT>>& initial_states, std::string& train_or_test);

    std::optional<MappingHistory<StateT>> base_rl_train(
        const std::vector<std::shared_ptr<StateT>>& train_initial_states, 
        const std::vector<std::shared_ptr<StateT>>& test_initial_states);
    void start_reanalyze_loop(bool& stop);
    std::optional<MappingHistory<StateT>> train(const std::vector<std::shared_ptr<StateT>>& train_states, const std::vector<std::shared_ptr<StateT>>& test_states, bool save_model_flag);
    void finish();
    void run();
    std::string get_path_to_write_map(const std::shared_ptr<StateT>& state,bool finetune_mode);
    void update_priorities(MappingDataset<StateT>& dataset, torch::Tensor tensor_indices);
    void add_train_states_with_curriculum_learning( std::vector<std::shared_ptr<StateT>>& cur_initial_states,
    const std::vector<std::shared_ptr<StateT>>& all_initial_states);

   
    
    template <typename T = ModelConfig>
    typename std::enable_if<std::is_same<T, GPRMConfig>::value || std::is_same<T, TransMapConfig>::value,std::vector<std::shared_ptr<StateT>>>::type
    get_eval_states() {
        auto dfg_config = this->global_config.getDFGConfig();
        auto train_config = this->global_config.getTrainConfig();

        auto path_config = this->global_config.getPathConfig();

        auto dfg_args = std::make_tuple(
            dfg_config
        );

        auto greedy_type = dfg_config.getGreedyType();

        bool calc_node_dists = greedy_type == EnumGreedyType::YOTT;

        std::tuple<bool> cgra_args = std::make_tuple(calc_node_dists);

        auto state_args = std::make_tuple(
            this->socket, this->server_k, this->global_config.getModelConfig(), this->global_config.getStateConfig()
        );
        auto eval_dot_path = path_config.getEvalDfgsDotPath();

        auto test_dfgs = create_dfgs<OrderedDFG>(eval_dot_path,dfg_args);

       
        return create_states<StateT, ModelConfig, OrderedDFG, CGRA>(
            this->global_config, test_dfgs, cgra_args, state_args
        );
    }

    
    template <typename T = ModelConfig>
    typename std::enable_if<!std::is_same<T, GPRMConfig>::value && !std::is_same<T, TransMapConfig>::value, std::vector<std::shared_ptr<StateT>>>::type
    get_eval_states() {
    
        auto dfg_config = this->global_config.getDFGConfig();
    

        auto train_config = this->global_config.getTrainConfig();
    

        auto path_config = this->global_config.getPathConfig();
    

        auto dfg_args = std::make_tuple(
            dfg_config
        );
    

        auto cgra_config = this->global_config.getCGRAConfig();
        auto greedy_type = dfg_config.getGreedyType();
        bool calc_node_dists = greedy_type == EnumGreedyType::YOTT;
        auto cgra_args = std::make_tuple(calc_node_dists);

        auto state_args = std::make_tuple(
        this->global_config.getStateConfig());

        auto eval_dot_path = path_config.getEvalDfgsDotPath();
        auto test_dfgs = create_dfgs<OrderedDFG>(eval_dot_path,dfg_args);
        
        return create_states<StateT, ModelConfig, OrderedDFG, CGRA>(
            this->global_config, test_dfgs, cgra_args, state_args
        );
    
    }




    template <typename T = ModelConfig>
    typename std::enable_if<std::is_same<T, GPRMConfig>::value || std::is_same<T, TransMapConfig>::value, std::tuple<std::vector<std::shared_ptr<StateT>>, std::vector<std::shared_ptr<StateT>>>>::type
    get_initial_states() {
        auto dfg_config = this->global_config.getDFGConfig();
        auto train_config = this->global_config.getTrainConfig();

        auto path_config = this->global_config.getPathConfig();

        auto dfg_args = std::make_tuple(
            dfg_config
        );

        auto greedy_type = dfg_config.getGreedyType();

        bool calc_node_dists = greedy_type == EnumGreedyType::YOTT;

        std::tuple<bool> cgra_args = std::make_tuple(calc_node_dists);

        auto state_args = std::make_tuple(
            this->socket, this->server_k, this->global_config.getModelConfig(), this->global_config.getStateConfig()
        );

        auto [train_initial_states, eval_intial_states] = create_train_eval_states<StateT, ModelConfig, OrderedDFG, CGRA>(
            this->global_config, path_config.getTrainDfgsDotPath(),
            path_config.getEvalDfgsDotPath(),
            dfg_args, cgra_args, state_args
        );
        int count = 0;
        for(auto& state:train_initial_states){
            state->set_id(count);
            count++;
        }

        return {train_initial_states, eval_intial_states};
    }

    
    template <typename T = ModelConfig>
    typename std::enable_if<!std::is_same<T, GPRMConfig>::value && !std::is_same<T, TransMapConfig>::value, std::tuple<std::vector<std::shared_ptr<StateT>>, std::vector<std::shared_ptr<StateT>>>>::type
    get_initial_states() {
    
        auto dfg_config = this->global_config.getDFGConfig();
    

        auto train_config = this->global_config.getTrainConfig();
    

        auto path_config = this->global_config.getPathConfig();
    

        auto dfg_args = std::make_tuple(
            dfg_config
        );
    

        auto cgra_config = this->global_config.getCGRAConfig();
        auto greedy_type = dfg_config.getGreedyType();
        bool calc_node_dists = greedy_type == EnumGreedyType::YOTT;
        auto cgra_args = std::make_tuple(calc_node_dists);

        auto state_args = std::make_tuple(
        this->global_config.getStateConfig());

            
        return create_train_eval_states<StateT,ModelConfig, OrderedDFG, CGRA>(
            this->global_config, path_config.getTrainDfgsDotPath(),
            path_config.getEvalDfgsDotPath(),
            dfg_args, cgra_args, state_args
        );
    
    }



    template <typename T = ModelConfig>
    typename std::enable_if<std::is_same<T, GPRMConfig>::value || std::is_same<T, TransMapConfig>::value,std::shared_ptr<StateT>>::type
    get_initial_state(std::shared_ptr<OrderedDFG> dfg, std::shared_ptr<CGRA> cgra) {
        auto temp_socket = std::make_shared<zmq::socket_t>(this->context, ZMQ_DEALER);
        temp_socket->set(zmq::sockopt::sndtimeo, -1);
        temp_socket->set(zmq::sockopt::rcvtimeo, -1);
        temp_socket->connect("tcp://localhost:" + global_config.getLauncherConfig().getServerPort());

        auto state_args = std::make_tuple(
            temp_socket, this->server_k, this->global_config.getModelConfig(), this->global_config.getStateConfig()
        );
        
        auto ret = create_initial_state<StateT, OrderedDFG, CGRA>(dfg, cgra, state_args);
        temp_socket->close();
        return ret;
    }

   
    template <typename T = ModelConfig>
    typename std::enable_if<!std::is_same<T, GPRMConfig>::value && !std::is_same<T, TransMapConfig>::value, std::shared_ptr<StateT>>::type
    get_initial_state(std::shared_ptr<OrderedDFG> dfg, std::shared_ptr<CGRA> cgra) {
            auto state_args = std::make_tuple(
            this->global_config.getStateConfig()
        );

        return create_initial_state<StateT, OrderedDFG, CGRA>(dfg, cgra, state_args);
    }


    
    void run_config_server()  {
        config_server<ModelConfig>(this->socket, this->server_k, this->global_config);  
    };

    void finish_socket(){
        this->socket->close();
        this->context.close();
    }

    void launcher_save_model(int cur_iter, bool train_finished, const torch::Generator& generator, 
        const double w_entropy_loss, bool force = false,  std::unordered_set<int>* mappings_to_save = nullptr, 
        std::unordered_map<int, MappingHistory<StateT>>* id_to_map = nullptr);


    std::thread get_reanalyze_thread(bool& stop);
    void finish_batch_requester(){
        if (this->batch_requester) {
            this->batch_requester.reset();
        }
    }
    void finish_reanalyze(std::thread& reanalyze_thread,bool& stop);
    void set_seed(int seed) {
        if (seed < 0) {
            std::cout << "[WARN] Seed must be a non-negative integer. Seting to 0." << std::endl;
            seed = 0;
        }
        std::cout << "[INFO] setting seed = " << seed << std::endl;
        torch::manual_seed(seed);
        std::srand(seed);
        torch::globalContext().setDeterministicCuDNN(true);
        torch::globalContext().setDeterministicAlgorithms(true, false);
        if (torch::cuda::is_available()) {
            torch::cuda::manual_seed_all(seed);

        }
    }
    bool should_calc_node_dists(){
        auto greedy_type = this->global_config.getDFGConfig().getGreedyType();
        return greedy_type == EnumGreedyType::YOTT;

    }
    std::string gen_hist_key(std::string category, std::string value){
        return std::string("hist") + std::string("_") + category + std::string("_") + value;
    }
    
    
    MappingHistory<StateT> map_state_with_zero_shot(const std::shared_ptr<StateT>& state){
        return this->config_self_mapping.map(state, false, 0, this->global_config.getTrainConfig().getSeed(), std::nullopt);
    } 

    void single_zero_shot_map(const std::string& filename, 
        const std::string& file_to_write_mapping,
        const int II,
        std::optional<std::unordered_map<std::string,int>>& node_to_actions){
        auto dfg = std::make_shared<OrderedDFG>(filename, this->global_config.getDFGConfig());
        auto cgra_config = global_config.getCGRAConfig();
        auto cgra = std::make_shared<CGRA>(II, cgra_config.getDimensionsList()[0], 
            init_homogeneous_pe_to_functionalities(cgra_config.getDimensionsList()[0].first * cgra_config.getDimensionsList()[0].second * II,
            cgra_config.getFunctionalitiesList()[0]), cgra_config.getConnectionsList()[0], this->should_calc_node_dists());
        auto state = this->get_initial_state(dfg, cgra);
        auto mapping = this->config_self_mapping.map(state, false, 0, this->global_config.getTrainConfig().getSeed(), node_to_actions);
        std::stringstream ss;
        mapping.write_mapping(ss);
        std::filesystem::path path_to_file(file_to_write_mapping);
        std::filesystem::create_directories(path_to_file.parent_path());
        std::ofstream file(file_to_write_mapping);
        if (file.is_open()) {
            file << ss.str();
            file.close();
        } else {
            std::cerr << "Error opening the file: " << file_to_write_mapping << "\n";
            throw std::runtime_error("Error opening the file");
        }
        std::cout << "[INFO] Mapping written to: " << file_to_write_mapping << "\n";

    }
    std::vector<std::shared_ptr<OrderedDFG>> get_unique_dfgs_from_states(const std::vector<std::shared_ptr<StateT>>& states) {
        std::vector<std::shared_ptr<OrderedDFG>> dfgs;
        std::unordered_set<std::string> dfg_names;
        for (const auto& state : states) {
            auto dfg = state->get_dfg();
            if (dfg_names.find(dfg->get_dfg_name()) == dfg_names.end()) {
                dfg_names.insert(dfg->get_dfg_name());
                dfgs.push_back(dfg);
            }
        }
        return dfgs;
    }
    bool is_isomorphic_to_any_dfg_in_list(
        const std::shared_ptr<OrderedDFG>& dfg, 
        const std::vector<std::shared_ptr<OrderedDFG>>& dfgs) {
        for (const auto& existing_dfg : dfgs) {
            if (dfg->are_structurally_isomorphic_ptr(existing_dfg)) {
                return true;
            }
        }
        return false;
    }

    void add_mapping_info(MappingHistory<StateT> mapping){
        auto state = mapping.get_last_state();
        auto id = state->get_id();
        auto node_to_pe = state->get_node_to_pe_();
        std::unordered_map<int,int> node_to_action;
        for(const auto& [node, pe]: node_to_pe){
            auto action = state->get_action_by_pe(pe);
            node_to_action[node] = action;
        }
        this->rl_map_inf[id] = node_to_action;
    }

    bool is_isomorphic_and_same_mapping_in_list(
        const std::shared_ptr<OrderedDFG>& dfg,
        const std::unordered_map<int,int>& node_to_pe,
        const std::vector<MappingHistory<StateT>>& mappings)
    {
        for (const auto& mapping : mappings) {
            auto cur_dfg = mapping.get_last_state()->get_dfg();

            if (cur_dfg->are_structurally_isomorphic_ptr(dfg)) {
                auto existing_map = mapping.get_last_state()->get_node_to_pe_();
                if (existing_map == node_to_pe) {
                    return true;
                }
            }

        }
        return false;
    }


    std::vector<MappingHistory<StateT>>
    clean_equal_mappings(const std::vector<MappingHistory<StateT>>& mappings) {
        
        tbb::concurrent_vector<MappingHistory<StateT>> cleaned_mappings;
    
        tbb::concurrent_unordered_map<
            std::tuple<int,int,int,int,int,int>,  
            DFGGroupMapping<StateT>
        > node_edge_to_dfgs;
    
        tbb::parallel_for(size_t(0), mappings.size(), [&](size_t i) {
            const auto& mapping = mappings[i];
            auto dfg = mapping.get_last_state()->get_dfg();
            auto curr_ii = mapping.get_last_state()->get_II();
            auto cgra = mapping.get_last_state()->get_cgra();
            auto dims = cgra->get_cgra_dimensions_();
            auto conn_bitset = cgra->get_connection_bitset();
            auto node_to_pe = mapping.get_last_state()->get_node_to_pe_();
    
            auto key = std::make_tuple(
                dims.first,
                dims.second,
                curr_ii,
                static_cast<int>(conn_bitset.to_ulong()),
                dfg->num_vertices(),
                dfg->get_num_edges()
            );
    
            auto& group = node_edge_to_dfgs[key];
    
            tbb::spin_mutex::scoped_lock lock(group.mutex);
    
            if (!is_isomorphic_and_same_mapping_in_list(dfg, node_to_pe, group.mappings)) {
                group.mappings.push_back(mapping);          
                cleaned_mappings.push_back(mapping);        
            } else {
                std::cout << "[INFO] Skipping mapping for DFG: "
                          << dfg->get_dfg_name()
                          << " (II=" << curr_ii
                          << ", dims=(" << dims.first << "," << dims.second << "))"
                          << " as it is isomorphic or has identical node-to-PE mapping.\n";
            }
        });
    
        return {cleaned_mappings.begin(), cleaned_mappings.end()};
    }
    MappingHistory<StateT> copy_mapping_history(MappingHistory<StateT> mapping)
    {
        return mapping;
    }

    void aug_and_save_supervised_data(
                                std::string path_to_data,
                                    std::string path_to_save,
                                bool del_old_file = false) {

        auto cgra_config = global_config.getCGRAConfig();
        auto train_config = global_config.getTrainConfig();

        std::vector<fs::path> dot_files;
        for (const auto& entry : fs::recursive_directory_iterator(path_to_data)) {
                if (entry.is_regular_file() && entry.path().extension() == ".dot") {
                dot_files.push_back(entry.path());
            }
        }

       
        bool aug_mapping = this->global_config.getTrainConfig().augment_mapping();
        if(aug_mapping){
            std::cout << "[INFO] Data augmentation is enabled.\n";
        }
        auto aug_is_incremental = train_config.augment_is_incremental();
        auto unroll_ii = cgra_config.getUnrollII();
        auto aug_dfgs = train_config.augment_dfgs_guided_by_curriculum();
        if(aug_dfgs){
            std::cout << "[INFO] DFG Augmentation os enabled.\n";
        }

        if(unroll_ii > 0){
            std::cout << "[INFO] Temporal augmentation is enabled with unroll II: " << unroll_ii << "\n";
        }
        // auto enum_training = this->global_config.getLauncherConfig().getTraining();
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, dot_files.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                // for (size_t i = 0; i < dot_files.size(); i++) {
                    try {
                        auto mapping_data_path = dot_files[i];
                mapping_data_path.replace_extension(".json");
                auto mapping_data = json::parse(std::ifstream(mapping_data_path));

                        auto dfg = std::make_shared<OrderedDFG>(
                            dot_files[i].string(), this->global_config.getDFGConfig());

                auto II = mapping_data["graph_properties"]["II"].get<int>();
                        auto dims = mapping_data["architecture_properties"]["dimensions"]
                                        .get<std::pair<int, int>>();

                        auto connections_dict =
                            mapping_data["architecture_properties"]["interconnections"]
                                .get<std::unordered_map<std::string, bool>>();
                std::unordered_set<EnumInterconnectionStyles> connections;
                        for (auto& [str, flag] : connections_dict) {
                            if (flag) {
                                if (str == "mesh") connections.insert(EnumInterconnectionStyles::MESH);
                                else if (str == "diagonal") connections.insert(EnumInterconnectionStyles::DIAGONAL);
                                else if (str == "one_hop") connections.insert(EnumInterconnectionStyles::ONE_HOP);
                                else if (str == "toroidal") connections.insert(EnumInterconnectionStyles::TOROIDAL);
                                else throw std::runtime_error("Invalid interconnection style: " + str);
                    } 
                }

                auto pe_to_functionalities =
                            init_homogeneous_pe_to_functionalities(dims.first * dims.second * II,
                    cgra_config.getFunctionalitiesList()[0]);
                auto cgra = std::make_shared<CGRA>(II, dims, 
                    pe_to_functionalities, connections , this->should_calc_node_dists());

                        const auto& coord_to_pe = cgra->get_coord_to_PE_();
                auto state = this->get_initial_state(dfg, cgra);
                        const auto& node_to_coord =
                            mapping_data["placement"]
                                .get<std::unordered_map<std::string,std::tuple<int,int,int>>>();

                std::unordered_map<std::string, int> node_to_action;
         
                for (const auto& node : dfg->get_placement_order()) {
                    if (node == -1) continue;
                    auto node_name = dfg->get_name_by_node(node);
                    auto [row, col, cycle] = node_to_coord.at(node_name);
                    if(dfg->is_tight_sched()){
                        cycle = dfg->get_tight_value_by_node(node) % II;
                    }else{
                        std::runtime_error("Use supervised data with tight scheduling.");
                    }
                    node_to_action[node_name] = coord_to_pe.at({cycle, row, col});

                }
            
                MappingHistory<StateT> mapping_history;
                    mapping_history.init_values(state);
                    auto cur_state = state;

                    for (const auto& node : dfg->get_placement_order()) {
                        if (node == -1) continue;
                        auto node_name = dfg->get_name_by_node(node);
                        auto action = node_to_action.at(node_name);
                        auto action_indice = cur_state->get_action_idx_by_action(action);
                        auto [next_state, reward, is_valid] =
                            this->environment.step(*cur_state, action);
                        cur_state = next_state;

                        mapping_history.append_state(next_state);
                        mapping_history.append_action(action);
                        mapping_history.append_action_indice(action_indice);
                        mapping_history.append_reward(reward);
                    }

                    std::vector<MappingHistory<StateT>> aug_mapping_history_list = {mapping_history};

                    MappingHistory<StateT> mapping_history_copy = copy_mapping_history(mapping_history);
                    if (aug_mapping) {
                        aug_mapping_history_list =
                            std::move(DataAugmenter::augment_mapping_ppo(mapping_history_copy, this->global_config, this->environment, aug_is_incremental));
                    
                        // if (aug_mapping_history_list.size() > 1) {
                        //     std::vector<MappingHistory<StateT>> tmp;
                        //     tmp.reserve(aug_mapping_history_list.size() - 1);
                        //     std::move(aug_mapping_history_list.begin() + 1, aug_mapping_history_list.end(), std::back_inserter(tmp));
                        //     aug_mapping_history_list.swap(tmp);
                        // }
                    
                        // aug_mapping_history_list = clean_mappings(aug_mapping_history_list);
                    }
                    
                    if(aug_dfgs){
                        // MappingHistory<StateT> mapping_history_copy = copy_mapping_history(mapping_history);
                        auto new_mappings = DataAugmenter::augment_mapping_by_placement_order_cutting(mapping_history, this->global_config, this->environment,
                                this->socket, this->server_k);
                        for(auto& new_map: new_mappings){
                            auto cur_dfg = new_map.get_last_state()->get_dfg();

                            aug_mapping_history_list.push_back(std::move(new_map));
                        }    
                    }

                    if (unroll_ii > 0){
                        // MappingHistory<StateT> mapping_history_copy = copy_mapping_history(mapping_history);
                        auto new_mappings = DataAugmenter::augment_temporal_mapping(mapping_history, this->global_config,
                            this->environment,
                            this->socket, this->server_k);
                        for(auto& new_map: new_mappings){
                            aug_mapping_history_list.push_back(std::move(new_map));
                        }
                    }
                    
                    std::cout << "[INFO] Processed file: " << dot_files[i]
                            << " with " << aug_mapping_history_list.size() << " mappings.\n";
                    for(auto map: aug_mapping_history_list){
                        if(!map.get_mapping_is_valid()){
                            throw std::runtime_error(
                                "Mapping is not valid for file: " + dot_files[i].string());
                        }                        
                    }

                    this->save_mappings(aug_mapping_history_list, path_to_save, dot_files[i].stem().string());

                    if(del_old_file){
                        fs::remove(dot_files[i]);
                        fs::remove(mapping_data_path);
                    }

                    } catch (const std::exception& e) {
                        auto msg_stream = std::stringstream();
                        msg_stream << "[ERROR] While processing " << dot_files[i]
                                << " : " << e.what() << "\n";
                                throw std::runtime_error(msg_stream.str());
                    }
                }
            }
        );
    }



    std::vector<MappingHistory<StateT>>
    generate_mappings(const std::string& path_to_data)
    {
        std::vector<MappingHistory<StateT>> mappings;
    
        std::cout << "[INFO] Loading mappings from : " << path_to_data << "\n";
    
        if (!fs::exists(path_to_data) || !fs::is_directory(path_to_data)) {
            std::cerr << "[ERROR] Path does not exist or is not a directory: "
                      << path_to_data << std::endl;
            return mappings;
        }
    
        std::vector<fs::path> dot_files;
        auto opts = fs::directory_options::skip_permission_denied |
                    fs::directory_options::follow_directory_symlink;
    
        for (const auto& entry : fs::recursive_directory_iterator(path_to_data, opts)) {
            if (entry.path().extension() == ".dot") {
                dot_files.push_back(entry.path());
            }
        }
    
        std::sort(dot_files.begin(), dot_files.end(),
                  [](const fs::path& a, const fs::path& b) {
                      return a.generic_string() < b.generic_string();
                  });
    
        if (dot_files.empty()) {
            std::cout << "[INFO] Found 0 mappings from RL data.\n";
            return mappings;
        }
    
        std::vector<MappingHistory<StateT>> mappings_per_file(dot_files.size());
        auto cgra_config = global_config.getCGRAConfig();
    
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, dot_files.size()),
            [&](const tbb::blocked_range<size_t>& r)
            {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    try {
                        const fs::path& dot_path = dot_files[i];
                        std::string stem = dot_path.stem().string();
    
                        size_t pos = stem.find('_');
                        if (pos == std::string::npos) {
                            throw std::runtime_error("Invalid filename format");
                        }
    
                        int id = std::stoi(stem.substr(0, pos));
    
                        fs::path json_path = dot_path;
                        json_path.replace_extension(".json");
    
                        auto mapping_data =
                            json::parse(std::ifstream(json_path));
    
                        auto dfg = std::make_shared<OrderedDFG>(
                            dot_path.string(),
                            this->global_config.getDFGConfig());
    
                        int II =
                            mapping_data["graph_properties"]["II"].get<int>();
    
                        auto dims =
                            mapping_data["architecture_properties"]["dimensions"]
                                .get<std::pair<int,int>>();
    
                        auto connections_dict =
                            mapping_data["architecture_properties"]["interconnections"]
                                .get<std::unordered_map<std::string,bool>>();
    
                        std::unordered_set<EnumInterconnectionStyles> connections;
                        for (const auto& [str, flag] : connections_dict) {
                            if (!flag) continue;
                            if (str == "mesh")
                                connections.insert(EnumInterconnectionStyles::MESH);
                            else if (str == "diagonal")
                                connections.insert(EnumInterconnectionStyles::DIAGONAL);
                            else if (str == "one_hop")
                                connections.insert(EnumInterconnectionStyles::ONE_HOP);
                            else if (str == "toroidal")
                                connections.insert(EnumInterconnectionStyles::TOROIDAL);
                            else
                                throw std::runtime_error("Invalid interconnection style: " + str);
                        }
    
                        auto pe_to_functionalities =
                            init_homogeneous_pe_to_functionalities(
                                dims.first * dims.second * II,
                                cgra_config.getFunctionalitiesList()[0]);
    
                        auto cgra = std::make_shared<CGRA>(
                            II,
                            dims,
                            pe_to_functionalities,
                            connections,
                            this->should_calc_node_dists());
    
                        const auto& coord_to_pe = cgra->get_coord_to_PE_();
    
                        auto state = this->get_initial_state(dfg, cgra);
                        state->set_id(id);
    
                        const auto& node_to_coord =
                            mapping_data["placement"]
                                .get<std::unordered_map<
                                    std::string,
                                    std::tuple<int,int,int>>>();
    
                        std::unordered_map<std::string,int> node_to_action;
                        for (const auto& node : dfg->get_placement_order()) {
                            if (node == -1) continue;
    
                            auto node_name = dfg->get_name_by_node(node);
                            auto [row, col, cycle] = node_to_coord.at(node_name);
    
                            if (dfg->is_tight_sched()) {
                                cycle %= II;
                            } else {
                                throw std::runtime_error(
                                    "Use supervised data with tight scheduling.");
                            }
    
                            node_to_action[node_name] =
                                coord_to_pe.at({cycle, row, col});
                        }
    
                        MappingHistory<StateT> mapping_history;
                        mapping_history.init_values(state);
    
                        auto cur_state = state;
    
                        for (const auto& node : dfg->get_placement_order()) {
                            if (node == -1) continue;
    
                            auto node_name = dfg->get_name_by_node(node);
                            int action = node_to_action.at(node_name);
                            int action_idx =
                                cur_state->get_action_idx_by_action(action);
    
                            auto [next_state, reward, is_valid] =
                                this->environment.step(*cur_state, action);
    
                            cur_state = next_state;
    
                            mapping_history.append_state(next_state);
                            mapping_history.append_action(action);
                            mapping_history.append_action_indice(action_idx);
                            mapping_history.append_reward(reward);
                        }
    
                        if (!mapping_history.get_mapping_is_valid()) {
                            throw std::runtime_error(
                                "Mapping is not valid for file: " +
                                dot_path.string());
                        }
    
                        mappings_per_file[i] = std::move(mapping_history);
                    }
                    catch (const std::exception& e) {
                        throw std::runtime_error(
                            "[ERROR] While processing " +
                            dot_files[i].string() + " : " + e.what());
                    }
                }
            }
        );
    
      
        for (auto& mh : mappings_per_file) {
            if (mh.size() > 0) {
                mappings.push_back(std::move(mh));
            }
        }
    
        std::cout << "[INFO] Found " << mappings.size()
                  << " mappings from RL data.\n";
    
        return mappings;
    }
    

    std::vector<MappingHistory<StateT>> gen_supervised_data_by_files(
        std::vector<std::string>& dot_files,
        std::vector<std::shared_ptr<OrderedDFG>>& test_dfgs,
        std::unordered_map<std::pair<int,int>, std::vector<std::shared_ptr<OrderedDFG>>>& node_edge_to_dfgs,
        std::optional<std::string> path_to_data_opt = std::nullopt) {

    std::string path_to_data; 
    if (path_to_data_opt.has_value()) {
        std::cout << "[INFO] Using custom path to data: " << path_to_data_opt.value() << "\n";
        path_to_data = path_to_data_opt.value();
    } else {
        path_to_data = this->global_config.getPathConfig().getTrainDfgsDotPath();
    }

    std::vector<MappingHistory<StateT>> mappings;
    auto cgra_config = global_config.getCGRAConfig();
    auto train_config = global_config.getTrainConfig();
    auto launcher_config = this->global_config.getLauncherConfig();

    std::vector<std::vector<MappingHistory<StateT>>> mappings_per_file(dot_files.size());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, dot_files.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                try {
                    auto mapping_data_str = dot_files[i];
                    std::filesystem::path mapping_data_path(mapping_data_str);
                    mapping_data_path.replace_extension(".json");
                    auto mapping_data = json::parse(std::ifstream(mapping_data_path));

                    auto dfg = std::make_shared<OrderedDFG>(
                        dot_files[i], this->global_config.getDFGConfig());

                    auto pair = std::make_pair(dfg->num_vertices(), dfg->get_num_edges());
                    if (node_edge_to_dfgs.find(pair) != node_edge_to_dfgs.end() &&
                        this->is_isomorphic_to_any_dfg_in_list(dfg, node_edge_to_dfgs.at(pair))) {
                        std::cout << "[INFO] Skipping DFG: " << dfg->get_dfg_name()
                                  << " as it is isomorphic to an existing DFG.\n";
                        continue;
                    }

                    auto II = mapping_data["graph_properties"]["II"].get<int>();
                    auto dims = mapping_data["architecture_properties"]["dimensions"]
                                    .get<std::pair<int,int>>();
                    auto connections_dict = mapping_data["architecture_properties"]["interconnections"]
                                                .get<std::unordered_map<std::string,bool>>();
                    std::unordered_set<EnumInterconnectionStyles> connections;
                    for (auto& [str, flag] : connections_dict) {
                        if (flag) {
                            if (str == "mesh") connections.insert(EnumInterconnectionStyles::MESH);
                            else if (str == "diagonal") connections.insert(EnumInterconnectionStyles::DIAGONAL);
                            else if (str == "one_hop") connections.insert(EnumInterconnectionStyles::ONE_HOP);
                            else if (str == "toroidal") connections.insert(EnumInterconnectionStyles::TOROIDAL);
                            else throw std::runtime_error("Invalid interconnection style: " + str);
                        } 
                    }

                    auto pe_to_functionalities =
                        init_homogeneous_pe_to_functionalities(dims.first * dims.second * II,
                            cgra_config.getFunctionalitiesList()[0]);
                    auto cgra = std::make_shared<CGRA>(II, dims, 
                        pe_to_functionalities, connections, this->should_calc_node_dists());
                    const auto& coord_to_pe = cgra->get_coord_to_PE_();
                    auto state = this->get_initial_state(dfg, cgra);
                    const auto& node_to_coord = mapping_data["placement"]
                                                    .get<std::unordered_map<std::string,std::tuple<int,int,int>>>();

                    std::unordered_map<std::string,int> node_to_action;
                    for (const auto& node : dfg->get_placement_order()) {
                        if (node == -1) continue;
                        auto node_name = dfg->get_name_by_node(node);
                        auto [row, col, cycle] = node_to_coord.at(node_name);
                        if (dfg->is_tight_sched()) cycle %= II;
                        else throw std::runtime_error("Use supervised data with tight scheduling.");
                        node_to_action[node_name] = coord_to_pe.at({cycle,row,col});
                    }

                    MappingHistory<StateT> mapping_history;
                    mapping_history.init_values(state);
                    auto cur_state = state;

                    for (const auto& node : dfg->get_placement_order()) {
                        if (node == -1) continue;
                        auto node_name = dfg->get_name_by_node(node);
                        auto action = node_to_action.at(node_name);
                        auto action_indice = cur_state->get_action_idx_by_action(action);
                        auto [next_state, reward, is_valid] = this->environment.step(*cur_state, action);
                        cur_state = next_state;

                        mapping_history.append_state(next_state);
                        mapping_history.append_action(action);
                        mapping_history.append_action_indice(action_indice);
                        mapping_history.append_reward(reward);
                    }

                    std::vector<MappingHistory<StateT>> aug_mapping_history_list = {mapping_history};

                  
                    for (auto& map : aug_mapping_history_list) {
                        if (!map.get_mapping_is_valid()) {
                            throw std::runtime_error("Mapping is not valid for file: " + dot_files[i]);
                        }
                    }

                    mappings_per_file[i] = std::move(aug_mapping_history_list);

                } catch (const std::exception& e) {
                    throw std::runtime_error("[ERROR] While processing " + dot_files[i] + " : " + e.what());
                    // std::cout << "[ERROR+REMOVE] While processing " << dot_files[i] << " : " << e.what() << std::endl;
                    // std::filesystem::remove(dot_files[i]);
                    
                    // std::filesystem::path p(dot_files[i]);
                    // p.replace_extension(".json");

                    // std::filesystem::remove(p);
                }
            }
        });

    for (const auto& vec : mappings_per_file) {
        std::move(vec.begin(), vec.end(), std::back_inserter(mappings));
    }

    return mappings;
}

    std::vector<MappingHistory<StateT>> gen_supervised_data(
        std::vector<std::shared_ptr<StateT>>& test_initial_states,
        std::optional<std::string> path_to_data_opt = std::nullopt) {

    std::string path_to_data; 
    if (path_to_data_opt.has_value()) {
        std::cout << "[INFO] Using custom path to data: " << path_to_data_opt.value() << "\n";
        path_to_data = path_to_data_opt.value();
    } else {
        path_to_data = this->global_config.getPathConfig().getTrainDfgsDotPath();
    }

    std::vector<MappingHistory<StateT>> mappings;
    auto cgra_config = global_config.getCGRAConfig();
    auto train_config = global_config.getTrainConfig();
    auto test_dfgs = this->get_unique_dfgs_from_states(test_initial_states); 
    auto node_edge_to_dfgs = create_node_edge_to_dfgs<OrderedDFG>(test_dfgs);
    auto launcher_config = this->global_config.getLauncherConfig();
    auto aug_test_only = launcher_config.getAugTestOnly();
    std::cout << "[INFO] Generating supervised data from: " << path_to_data << "\n";

    std::vector<fs::path> dot_files;
    for (const auto& entry : fs::recursive_directory_iterator(path_to_data)) {
        if (entry.is_regular_file() && entry.path().extension() == ".dot") {
            dot_files.push_back(entry.path());
        }
    }
    
    std::sort(dot_files.begin(), dot_files.end(),
    [](const fs::path& a, const fs::path& b) {
        return a.generic_string() < b.generic_string();
    });

    std::vector<std::vector<MappingHistory<StateT>>> mappings_per_file(dot_files.size());

    bool aug_mapping = train_config.augment_mapping() && (!aug_test_only);
    auto aug_is_incremental = train_config.augment_is_incremental() && (!aug_test_only);
    auto unroll_ii = cgra_config.getUnrollII() && (!aug_test_only);
    auto aug_dfgs = train_config.augment_dfgs_guided_by_curriculum() && (!aug_test_only);

    if (aug_mapping) std::cout << "[INFO] Data augmentation is enabled.\n";
    if (aug_dfgs) std::cout << "[INFO] DFG augmentation is enabled.\n";
    if (unroll_ii > 0) std::cout << "[INFO] Temporal augmentation enabled with unroll II: " << unroll_ii << "\n";

    tbb::parallel_for(tbb::blocked_range<size_t>(0, dot_files.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                try {
                    auto mapping_data_path = dot_files[i];
                    mapping_data_path.replace_extension(".json");
                    auto mapping_data = json::parse(std::ifstream(mapping_data_path));

                    auto dfg = std::make_shared<OrderedDFG>(
                        dot_files[i].string(), this->global_config.getDFGConfig());

                    auto pair = std::make_pair(dfg->num_vertices(), dfg->get_num_edges());
                    if (node_edge_to_dfgs.find(pair) != node_edge_to_dfgs.end() &&
                        this->is_isomorphic_to_any_dfg_in_list(dfg, node_edge_to_dfgs.at(pair))) {
                        std::cout << "[INFO] Skipping DFG: " << dfg->get_dfg_name()
                                  << " as it is isomorphic to an existing DFG.\n";
                        continue;
                    }

                    auto II = mapping_data["graph_properties"]["II"].get<int>();
                    auto dims = mapping_data["architecture_properties"]["dimensions"]
                                    .get<std::pair<int,int>>();
                    auto connections_dict = mapping_data["architecture_properties"]["interconnections"]
                                                .get<std::unordered_map<std::string,bool>>();
                    std::unordered_set<EnumInterconnectionStyles> connections;
                    for (auto& [str, flag] : connections_dict) {
                        if (flag) {
                            if (str == "mesh") connections.insert(EnumInterconnectionStyles::MESH);
                            else if (str == "diagonal") connections.insert(EnumInterconnectionStyles::DIAGONAL);
                            else if (str == "one_hop") connections.insert(EnumInterconnectionStyles::ONE_HOP);
                            else if (str == "toroidal") connections.insert(EnumInterconnectionStyles::TOROIDAL);
                            else throw std::runtime_error("Invalid interconnection style: " + str);
                        } 
                    }

                    auto pe_to_functionalities =
                        init_homogeneous_pe_to_functionalities(dims.first * dims.second * II,
                            cgra_config.getFunctionalitiesList()[0]);
                    auto cgra = std::make_shared<CGRA>(II, dims, 
                        pe_to_functionalities, connections, this->should_calc_node_dists());
                    const auto& coord_to_pe = cgra->get_coord_to_PE_();
                    auto state = this->get_initial_state(dfg, cgra);
                    const auto& node_to_coord = mapping_data["placement"]
                                                    .get<std::unordered_map<std::string,std::tuple<int,int,int>>>();

                    std::unordered_map<std::string,int> node_to_action;
                    for (const auto& node : dfg->get_placement_order()) {
                        if (node == -1) continue;
                        auto node_name = dfg->get_name_by_node(node);
                        auto [row, col, cycle] = node_to_coord.at(node_name);
                        if (dfg->is_tight_sched()) cycle %= II;
                        else throw std::runtime_error("Use supervised data with tight scheduling.");
                        node_to_action[node_name] = coord_to_pe.at({cycle,row,col});
                    }

                    MappingHistory<StateT> mapping_history;
                    mapping_history.init_values(state);
                    auto cur_state = state;

                    for (const auto& node : dfg->get_placement_order()) {
                        if (node == -1) continue;
                        auto node_name = dfg->get_name_by_node(node);
                        auto action = node_to_action.at(node_name);
                        auto action_indice = cur_state->get_action_idx_by_action(action);
                        auto [next_state, reward, is_valid] = this->environment.step(*cur_state, action);
                        cur_state = next_state;

                        mapping_history.append_state(next_state);
                        mapping_history.append_action(action);
                        mapping_history.append_action_indice(action_indice);
                        mapping_history.append_reward(reward);
                    }

                    std::vector<MappingHistory<StateT>> aug_mapping_history_list = {mapping_history};

                    if (aug_mapping) {
                        MappingHistory<StateT> mapping_history_copy = copy_mapping_history(mapping_history);
                        aug_mapping_history_list = std::move(
                            DataAugmenter::augment_mapping_ppo(mapping_history_copy, this->global_config, this->environment, aug_is_incremental)
                        );

                        // if (aug_mapping_history_list.size() > 1) {
                        //     std::vector<MappingHistory<StateT>> tmp;
                        //     tmp.reserve(aug_mapping_history_list.size()-1);
                        //     std::move(aug_mapping_history_list.begin()+1, aug_mapping_history_list.end(), std::back_inserter(tmp));
                        //     aug_mapping_history_list.swap(tmp);
                        // }
                    }

                    if (aug_dfgs) {
                        auto new_mappings = DataAugmenter::augment_mapping_by_placement_order_cutting(
                            mapping_history, this->global_config, this->environment, this->socket, this->server_k);
                        for (auto& new_map : new_mappings) {
                            auto cur_dfg = new_map.get_last_state()->get_dfg();
                            auto pair = std::make_pair(cur_dfg->num_vertices(), cur_dfg->get_num_edges());
                            if (node_edge_to_dfgs.find(pair) != node_edge_to_dfgs.end() &&
                                this->is_isomorphic_to_any_dfg_in_list(cur_dfg, node_edge_to_dfgs.at(pair))) {
                                continue;
                            }
                            aug_mapping_history_list.push_back(std::move(new_map));
                        }
                    }

                    if (unroll_ii > 0) {
                        auto new_mappings = DataAugmenter::augment_temporal_mapping(
                            mapping_history, this->global_config, this->environment, this->socket, this->server_k);
                        for (auto& new_map : new_mappings) {
                            aug_mapping_history_list.push_back(std::move(new_map));
                        }
                    }

                    for (auto& map : aug_mapping_history_list) {
                        if (!map.get_mapping_is_valid()) {
                            throw std::runtime_error("Mapping is not valid for file: " + dot_files[i].string());
                        }
                    }

                    mappings_per_file[i] = std::move(aug_mapping_history_list);

                } catch (const std::exception& e) {
                    throw std::runtime_error("[ERROR] While processing " + dot_files[i].string() + " : " + e.what());
                }
            }
        });

    for (const auto& vec : mappings_per_file) {
        std::move(vec.begin(), vec.end(), std::back_inserter(mappings));
    }

    if (aug_dfgs) {
        mappings = this->clean_equal_mappings(mappings);
    }

    std::cout << "[INFO] Generated " << mappings.size() << " mappings from supervised data.\n";
    return mappings;
}


std::vector<std::shared_ptr<StateT>> gen_rl_data_from_supervised_data(
    std::vector<std::shared_ptr<StateT>>& test_initial_states,
    std::optional<std::string> path_to_data_opt = std::nullopt) {

    std::string path_to_data; 
    if (path_to_data_opt.has_value()) {
        std::cout << "[INFO] Using custom path to data: " << path_to_data_opt.value() << "\n";
        path_to_data = path_to_data_opt.value();
    } else {
        path_to_data = this->global_config.getPathConfig().getTrainDfgsDotPath();
    }

    auto cgra_config = global_config.getCGRAConfig();
    auto train_config = global_config.getTrainConfig();
    auto test_dfgs = this->get_unique_dfgs_from_states(test_initial_states); 
    auto node_edge_to_dfgs = create_node_edge_to_dfgs<OrderedDFG>(test_dfgs);

    std::cout << "[INFO] Generating supervised data from: " << path_to_data << "\n";

    std::vector<fs::path> dot_files;
    for (const auto& entry : fs::recursive_directory_iterator(path_to_data)) {
        if (entry.is_regular_file() && entry.path().extension() == ".dot") {
            dot_files.push_back(entry.path());
        }
    }
    std::sort(dot_files.begin(), dot_files.end(),
    [](const fs::path& a, const fs::path& b) {
        return a.generic_string() < b.generic_string();
    });

    std::vector<std::shared_ptr<StateT>> train_states(dot_files.size());

    at::parallel_for(0, dot_files.size(), 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
                try {
                    auto mapping_data_path = dot_files[i];
                    mapping_data_path.replace_extension(".json");
                    auto mapping_data = json::parse(std::ifstream(mapping_data_path));

                    auto dfg = std::make_shared<OrderedDFG>(
                        dot_files[i].string(), this->global_config.getDFGConfig());

                    auto pair = std::make_pair(dfg->num_vertices(), dfg->get_num_edges());
                    if (node_edge_to_dfgs.find(pair) != node_edge_to_dfgs.end() &&
                        this->is_isomorphic_to_any_dfg_in_list(dfg, node_edge_to_dfgs.at(pair))) {
                        std::cout << "[INFO] Skipping DFG: " << dfg->get_dfg_name()
                                << " as it is isomorphic to an existing DFG.\n";
                        continue;
                    }

                    auto II = mapping_data["graph_properties"]["II"].get<int>();
                    auto dims = mapping_data["architecture_properties"]["dimensions"]
                                    .get<std::pair<int,int>>();
                    auto connections_dict = mapping_data["architecture_properties"]["interconnections"]
                                                .get<std::unordered_map<std::string,bool>>();
                    std::unordered_set<EnumInterconnectionStyles> connections;
                    for (auto& [str, flag] : connections_dict) {
                        if (flag) {
                            if (str == "mesh") connections.insert(EnumInterconnectionStyles::MESH);
                            else if (str == "diagonal") connections.insert(EnumInterconnectionStyles::DIAGONAL);
                            else if (str == "one_hop") connections.insert(EnumInterconnectionStyles::ONE_HOP);
                            else if (str == "toroidal") connections.insert(EnumInterconnectionStyles::TOROIDAL);
                            else throw std::runtime_error("Invalid interconnection style: " + str);
                        } 
                    }

                    auto pe_to_functionalities =
                        init_homogeneous_pe_to_functionalities(dims.first * dims.second * II,
                            cgra_config.getFunctionalitiesList()[0]);
                    auto cgra = std::make_shared<CGRA>(II, dims, 
                        pe_to_functionalities, connections, this->should_calc_node_dists());
                    auto state = this->get_initial_state(dfg, cgra);
                    train_states[i] =  std::move(state);
                
                } catch (const std::exception& e) {
                    throw std::runtime_error("[ERROR] While processing " + dot_files[i].string() + " : " + e.what());
                }
            }
        });

    // Remove empty states due to isomorphic DFGs
    std::vector<std::shared_ptr<StateT>> cleaned_train_states;
    int i = 0;
    for (const auto& state : train_states) {
        if (state != nullptr) {
            state->set_id(i);
            cleaned_train_states.push_back(state);
            i ++;
        }
    }

    std::cout << "[INFO] Generated " << cleaned_train_states.size() << " initial states from supervised data.\n";
    return cleaned_train_states;
}

    void map_with_zero_shot(CSVMappingWriter& csv_writer){
        auto cgra_config = global_config.getCGRAConfig();
        auto launcher_config = global_config.getLauncherConfig();
        auto connections_list = cgra_config.getConnectionsList();
        auto max_II = cgra_config.getUnrollII();
        auto dot_path = global_config.getPathConfig().getEvalDfgsDotPath();
        auto save_mapping_path = global_config.getPathConfig().getSaveCheckpointPath();
    
        std::unordered_map<int,std::unordered_map<std::pair<int,int>,std::vector<std::shared_ptr<CGRA>>>> II_to_dims_to_cgras;
    
        auto [dfgs,_] = initialize_dfgs<OrderedDFG>(dot_path, false, global_config.getDFGConfig());
    
        for(auto& dfg: dfgs){
            for(auto& dims: cgra_config.getDimensionsList()){
    
                auto MII = static_cast<int>(std::ceil(
                    static_cast<double>(dfg->num_vertices()) /
                    static_cast<double>(dims.first*dims.second)
                ));
    
                std::vector<bool> valid_mapping_found;
                int total_valid_mappings = 0;
    
                for(int i = MII; i <= MII + max_II; i++){
    
                    std::vector<std::shared_ptr<CGRA>> cgras;
                    bool mapping_is_valid = false;
    
                    if(II_to_dims_to_cgras.find(i) == II_to_dims_to_cgras.end()){
                        II_to_dims_to_cgras[i][dims] = {};
                        for(auto& conn : connections_list){
                            auto pe_to_functionalities =
                                init_homogeneous_pe_to_functionalities(
                                    dims.first * dims.second * i,
                                    cgra_config.getFunctionalitiesList()[0]
                                );
                            II_to_dims_to_cgras[i][dims].emplace_back(
                                std::make_shared<CGRA>(
                                    i, dims, pe_to_functionalities, conn, this->should_calc_node_dists()
                                )
                            );
                        }
                    }else{
                        auto& dims_map = II_to_dims_to_cgras.at(i);
                        if(dims_map.find(dims) == dims_map.end()){
                            dims_map[dims] = {};
                            for(auto& conn : connections_list){
                                auto pe_to_functionalities =
                                    init_homogeneous_pe_to_functionalities(
                                        dims.first * dims.second * i,
                                        cgra_config.getFunctionalitiesList()[0]
                                    );
                                dims_map[dims].emplace_back(
                                    std::make_shared<CGRA>(
                                        i, dims, pe_to_functionalities, conn, this->should_calc_node_dists()
                                    )
                                );
                            }
                        }
                    }
    
                    cgras = II_to_dims_to_cgras[i][dims];
    
                    if(i == MII){
                        valid_mapping_found.resize(cgras.size(), false);
                    }
    
                    int count = 0;
    
                    for(auto& cgra: cgras){
    
                        if(valid_mapping_found[count]){
                            count++;
                            continue;
                        }
    
                        auto state = this->get_initial_state(dfg, cgra);
    
                        std::string raw_path = save_mapping_path +std::format(
                            "{}/{}/{}/{}x{}/{}/II_{}.md",
                            get_model_name_by_enum(launcher_config.getModel()),
                            csv_writer.get_key(),
                            dfg->get_dfg_name(),
                            dims.first,
                            dims.second,
                            std::bitset<4>(gen_bin_by_styles(cgra->get_interconnection_styles())).to_string(),
                            i
                        );
    
                        std::filesystem::path path_to_file = std::filesystem::path(raw_path).lexically_normal();
                        std::filesystem::path parent = path_to_file.parent_path();
    
                        if (parent.empty()){
                            throw std::runtime_error("Invalid parent path");
                        }
    
                        std::error_code ec;
                        std::filesystem::create_directories(parent, ec);
                        if (ec){
                            throw std::runtime_error("Could not create directories");
                        }                        
                        auto mapping = this->map_state_with_zero_shot(state);

                        // mapping.print();
                        std::stringstream ss;
                        mapping.write_mapping(ss);
    
                        std::ofstream file(path_to_file);
                        if(!file.is_open()){
                            throw std::runtime_error("Error opening mapping file");
                        }
    
                        file << ss.str();
                        file.close();
    
                        mapping_is_valid = mapping.get_mapping_is_valid();
                        csv_writer.write_data(global_config, mapping, true);
    
                        if(mapping_is_valid){
                            valid_mapping_found[count] = true;
                            total_valid_mappings++;
                        }
    
                        count++;
                    }
    
                    if(total_valid_mappings == static_cast<int>(valid_mapping_found.size())){
                        break;
                    }
                }
            }
        }
    }
    
    std::vector<std::vector<c10::IValue>> generate_batches(std::vector<MappingHistory<StateT>>& mappings, int batch_size){
        auto data_gen_start = std::chrono::high_resolution_clock::now();

        for (auto& mapping : mappings) {
            this->supervised_buffer->add_mapping(std::move(mapping));
        }

        auto data_gen_end = std::chrono::high_resolution_clock::now();
        std::cout << "[INFO] Supervised data generation completed in "
                  << std::chrono::duration<double>(data_gen_end - data_gen_start).count()
                  << " seconds." << std::endl;

        auto dataset = this->supervised_buffer->get_batch(EnumReplayBuffer::SUPERVISED_BUFFER);

        std::cout << "[INFO] Dataset size: " << dataset.size() << std::endl;
        

        std::vector<std::vector<int>> grouped_idxs;
        auto train_config = this->global_config.getTrainConfig();
        grouped_idxs = create_batch_size_idxs(dataset.size(), batch_size, train_config.getShuffleTrainData(), torch::make_generator<torch::CPUGeneratorImpl>(train_config.getSeed()));
        std::cout << "[INFO] Training batches created. Number of batches: " << grouped_idxs.size()
                  << " with mini batch size " << batch_size << std::endl;

        std::vector<std::vector<c10::IValue>> all_train_data(grouped_idxs.size());
        auto collate_start = std::chrono::steady_clock::now();
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, grouped_idxs.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t ip = r.begin(); ip != r.end(); ++ip) {
                    all_train_data[ip] = dataset.get_train_data(this->collate, grouped_idxs[ip]);
                }
            }
        );
        auto collate_end = std::chrono::steady_clock::now();
        std::cout << "[INFO] Collate time: "
                  << std::chrono::duration<double>(collate_end - collate_start).count()
                  << " seconds." << std::endl;
        this->supervised_buffer->clear_buffer();
        return all_train_data;
    }

    std::vector<std::vector<c10::IValue>> generate_batches(std::vector<std::reference_wrapper<MappingHistory<StateT>>>& mappings, int batch_size){
        auto data_gen_start = std::chrono::high_resolution_clock::now();

        for (auto& mapping : mappings) {
            this->supervised_buffer->add_mapping(mapping);
        }

        auto data_gen_end = std::chrono::high_resolution_clock::now();
        std::cout << "[INFO] Supervised data generation completed in "
                  << std::chrono::duration<double>(data_gen_end - data_gen_start).count()
                  << " seconds." << std::endl;

        auto dataset = this->supervised_buffer->get_batch(EnumReplayBuffer::SUPERVISED_BUFFER);

        std::cout << "[INFO] Dataset size: " << dataset.size() << std::endl;
        

        std::vector<std::vector<int>> grouped_idxs;
        auto train_config = this->global_config.getTrainConfig();
        grouped_idxs = create_batch_size_idxs(dataset.size(), batch_size, train_config.getShuffleTrainData(), torch::make_generator<torch::CPUGeneratorImpl>(train_config.getSeed()));
        std::cout << "[INFO] Training batches created. Number of batches: " << grouped_idxs.size()
                  << " with mini batch size " << batch_size << std::endl;

        std::vector<std::vector<c10::IValue>> all_train_data(grouped_idxs.size());
        auto collate_start = std::chrono::steady_clock::now();
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, grouped_idxs.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t ip = r.begin(); ip != r.end(); ++ip) {
                    all_train_data[ip] = dataset.get_train_data(this->collate, grouped_idxs[ip]);
                }
            }
        );
        auto collate_end = std::chrono::steady_clock::now();
        std::cout << "[INFO] Collate time: "
                  << std::chrono::duration<double>(collate_end - collate_start).count()
                  << " seconds." << std::endl;
        this->supervised_buffer->clear_buffer();
        return all_train_data;
    }
    
    void send_data_to_server(std::vector<std::vector<c10::IValue>>& all_train_data){
        int num_batches = all_train_data.size();
        create_list_with_size(this->socket, server_k, num_batches);
    
    
        tbb::parallel_for(tbb::blocked_range<int>(0, num_batches),
        [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); i++) {
                std::shared_ptr<zmq::socket_t> socket = std::make_shared<zmq::socket_t>(context, ZMQ_DEALER);
                socket->set(zmq::sockopt::sndtimeo, -1);
                socket->set(zmq::sockopt::rcvtimeo, -1);
                socket->connect("tcp://localhost:" + global_config.getLauncherConfig().getServerPort());
    
                auto& batch = all_train_data[i];
                send_supervised_data_parallel(socket, batch, server_k, i);
                all_train_data[i].clear();
                all_train_data[i].shrink_to_fit();
                socket->close();

            }
        }
    );
        // context.close();
        std::cout << "[INFO] Supervised data sent." << std::endl;
    }

    void save_supervised_data_server(std::string& path){
        save_supervised_data(socket, server_k, path);
        std::cout << "[INFO] Supervised data saved to " << path << "." << std::endl;
    }

    void save_block_of_supervised_data(std::string& path){
        save_block_of_data(socket, server_k, path);
    }

    TrainConfig& get_train_config() {
        return this->global_config.getTrainConfig();
    }


void train_supervised(std::vector<std::shared_ptr<StateT>>& test_initial_states) {
    static volatile sig_atomic_t ctrl_c_pressed = 0;
    auto handler = [](int sig){
        if (sig == SIGINT) ctrl_c_pressed = 1;
    };
    std::signal(SIGINT, handler);

    auto train_config = this->global_config.getTrainConfig(); 
    auto path_config = this->global_config.getPathConfig();
    auto buffer_config = this->global_config.getBufferConfig();

    auto batch_size = train_config.getBatchSizeStates();
    auto shuffle = train_config.getShuffleTrainData();
    auto steps = train_config.getSteps();
    int initial_iter = train_config.getInitialIter();
    auto dummy_generator = torch::make_generator<torch::CPUGeneratorImpl>(train_config.getSeed());

    int last_saved_iter = initial_iter;

    std::cout << "[INFO] Supervised Training Configuration:" << std::endl;
    std::cout << "[INFO] Steps: " << steps << std::endl;
    std::cout << "[INFO] Batch size: " << batch_size << std::endl;
    std::cout << "[INFO] Shuffle: " << shuffle << std::endl;
    std::cout << "[INFO] Initial iteration: " << initial_iter << std::endl;

    auto load_sup_data_path = train_config.getPathToLoadSupervisedData();
    std::cout << load_sup_data_path << std::endl;

    if(load_sup_data_path != ""){
        std::cout << "[INFO] Loading supervised dataset " << load_sup_data_path << std::endl;
        load_supervised_data(socket, server_k, load_sup_data_path);
    } else {
        if (initial_iter == 0) {
            std::cout << "[INFO] Generating supervised data..." << std::endl;

            auto mappings = this->gen_supervised_data(test_initial_states);
            if(ctrl_c_pressed){
                std::cout << "[INFO] CTRL+C pressed during data generation." << std::endl;
            }

            for (auto& mapping : mappings) {
                this->buffer->add_mapping(std::move(mapping));
            }

            auto dataset = buffer->get_batch();
            std::vector<std::vector<int>> grouped_idxs =
                create_batch_size_idxs(dataset.size(), batch_size, shuffle, dummy_generator);

            std::vector<std::vector<c10::IValue>> all_train_data(grouped_idxs.size());
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, grouped_idxs.size()),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t ip = r.begin(); ip != r.end(); ++ip) {
                        if(ctrl_c_pressed) return;
                        all_train_data[ip] = dataset.get_train_data(this->collate, grouped_idxs[ip]);
                    }
                }
            );

            buffer->clear_buffer();

            int num_batches = all_train_data.size();
            int i = 1;
            for(auto& batch: all_train_data){
                if(ctrl_c_pressed) break;
                send_supervised_data(this->socket, batch, this->server_k, i == num_batches);
                i++;
            }
        } else {
            std::cout << "[INFO] Loading supervised dataset from checkpoint..." << std::endl;
            load_dataset_from_checkpoint(this->socket, this->server_k);
        }
    }

    std::cout << "[INFO] Number of test states " << test_initial_states.size() << std::endl;

    auto dummy_data = std::vector<c10::IValue>({torch::tensor({0.0}, torch::kFloat32)});
    auto dummy_train_states = std::vector<std::shared_ptr<StateT>>();
    if(path_config.getLoadCheckpointPath() != ""){
        load_timers();
    }else{
        initialize_timers();
    }
    for (auto step = initial_iter; step <= steps; step++) {
        bool finish_loop = ctrl_c_pressed ||  time_exceeded();  

        auto step_start = std::chrono::high_resolution_clock::now();

        auto [eval_train_metrics_dict, eval_test_metrics_dict,
              train_hist_dict, test_hist_dict, eval_mapping] =
            eval_step(step, steps, test_initial_states, dummy_train_states, finish_loop);

        bool skip_train = (step == steps);

        std::unordered_map<std::string, double> train_dict;

        if (!skip_train && !finish_loop) {
            std::cout << "[INFO] Performing supervised training on server at step " << step << std::endl;
            train_dict = server_train(socket, dummy_data, step, 0, server_k);
            std::cout << "[INFO] Supervised training completed for step " << step << std::endl;
        }

        for (auto& pair : eval_train_metrics_dict)
            eval_test_metrics_dict[pair.first] = pair.second;

        write_tensorboard(socket, server_k, train_dict,
                          eval_test_metrics_dict, train_hist_dict,
                          test_hist_dict, step);
                          
        update_last_time();
        this->launcher_save_model(step, step == steps, dummy_generator, 0);
        if (!finish_loop){
            last_saved_iter = step;
            auto step_end = std::chrono::high_resolution_clock::now();
            std::cout << "[INFO] Step " << step << " completed in "
                      << std::chrono::duration<double>(step_end - step_start).count()
                      << " seconds.\n" << std::endl;
        } else { break;}
    }

    if(ctrl_c_pressed){
        std::cout << "[INFO] CTRL+C detected. Performing final evaluation and saving model..." << std::endl;
        this->launcher_save_model(last_saved_iter,
                                  last_saved_iter == steps,
                                  dummy_generator,
                                  0,
                                  true);
    }
}

    void save_mapping(
        const MappingHistory<StateT>& mapping,
        const std::string& path_to_save,
        std::string filename_prefix,
        std::string path_prefix = "")
    {
        nlohmann::json js;
    
        struct FileData {
            std::string json_path;
            std::string dot_path;
            std::string json_content;
            std::string dot_content;
        };


        auto last_state = mapping.get_last_state();
        auto dfg = last_state->get_dfg();
        FileData file;
            auto cgra = last_state->get_cgra();
            auto dims = cgra->get_cgra_dimensions_();
            nlohmann::json local_js;
            local_js["graph_properties"]["node_count"] = dfg->num_vertices();
            local_js["graph_properties"]["edge_count"] = dfg->get_num_edges();
            local_js["graph_properties"]["II"] = last_state->get_II();
            local_js["architecture_properties"]["dimensions"] = {dims.first, dims.second};
            local_js["architecture_properties"]["interconnections"] = cgra->get_interconnection_styles_json();
    
            local_js["placement"] = nlohmann::json::object();
            const auto& node_to_pe = last_state->get_node_to_pe_();
            for (const auto& [node, pe] : node_to_pe) {
                auto node_name = dfg->get_name_by_node(node);
                auto coord = cgra->get_coord_by_PE_(pe);
                local_js["placement"][node_name] = {
                    std::get<1>(coord),
                    std::get<2>(coord),
                    std::get<0>(coord)
                };
            }
    
            auto base_filename = std::format("{}/{}/{}x{}/{}/{}/{}_{}",
                path_prefix,
                last_state->get_II(),
                dims.first,
                dims.second,
                std::bitset<4>(gen_bin_by_styles(cgra->get_interconnection_styles())).to_string(),
                dfg->num_vertices(),
                filename_prefix,
                dfg->get_dfg_name()            
            );        
            file.json_path = path_to_save + base_filename ;
            file.dot_path = path_to_save + base_filename ;
            file.json_content = local_js.dump(4);
            file.dot_content = dfg->to_dot_string();
        

            const auto& f = file;
                    
            std::filesystem::create_directories(std::filesystem::path(f.json_path).parent_path());
        
            std::ofstream json_file(f.json_path + ".json");
            if (!json_file) {
                std::cerr << "Error opening JSON file: " << f.json_path << "\n";
                throw std::runtime_error("Error opening JSON file");
            }
            json_file << f.json_content;
        
            std::ofstream dot_file(f.dot_path + ".dot");
            if (!dot_file) {
                std::cerr << "Error opening DOT file: " << f.dot_path << "\n";
                throw std::runtime_error("Error opening DOT file");
            }
            dot_file << f.dot_content;

            json_file.close();
            dot_file.close();
            // std::cout << "Saved file \n" << std::flush;
        }            

        
    

    void save_mappings(
        const std::vector<MappingHistory<StateT>>& mappings,
        const std::string& path_to_save,
        const std::string& filename_prefix,
        const std::string& path_prefix = "")
    {
        struct FileData {
            std::filesystem::path base_path;
            std::string json_content;
            std::string dot_content;
        };
    
        std::vector<FileData> files(mappings.size());
    
        tbb::parallel_for(size_t(0), mappings.size(), [&](size_t i) {
            const auto& mapping = mappings[i];
            auto last_state = mapping.get_last_state();
            auto dfg = last_state->get_dfg();
    
            if (dfg->is_disconnected()) {
                return;
            }
    
            auto cgra = last_state->get_cgra();
            auto dims = cgra->get_cgra_dimensions_();
    
            nlohmann::json local_js;
            local_js["graph_properties"]["node_count"] = dfg->num_vertices();
            local_js["graph_properties"]["edge_count"] = dfg->get_num_edges();
            local_js["graph_properties"]["II"] = last_state->get_II();
            local_js["architecture_properties"]["dimensions"] = {dims.first, dims.second};
            local_js["architecture_properties"]["interconnections"] =
                cgra->get_interconnection_styles_json();
    
            local_js["placement"] = nlohmann::json::object();
            const auto& node_to_pe = last_state->get_node_to_pe_();
            for (const auto& [node, pe] : node_to_pe) {
                auto node_name = dfg->get_name_by_node(node);
                auto coord = cgra->get_coord_by_PE_(pe);
                local_js["placement"][node_name] = {
                    std::get<1>(coord),
                    std::get<2>(coord),
                    std::get<0>(coord)
                };
            }
    
            std::string base_filename = std::format(
                "{}/{}/{}x{}/{}/{}/{}",
                path_prefix,
                last_state->get_II(),
                dims.first,
                dims.second,
                std::bitset<4>(
                    gen_bin_by_styles(cgra->get_interconnection_styles())
                ).to_string(),
                dfg->num_vertices(),
                filename_prefix
            );
    
            files[i].base_path =
                std::filesystem::path(path_to_save) / base_filename;
            files[i].json_content = local_js.dump(4);
            files[i].dot_content = dfg->to_dot_string();
        });
    
        for (size_t i = 0; i < files.size(); ++i) {
            const auto& f = files[i];
            if (f.base_path.empty()) {
                continue;
            }
    
            std::filesystem::create_directories(f.base_path.parent_path());
    
            auto json_file_path = f.base_path.string() + std::format("_{}.json", i);
            auto dot_file_path  = f.base_path.string() + std::format("_{}.dot", i);
    
            std::ofstream json_file(json_file_path);
            if (!json_file) {
                throw std::runtime_error(
                    "Error opening JSON file: " + json_file_path
                );
            }
            json_file << f.json_content;
    
            std::ofstream dot_file(dot_file_path);
            if (!dot_file) {
                throw std::runtime_error(
                    "Error opening DOT file: " + dot_file_path
                );
            }
            dot_file << f.dot_content;
        }
    
        std::cout << "Saved " << files.size() << " files\n" << std::flush;
    }
        

    
    std::tuple<std::vector<MappingHistory<StateT>>, std::unordered_map<std::string, double> > gen_mappings_only(
        SelfMapping<StateT,ModelConfig>& self_mapping,  std::vector<std::shared_ptr<StateT>>& train_initial_states, double train_ratio,
    std::unordered_map<std::pair<int,int>, std::vector<double>>& dims_to_action_count, int initial_seed) 
    {
        std::optional<MappingHistory<StateT>> result = std::nullopt;

        auto train_config = global_config.getTrainConfig();
        auto num_samples_per_mapping = train_config.getNumSamplesPerMapping();
        auto use_efficient_sampler = train_config.getUseEfficientSampling();
        if(use_efficient_sampler) assert(num_samples_per_mapping == 1);
        std::unordered_map<std::string, double> data_gen_metrics_dict;

        tbb::enumerable_thread_specific<std::vector<MappingHistory<StateT>>> thread_buffers;
        tbb::enumerable_thread_specific<double> thread_reward_sums;

        // tbb::enumerable_thread_specific<double> thread_total_nodes_count;
        tbb::enumerable_thread_specific<double> thread_total_mapped_nodes_count;

        // tbb::enumerable_thread_specific<double> thread_total_edges_count;
        tbb::enumerable_thread_specific<double> thread_total_mapped_edges_count;

        // tbb::enumerable_thread_specific<double> thread_total_node_convergences_count;
        tbb::enumerable_thread_specific<double> thread_total_node_valid_convergences_count;


        // tbb::enumerable_thread_specific<double> thread_total_edge_convergences_count;
        tbb::enumerable_thread_specific<double> thread_total_edge_valid_convergences_count;

        tbb::enumerable_thread_specific<double> thread_total_valid_mapping_count;

        tbb::enumerable_thread_specific<double> thread_total_routing_nodes_count;
        tbb::enumerable_thread_specific<int> thread_step_counts;

        tbb::enumerable_thread_specific<std::optional<MappingHistory<StateT>>> thread_best_mapping;

    #ifdef DEBUG
        std::cout << "\n[DEBUG] gen_train_data - Starting generation\n";
        std::cout << "[DEBUG] Number of initial states: " << train_initial_states.size() << "\n";
        std::cout << "[DEBUG] Train ratio: " << train_ratio << "\n";
        std::cout << "[DEBUG] Samples per mapping: " << num_samples_per_mapping << "\n";
        std::cout << "[DEBUG] Mapping mode: " << (this->mapping_mode ? "ON" : "OFF") << "\n";
    #endif

        
    bool aug_mapping = this->global_config.getTrainConfig().augment_mapping();
    // auto enum_training = this->global_config.getLauncherConfig().getTraining();
    if (aug_mapping) {
        std::cout << "[INFO] Augmenting mappings is enabled." << std::endl;
    }
    arena.execute([&] {
        tbb::task_group tg;
    
        for (size_t i = 0; i < train_initial_states.size(); ++i) {

            auto& state = train_initial_states[i];
            for (int k = 0; k < num_samples_per_mapping; ++k) {
                tg.run([&, i, k] {
                    int seed = initial_seed + k;
    
                
                    auto mapping_history = self_mapping.map(state, true, train_ratio, seed, std::nullopt);
    
                    double reward = mapping_history.calc_sum_reward();
                    int steps = mapping_history.get_episode_length() - 1;
                    thread_reward_sums.local() += reward;
                    thread_step_counts.local() += steps;
    
                    auto mapping_is_valid = mapping_history.get_mapping_is_valid();
                    thread_total_valid_mapping_count.local() += mapping_is_valid ? 1. : 0.;
    
                    auto total_nodes = mapping_history.get_num_dfg_nodes();
                    thread_total_mapped_nodes_count.local() +=
                        static_cast<double>(mapping_history.get_num_mapped_dfg_nodes()) / static_cast<double>(total_nodes);
    
                    auto total_edges = mapping_history.get_num_dfg_edges();
                    thread_total_mapped_edges_count.local() +=
                        static_cast<double>(mapping_history.get_num_mapped_dfg_edges()) / static_cast<double>(total_edges);
    
                    auto total_node_convergences = mapping_history.get_num_dfg_node_convergences();
                    thread_total_node_valid_convergences_count.local() += total_node_convergences == 0 ? 1 :
                        static_cast<double>(mapping_history.get_num_valid_dfg_node_convergences()) /
                        static_cast<double>(total_node_convergences);
    
                    auto total_edge_convergences = mapping_history.get_num_dfg_edge_convergences();
                    thread_total_edge_valid_convergences_count.local() += total_edge_convergences == 0 ? 1 :
                        static_cast<double>(mapping_history.get_num_valid_dfg_edge_convergences()) /
                        static_cast<double>(total_edge_convergences);
    
                    thread_total_routing_nodes_count.local() += mapping_history.get_total_routing_nodes();
    
                    auto& local_buf = thread_buffers.local();
                 
                    local_buf.emplace_back(std::move(mapping_history));
                 
                    
                    if (this->mapping_mode && mapping_is_valid) {
                        auto& local_best = thread_best_mapping.local();
                        if (!local_best || reward > local_best->calc_sum_reward()) {
                            local_best = local_buf.back();
                        }
                    }
                });
            }
        }
    
        tg.wait();
    });

    std::vector<MappingHistory<StateT>> mappings;

        for (auto& local_buffer : thread_buffers) {
            for (auto& mapping : local_buffer) {
              mappings.push_back(std::move(mapping));
            }
        }


        double total_reward_sum = 0.0;
        // double total_nodes = 0.0;
        double total_mapped_nodes = 0.0;

        // double total_edges = 0.0;
        double total_mapped_edges = 0.0;

        // double total_node_convergences = 0.0;
        double total_node_valid_convergences = 0.0;

        // double total_edge_convergences = 0.0;
        double total_edge_valid_convergences = 0.0;

        double total_routing_nodes = 0.0;


        int total_step_count = 0;
        for (auto& val : thread_reward_sums) total_reward_sum += val;
        for (auto& cnt : thread_step_counts) total_step_count += cnt;
        
        // for (auto& cnt : thread_total_nodes_count) total_nodes += cnt;
        for (auto& cnt : thread_total_mapped_nodes_count) total_mapped_nodes += cnt;
        
        // for (auto& cnt : thread_total_edges_count) total_edges += cnt;
        for (auto& cnt : thread_total_mapped_edges_count) total_mapped_edges += cnt;
        
        // for (auto& cnt : thread_total_node_convergences_count) total_node_convergences += cnt;
        for (auto& cnt : thread_total_node_valid_convergences_count) total_node_valid_convergences += cnt;
        
        // for (auto& cnt : thread_total_edge_convergences_count) total_edge_convergences += cnt;
        for (auto& cnt : thread_total_edge_valid_convergences_count) total_edge_valid_convergences += cnt;

        for(auto cnt: thread_total_routing_nodes_count){
            total_routing_nodes += cnt;
        }
        int total_valid_mapping = 0;
        for(auto cnt: thread_total_valid_mapping_count) total_valid_mapping += cnt;

        auto total_mappings = static_cast<double>(train_initial_states.size()* num_samples_per_mapping);
        double average_reward = (total_step_count > 0)
                                ? (total_reward_sum / total_step_count)
                                : 0.0;
        double pct_mapped_nodes = (total_mappings > 0)
                                ? (total_mapped_nodes / total_mappings)
                                : 0.0;
        double pct_mapped_edges = (total_mappings > 0)
                                ? (total_mapped_edges / total_mappings)
                                : 0.0;
        double pct_valid_node_convergences = (total_mappings > 0)
                                ? (total_node_valid_convergences / total_mappings)
                                : 0.0;
        double pct_valid_edge_convergences = (total_mappings > 0)
                                ? (total_edge_valid_convergences / total_mappings)
                                : 0.0;
        double average_return = (total_mappings > 0) ? total_reward_sum / total_mappings : 0.0;

        double average_routing_nodes = total_routing_nodes / total_mappings;

        double valid_mapping_rate = total_valid_mapping / total_mappings;
        std::cout << "[INFO] Mean reward: " << average_reward << std::endl;

    #ifdef DEBUG
        std::cout << "[DEBUG] gen_train_data - Completed\n";
        std::cout << "[DEBUG] Final buffer size: " << buffer->size() << "\n";
        std::cout << "[DEBUG] Result " << (result.has_value() ? "contains" : "does not contain")
                << " a valid mapping\n";
    #endif
        data_gen_metrics_dict["MappingGen/mean_reward"] = average_reward;
        data_gen_metrics_dict["MappingGen/pct_mapped_nodes_per_mapping"] = pct_mapped_nodes;
        data_gen_metrics_dict["MappingGen/pct_mapped_edges_per_mapping"] = pct_mapped_edges;
        data_gen_metrics_dict["MappingGen/pct_valid_node_convergences_per_mapping"] = pct_valid_node_convergences;
        data_gen_metrics_dict["MappingGen/pct_valid_edge_convergences_per_mapping"] = pct_valid_edge_convergences;
        data_gen_metrics_dict["MappingGen/mean_return"] = average_return;
        data_gen_metrics_dict["MappingGen/mean_rout_nodes_per_mapping"] = average_routing_nodes;
        data_gen_metrics_dict["MappingGen/valid_mapping_rate"] = valid_mapping_rate;
        data_gen_metrics_dict["MappingGen/mean_episode_length"] = (total_step_count + total_mappings) / total_mappings;

        return std::make_tuple(std::move(mappings), std::move(data_gen_metrics_dict));
    }

    bool time_exceeded() {
        auto launcher_config = this->global_config.getLauncherConfig();
        auto max_exp_time = launcher_config.getMaxExpTime(); // in seconds
        return elapsed_time >= max_exp_time;
    }


    std::optional<MappingHistory<StateT>> hybrid_training(
         std::vector<MappingHistory<StateT>>& mappings,
         std::vector<std::shared_ptr<StateT>>& rl_train_states,
        const std::vector<std::shared_ptr<StateT>>& test_initial_states
    ) {
        static volatile sig_atomic_t ctrl_c_pressed = 0;

        auto handler = [](int sig){
            if (sig == SIGINT)
                ctrl_c_pressed = 1;
        };
        std::signal(SIGINT, handler);



        auto train_config = this->global_config.getTrainConfig(); 
        auto path_config = this->global_config.getPathConfig();
        auto ppo_config = this->global_config.getPPOConfig();
        auto buffer_config = this->global_config.getBufferConfig();
        auto clear_mcts_buffer_after_train = this->mcts_global_config.getBufferConfig().getClearBufferAfterTraining();
    
        int steps = train_config.getSteps();                                 
        int buffer_size_to_start_train = train_config.getBufferSizeToStartTraining();
        // int mcts_buffer_capacity = buffer_config.getBufferSize();         
    
        std::vector<std::shared_ptr<StateT>> train_initial_states;
        train_initial_states.reserve(mappings.size());
        std::unordered_map<int, MappingHistory<StateT>> id_to_mapping;
        int count = 0;
        for (auto& mapping : mappings) {
            auto initial_state = mapping.get_first_state();
            train_initial_states.push_back(initial_state);
            initial_state->set_id(count);
            assert(!id_to_mapping.count(initial_state->get_id()));
            id_to_mapping[initial_state->get_id()] = std::move(mapping);
            count++;
        }

        //load found mappings if exists
        auto path_to_found_mappings = path_config.getSaveCheckpointPath() + "found_mappings/";
        //create if dont exist
        std::filesystem::create_directories(path_to_found_mappings);
        auto found_mappings = generate_mappings(path_to_found_mappings);
        
        for(auto& mapping: found_mappings){
            auto initial_state = mapping.get_first_state();
            assert(!id_to_mapping.count(initial_state->get_id()));
            id_to_mapping[initial_state->get_id()] = std::move(mapping);
        }

        for (auto& state : rl_train_states) {
            state->set_id(count);
            count++;
        }
    
        auto cur_train_states = this->get_train_states(train_initial_states);
        auto batch_size = train_config.getBatchSizeStates();
        auto shuffle = train_config.getShuffleTrainData();
        auto iterations = train_config.getIterations();
        auto PER = buffer_config.getPER();
        int initial_iter = train_config.getInitialIter();
        auto update_model_interval = train_config.getUpdateModelInterval();
        int num_envs = train_config.getNumEnvs();
        auto supervised_train_interval = train_config.getSupervisedTrainInterval();
        auto mcts_train_interval = train_config.getMCTSTrainInterval();
        auto ppo_train_interval = train_config.getPPOTrainInterval();
        auto rl_train_start = train_config.getRLTrainStart();

        torch::Generator generator = torch::make_generator<torch::CPUGeneratorImpl>(train_config.getSeed());
    
        // Load generator checkpoint if exists
        if (!path_config.getLoadCheckpointPath().empty()) {
            std::cout << "[INFO] Loading env generator from " << path_config.getLoadCheckpointPath() << std::endl;
            torch::Tensor gen_state;
            torch::load(gen_state, path_config.getLoadCheckpointPath() + "env_generator_state.pt");
            generator.set_state(gen_state);
        }
    
        std::vector<double> sampled_idx_to_count(cur_train_states.size(), 0.0);
        std::unordered_map<std::pair<int,int>, std::vector<double>> dims_to_action_count;
        int max_graph_size = 0;
        int step = 0;
    
        for (auto& state : cur_train_states)
            max_graph_size = std::max(max_graph_size, state->get_num_dfg_nodes());
        std::cout << "[INFO] Max graph size: " << max_graph_size << "\n";
        this->env_sampler.init_priorities(mappings.size() + rl_train_states.size());

       
        auto use_efficient_sampler = train_config.getUseEfficientSampling();

        if (use_efficient_sampler && path_config.getLoadCheckpointPath() != "") {
            std::string sampler_path = path_config.getLoadCheckpointPath() + "env_sampler.bin";
            std::ifstream ifs(sampler_path, std::ios::binary);
            if (ifs.good()) {
                std::cout << "[INFO] Loading env sampler from " << sampler_path << std::endl;
                try {
                    cereal::BinaryInputArchive archive(ifs);
                    archive(this->env_sampler);
                    // this->env_sampler.update_states(cur_train_states);
                } catch (const std::exception& e) {
                    std::cerr << "[WARN] Failed to load env_sampler: " << e.what() << std::endl;
                }
            }

            
            rl_train_states = this->env_sampler.add_data_if_exists(rl_train_states);

            if (this->use_curriculum_learning) {
                cur_train_states = this->map_curriculum.fill_states_until_current_index(cur_train_states);
            }

            for (auto& s : cur_train_states){
                this->env_sampler.add_old_sample(s);
            }
        }

        
        int last_saved_iter = initial_iter;
        double last_entropy_loss = 0.0;
        if(path_config.getLoadCheckpointPath() != ""){
            load_timers();
        }else{
            initialize_timers();
        }

        std::unordered_set<int> mappings_to_save;
        while(!ctrl_c_pressed && !time_exceeded()) {
            for (int i = initial_iter; i <= steps; i++) {
                auto finish_loop = ctrl_c_pressed || time_exceeded();
                int N = static_cast<int>(cur_train_states.size());
                int sample_size = (num_envs != -1) ? std::min(num_envs, N) : N;
        
                std::vector<std::shared_ptr<StateT>> real_train_states;
                if (train_config.getUseEfficientSampling() && id_to_mapping.size() > 0) {
                    real_train_states = this->env_sampler.sample_states(generator);
                } else {
                    torch::Tensor sampled_indices = torch::randperm(N, generator).slice(0, 0, sample_size);
                    real_train_states.reserve(sample_size);
                    for (int j = 0; j < sampled_indices.size(0); j++) {
                        int idx = sampled_indices[j].item<int>();
                        sampled_idx_to_count[idx] += 1.0;
                        real_train_states.push_back(cur_train_states[idx]);
                    }
                }
        
                auto iter_start = std::chrono::steady_clock::now();
                std::cout << "[INFO] Iteration " << i << " started." << std::endl;
                std::cout << "[INFO] Number of environments (Supervised + PPO): " << real_train_states.size() << std::endl;
                auto w_entropy_loss = compute_entropy_bonus(i, steps, ppo_config);
                bool skip_train = (i == steps);
        
                std::unordered_map<std::string, double> train_metrics_dict;
                double mean_raw_adv = 0.0, std_raw_adv = 0.0, mean_values = 0.0, std_values = 0.0;
                std::unordered_map<std::string, double> train_dict;
        
                auto [eval_train_metrics_dict, eval_test_metrics_dict, train_hist_dict, test_hist_dict, eval_mapping] =
                    eval_step(i, steps, test_initial_states, cur_train_states, finish_loop);
                if (eval_mapping.has_value()) return eval_mapping;
        
                // ---------------- Supervised Training ----------------
                if (!finish_loop && !skip_train && (i % supervised_train_interval == 0) && !real_train_states.empty()) {
                    std::vector<std::reference_wrapper<MappingHistory<StateT>>> real_train_mappings;
                    real_train_mappings.reserve(real_train_states.size());
                    for (const auto& state : real_train_states) {
                        auto it = id_to_mapping.find(state->get_id());
                        if (it != id_to_mapping.end())
                            real_train_mappings.push_back(std::ref(it->second));
                    }
        
                    auto supervised_train_batches = this->generate_batches(real_train_mappings, batch_size);
                    std::unordered_map<std::string, double> supervised_train_dict;
        
                    for (const auto& data : supervised_train_batches) {
                        auto batch_metrics = server_train(socket, data, step, 0, server_k, static_cast<int>(EnumTraining::SUPERVISED));
                        for (const auto& p : batch_metrics)
                            supervised_train_dict["Supervised/" + p.first] += p.second;
                    }
        
                    if (!supervised_train_batches.empty()) {
                        for (auto& p : supervised_train_dict)
                            p.second /= static_cast<double>(supervised_train_batches.size());
                    }
        
                    for (const auto& p : supervised_train_dict)
                        train_dict[p.first] = p.second;
                }
        
                // ---------------- PPO Data Generation ----------------
                if (i >= rl_train_start && (i % ppo_train_interval == 0) && !finish_loop && !skip_train && !real_train_states.empty()) {
                    double train_ratio = static_cast<double>(i) / steps;
                    auto data_start = std::chrono::steady_clock::now();
        
                    auto [mapping, gen_data_dict] = this->gen_train_data(this->config_self_mapping, real_train_states, train_ratio, dims_to_action_count, train_config.getSeed());
        
                    // EMA updates for curriculum learning
                    auto cur_mean_rout_nodes = gen_data_dict["MappingGen/mean_rout_nodes_per_mapping"];
                    auto cur_mean_valid_map_rate = gen_data_dict["MappingGen/valid_mapping_rate"];
                    this->ema_rout_nodes = this->calc_ema(this->ema_rout_nodes, cur_mean_rout_nodes, this->ema_alpha);
                    this->ema_valid_map_rate = this->calc_ema(this->ema_valid_map_rate, cur_mean_valid_map_rate, this->ema_alpha);
                    gen_data_dict["Control/ema_rout_nodes"] = this->ema_rout_nodes;
                    gen_data_dict["Control/ema_valid_map_rate"] = this->ema_valid_map_rate;
        
                    for (auto& pair : gen_data_dict)
                        train_metrics_dict["PPO/" + pair.first] = pair.second;
        
                    if (mapping.has_value()) return mapping;
        
                    auto data_end = std::chrono::steady_clock::now();
                    std::cout << "[INFO] Data generation took "
                            << std::chrono::duration<double>(data_end - data_start).count()
                            << " seconds." << std::endl;
        
                    // ---------------- PPO Training ----------------
                    if ((i + 1) % update_model_interval == 0 && this->buffer->get_num_data_in_buffer() > 0) {
                        auto batch_start = std::chrono::steady_clock::now();
                        auto dataset = this->buffer->get_batch();
                        auto batch_end = std::chrono::steady_clock::now();
                        std::cout << "[INFO] Getting batch - Time: "
                                << std::chrono::duration<double>(batch_end - batch_start).count()
                                << std::endl;
        
                        mean_raw_adv = dataset.get_raw_mean_advantages();
                        std_raw_adv = dataset.get_raw_std_advantages();
                        mean_values = dataset.get_mean_values();
                        std_values = dataset.get_std_values();
        
                        bool first_train = true;
                        auto grouped_idxs = create_batch_size_idxs(dataset.size(), batch_size, shuffle, generator);
                        std::vector<std::vector<c10::IValue>> all_train_data(grouped_idxs.size());
        
                        tbb::parallel_for(
                            tbb::blocked_range<size_t>(0, grouped_idxs.size()),
                            [&](const tbb::blocked_range<size_t>& r) {
                                for (size_t ip = r.begin(); ip != r.end(); ++ip)
                                    all_train_data[ip] = dataset.get_train_data(this->collate, grouped_idxs[ip]);
                            }
                        );
        
                        std::unordered_map<std::string, double> ppo_accum_metrics;
                        for (int k = 0; k < iterations; k++) {
                            int idx_count = 0;
                            for (auto& cur_train_data : all_train_data) {
                                auto cur_train_batch = server_train(socket, cur_train_data, step, w_entropy_loss, server_k, static_cast<int>(EnumTraining::PPO));
                                step++;
        
                                if (first_train) {
                                    for (auto& p : cur_train_batch) ppo_accum_metrics["PPO/" + p.first] = p.second;
                                    first_train = false;
                                } else {
                                    for (auto& p : cur_train_batch) ppo_accum_metrics["PPO/" + p.first] += p.second;
                                }
        
                                if ((k == iterations - 1) && PER) {
                                    auto tensor_indices = torch::tensor(grouped_idxs[idx_count], torch::kInt32);
                                    this->update_priorities(dataset, tensor_indices);
                                }
                                idx_count++;
                            }
                        }
        
                        double denom = static_cast<double>(grouped_idxs.size() * iterations);
                        for (auto& p : ppo_accum_metrics)
                            p.second /= denom;
        
                        // merge em train_dict
                        for (auto& p : ppo_accum_metrics)
                            train_dict[p.first] = p.second;
        
                        std::cout << "[INFO] PPO training completed for this iteration." << std::endl;
                    }
                }
        
                // ---------------- MCTS ----------------
                if (i >= rl_train_start && (i % mcts_train_interval == 0) && !finish_loop && !skip_train && rl_train_states.size() > 0) {
                    double train_ratio = static_cast<double>(i) / static_cast<double>(steps);
                    auto rate = train_config.getGoodSampleRate();
                    int N = static_cast<int>(rl_train_states.size());
                    int sample_size = (num_envs != -1) ? std::min(num_envs, N) : N;
                    sample_size = std::max(sample_size, static_cast<int>(rate * N));
                    torch::Tensor sampled_indices = torch::randperm(N, generator).slice(0, 0, sample_size);
        
                    std::vector<std::shared_ptr<StateT>> mcts_train_states;
                    mcts_train_states.reserve(sample_size);
                    for (int j = 0; j < sampled_indices.size(0); j++) {
                        int idx = sampled_indices[j].item<int>();
                        mcts_train_states.push_back(rl_train_states[idx]);
                    }
        
                    std::cout << "[INFO] Generating MCTS training data from " << mcts_train_states.size() << " initial states." << std::endl;
                    auto [mappings_out, gen_data_dict] = this->gen_mappings_only(this->mcts_self_mapping, mcts_train_states, train_ratio, dims_to_action_count, train_config.getSeed());
        
                    for (auto& mapping : mappings_out) {
                        if (mapping.get_mapping_is_valid() && mapping.get_total_routing_nodes() == 0) {
                            auto initial_state = mapping.get_first_state();
                            mappings_to_save.insert(initial_state->get_id());
                           
                            id_to_mapping[initial_state->get_id()] = std::move(mapping);
                            this->env_sampler.add_sample(initial_state);
                            //clean from rl_train_states finding state_id and swapping
                            auto it = std::find_if(rl_train_states.begin(), rl_train_states.end(),
                                [&](const std::shared_ptr<StateT>& state) {
                                    return state->get_id() == initial_state->get_id();
                                });
                            
                                std::swap(*it, rl_train_states.back());
                                rl_train_states.pop_back();
                        } else {
                            this->mcts_buffer->add_mapping(std::move(mapping));
                        }
                    }
        
                    for (auto& kv : gen_data_dict)
                        train_metrics_dict["MCTS/" + kv.first] = kv.second;
        
                    if (this->mcts_buffer->get_num_data_in_buffer() >= buffer_size_to_start_train) {
                        auto cur_mean_rout_nodes = gen_data_dict["MappingGen/mean_rout_nodes_per_mapping"];
                        auto cur_mean_valid_map_rate = gen_data_dict["MappingGen/valid_mapping_rate"];
                        this->ema_rout_nodes = this->calc_ema(this->ema_rout_nodes, cur_mean_rout_nodes, this->ema_alpha);
                        this->ema_valid_map_rate = this->calc_ema(this->ema_valid_map_rate, cur_mean_valid_map_rate, this->ema_alpha);
                        train_metrics_dict["MCTS/Control/ema_rout_nodes"] = this->ema_rout_nodes;
                        train_metrics_dict["MCTS/Control/ema_valid_map_rate"] = this->ema_valid_map_rate;
        
                        if ((i + 1) % update_model_interval == 0) {
                            auto dataset = this->mcts_buffer->get_batch();
        
                            mean_raw_adv = dataset.get_raw_mean_advantages();
                            std_raw_adv = dataset.get_raw_std_advantages();
                            mean_values = dataset.get_mean_values();
                            std_values = dataset.get_std_values();
        
                            auto grouped_idxs = create_batch_size_idxs(dataset.size(), batch_size, shuffle, generator);
                            std::vector<std::vector<c10::IValue>> all_train_data(grouped_idxs.size());
                            std::vector<torch::Tensor> all_train_indices(grouped_idxs.size());
        
                            tbb::parallel_for(
                                tbb::blocked_range<size_t>(0, grouped_idxs.size()),
                                [&](const tbb::blocked_range<size_t>& r) {
                                    for (size_t ip = r.begin(); ip != r.end(); ++ip) {
                                        all_train_data[ip] = dataset.get_train_data(this->collate, grouped_idxs[ip]);
                                        all_train_indices[ip] = torch::tensor(grouped_idxs[ip], torch::kInt32);
                                    }
                                }
                            );
        
                            bool first_train = true;
                            std::unordered_map<std::string, double> mcts_accum;
        
                            for (size_t k = 0; k < all_train_data.size(); ++k) {
                                auto cur_train_dict = server_train(socket, all_train_data[k], i, std::nullopt, server_k, static_cast<int>(EnumTraining::MCTS));
                                if (first_train) {
                                    for (const auto& p : cur_train_dict) mcts_accum["MCTS/" + p.first] = p.second;
                                    first_train = false;
                                } else {
                                    for (const auto& p : cur_train_dict) mcts_accum["MCTS/" + p.first] += p.second;
                                }
                                if (PER) this->update_priorities(dataset, all_train_indices[k]);
                            }
        
                            for (auto& p : mcts_accum)
                                p.second /= static_cast<double>(all_train_data.size());
        
                            for (auto& p : mcts_accum)
                                train_dict[p.first] = p.second;
                        }
                    }
                }
        
                // ---------------- Control / logging metrics ----------------
                train_metrics_dict["Control/num_data_in_buffer"] = this->buffer->get_num_data_in_buffer();
                train_metrics_dict["Control/num_initial_train_states"] = static_cast<double>(cur_train_states.size());
                train_metrics_dict["Control/trained_envs_per_step"] = static_cast<double>(real_train_states.size());
                train_metrics_dict["Control/mean_raw_adv"] = mean_raw_adv;
                train_metrics_dict["Control/std_raw_adv"] = std_raw_adv;
                train_metrics_dict["Control/mean_pred_values"] = mean_values;
                train_metrics_dict["Control/std_pred_values"] = std_values;
                double mean_rout_nodes_threshold = this->curriculum_progress.get_mean_rout_nodes_threshold();
                double valid_map_rate_threshold = this->curriculum_progress.get_valid_map_rate_threshold();
                train_metrics_dict["Control/mean_rout_nodes_threshold"] = mean_rout_nodes_threshold;
                train_metrics_dict["Control/valid_map_rate_threshold"] = valid_map_rate_threshold;
        
                if (train_config.getUseEfficientSampling()) {
                    train_metrics_dict["Control/num_good_states"] = static_cast<double>(this->env_sampler.get_num_good_states());
                    train_metrics_dict["Control/num_bad_states"] = static_cast<double>(this->env_sampler.get_num_bad_states());
                }
        
                // train_metrics_dict["Control/mcts_buffer_capacity"] = static_cast<double>(mcts_buffer_capacity);
                train_metrics_dict["Control/mcts_buffer_size_to_start_train"] = static_cast<double>(buffer_size_to_start_train);
                train_dict["Control/num_supervised_mapping_in_buffer"] = static_cast<double>(this->supervised_buffer->get_num_data_in_buffer());
        
                for (auto& pair : train_metrics_dict)
                    eval_test_metrics_dict[pair.first] = pair.second;
                for (auto& pair : eval_train_metrics_dict)
                    eval_test_metrics_dict[pair.first] = pair.second;
        
                if (i == steps) {
                    if (train_config.getUseEfficientSampling())
                        train_hist_dict["Control/sampled_idx_count"] = this->env_sampler.get_visit_counts();
                    else
                        train_hist_dict["Control/sampled_idx_count"] = std::ref(sampled_idx_to_count);
                }
        
                write_tensorboard(socket, server_k, train_dict, eval_test_metrics_dict, train_hist_dict, test_hist_dict, i);
        
                if (buffer_config.getClearBufferAfterTraining())
                    this->buffer->clear_buffer();
                
                if( clear_mcts_buffer_after_train )
                    this->mcts_buffer = std::make_shared<Buffer<StateT, ModelConfig>>(mcts_global_config, this->socket, this->server_k, this->batch_requester, "infer");
        
                update_last_time();
                this->launcher_save_model(i, i == steps, generator, w_entropy_loss, false, &mappings_to_save, &id_to_mapping);
              
                auto iter_end = std::chrono::steady_clock::now();
                std::cout << "[INFO] Iteration " << i << " completed in "
                        << std::chrono::duration<double>(iter_end - iter_start).count()
                        << " seconds.\n" << std::endl;
                if(finish_loop) {
                    break;
                }
                else{  
                    last_saved_iter = i;
                    last_entropy_loss = w_entropy_loss;
                };
            }
            if(!ctrl_c_pressed && !time_exceeded()){
                std::cout << "[INFO] Hybrid PPO training completed." << std::endl << std::endl;
                return std::nullopt;
            }
        }
        std::cout << "[INFO] CTRL+C pressed or Time exceeded. Saving model and exiting." << std::endl;
        this->launcher_save_model(last_saved_iter, last_saved_iter == steps, generator, last_entropy_loss, true, &mappings_to_save, &id_to_mapping );
        return std::nullopt;
    }
    
    
};


#include "src/tpp/launcher.tpp"
#endif
