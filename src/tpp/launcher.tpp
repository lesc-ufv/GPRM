#include "src/hpp/launcher.hpp"
    template<typename StateT,typename ModelConfig>
    Launcher<StateT,ModelConfig>::Launcher() {};

    template<typename StateT, typename ModelConfig>
    Launcher<StateT, ModelConfig>::Launcher(GlobalConfig<ModelConfig>& global_config, std::optional<GlobalConfig<ModelConfig>> opt_mcts_global_config) 
        : global_config(global_config),thread_control(tbb::global_control(tbb::global_control::max_allowed_parallelism, 
          global_config.getLauncherConfig().getNumThreads() != -1
        ? global_config.getLauncherConfig().getNumThreads() : std::thread::hardware_concurrency())),
        arena(global_config.getLauncherConfig().getNumThreads() != -1
        ? global_config.getLauncherConfig().getNumThreads() : std::thread::hardware_concurrency()) {
            std::ios::sync_with_stdio(false);
            std::cin.tie(nullptr);    
        
        #ifdef DEBUG
            std::cout << "[DEBUG] Initializing Launcher..." << std::endl;
        #endif
            // int cpus = std::thread::hardware_concurrency();
            // torch::set_num_threads(cpus);
            // torch::set_num_interop_threads(cpus);

            std::cout << "[WARN] Use Hybrid training only with all training states." << std::endl;

            auto& context = ZmqContextHolder::instance();;
            this->socket = std::make_shared<zmq::socket_t>(context, ZMQ_DEALER);
            int timeout = -1;
            this->socket->set(zmq::sockopt::sndtimeo, timeout);
            this->socket->set(zmq::sockopt::rcvtimeo, timeout);

            auto socket_br = std::make_shared<zmq::socket_t>(context, ZMQ_DEALER);
            socket_br->set(zmq::sockopt::sndtimeo, timeout);
            socket_br->set(zmq::sockopt::rcvtimeo, timeout);
            #ifdef DEBUG
            std::cout << "[DEBUG] ZeroMQ socket created and configured." << std::endl;
            #endif
            auto& train_config = this->global_config.getTrainConfig();
            this->ema_alpha = train_config.getEmaAlpha();
            auto launcher_config = this->global_config.getLauncherConfig();
            socket->connect("tcp://localhost:" + launcher_config.getServerPort());
            socket_br->connect("tcp://localhost:" + launcher_config.getServerPort());

            RNG::set_seed(global_config.getTrainConfig().getSeed());

        #ifdef DEBUG
            std::cout << "[DEBUG] Connected to server at tcp://localhost:" << launcher_config.getServerPort() << std::endl;
        #endif

            this->environment = Environment<StateT>(launcher_config.getEnvironment());
            this->collate = Collate<StateT>(launcher_config.getModel());
            std::string func_name = "infer";
            bool use_batch_requester = launcher_config.getUseBatchRequester();
            this->mcts_global_config = opt_mcts_global_config.has_value() ? 
                opt_mcts_global_config.value() : global_config;
            this->batch_requester =  use_batch_requester ? 
                std::make_optional(std::make_shared<BatchRequester<StateT>>(std::move(socket_br), global_config.getLauncherConfig().getBatchSizeRequester(), this->collate, this->server_k)) : 
                std::nullopt;
           
            this->config_self_mapping = SelfMapping<StateT, ModelConfig>(this->global_config, this->server_k, func_name, this->environment, batch_requester);
            this->mcts_self_mapping = SelfMapping<StateT, ModelConfig>(mcts_global_config, this->server_k, func_name, this->environment, batch_requester);
            this->buffer = std::make_shared<Buffer<StateT, ModelConfig>>(global_config, this->socket, this->server_k, this->batch_requester, func_name);
            this->supervised_buffer = std::make_shared<Buffer<StateT, ModelConfig>>(global_config, this->socket, this->server_k, this->batch_requester, func_name);
            this->supervised_buffer->rebuild_functions(EnumReplayBuffer::SUPERVISED_BUFFER);
            this->mcts_buffer = std::make_shared<Buffer<StateT, ModelConfig>>(mcts_global_config, this->socket, this->server_k, this->batch_requester, func_name);
            this->use_curriculum_learning = train_config.getUseCurriculumLearning();

        #ifdef DEBUG
            std::cout << "[DEBUG] Environment, Collate, SelfMapping, Buffer, and Reanalyze initialized." << std::endl;
        #endif


            auto task = launcher_config.getEnumTaskType();
            auto& buffer_config = this->global_config.getBufferConfig();
            this->mapping_mode = (task == EnumTaskType::FINETUNE_MAP) || (EnumTaskType::ZERO_SHOT_MAP == task);
            this->use_reanalyze = buffer_config.getUseReanalyze();
            this->using_mcts = launcher_config.getSelfMapping() != EnumSelfMapping::POLICY_LOGITS;

            if(launcher_config.getReplayBuffer() == EnumReplayBuffer::SMARTMAP_PRE_TRAINING_BUFFER){
                assert(!buffer_config.getPER());
            }

            set_seed(train_config.getSeed());
            std::mutex mtx;
            std::set<std::thread::id> threads_used;
            auto num_threads = launcher_config.getNumThreads() != -1 ? 
                launcher_config.getNumThreads() : std::thread::hardware_concurrency();
            arena.execute([&] {
                tbb::task_group tg;
                for (int i = 0; i < static_cast<int>(num_threads); ++i) {
                    tg.run([&] {
                        std::cout << "[INFO] Thread " << std::this_thread::get_id() << " is running." << std::endl;
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                        std::lock_guard<std::mutex> lock(mtx);
                        threads_used.insert(std::this_thread::get_id());
                            std::cout << "[DEBUG] Thread " << std::this_thread::get_id() << " has been added to the set." << std::endl;

                    });
                }
                tg.wait();
            });
        
        
            std::cout << "[INFO] Max threads: " << threads_used.size() << std::endl;
        
        #ifdef DEBUG
            std::cout << "[DEBUG] Mapping mode: " << this->mapping_mode << std::endl;
            std::cout << "[DEBUG] Use reanalyze: " << this->use_reanalyze << std::endl;
            std::cout << "[DEBUG] Use curriculum learning: " << this->use_curriculum_learning << std::endl;
            std::cout << "[DEBUG] Using MCTS: " << this->using_mcts << std::endl;
        #endif

        #ifdef DEBUG
            std::cout << "[DEBUG] Launcher initialized successfully." << std::endl;
        #endif

        auto& path_config = this->global_config.getPathConfig();
        if (this->use_curriculum_learning){
            std::cout << "[INFO] Using curriculum learning." << std::endl;
            this->map_curriculum = MapCurriculumLearning<StateT,ModelConfig>();
            this->curriculum_progress = CurriculumProgress<ModelConfig>(this->global_config);
        }
        
        auto load_model_path = path_config.get_load_model_path(); 

        if(path_config.getLoadCheckpointPath() != "" && !this->mapping_mode){
            std::cout << "[INFO] Loading torch internal generator state from " << path_config.getLoadCheckpointPath() << std::endl;
            torch::Tensor loaded_state;
            torch::load(loaded_state, path_config.getLoadCheckpointPath() + "torch_internal_state.pt");
            auto gen = at::detail::getDefaultCPUGenerator();
            gen.set_state(loaded_state);
            std::cout << "[INFO] Torch internal generator state loaded successfully." << std::endl;

            std::cout << "[INFO] Loading buffer from " << path_config.getLoadCheckpointPath() << "buffer.bin" << std::endl;
            std::ifstream is(path_config.getLoadCheckpointPath() + "buffer.bin", std::ios::binary);
            std::cout << "good=" << is.good()
            // ######
                    << " eof=" << is.eof()
                    << " fail=" << is.fail() << std::endl;

            is.seekg(0, std::ios::end);
            std::cout << "size=" << is.tellg() << std::endl;
            is.seekg(0);
            // ########
            auto skip_buffer = train_config.getSkipLoadingBuffer();
            if(!skip_buffer){
                cereal::BinaryInputArchive archive(is);
                archive(this->buffer);
                this->buffer->set_socket(this->socket);
                this->buffer->rebuild_from_config(this->global_config);
                this->buffer->set_batch_requester(this->batch_requester);
                std::cout << "[INFO] Buffer loaded successfully with " << this->buffer->get_num_data_in_buffer() << " mappings." << std::endl;
            }

            if(this->use_curriculum_learning) {
                std::cout << "[INFO] Loading map curriculum learning from " << path_config.getLoadCheckpointPath() << "curriculum.bin" << std::endl;
                std::ifstream mcl_is(path_config.getLoadCheckpointPath() + "curriculum.bin", std::ios::binary);
                cereal::BinaryInputArchive mcl_archive(mcl_is);
                mcl_archive(this->map_curriculum);

                std::ifstream cp(path_config.getLoadCheckpointPath() + "curriculum_progress.bin", std::ios::binary);
                cereal::BinaryInputArchive cp_archive(cp);
                cp_archive(this->curriculum_progress);
                std::cout << "[INFO] Map curriculum learning loaded successfully." << std::endl;
            }
            auto enum_training = launcher_config.getTraining();
            if(enum_training == EnumTraining::HYBRID){
                std::cout << "[INFO] Loading supervised buffer from " << path_config.getLoadCheckpointPath() << "supervised_hybrid_buffer.bin" << std::endl;
                std::ifstream sup_is(path_config.getLoadCheckpointPath() + "supervised_buffer.bin", std::ios::binary);
                cereal::BinaryInputArchive sup_archive(sup_is);
                sup_archive(this->supervised_buffer);
                this->supervised_buffer->rebuild_from_config(this->global_config);
                std::cout << "[INFO] Supervised buffer loaded successfully." << std::endl;

                //loading traiing data
                std::cout << "[INFO] Loading training data from " << path_config.getLoadCheckpointPath() << "training_data.bin" << std::endl;
                std::ifstream td_is(path_config.getLoadCheckpointPath() + "training_data.bin", std::ios::binary);
                cereal::BinaryInputArchive td_archive(td_is);
                td_archive(this->training_data);

                //loading reanalyze
                std::cout << "[INFO] Loading reanalyze from " << path_config.getLoadCheckpointPath() << "reanalyze.bin" << std::endl;
                std::ifstream r_is(path_config.getLoadCheckpointPath() + "reanalyze.bin", std::ios::binary);
                cereal::BinaryInputArchive r_archive(r_is);
                r_archive(this->reanalyze);
                std::cout << "[INFO] Training data loaded successfully." << std::endl;
            }

            json js;
            std::ifstream file(path_config.getLoadCheckpointPath() + "train_info.json");
            if (file.is_open()) {
                file >> js;
                std::cout << "[INFO] Loading data from train_info.json \n";
                train_config.setInitialIter(static_cast<int>(js["final_iter"]) + 1);
                path_config.setLoadCheckpointPath(path_config.getLoadCheckpointPath());
                auto& ppo_config = this->global_config.getPPOConfig();
                ppo_config.setInitialEntropyCoef(js["entropy_coef"]);
                this->ema_rout_nodes = js["ema_rout_nodes"];
                this->ema_valid_map_rate = js["ema_valid_map_rate"];
            } else {
                std::cerr << "[Error] Error opening file: " << path_config.getLoadCheckpointPath() + "train_info.json" << "\n";
                throw;
            }
            file.close();
        }
        // else if(load_model_path != ""){
        //     std::cout << "[INFO] Loading model from " << load_model_path << std::endl;
        //     load_model_server(this->socket, this->server_k, load_model_path);
        // }

        this->reanalyze = Reanalyze<StateT, ModelConfig>(this->buffer, this->config_self_mapping);
        this->mcts_reanalyze = Reanalyze<StateT, ModelConfig>(this->mcts_buffer, this->mcts_self_mapping);
        auto mcts_config = mcts_global_config.getMCTSConfig();
        std::cout << "Return optimal mapping enabled: " << mcts_config.getReturnOptimalMapping() << std::endl;

    }







    template<typename StateT, typename ModelConfig>
    void Launcher<StateT, ModelConfig>::add_train_states_with_curriculum_learning(
        std::vector<std::shared_ptr<StateT>>& cur_initial_states,
        const std::vector<std::shared_ptr<StateT>>& all_initial_states) {
        #ifdef DEBUG

            std::cout << "[DEBUG] Adding train states with curriculum learning..." << std::endl;
    #endif

        int cur_idx = cur_initial_states.size();
        if (cur_idx >= static_cast<int>(all_initial_states.size())) {
    #ifdef DEBUG
            std::cout << "[DEBUG] No more states to add. Exiting function." << std::endl;
    #endif
            return;
        }

        auto cur_state = all_initial_states[cur_idx];
        int cur_dfg_size = cur_state->get_num_dfg_nodes();

    #ifdef DEBUG
        std::cout << "[DEBUG] Starting with index: " << cur_idx << ", DFG size: " << cur_dfg_size << std::endl;
    #endif

        std::cout << "[INFO] Adding train states with curriculum learning with index: " << cur_idx << ", DFG size: " << cur_dfg_size << std::endl;


        while (cur_idx < static_cast<int>(all_initial_states.size())) {
            cur_state = all_initial_states[cur_idx];
            if (cur_state->get_num_dfg_nodes() != cur_dfg_size) {
                break;
            }

            cur_initial_states.push_back(cur_state);
            cur_idx++;

    #ifdef DEBUG
            std::cout << "[DEBUG] Added state at index " << cur_idx - 1 
                    << ", DFG size: " << cur_dfg_size << std::endl;
    #endif
        }

    #ifdef DEBUG
        std::cout << "[DEBUG] Finished adding states. Total states now: " 
                << cur_initial_states.size() << std::endl;
    #endif
    }

    template<typename StateT, typename ModelConfig>
    std::vector<std::shared_ptr<StateT>> Launcher<StateT, ModelConfig>::get_train_states(
        const std::vector<std::shared_ptr<StateT>>& all_initial_states) {
        
            std::vector<std::shared_ptr<StateT>> cur_initial_states;

        #ifdef DEBUG
            std::cout << "[DEBUG] Fetching train states..." << std::endl;
        #endif
        
        auto use_efficient_sampler = this->global_config.getTrainConfig().getUseEfficientSampling();
        if (this->use_curriculum_learning) {
                #ifdef DEBUG
                        std::cout << "[DEBUG] Using curriculum learning..." << std::endl;
                #endif  
                this->map_curriculum.build_curriculum_states(all_initial_states, this->global_config);
                auto path_config = this->global_config.getPathConfig();
                bool upd_idx = path_config.getLoadCheckpointPath() != "";
                this->map_curriculum.add_next_states(cur_initial_states, upd_idx);
                bool is_dyn_progress = this->global_config.getTrainConfig().getEnumCurriculumProgress() == EnumCurriculumProgress::DYN_ADAPTIVE;
                if (is_dyn_progress){
                    int total_curriculum_steps = this->map_curriculum.get_total_curriculum_steps();
                    this->curriculum_progress.set_total_curriculum_steps(total_curriculum_steps);
                }
                if(use_efficient_sampler) this->env_sampler = EnvSampler<StateT,ModelConfig>(cur_initial_states, this->global_config);
                // add_train_states_with_curriculum_learning(cur_initial_states, all_initial_states);
            } else {
        #ifdef DEBUG
                std::cout << "[DEBUG] Curriculum learning disabled. Using all initial states." << std::endl;
        #endif
                cur_initial_states = std::ref(all_initial_states); 
                if(use_efficient_sampler) this->env_sampler = EnvSampler<StateT,ModelConfig>(all_initial_states, this->global_config);
            }
            if(use_efficient_sampler) this->env_sampler.init_priorities(all_initial_states.size());
          
        #ifdef DEBUG
            std::cout << "[DEBUG] Returning " << cur_initial_states.size() << " train states." << std::endl;
        #endif

        return cur_initial_states;
    }


    template<typename StateT, typename ModelConfig>
    std::tuple<std::unordered_map<std::string, double>, 
            std::unordered_map<std::string, double>, 
            std::unordered_map<std::string, std::vector<double>>, 
            std::unordered_map<std::string, std::vector<double>>, 

            std::optional<MappingHistory<StateT>>>  
    Launcher<StateT, ModelConfig>::eval_step(
        const int& i, const int& steps, 
        const std::vector<std::shared_ptr<StateT>>& test_initial_states, 
        const std::vector<std::shared_ptr<StateT>>& train_initial_states,
    bool force) {

        auto train_config = global_config.getTrainConfig();
        auto step_to_start_eval = train_config.getStepToStartEval();
        auto eval_first_step  = train_config.getEvalFirstStep();
        
        std::unordered_map<std::string, double> test_metrics_dict, train_metrics_dict;
        std::unordered_map<std::string, std::vector<double>> test_hist_dict, train_hist_dict;
    
        std::optional<MappingHistory<StateT>> mapping = std::nullopt;
        std::string test_str = "Test/";
        std::string train_str = "Train/";
        
        #ifdef DEBUG
            std::cout << "[DEBUG] Evaluation step " << i << " out of " << steps << std::endl;
        #endif
    
            if ( (i >= step_to_start_eval) && ((i + 1) % train_config.getEvalIntervalTrain() == 0) ) {
                auto eval_start = std::chrono::high_resolution_clock::now();
                
                std::cout << "[INFO] Evaluating model with train data..." << std::endl;
                #ifdef DEBUG
                
                std::cout << "[DEBUG] Performing evaluation (train) at step " << i << std::endl;
                #endif
                
                auto [cur_train_metrics_dict, cur_train_hist_dict,  train_mapping] = this->eval(train_initial_states, train_str);
                train_metrics_dict = std::move(cur_train_metrics_dict);
                train_hist_dict = std::move(cur_train_hist_dict);
                
                if (train_mapping.has_value()) {
                    mapping = std::move(train_mapping);

                }
                auto eval_end = std::chrono::high_resolution_clock::now();
                std::cout << "[INFO] Eval time: " << std::chrono::duration<double>(eval_end - eval_start).count() << " seconds - step: " << i << std::endl;
            }

            if ( force || (i == 0 && eval_first_step ) || (i == steps) || (((i >= step_to_start_eval) && (i !=0  && i  % train_config.getEvalIntervalTest() == 0)))) {
                auto eval_start = std::chrono::high_resolution_clock::now();

                std::cout << "[INFO] Evaluating model with test data..." << std::endl;

                #ifdef DEBUG
                        std::cout << "[DEBUG] Performing evaluation (test) at step " << i << std::endl;
                #endif
                        auto [cur_test_metrics_dict, cur_test_hist_dict, test_mapping] = this->eval(test_initial_states, test_str);
                        test_metrics_dict = std::move(cur_test_metrics_dict);
                        test_hist_dict = std::move(cur_test_hist_dict);
            
                        if (test_mapping.has_value()) {
                            mapping = std::move(test_mapping);
                        }
                auto eval_end = std::chrono::high_resolution_clock::now();
                std::cout << "[INFO] Eval time: " << std::chrono::duration<double>(eval_end - eval_start).count() << " - step: " << i << std::endl;
            
                    }

        #ifdef DEBUG
            std::cout << "[DEBUG] Evaluation completed at step " << i << std::endl;
        #endif

        return { std::move(train_metrics_dict), std::move(test_metrics_dict), std::move(train_hist_dict), std::move(test_hist_dict),
                std::move(mapping) };
    }

    template<typename StateT, typename ModelConfig>
    std::optional<MappingHistory<StateT>> Launcher<StateT, ModelConfig>::base_rl_train( 
        const std::vector<std::shared_ptr<StateT>>& train_initial_states,
        const std::vector<std::shared_ptr<StateT>>& test_initial_states
    ){
        static volatile sig_atomic_t ctrl_c_pressed = 0;

        auto handler = [](int sig){
            if (sig == SIGINT)
                ctrl_c_pressed = 1;
        };
        
        std::signal(SIGINT, handler);

        auto train_config = this->global_config.getTrainConfig(); 
        auto path_config = this->global_config.getPathConfig();
        auto buffer_config = this->global_config.getBufferConfig();

        auto cur_train_states = this->get_train_states(train_initial_states);
        auto batch_size = train_config.getBatchSizeStates();
        auto shuffle = train_config.getShuffleTrainData();
        auto PER = buffer_config.getPER();
        auto steps = train_config.getSteps();
        int initial_iter = train_config.getInitialIter();
        auto update_model_interval = train_config.getUpdateModelInterval();
        int num_envs = train_config.getNumEnvs();
        int buffer_size_to_start_train = train_config.getBufferSizeToStartTraining();
        torch::Generator generator = torch::make_generator<torch::CPUGeneratorImpl>(train_config.getSeed());

        if (path_config.getLoadCheckpointPath() != "") {
            std::cout << "[INFO] Loading env generator from " << path_config.getLoadCheckpointPath() << std::endl;
            try {
                torch::Tensor gen_state;
                torch::load(gen_state, path_config.getLoadCheckpointPath() + "env_generator_state.pt");
                generator.set_state(gen_state);
            } catch (const c10::Error& e) {
                std::cerr << "[WARN] Failed to load env generator state: " << e.what() << std::endl;
            }
        }

        auto num_samples_per_mapping = train_config.getNumSamplesPerMapping();
        auto use_efficient_sampler = train_config.getUseEfficientSampling();

        std::cout << "[INFO] Steps: " << steps << "\n";
        std::cout << "[INFO] PER: " << PER << "\n";
        std::cout << "[INFO] Samples per mapping: " << num_samples_per_mapping << "\n";
        std::cout << "[INFO] Mapping mode: " << (this->mapping_mode ? "ON" : "OFF") << "\n";
        std::cout << "[INFO] Efficient sampler: " << (use_efficient_sampler ? "ON" : "OFF") << "\n";

        std::vector<double> sampled_idx_to_count(train_initial_states.size(), 0.0);
        std::unordered_map<std::pair<int,int>, std::vector<double>> dims_to_action_count;
        int max_graph_size = 0;
        for (auto& state : train_initial_states)
            max_graph_size = std::max(max_graph_size, state->get_num_dfg_nodes());
        std::cout << "[INFO] Max graph size: " << max_graph_size << "\n";

      
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
            
            if (this->use_curriculum_learning) {
                cur_train_states = this->map_curriculum.fill_states_until_current_index(cur_train_states);
            }

            for (auto& s : cur_train_states){
                this->env_sampler.add_old_sample(s);
            }
        }



        int last_saved_iter = initial_iter;
        
        if(path_config.getLoadCheckpointPath() != ""){
            load_timers();
        }else{
            initialize_timers();
        }

        while(!ctrl_c_pressed && !time_exceeded()){
            for (int i = initial_iter; i <= steps; i++) {
                bool finish_loop = ctrl_c_pressed ||  time_exceeded(); 
                
                auto initial_time = std::chrono::high_resolution_clock::now();
                std::unordered_map<std::string, double> train_metrics_dict;
    
                int N = cur_train_states.size();
                int sample_size = num_envs != -1 ? std::min(num_envs, N) : N;
    
                torch::Tensor sampled_indices = torch::randperm(N, generator).slice(0, 0, sample_size);
    
                std::vector<std::shared_ptr<StateT>> real_train_states;
                if (!use_efficient_sampler) {
                    real_train_states.reserve(sample_size);
                    for (int j = 0; j < sampled_indices.size(0); j++) {
                        int idx = sampled_indices[j].item<int>();
                        sampled_idx_to_count[idx] += 1;
                        real_train_states.push_back(cur_train_states[idx]);
                    }
                } else {
                    real_train_states = this->env_sampler.sample_states(generator);
                }
    
                std::cout << "[INFO] RL Training Step " << i << "/" << steps << std::endl;
                bool skip_train = (i == steps) ;
    
                auto [eval_train_metrics_dict, eval_test_metrics_dict, train_hist_dict, test_hist_dict, eval_mapping] =
                    eval_step(i, steps, test_initial_states, cur_train_states, finish_loop);
    
                if (eval_mapping.has_value()) {
                    std::cout << "[INFO] Evaluation mapping found. Returning result." << std::endl;
                    return eval_mapping;
                }
    
                double mean_raw_adv = 0.0, std_raw_adv = 0.0, mean_values = 0.0, std_values = 0.0;
                std::unordered_map<std::string, double> train_dict;
    
                if (!finish_loop && !skip_train) {
                    double train_ratio = static_cast<double>(i) / static_cast<double>(steps);
                    std::cout << "[INFO] Generating training data from " << real_train_states.size() << " initial states." << std::endl;
    
                    auto [mapping, gen_data_dict] = this->gen_train_data(this->config_self_mapping,real_train_states, train_ratio, dims_to_action_count, train_config.getSeed() + 100 * i);
                    if (mapping.has_value())
                        return mapping;
                        
                    if (this->buffer->get_num_data_in_buffer() >= buffer_size_to_start_train){
                        auto cur_mean_rout_nodes = gen_data_dict["MappingGen/mean_rout_nodes_per_mapping"];
                        auto cur_mean_valid_map_rate = gen_data_dict["MappingGen/valid_mapping_rate"];
                        this->ema_rout_nodes = this->calc_ema(this->ema_rout_nodes, cur_mean_rout_nodes, this->ema_alpha);
                        this->ema_valid_map_rate = this->calc_ema(this->ema_valid_map_rate, cur_mean_valid_map_rate, this->ema_alpha);
                        gen_data_dict["Control/ema_rout_nodes"] = this->ema_rout_nodes;
                        gen_data_dict["Control/ema_valid_map_rate"] = this->ema_valid_map_rate;
        
                        for (auto& kv : gen_data_dict)
                            train_metrics_dict[kv.first] = kv.second;
                            
                        if ((i + 1) % update_model_interval == 0) {
                            auto dataset = buffer->get_batch();
        
                            mean_raw_adv = dataset.get_raw_mean_advantages();
                            std_raw_adv = dataset.get_raw_std_advantages();
                            mean_values = dataset.get_mean_values();
                            std_values = dataset.get_std_values();
        
                            auto grouped_idxs = create_batch_size_idxs(dataset.size(), batch_size, shuffle, generator);
                            std::vector<std::vector<c10::IValue>> all_train_data(grouped_idxs.size());
                            std::vector<torch::Tensor> all_train_indices(grouped_idxs.size());
                            at::parallel_for(0, grouped_idxs.size(), 0, [&](int64_t begin, int64_t end) {
                                for (int64_t ip = begin; ip < end; ++ip) {
                                    all_train_data[ip] =
                                        dataset.get_train_data(this->collate, grouped_idxs[ip]);
                                    all_train_indices[ip] =
                                        torch::tensor(grouped_idxs[ip], torch::kInt32);
                                }
                            });
                            
                            std::cout << "[INFO] Collate - Number of batches: " << grouped_idxs.size() << " - Mini batch size: " << batch_size << std::endl;
        
                            bool first_train = true;
        
                            for (size_t k = 0; k < all_train_data.size(); ++k) {
                                auto cur_train_dict = server_train(socket, all_train_data[k], i, std::nullopt, server_k);
                                if (first_train) {
                                    train_dict = cur_train_dict;
                                    first_train = false;
                                } else {
                                    for (auto& p : cur_train_dict)
                                        train_dict[p.first] += p.second;
                                }
                                if (PER)
                                    this->update_priorities(dataset, all_train_indices[k]);
                            }
        
                            for (auto& p : train_dict)
                                train_dict[p.first] /= static_cast<double>(all_train_data.size());
                        }
                    }
                }
    
                train_metrics_dict["Control/num_data_in_buffer"] = this->buffer->get_num_data_in_buffer();
                train_metrics_dict["Control/num_initial_train_states"] = cur_train_states.size();
                train_metrics_dict["Control/trained_envs_per_step"] = real_train_states.size();
                train_metrics_dict["Control/mean_raw_adv"] = mean_raw_adv;
                train_metrics_dict["Control/std_raw_adv"] = std_raw_adv;
                train_metrics_dict["Control/mean_pred_values"] = mean_values;
                train_metrics_dict["Control/std_pred_values"] = std_values;
                double mean_rout_nodes_threshold = this->curriculum_progress.get_mean_rout_nodes_threshold();
                double valid_map_rate_threshold = this->curriculum_progress.get_valid_map_rate_threshold();
                train_metrics_dict["Control/mean_rout_nodes_threshold"] = mean_rout_nodes_threshold;
                train_metrics_dict["Control/valid_map_rate_threshold"] = valid_map_rate_threshold;
    
                if (use_efficient_sampler) {
                    train_metrics_dict["Control/num_good_states"] = this->env_sampler.get_num_good_states();
                    train_metrics_dict["Control/num_bad_states"] = this->env_sampler.get_num_bad_states();
                }
    
                for (auto& p : train_metrics_dict)
                    eval_test_metrics_dict[p.first] = p.second;
                for (auto& p : eval_train_metrics_dict)
                    eval_test_metrics_dict[p.first] = p.second;
    
                // if (i == steps)
                    // train_hist_dict["Control/sampled_idx_count"] = std::ref(sampled_idx_to_count);
                if (test_hist_dict.size() != 0 && use_efficient_sampler)
                    train_hist_dict["Control/sampled_idx_count"] = this->env_sampler.get_visit_counts();
    
                write_tensorboard(socket, server_k, train_dict, eval_test_metrics_dict, train_hist_dict, test_hist_dict, i);
    
                if (this->use_curriculum_learning) {
                    if (this->curriculum_progress.should_increase_progress(i, this->ema_rout_nodes, this->ema_valid_map_rate)) {
                        auto new_states = this->map_curriculum.add_next_states(cur_train_states);
                        if (use_efficient_sampler) {
                            for (auto& s : new_states)
                                this->env_sampler.add_sample(s);
                        }
                    }
                }
    
                if (use_efficient_sampler)
                    this->env_sampler.update_states(real_train_states);
    
                if (buffer_config.getClearBufferAfterTraining())
                    this->buffer->clear_buffer();
    
                update_last_time();
                this->launcher_save_model(i, i == steps, generator, 0);
                auto final_time = std::chrono::high_resolution_clock::now();
                std::cout << "[INFO] Step " << i << " completed - Time (s): " <<  std::chrono::duration_cast<std::chrono::seconds>(final_time - initial_time) << std::endl << std::endl;
                
                if(finish_loop) {
                    break;
                }
                else{  
                    last_saved_iter = i;
                };
            }
            std::cout << "[INFO] Base RL training completed." << std::endl;
        }
        if(ctrl_c_pressed || time_exceeded()){
            std::cout << "[INFO] CTRL+C pressed or Time exceeded. Saving model and exiting." << std::endl;
            this->launcher_save_model(last_saved_iter, last_saved_iter == steps, generator, 0, true);
        }
        return std::nullopt;

    }


    template<typename StateT, typename ModelConfig>
    void Launcher<StateT, ModelConfig>::update_priorities(MappingDataset<StateT>& dataset, torch::Tensor tensor_indices){
        auto mapping_indices = dataset.get_mapping_idxs_by_idxs(tensor_indices);
        auto state_indices = dataset.get_state_idxs_by_idxs(tensor_indices);
        torch::Tensor new_priorities;
        auto skip_priorities_req = this->global_config.getLauncherConfig().getReplayBuffer() == EnumReplayBuffer::MAPZERO_BUFFER;
        if(!skip_priorities_req) new_priorities = get_last_priorities(this->socket, this->server_k)[0];
        buffer->update_priorities(mapping_indices, state_indices, new_priorities);
    }


    template<typename StateT, typename ModelConfig>
    std::tuple<std::optional<MappingHistory<StateT>>, std::unordered_map<std::string, double>> Launcher<StateT, ModelConfig>::gen_train_data(
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
    auto enum_training = this->global_config.getLauncherConfig().getTraining();
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
                    
                    if (aug_mapping) {
                        std::vector<MappingHistory<StateT>> aug_list;
                        auto aug_is_incremental = train_config.augment_is_incremental();
    
                        if ((enum_training == EnumTraining::PPO) || (enum_training == EnumTraining::POLICY_GRADIENT)) {
                            aug_list = DataAugmenter::augment_mapping_ppo(mapping_history, this->global_config, this->environment, aug_is_incremental);
                        } else if (enum_training == EnumTraining::MCTS) {
                            aug_list = DataAugmenter::augment_mapping_mcts(mapping_history, this->global_config, this->environment, aug_is_incremental);
                        }
    
                        for (auto& map_his : aug_list) {
                            local_buf.emplace_back(std::move(map_his));
                        }
                    } else {
                        local_buf.emplace_back(std::move(mapping_history));
                    }
                    
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

        for (auto& local_buffer : thread_buffers) {
            for (auto& mapping : local_buffer) {
                // auto actions = mapping.get_actions();
                // auto states = mapping.get_states();
                // auto dims = states[0]->get_cgra_dimensions_();
                // for(int i = 1; i < static_cast<int>(actions.size());i++ ){
                //     dims_to_action_count[dims].at(actions[i]) += 1.0;
                // }
                if(use_efficient_sampler){
                    auto id = mapping.get_first_state()->get_id();
                    auto mapping_is_valid = mapping.get_mapping_is_valid();
                    this->env_sampler.update_ema_rout_nodes(id, mapping.get_total_routing_nodes());
                    this->env_sampler.update_ema_valid_map_rate(id, mapping_is_valid);
                    this->env_sampler.update_last_return(id, mapping.calc_sum_reward());

                    // this->env_sampler.update_last_return(id, this->env_sampler.get_abs_gae_score(mapping));
                    this->env_sampler.compute_score(mapping);
                    
                }

                if(enum_training == EnumTraining::HYBRID && mapping.get_mapping_is_valid() && mapping.get_total_routing_nodes() == 0){
                    // mapping.set_skip_initial_state(true);
                    mapping.print();
                    // int a;
                    // std::cin >> a;
                    this->supervised_buffer->add_mapping(std::move(mapping));

                    // if(!this->training_data.modified) this->training_data.modified = true;
                    this->training_data.total_mappings += 1;
                }else{
                    // mapping.print();
                    // int a; std::cin >> a;
                    buffer->add_mapping(std::move(mapping));
                }
            }
        }
   

        for (auto& local_best : thread_best_mapping) {
            if (local_best && (!result || local_best->calc_sum_reward() > result->calc_sum_reward())) {
                result = std::move(local_best);
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

        return std::make_tuple(std::move(result), std::move(data_gen_metrics_dict));
    }

    template<typename StateT,typename ModelConfig>
    void Launcher<StateT,ModelConfig>::launcher_save_model(int cur_iter, bool train_finished, const torch::Generator& generator, 
    const double w_entropy_loss, bool force, std::unordered_set<int>* mappings_to_save, 
    std::unordered_map<int, MappingHistory<StateT>>* id_to_map ){
        auto launcher_config = this->global_config.getLauncherConfig();
        auto train_config = this->global_config.getTrainConfig();
        auto path_config = this->global_config.getPathConfig();
        auto enum_training = launcher_config.getTraining();
        if( force  ||( launcher_config.getSaveModel() && (train_finished || (cur_iter % train_config.getSaveInterval() == 0))) ){
            std::cout << "[INFO] Model saved at iteration " << cur_iter << std::endl;
            update_last_time();
            save_timers();
            std::unordered_map<std::string,double> extra_data;
            extra_data["ema_rout_nodes"] = this->ema_rout_nodes;
            extra_data["ema_valid_map_rate"] = this->ema_valid_map_rate;
            save_model(socket, server_k, cur_iter,train_finished, w_entropy_loss, extra_data);
            torch::serialize::OutputArchive archive;
            auto gen_state = generator.get_state();
            
            torch::save(gen_state, path_config.getSaveCheckpointPath() + "env_generator_state.pt");
            auto torch_internal_state = at::detail::getDefaultCPUGenerator().get_state();

            torch::save(torch_internal_state, path_config.getSaveCheckpointPath() + "torch_internal_state.pt");
            std::ofstream os(path_config.getSaveCheckpointPath() + "buffer.bin", std::ios::binary);
            cereal::BinaryOutputArchive archive_buffer(os);
            archive_buffer(buffer);

            if(this->use_curriculum_learning){
                std::ofstream os_curriculum(path_config.getSaveCheckpointPath() + "curriculum.bin", std::ios::binary);
                cereal::BinaryOutputArchive archive_curriculum(os_curriculum);
                archive_curriculum(this->map_curriculum);

                std::ofstream os_curriculum_prog(path_config.getSaveCheckpointPath() + "curriculum_progress.bin", std::ios::binary);
                cereal::BinaryOutputArchive archive_curriculum_prog(os_curriculum_prog);
                archive_curriculum_prog(this->curriculum_progress);
            }



            bool use_efficient_sampler = this->global_config.getTrainConfig().getUseEfficientSampling();
            if(use_efficient_sampler){
                std::ofstream os_env_sampler(path_config.getSaveCheckpointPath() + "env_sampler.bin", std::ios::binary);
                cereal::BinaryOutputArchive archive_env_sampler(os_env_sampler);
                archive_env_sampler(this->env_sampler);
            }

            if(enum_training == EnumTraining::HYBRID){
                std::ofstream os_hybrid_buffer(path_config.getSaveCheckpointPath() + "supervised_hybrid_buffer.bin", std::ios::binary);
                cereal::BinaryOutputArchive archive_hybrid_buffer(os_hybrid_buffer);
                archive_hybrid_buffer(this->supervised_buffer);

                std::ofstream os_training_data(path_config.getSaveCheckpointPath() + "training_data.bin", std::ios::binary);
                cereal::BinaryOutputArchive archive_training_data(os_training_data);
                archive_training_data(this->curriculum_progress);
            }

            std::ofstream os_reanalyze(path_config.getSaveCheckpointPath() + "reanalyze.bin", std::ios::binary);
            cereal::BinaryOutputArchive arc_reanalyze(os_reanalyze);
            arc_reanalyze(this->reanalyze);
            
            if (mappings_to_save == nullptr || id_to_map == nullptr) return;

            for(auto id: *mappings_to_save){
                this->save_mapping(
                    id_to_map->at(id),
                    path_config.getSaveCheckpointPath(),
                    std::to_string(id) ,
                    "found_mappings"
                );
            }

            mappings_to_save->clear();
        }
    }
    // int Launcher<StateT,ModelConfig>::load_iter(){   
    //     auto path_config = this->global_config.getPathConfig();

    //     if(path_config.getLoadPathModel() != "none"){
    //         json js;
    //         std::ifstream file(path_config.getLoadPathModel());
    //     }
    // }

   
    template<typename StateT, typename ModelConfig>
    std::optional<MappingHistory<StateT>> Launcher<StateT, ModelConfig>::ppo_train(
        const std::vector<std::shared_ptr<StateT>>& train_initial_states,
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
        std::string func_name = "ppo";

        auto cur_train_states = this->get_train_states(train_initial_states);
        auto iterations = train_config.getIterations();
        auto batch_size = train_config.getBatchSizeStates();
        auto shuffle = train_config.getShuffleTrainData();
        auto epochs = train_config.getEpochs();
        auto PER = buffer_config.getPER();
        int initial_iter = train_config.getInitialIter();
        auto update_model_interval = train_config.getUpdateModelInterval();
        int num_envs = train_config.getNumEnvs();   
        torch::Generator generator = torch::make_generator<torch::CPUGeneratorImpl>(train_config.getSeed());
        
        if(path_config.getLoadCheckpointPath() != ""){
            std::cout << "[INFO] Loading env generator from " << path_config.getLoadCheckpointPath() << std::endl;
            torch::serialize::InputArchive archive;
            torch::Tensor gen_state;
            torch::load(gen_state,path_config.getLoadCheckpointPath() + "env_generator_state.pt");
            generator.set_state(gen_state);
        }

        auto num_samples_per_mapping = train_config.getNumSamplesPerMapping();
        auto use_efficient_sampler = train_config.getUseEfficientSampling();
        std::cout << "[INFO] Epochs: " << train_config.getEpochs() << "\n";
        std::cout << "[INFO] PER: " << PER << "\n";
        std::cout << "[INFO] Samples per mapping: " << num_samples_per_mapping << "\n";
        std::cout << "[INFO] Mapping mode: " << (this->mapping_mode ? "ON" : "OFF") << "\n";
        std::cout << "[INFO] Using efficient sampler: " << (use_efficient_sampler ? "ON" : "OFF") << "\n";
        std::cout << "[INFO] Number of initial train states: " << cur_train_states.size() << "\n";
        std::vector<double> sampled_idx_to_count = std::vector<double>(train_initial_states.size(),0.0);
        std::unordered_map<std::pair<int,int>, std::vector<double>> dims_to_action_count;
        int max_graph_size = 0;
        int step = 0;
        
        for(auto& state: train_initial_states){
            max_graph_size = std::max(max_graph_size, state->get_num_dfg_nodes());
        }
        std::cout << "[INFO] Max graph size: " << max_graph_size << "\n";
        // for(const auto& dims: this->global_config.getCGRAConfig().getDimensionsList()){
        //     int max_II = std::ceil(static_cast<double>(max_graph_size) / static_cast<double>(dims.first*dims.second));
        //     dims_to_action_count[dims] = std::vector<double>(dims.first*dims.second*(this->global_config.getCGRAConfig().getUnrollII() + max_II), 0.0);
            
        // }

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
            
            if (this->use_curriculum_learning) {
                cur_train_states = this->map_curriculum.fill_states_until_current_index(cur_train_states);
            }

            for (auto& s : cur_train_states){
                this->env_sampler.add_old_sample(s);
            }
        }

        int seed = train_config.getSeed();
        int last_saved_iter = initial_iter;
        double last_entropy_loss = 0.0;
        if(path_config.getLoadCheckpointPath() != ""){
            load_timers();
        }else{
            initialize_timers();
        }
        while(!ctrl_c_pressed && !time_exceeded()){
            for (int i = initial_iter; i <= iterations; i++) {
                bool finish_loop = ctrl_c_pressed ||  time_exceeded(); 
                int N = cur_train_states.size();
                int sample_size = num_envs != -1 ? std::min(num_envs, N) : N;
                torch::Tensor sampled_indices = torch::randperm(N, generator).slice(0,0,sample_size);
                #ifdef DEBUG
                    std::cout << "[DEBUG] Sampled indices: " << sampled_indices << std::endl;
                #endif
         
                std::vector<std::shared_ptr<StateT>> real_train_states;
                real_train_states.reserve(sample_size);
                if(!use_efficient_sampler){
                    for (int j = 0; j < sampled_indices.size(0); j++) {
                        int idx = sampled_indices[j].item<int>();
                        sampled_idx_to_count[idx] += 1;
                        real_train_states.push_back(cur_train_states[idx]);
                    }
                }else{
                    real_train_states = this->env_sampler.sample_states(generator);
                }
    
    
                auto iter_start = std::chrono::steady_clock::now();
                std::cout << "[INFO] Iteration " << i << " started." << std::endl;
    
                auto w_entropy_loss = compute_entropy_bonus(i, iterations, ppo_config);
                auto skip_train = i == iterations;
    
                std::unordered_map<std::string, double> train_metrics_dict;
    
                auto [eval_train_metrics_dict, eval_test_metrics_dict, train_hist_dict, test_hist_dict, eval_mapping] = eval_step(i, iterations, test_initial_states, cur_train_states, finish_loop);
                
                if (eval_mapping.has_value()) return eval_mapping;
                
                double mean_raw_adv, std_raw_adv, mean_values, std_values;
                
                std::unordered_map<std::string, double> train_dict;
    
                if (!finish_loop && !skip_train) {
                    double train_ratio = static_cast<double>(i) / static_cast<double>(iterations);
                    auto data_start = std::chrono::steady_clock::now();
                    std::cout << "[INFO] Generating training data from " << real_train_states.size() << " initial states." << std::endl;
            
                    auto [mapping, gen_data_dict] = this->gen_train_data(this->config_self_mapping,real_train_states, train_ratio,dims_to_action_count, seed);
                    
                    auto cur_mean_rout_nodes = gen_data_dict["MappingGen/mean_rout_nodes_per_mapping"];
                    auto cur_mean_valid_map_rate = gen_data_dict["MappingGen/valid_mapping_rate"];
                    this->ema_rout_nodes = this->calc_ema(this->ema_rout_nodes, cur_mean_rout_nodes, this->ema_alpha);
                    this->ema_valid_map_rate = this->calc_ema(this->ema_valid_map_rate, cur_mean_valid_map_rate, this->ema_alpha);
                    gen_data_dict["Control/ema_rout_nodes"] = this->ema_rout_nodes;
                    gen_data_dict["Control/ema_valid_map_rate"] = this->ema_valid_map_rate;
    
                    auto data_end = std::chrono::steady_clock::now();
    
                    for(auto&pair: gen_data_dict){
                        train_metrics_dict[pair.first] = pair.second;
                    }
    
    
                    std::chrono::duration<double> data_duration = data_end - data_start;
                    std::cout << "[INFO] Data generation took " << data_duration.count() << " seconds." << std::endl;
                    
                    if (mapping.has_value()) return mapping;
    
                    if( (i+1) % update_model_interval == 0){
                        auto batch_start = std::chrono::steady_clock::now();
                        auto dataset = buffer->get_batch();
                        auto batch_end = std::chrono::steady_clock::now();
                        std::chrono::duration<double> batch_duration = batch_end - batch_start;
                        std::cout << "[INFO] Getting batch - Time: " << batch_duration.count() << std::endl;
    
                        mean_raw_adv = dataset.get_raw_mean_advantages();
                        std_raw_adv = dataset.get_raw_std_advantages();
                        mean_values = dataset.get_mean_values();
                        std_values = dataset.get_std_values();
                        
                        bool first_train = true;
                        std::vector<std::vector<int>> grouped_idxs;
                        auto train_start = std::chrono::steady_clock::now();
                        auto collate_start = std::chrono::steady_clock::now();
                        grouped_idxs = create_batch_size_idxs(dataset.size(), batch_size, shuffle, generator);
                        std::vector<std::vector<c10::IValue>> all_train_data(grouped_idxs.size());
                        at::parallel_for(0, grouped_idxs.size(), 0, [&](int64_t begin, int64_t end) {
                            for (int64_t ip = begin; ip < end; ++ip) {
                                all_train_data[ip] =
                                    dataset.get_train_data(this->collate, grouped_idxs[ip]);
                            }
                        });
                        auto collate_end = std::chrono::steady_clock::now();
                        std::chrono::duration<double> collate_duration = collate_end - collate_start;
                        std::cout << "[INFO] Collate - Number of batches: " << grouped_idxs.size() << " - Mini batch size: " << batch_size << " - Time: " << collate_duration.count() << std::endl;
                        for (int k = 0; k < epochs; k++) {
                            int idx_count = 0;
                            for (auto& cur_train_data : all_train_data) {
                                    // std::vector<c10::IValue> cur_train_data;
                                    // cur_train_data = dataset.get_train_data(this->collate, indices);
                                    
                                    auto cur_train_dict = server_train(socket, cur_train_data, step, w_entropy_loss, server_k);
                                    step++;
                                    if (first_train) {
                                        train_dict = cur_train_dict;
                                        first_train = false;
                                    } else {
                                        for (auto& pair : cur_train_dict) {
                                            train_dict[pair.first] += pair.second;
                                        }
                                    }
    
                                    if ((k == epochs - 1) && PER) {
                                        auto tensor_indices = torch::tensor(grouped_idxs[idx_count], torch::kInt32);
                                        this->update_priorities(dataset, tensor_indices);
                                    }
                                    idx_count +=1;
                                }
                            }
    
                        for (auto& pair : train_dict) {
                            train_dict[pair.first] /= (static_cast<double>(grouped_idxs.size()) * epochs);
                        }
    
                        auto train_end = std::chrono::steady_clock::now();
                        std::chrono::duration<double> train_duration = train_end - train_start;
                        std::cout << "[INFO] Training took " << train_duration.count() << " seconds." << std::endl;
                    }
                    
                }
    
                train_metrics_dict["Control/num_data_in_buffer"] = this->buffer->get_num_data_in_buffer();
                train_metrics_dict["Control/num_initial_train_states"] = cur_train_states.size();
                train_metrics_dict["Control/trained_envs_per_step"] = real_train_states.size();
                train_metrics_dict["Control/mean_raw_adv"] = mean_raw_adv;
                train_metrics_dict["Control/std_raw_adv"] = std_raw_adv;
                train_metrics_dict["Control/mean_pred_values"] = mean_values;
                train_metrics_dict["Control/std_pred_values"] = std_values;
                double mean_rout_nodes_threshold = this->curriculum_progress.get_mean_rout_nodes_threshold();
                double valid_map_rate_threshold = this->curriculum_progress.get_valid_map_rate_threshold();
                train_metrics_dict["Control/mean_rout_nodes_threshold"] = mean_rout_nodes_threshold;
                train_metrics_dict["Control/valid_map_rate_threshold"] = valid_map_rate_threshold;
                if(use_efficient_sampler){
                    train_metrics_dict["Control/num_good_states"] = this->env_sampler.get_num_good_states();
                    train_metrics_dict["Control/num_bad_states"] = this->env_sampler.get_num_bad_states();
    
                }
    
                
                for (auto& pair : train_metrics_dict)
                    eval_test_metrics_dict[pair.first] = pair.second;
                for (auto& pair : eval_train_metrics_dict)
                    eval_test_metrics_dict[pair.first] = pair.second;
                if (i == iterations){
                    train_hist_dict["Control/sampled_idx_count"] = std::ref(sampled_idx_to_count);
                }
    
                if(test_hist_dict.size() != 0){
                    if(use_efficient_sampler){
                        train_hist_dict["Control/sampled_idx_count"] = this->env_sampler.get_visit_counts();
    
                    }
                }
                // for(auto& pair: dims_to_action_count){
                //     std::string key = "Control/dims_to_action_count_" + std::to_string(pair.first.first) + "_" + std::to_string(pair.first.second);
                //     train_hist_dict[key] = std::ref(pair.second);
                // }
                write_tensorboard(socket, server_k, train_dict, eval_test_metrics_dict, train_hist_dict, test_hist_dict, i);
             
                if (this->use_curriculum_learning) {
                    if(this->curriculum_progress.should_increase_progress(i,this->ema_rout_nodes, this->ema_valid_map_rate)){
                        auto add_states = this->map_curriculum.add_next_states(cur_train_states);
                        if(use_efficient_sampler){
                            for(auto& state: add_states){
                                this->env_sampler.add_sample(state);
                            }
                        }
                    }
                }
    
                if(use_efficient_sampler){
                    this->env_sampler.update_states(real_train_states);
                }
    
                if (buffer_config.getClearBufferAfterTraining()){
                    this->buffer->clear_buffer();
                }
                update_last_time();
                this->launcher_save_model(i, i == iterations, generator, w_entropy_loss);
    
                auto iter_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> iter_duration = iter_end - iter_start;
                std::cout << "[INFO] Iteration " << i << " completed in " << iter_duration.count() << " seconds.\n" << std::endl;

                if(finish_loop) {
                    break;
                }
                else{  
                    last_saved_iter = i;
                    last_entropy_loss = w_entropy_loss;

                };
            }
            if(!ctrl_c_pressed && !time_exceeded()){ 

                std::cout << "[INFO] PPO training completed." << std::endl << std::endl;
                return std::nullopt;
            }
        }
        std::cout << "[INFO] CTRL+C pressed or Time exceeded. Saving model and exiting." << std::endl;
        this->launcher_save_model(last_saved_iter, last_saved_iter == iterations, generator, last_entropy_loss, true);
        return std::nullopt;

    }


    template<typename StateT,typename ModelConfig>
    void Launcher<StateT,ModelConfig>::start_reanalyze_loop(bool& stop){
        std::cout << "[Thread] Reanalyze thread started\n" << std::flush;
        while(!stop){
            // std::cout << "[Thread] Calling reanalyze...\n" << std::flush;
            try {
                if(this->time_exceeded()) {
                    stop = true;
                    break;
                }
                this->reanalyze.reanalyze();
            } catch (const std::exception& e) {
                std::cerr << "[Thread] Exception during reanalyze: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[Thread] Unknown exception during reanalyze\n";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
        }
        std::cout << "[Thread] Reanalyze thread stopped\n" << std::flush;
    }

    template<typename StateT,typename ModelConfig>
    std::thread Launcher<StateT,ModelConfig>::get_reanalyze_thread(bool& stop){
    return std::thread(&Launcher<StateT, ModelConfig>::start_reanalyze_loop, this, std::ref(stop));   
    }



    template<typename StateT,typename ModelConfig>
    void Launcher<StateT,ModelConfig>::finish_reanalyze(std::thread& reanalyze_thread,bool& stop){
        stop = true;
        if(reanalyze_thread.joinable()) reanalyze_thread.join();   
    }


    template<typename StateT,typename ModelConfig>
    std::optional<MappingHistory<StateT>> Launcher<StateT,ModelConfig>::train(const std::vector<std::shared_ptr<StateT>>& train_states, const std::vector<std::shared_ptr<StateT>>& test_states, bool save_model_flag){
        std::string config_text;
        std::stringstream ss;
        this->global_config.print_ss(ss);
        ss.str(config_text);
        add_text_to_tensorboard(socket, server_k,config_text);
        auto launcher_config = this->global_config.getLauncherConfig();
        auto train_type = launcher_config.getTraining();
        bool stop = false;
        if(train_type == EnumTraining::MCTS){
            auto self_mapping_type = launcher_config.getSelfMapping();
            assert( (self_mapping_type == EnumSelfMapping::MAPZERO_MCTS) || (self_mapping_type == EnumSelfMapping::SMARTMAP_MCTS));
        }
        std::thread reanalyze_thread;
        if(this->use_reanalyze){
            auto buffer_type = launcher_config.getReplayBuffer();
            assert(buffer_type != EnumReplayBuffer::SMARTMAP_PRE_TRAINING_BUFFER);
            reanalyze_thread = this->get_reanalyze_thread(stop); 
        }
        std::optional<MappingHistory<StateT>> mapping_history;
        throw std::runtime_error("Add load iter if load path model is != none");
        if((train_type == EnumTraining::MCTS) || (train_type == EnumTraining::POLICY_GRADIENT)){
            mapping_history = std::move(this->base_rl_train(train_states, test_states));
        }else if(train_type == EnumTraining::PPO){
            mapping_history = std::move(this->ppo_train(train_states, test_states));
        }
        
        
        this->finish_reanalyze(reanalyze_thread,stop);
        return (mapping_history.has_value() ? mapping_history : std::nullopt);

    }


    template<typename StateT, typename ModelConfig>
    std::tuple<std::unordered_map<std::string, double>, std::unordered_map<std::string, std::vector<double>>, std::optional<MappingHistory<StateT>>> 
    Launcher<StateT, ModelConfig>::eval(
        const std::vector<std::shared_ptr<StateT>>& initial_states, 
        std::string& train_or_test) 
    {
        torch::Tensor test_sum_rewards = torch::tensor(0.0);
        torch::Tensor mean_tree_depth = torch::tensor(0.0);
        double sum_episode_length = 0;
        double num_valid_mappings = 0;
        // double total_nodes = 0;
        // double total_edges = 0;
        // double total_convergent_nodes = 0;
        // double total_convergent_edges = 0;
        double total_mapped_nodes = 0;
        double total_mapped_edges = 0;
        double total_valid_convergent_nodes = 0;
        double total_valid_convergent_edges = 0;
        double total_routing_nodes = 0;
        std::optional<MappingHistory<StateT>> valid_mapping_history = std::nullopt;
        std::unordered_map<std::string, double> dict;
        std::unordered_map<std::string, std::vector<double>> hist_dict;
        std::mutex accum_mutex;

        arena.execute([&] {
            tbb::parallel_for(size_t(0), initial_states.size(), [&](size_t i) {

            // for(int i = 0; i < static_cast<int>(initial_states.size()); i++){
                auto& state = initial_states[i];
        
                auto mapping_history = this->config_self_mapping.map(state, false, 1, this->global_config.getTrainConfig().getSeed(), std::nullopt);
        
                bool is_valid = mapping_history.get_mapping_is_valid();
        
                {
                    std::lock_guard<std::mutex> lock(accum_mutex);
        
                    if (this->mapping_mode && is_valid && !valid_mapping_history.has_value()) {
                        valid_mapping_history = mapping_history;
                        
                    }
        
                    sum_episode_length += mapping_history.get_episode_length();
        
                    auto total_nodes = mapping_history.get_num_dfg_nodes();
                    auto total_edges = mapping_history.get_num_dfg_edges();
                    auto total_convergent_nodes = mapping_history.get_num_dfg_node_convergences();
                    auto total_convergent_edges = mapping_history.get_num_dfg_edge_convergences();
        
                    total_mapped_nodes += static_cast<double>(mapping_history.get_num_mapped_dfg_nodes()) / static_cast<double>(total_nodes);
                    total_mapped_edges += static_cast<double>(mapping_history.get_num_mapped_dfg_edges()) / static_cast<double>(total_edges);
        
                    total_valid_convergent_nodes += total_convergent_nodes == 0? 1 :static_cast<double>(mapping_history.get_num_valid_dfg_node_convergences()) /
                                                     (static_cast<double>(total_convergent_nodes));
                    total_valid_convergent_edges += total_convergent_edges == 0 ? 1 : static_cast<double>(mapping_history.get_num_valid_dfg_edge_convergences()) /
                                                     ( static_cast<double>(total_convergent_edges));
        
                    total_routing_nodes += mapping_history.get_total_routing_nodes();
        
                    test_sum_rewards += mapping_history.calc_sum_reward();
        
                    if (this->using_mcts) {
                        const auto& tree_depths = mapping_history.get_tree_depths();
                        auto tensor_tree_depths = torch::tensor(tree_depths, torch::kFloat32);
                        mean_tree_depth += tensor_tree_depths.mean();
                    }
        
                    num_valid_mappings += static_cast<int>(is_valid);
        
                    if (is_valid) {
                        hist_dict[train_or_test + "valid_map_per_dfg_size"].push_back(state->get_num_dfg_nodes());
                        hist_dict[train_or_test + "valid_map_per_arch_size"].push_back(state->get_cgra_dimensions_().first * state->get_cgra_dimensions_().second);
                        hist_dict[train_or_test + "valid_map_per_II"].push_back(state->get_II());
                        double conn_style_value = gen_bin_by_styles(state->get_interconnection_styles());
                        hist_dict[train_or_test + "valid_map_per_conn_style"].push_back(conn_style_value);
                    }
                }
            }
        );
        }
    );
        

        if (valid_mapping_history.has_value()) {
            std::unordered_map<std::string, double> result_map;
            std::unordered_map<std::string, std::vector<double>> hist_map;

            return {result_map, hist_map,valid_mapping_history};
        }

        double num_states = static_cast<double>(initial_states.size());
        auto test_mean_return = test_sum_rewards.item<double>() / num_states;
        auto two_mean_tree_depth = mean_tree_depth.item<double>() / num_states;
        double mean_episode_length = static_cast<double>(sum_episode_length) / num_states;
        double valid_mappings_rate = static_cast<double>(num_valid_mappings) / num_states;
        // for(auto pair: hist_dict){
        //     std::cout << pair.first << "\n";
        //     print_vector(pair.second);
        // }
        //  dict = {
        //     {train_or_test + "mean_return", test_mean_return},
        //     {train_or_test + "mean_episode_length", mean_episode_length},
        //     {train_or_test + "valid_mappings_rate", valid_mappings_rate}
        // };
        dict[train_or_test + "mean_return"] = test_mean_return;
        dict[train_or_test + "mean_episode_length"] = mean_episode_length;
        dict[train_or_test + "valid_mappings_rate"] = valid_mappings_rate;
        dict[train_or_test + "mean_reward_eval"] = test_sum_rewards.item<double>() / static_cast<double>(sum_episode_length - num_states);
        dict[train_or_test + "pct_mapped_nodes_per_mapping"] = total_mapped_nodes / num_states;
        dict[train_or_test + "pct_mapped_edges_per_mapping"] = total_mapped_edges / num_states;
        dict[train_or_test + "pct_valid_node_convergences_per_mapping"] = total_valid_convergent_nodes / num_states;
        dict[train_or_test + "pct_valid_edge_convergences_per_mapping"] = total_valid_convergent_edges / num_states;
        dict[train_or_test + "mean_routing_nodes_per_mapping"] = total_routing_nodes / num_states;


        if (this->using_mcts) {
            dict[train_or_test + "2x_mean_tree_depth"] = two_mean_tree_depth;
        }

        return {std::move(dict), std::move(hist_dict), std::nullopt};
    }


    template<typename StateT,typename ModelConfig>
    void Launcher<StateT,ModelConfig>::finish(){
        finish_server(socket, server_k);
        this->finish_socket();
        auto use_batch_requester = this->global_config.getLauncherConfig().getUseBatchRequester();
        if(use_batch_requester){
            this->batch_requester.reset();
        }
    }



    template<typename StateT,typename ModelConfig>
    std::vector<MappingHistory<StateT>>Launcher<StateT,ModelConfig>::map_with_zero_shot(const std::vector<std::shared_ptr<StateT>>& states){
        std::vector<MappingHistory<StateT>> mapping_hist_list;
        
        for(auto& state: states){
            auto mapping_history = this->config_self_mapping.map(state, false,1,this->global_config.getTrainConfig().getSeed(),std::nullopt);
            mapping_hist_list.emplace_back(std::move(mapping_history));
        }

        return mapping_hist_list;
    }

    template<typename StateT,typename ModelConfig>
    std::string Launcher<StateT,ModelConfig>::get_path_to_write_map(const std::shared_ptr<StateT>& state,bool finetune_mode){
        return state->get_dfg_name() + (finetune_mode ? "/finetune/" : "/zero_shot/");
    }

    template<typename StateT,typename ModelConfig>
    std::vector<MappingHistory<StateT>> Launcher<StateT,ModelConfig>::map_with_finetune(const std::vector<std::shared_ptr<StateT>>& states){
        std::vector<MappingHistory<StateT>> mapping_hist_list;

        for(auto& state: states){
            auto mapping_history = this->train({state}, {state}, false);
            if(!mapping_history.has_value()){
                mapping_history = this->map_with_zero_shot({state})[0];
            }
            mapping_hist_list.emplace_back(std::move(mapping_history.value()));

        }
        return mapping_hist_list;
    }

    template<typename StateT,typename ModelConfig>
    void Launcher<StateT,ModelConfig>::run(){
        auto launcher_config = this->global_config.getLauncherConfig();
        auto task_type = launcher_config.getEnumTaskType();
        this->run_config_server();

        std::vector<MappingHistory<StateT>> mapping_hist_list;

        auto [train_states, test_states] = this->get_initial_states();


        if(task_type == EnumTaskType::TRAIN){
            auto _ = this->train(train_states,test_states, true);
        } else if (task_type == EnumTaskType::ZERO_SHOT_MAP){
            mapping_hist_list = this->map_with_zero_shot(test_states);
            
        } else if(task_type == EnumTaskType::FINETUNE_MAP){
            mapping_hist_list = this->map_with_finetune(test_states);
        }

        if(task_type != EnumTaskType::TRAIN){
            bool finetune_mode = task_type ==  EnumTaskType::FINETUNE_MAP;
            for (auto& mapping_history: mapping_hist_list)
            {   std::stringstream ss;
                auto state = mapping_history.get_state_by_idx(0);
                auto path_to_write_map = this->get_path_to_write_map(state,finetune_mode);
                mapping_history.write_mapping(ss);
                throw std::runtime_error("write ss to file");

            }
        }
        this->finish();
    }
