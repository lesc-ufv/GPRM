#include "src/hpp/utils/util_train.hpp"

template <typename DFGT, typename... Args>
std::tuple<std::vector<std::shared_ptr<DFGT>>, std::vector<std::pair<int, int>>>
initialize_dfgs(const std::string& dfgs_dot_path, bool sort, Args&... args) {
    tbb::concurrent_vector<std::shared_ptr<DFGT>> dfgs;
    tbb::concurrent_vector<std::pair<int, int>> idx_dfg_size_pair;

    std::vector<fs::path> dot_files;

    std::function<void(const fs::path&)> collect_dot_files = [&](const fs::path& dir_path) {
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_directory()) {
                collect_dot_files(entry.path());
            } else if (entry.is_regular_file() && entry.path().extension() == ".dot") {
                dot_files.push_back(entry.path());
            }
        }
    };

    collect_dot_files(dfgs_dot_path);

    tbb::parallel_for_each(dot_files.begin(), dot_files.end(), [&](const fs::path& dot_path) {
        auto dfg = std::make_shared<DFGT>(dot_path.string(), args...);
        int size = dfg->get_num_nodes();

        #ifdef DEBUG
            print_line(50);
            std::cout << "Creating DFG " << dot_path << std::endl << std::flush;
            // dfg->print();
            print_line(50);
        #endif

        int idx = dfgs.size(); 
        dfgs.push_back(dfg);
        idx_dfg_size_pair.push_back({idx, size});
    });

    std::vector<std::shared_ptr<DFGT>> dfgs_vec(dfgs.begin(), dfgs.end());
    std::vector<std::pair<int, int>> idx_size_vec(idx_dfg_size_pair.begin(), idx_dfg_size_pair.end());

  
    return {dfgs_vec, idx_size_vec};
}



template <typename CGRAT,typename... Args>
std::unordered_map<std::pair<int,int>,std::unordered_map<int,std::vector<std::shared_ptr<CGRAT>>>> 
    create_dim_to_II_to_homogeneous_cgras(const std::vector<std::unordered_set<EnumInterconnectionStyles>>& interconnection_styles_list,
                                        std::unordered_map<std::pair<int,int>,std::unordered_set<int>>& set_IIs_by_dims, 
                                        const std::vector<std::pair<int,int>>& dimensions_list,
                                        const std::vector<std::unordered_set<EnumFunctionalities>>& functionalities_list, 
                                        Args&... args){
    std::unordered_map<std::pair<int,int>,std::unordered_map<int,std::vector<std::shared_ptr<CGRAT>>>> dims_to_II_to_cgras;
    for(auto& arch_dims: dimensions_list){
        for (auto& II: set_IIs_by_dims.at(arch_dims)){
            auto cgras = initialize_homogeneous_cgras<CGRAT>(interconnection_styles_list, II, arch_dims,functionalities_list, args...); 
            dims_to_II_to_cgras[arch_dims][II] = cgras;
        }
    }
    return dims_to_II_to_cgras;
}

template <typename DFGT>
void add_set_IIs_by_dims(std::unordered_map<std::pair<int,int>,std::unordered_set<int>>& set_IIs_by_dims, std::vector<std::shared_ptr<DFGT>>& dfgs,
const std::vector<std::pair<int,int>>& dimensions_list, int unroll_II){
    std::vector<int> num_spatial_PEs_list = {};
    for(auto& dimensions: dimensions_list){
        num_spatial_PEs_list.emplace_back(dimensions.first * dimensions.second);
    }
    for (auto& dfg: dfgs){
        for(int i = 0; i< static_cast<int>(dimensions_list.size()); i++){
            auto II = calc_II(dfg->get_num_nodes(), num_spatial_PEs_list[i]);
            for(int j = 0; j <= unroll_II; j++) set_IIs_by_dims[dimensions_list[i]].insert(II + j);
        }
    }
   
}



template <typename StateT, typename DFGT, typename CGRAT, typename... Args>
void add_initial_states_by_dfg(std::vector<std::shared_ptr<StateT>>& initial_states,
                               std::shared_ptr<DFGT>& dfg,
                               std::unordered_map<std::pair<int, int>, std::unordered_map<int, std::vector<std::shared_ptr<CGRAT>>>>& dims_to_II_to_cgras,
                               const std::vector<std::pair<int, int>>& dimensions_list,
                               int unroll_II,
                               Args&... args)
{
    using Clock = std::chrono::high_resolution_clock;
    auto start_time = Clock::now();

    std::vector<int> num_spatial_PEs_list;
    for (const auto& dimensions : dimensions_list) {
        num_spatial_PEs_list.emplace_back(dimensions.first * dimensions.second);
    }

    std::cout << "[Info] Adding initial states with DFG " << dfg->get_dfg_name() << std::endl;

    for (int i = 0; i < static_cast<int>(dimensions_list.size()); i++) {
        const auto& dims = dimensions_list[i];
        auto II = calc_II(dfg->get_num_nodes(), num_spatial_PEs_list[i]);

        std::cout << "[Info] Dimensions: " << dims.first << "x" << dims.second
                  << ", Initial II: " << II << std::endl;

        for (int j = 0; j <= unroll_II; j++) {
            int current_II = II + j;
            const auto& cgra_list = dims_to_II_to_cgras.at(dims).at(current_II);

            std::cout << "  └─ [Info] Trying II = " << current_II
                      << " with " << cgra_list.size() << " CGRA(s)" << std::endl;

            for (const auto& cgra : cgra_list) {
                bool valid_state = is_valid_state(dfg, cgra);
                if(!valid_state) {
                    auto& dfg_name = dfg->get_dfg_name();
                    auto& cgra_dims = cgra->get_cgra_dimensions_();
                    auto cgra_II = cgra->get_II();
                    std::cout << "     └─ [Skipped] Invalid state for DFG: " << dfg_name
                              << ", CGRA Dimensions: " << cgra_dims.first << "x" << cgra_dims.second
                              << ", II: " << cgra_II << std::endl;
                    continue;
                }
                initial_states.emplace_back(std::make_shared<StateT>(dfg, cgra, args...));
                std::cout << "     └─ [Added] New state for II = " << current_II << std::endl;
            }
        }
    }

    auto end_time = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "[Done] Total initial states added: " << initial_states.size()
              << " in " << duration << " ms." << std::endl;
}

template <typename StateT, typename DFGT, typename CGRAT, typename... Args>
void add_initial_states(std::vector<std::shared_ptr<StateT>>& initial_states, std::vector<std::shared_ptr<DFGT>>& dfgs,
                        std::unordered_map<std::pair<int,int>,std::unordered_map<int,std::vector<std::shared_ptr<CGRAT>>>> & dims_to_II_to_cgras, 
                        const std::vector<std::pair<int,int>>& dimensions_list, int unroll_II,
                        Args&...  args
){
#ifdef DEBUG
    std::cout << "Entering add_initial_states..." << std::endl;
    std::cout << "Initial size of initial_states: " << initial_states.size() << std::endl;
    std::cout << "Number of DFGs: " << dfgs.size() << std::endl;
#endif
    for (auto& dfg: dfgs){
#ifdef DEBUG
        std::cout << "Processing DFG: " << &dfg << std::endl;
#endif
        add_initial_states_by_dfg<StateT,DFGT,CGRAT>(initial_states, dfg, dims_to_II_to_cgras, dimensions_list, unroll_II, args...);
    }
#ifdef DEBUG
    std::cout << "Exiting add_initial_states..." << std::endl;
    std::cout << "Final size of initial_states: " << initial_states.size() << std::endl;
#endif
}


template<typename DFGT, typename... DFGArgs>
std::vector<std::shared_ptr<DFGT>> create_dfgs(std::string dot_path, std::tuple<DFGArgs...>& dfg_args){
    #ifdef DEBUG
        std::cout << "Entering create_states with dot_path: " << dot_path  << std::endl;
    #endif
        auto [dfgs, idx_dfg_size_pair] = std::apply(
            [&](auto&... dfg) { 
    #ifdef DEBUG
                std::cout << "Initializing DFGs..." << std::endl;
    #endif
                return initialize_dfgs<DFGT>(dot_path, false, dfg...); 
            },
            dfg_args
        );

    // #ifdef DEBUG
        std::cout << "[INFO] Created DFGs. Number of DFGs: " << dfgs.size() << std::endl;
    // #endif  
    return dfgs;
}
template <typename StateT, typename ModelConfig, typename DFGT, typename CGRAT, typename... CGRAArgs, typename... StateArgs>
std::vector<std::shared_ptr<StateT>> create_states(
    GlobalConfig<ModelConfig>& global_config,
    std::vector<std::shared_ptr<DFGT>> dfgs, 
    std::tuple<CGRAArgs...>& cgra_args, 
    std::tuple<StateArgs...>& state_args
) {

    std::unordered_map<std::pair<int, int>, std::unordered_set<int>> set_IIs_by_dims;
    auto cgra_config = global_config.getCGRAConfig();

    auto dimensions_list = cgra_config.getDimensionsList();
#ifdef DEBUG
    std::cout << "Dimensions list size: " << dimensions_list.size() << std::endl;
#endif
    add_set_IIs_by_dims<DFGT>(set_IIs_by_dims, dfgs, dimensions_list, cgra_config.getUnrollII());

#ifdef DEBUG
    std::cout << "Creating dim_to_II_to_cgras..." << std::endl;
#endif
    auto dim_to_II_to_cgras = std::apply(
        [&](auto&... cgra) {
#ifdef DEBUG
            std::cout << "Applying create_dim_to_II_to_homogeneous_cgras..." << std::endl;
#endif
            return create_dim_to_II_to_homogeneous_cgras<CGRAT>(
                cgra_config.getConnectionsList(),
                set_IIs_by_dims,
                dimensions_list,
                cgra_config.getFunctionalitiesList(),
                cgra...
            );
        },
        cgra_args
    );

#ifdef DEBUG
    std::cout << "dim_to_II_to_cgras created. Number of entries: " << dim_to_II_to_cgras.size() << std::endl;
#endif
int num_dims = dimensions_list.size();
int num_connections = cgra_config.getConnectionsList().size();
int num_functionalities = cgra_config.getFunctionalitiesList().size();

    int estimated_size = dfgs.size() * num_dims * num_connections * num_functionalities;
    std::vector<std::shared_ptr<StateT>> initial_states = {};
    initial_states.reserve(estimated_size);

#ifdef DEBUG
    std::cout << "Reserving space for initial states. Estimated size: " << estimated_size << std::endl;
#endif
    std::apply(
        [&](auto&... state) {
#ifdef DEBUG
            std::cout << "Adding initial states..." << std::endl;
#endif      
            add_initial_states<StateT, DFGT, CGRAT>(
                initial_states,
                dfgs,
                dim_to_II_to_cgras,
                cgra_config.getDimensionsList(),
                cgra_config.getUnrollII(),
                state...
            );
        },
        state_args
    );

#ifdef DEBUG
    std::cout << "Returning initial states with size: " << initial_states.size() << std::endl;
#endif
    return initial_states;
}

template <typename StateT, typename ModelConfig, typename DFGT, typename CGRAT, typename... DFGArgs, typename... CGRAArgs, typename... StateArgs>
std::tuple<std::vector<std::shared_ptr<StateT>>, std::vector<std::shared_ptr<StateT>>> 
create_train_eval_states(
    GlobalConfig<ModelConfig> global_config,
    const std::string& train_dot_path, 
    const std::string& eval_dot_path, 
    std::tuple<DFGArgs...>& dfg_args,   
    std::tuple<CGRAArgs...>& cgra_args, 
    std::tuple<StateArgs...>& state_args 
) {

#ifdef DEBUG
    std::cout << "Entering create_train_eval_states..." << std::endl << std::flush;
#endif
    std::vector<std::shared_ptr<StateT>> train_initial_states, eval_initial_states;
    std::vector<std::shared_ptr<DFGT>> train_dfgs, test_dfgs;
    if (!train_dot_path.empty()) {
        #ifdef DEBUG
                std::cout << "Creating DFGs dot_path: " << train_dot_path << std::endl << std::flush;
        #endif  
        train_dfgs = create_dfgs<DFGT>(train_dot_path,dfg_args);
    }

    if (!eval_dot_path.empty()) {
        #ifdef DEBUG
            std::cout << "Creating eval initial states from dot_path: " << eval_dot_path << std::endl << std::flush;
        #endif
        test_dfgs = create_dfgs<DFGT>(eval_dot_path,dfg_args);

    }

    auto train_config = global_config.getTrainConfig();
    auto aug_dfgs = train_config.augment_dfgs_guided_by_curriculum();
    auto min_dfg_to_augment = train_config.getMinDFGSizeToAugment();
    if(aug_dfgs){
        std::cout << "[INFO] Augmenting Training DFGs..." << std::endl << std::flush;

        auto original_dfgs = train_dfgs; 
        tbb::concurrent_vector<std::shared_ptr<OrderedDFG>> train_dfgs_parallel;
        tbb::parallel_for(size_t(0), original_dfgs.size(), [&](size_t i) {
            auto& dfg = original_dfgs[i];
            auto new_dfgs = DataAugmenter::augment_dfg(dfg, global_config, min_dfg_to_augment);
        
            for (int j = 0; j < static_cast<int>(new_dfgs.size()); j++) {
                train_dfgs_parallel.push_back(std::move(new_dfgs[j]));
            }
        });
        
        train_dfgs.assign(train_dfgs_parallel.begin(), train_dfgs_parallel.end());
        std::cout << "[INFO] Training DFGs after augmentation: " << train_dfgs.size() << std::endl;
        std::cout << "[INFO] Cleaning Isomorphic DFGs into training data" << std::endl;
        train_dfgs = clean_isomorphic_dfgs_curriculum_dfg_augmentation(train_dfgs);
        std::cout << "[INFO] Training DFGs after cleaning: " << train_dfgs.size() << std::endl;

        if(train_dot_path != eval_dot_path){
            std::cout << "[INFO] Cleaning Isomorphic DFGs into training data with test data" << std::endl;
            train_dfgs = clean_isomorphic_dfgs(train_dfgs,test_dfgs);
            std::cout << "[INFO] Training DFGs after cleaning: " << train_dfgs.size() << std::endl;
        }
    }
    if(!train_dfgs.empty()){
        train_initial_states = create_states<StateT, ModelConfig, DFGT, CGRAT>(
            global_config, train_dfgs, cgra_args, state_args
        );
        bool use_efficient_sampling = train_config.getUseEfficientSampling();
        if (use_efficient_sampling){
            std::sort(train_initial_states.begin(), train_initial_states.end(),
                [](const std::shared_ptr<StateT>& a, const std::shared_ptr<StateT>& b) {
                    auto a_dims = a->get_cgra_dimensions_();
                    auto b_dims = b->get_cgra_dimensions_();
                    auto a_II = a->get_II();
                    auto b_II = b->get_II();
                    auto a_conn_name = gen_bin_by_styles(a->get_interconnection_styles());
                    auto b_conn_name = gen_bin_by_styles(b->get_interconnection_styles());
                    return std::format("{}_{}x{}_{}_{}",a->get_dfg_name(),a_dims.first,a_dims.second,a_II,a_conn_name) < \
                    std::format("{}_{}x{}_{}_{}",b->get_dfg_name(),b_dims.first,b_dims.second,b_II,b_conn_name);
                });
                
            for(int i = 0; i < static_cast<int>(train_initial_states.size()); i++){
                train_initial_states[i]->set_id(i);
                // assert(train_initial_states[i]->get_id() == i);
            }
        }    
    }    
    
    std::cout << "[INFO] Train initial states created. Size: " << train_initial_states.size() << std::endl << std::flush;
        eval_initial_states = create_states<StateT,  ModelConfig, DFGT, CGRAT>(
            global_config, test_dfgs, cgra_args, state_args
        );
    std::cout << "[INFO] Test initial states created. Size: " << eval_initial_states.size() << std::endl << std::flush;

    #ifdef DEBUG
        std::cout << "Returning train and test initial states..." << std::endl << std::flush;
    #endif
        return std::make_tuple(std::move(train_initial_states), std::move(eval_initial_states));
}
