#ifndef UTIL_TRAIN_HPP
#define UTIL_TRAIN_HPP

#include <iostream>
#include <torch/torch.h>
#include <utility>
#include <filesystem>
#include "src/hpp/enums/enum_interconnection_styles.hpp"
#include "src/hpp/enums/enum_functionalities.hpp"
#include "src/hpp/configs/cgra_config.hpp"
#include <vector>
#include <utility> 
#include "src/hpp/utils/cgra/util_cgra.hpp"
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/blocked_range.h>
#include <tbb/spin_mutex.h>
#include <memory>
#include <chrono>  
#include <future>
#include <mutex>  
#include "src/hpp/utils/hashes.hpp"
#include "src/hpp/rl/data_augmenter.hpp"
#include <tbb/parallel_for_each.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
namespace fs = std::filesystem;
torch::Tensor create_batch_edge_indices(const std::vector<int>& node_sizes, const std::vector<torch::Tensor>& edge_indices);
torch::Tensor create_batch_edge_indices(const int& max_nodes, const std::vector<torch::Tensor>& edge_indices);
void add_and_pad_graph(torch::Tensor& graph_features, torch::Tensor& graph_pad_mask, const int& graph_size,
torch::Tensor& batch_graph_features, torch::Tensor& batch_graph_pad_mask);
torch::Tensor make_advantages(const torch::Tensor& rewards, const torch::Tensor& values, const double& discount);
torch::Tensor compute_n_step_target_value(const torch::Tensor& rewards, const torch::Tensor& values, const int& td_steps, const double& discount, const int& index);
torch::Tensor compute_target_values_with_gae(const torch::Tensor& rewards, const torch::Tensor& values, const double& lambda, const double& discount);
torch::Tensor compute_n_step_target_values(const torch::Tensor& rewards, const torch::Tensor& values, const int& td_steps, const double& discount);

torch::Tensor create_indices(int total_size, bool shuffle, torch::Dtype type, const torch::Generator& generator);
std::vector<std::vector<int>> create_batch_size_idxs(int total_size, int batch_size, bool shuffle,  const torch::Generator& generator);

template <typename DFGT, typename... Args>
std::tuple<std::vector<std::shared_ptr<DFGT>>, std::vector<std::pair<int, int>>> initialize_dfgs(const std::string& dfgs_dot_path, bool sort,
                                                                                 Args&... args);

template <typename CGRAT,typename... Args>
std::unordered_map<std::pair<int,int>,std::unordered_map<int,std::vector<std::shared_ptr<CGRAT>>>> 
    create_dim_to_II_to_homogeneous_cgras(const std::vector<std::unordered_set<EnumInterconnectionStyles>>& interconnection_styles_list,
                                        std::unordered_map<std::pair<int,int>,std::unordered_set<int>>& set_IIs_by_dims, 
                                        const std::vector<std::pair<int,int>>& dimensions_list,
                                        const std::vector<std::unordered_set<EnumFunctionalities>>& functionalities_list, 
                                        Args&... args);

template <typename DFGT>
void add_set_IIs_by_dims(std::unordered_map<std::pair<int,int>,std::unordered_set<int>>& set_IIs_by_dims, std::vector<std::shared_ptr<DFGT>>& dfgs,
const std::vector<std::pair<int,int>>& dimensions_list, int unroll_II);

template <typename StateT, typename DFGT, typename CGRAT, typename... Args>
void add_initial_states_by_dfg(std::vector<std::shared_ptr<StateT>>& initial_states, std::shared_ptr<DFGT>& dfg,
                        std::unordered_map<std::pair<int,int>,std::unordered_map<int,std::vector<std::shared_ptr<CGRAT>>>> & dims_to_II_to_cgras, 
                        const std::vector<std::pair<int,int>>& dimensions_list,  int unroll_II,Args&... args);


template <typename StateT, typename DFGT, typename CGRAT, typename... Args>
void add_initial_states(std::vector<std::shared_ptr<StateT>>& initial_states, std::vector<std::shared_ptr<DFGT>>& dfgs,
                        std::unordered_map<std::pair<int,int>,std::unordered_map<int,std::vector<std::shared_ptr<CGRAT>>>> & dims_to_II_to_cgras, 
                        const std::vector<std::pair<int,int>>& dimensions_list,int unroll_II, Args&... args);

template <typename StateT, typename ModelConfig, typename DFGT, typename CGRAT,typename... CGRAArgs, typename... StateArgs>
std::vector<std::shared_ptr<StateT>> create_states(
    GlobalConfig<ModelConfig>& global_config,
    std::vector<std::shared_ptr<DFGT>> dfgs, 
    std::tuple<CGRAArgs...>& cgra_args, 
    std::tuple<StateArgs...>& state_args
);

torch::Tensor compute_gae(const torch::Tensor& rewards, const torch::Tensor& values, const double& lambda, const double& discount);

template<typename DFG, typename CGRA>
bool is_valid_state(std::shared_ptr<DFG> dfg, std::shared_ptr<CGRA> cgra) {
    // #define DEBUG
    if (dfg->num_vertices() > cgra->total_PEs()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] Invalid: Number of vertices in DFG is greater than total PEs in CGRA." << std::endl;
        #endif
        return false;
    }

    // if (dfg->is_disconnected()) {
    //     #ifdef DEBUG
    //     std::cout << "[DEBUG] Invalid: DFG is disconnected." << std::endl;
    //     #endif
    //     return false;
    // }

    auto dfg_IO_to_count = dfg->get_IO_to_count();
    auto cgra_IO_to_count = cgra->get_IO_to_count();

    std::vector<std::tuple<std::pair<int,int>, int, int>> dfg_IO_pairs; 
    for (const auto& [io_pair, count] : dfg_IO_to_count) {
        int sum_io = io_pair.first + io_pair.second;
        dfg_IO_pairs.emplace_back(io_pair, sum_io, count);
    }


    std::sort(dfg_IO_pairs.begin(), dfg_IO_pairs.end(),
        [](const auto& a, const auto& b) {
            if (std::get<1>(a) != std::get<1>(b))
                return std::get<1>(a) > std::get<1>(b); 
            return std::get<0>(a).first > std::get<0>(b).first;
        });

    std::vector<std::tuple<std::pair<int,int>, int,int>> cgra_IO_pairs; 
    for (const auto& [io_pair, count] : cgra_IO_to_count) {
        int sum_io = io_pair.first + io_pair.second;
        cgra_IO_pairs.emplace_back(io_pair, sum_io, count);
    }

    std::sort(cgra_IO_pairs.begin(), cgra_IO_pairs.end(),
        [](const auto& a, const auto& b) {
            if (std::get<1>(a) != std::get<1>(b))
                return std::get<1>(a) > std::get<1>(b);
            return std::get<0>(a).first > std::get<0>(b).first;
        });

    #ifdef DEBUG
    std::cout << "[DEBUG] DFG IO pairs: ";
    for (const auto& [io_pair, sum_io, count] : dfg_IO_pairs) {
        std::cout << "(" << io_pair.first << "," << io_pair.second
                << ") sum=" << sum_io << " count=" << count << " | ";
    }
    std::cout << std::endl;

    std::cout << "[DEBUG] CGRA IO pairs: ";
    for (const auto& [io_pair, sum_io, count] : cgra_IO_pairs) {
        std::cout << "(" << io_pair.first << "," << io_pair.second
                << ") sum=" << sum_io << " count=" << count << " | ";
    }
    std::cout << std::endl;
    #endif

    std::unordered_map<std::pair<int,int>, int> cgra_IO_count_map;


    for (const auto& cgra_IO_tuple : cgra_IO_pairs) {
        cgra_IO_count_map[std::get<0>(cgra_IO_tuple)] = 0;
    }

    for (const auto& [dfg_IO_pair, sum, count_dfg] : dfg_IO_pairs) {
        bool found_PE = false;
        #ifdef DEBUG
        std::cout << "[DEBUG] Trying to map DFG IO pair (" << dfg_IO_pair.first << ", " << dfg_IO_pair.second << ")..." << std::endl;
        #endif

        auto cur_count = count_dfg;
        for (const auto& [cgra_IO_pair, _, count_cgra] : cgra_IO_pairs) {
            #ifdef DEBUG
            std::cout << "[DEBUG] Comparing with CGRA IO pair (" << cgra_IO_pair.first << ", " << cgra_IO_pair.second << ")..." << std::endl;
            #endif
            if (dfg_IO_pair.first <= cgra_IO_pair.first &&
                dfg_IO_pair.second <= cgra_IO_pair.second) {

                if (count_cgra - cgra_IO_count_map[cgra_IO_pair] >= cur_count) {
                    #ifdef DEBUG
                    std::cout << "[DEBUG] Found a match! Updating CGRA IO count for (" 
                              << cgra_IO_pair.first << ", " << cgra_IO_pair.second << ")" << std::endl;
                    #endif
                    cgra_IO_count_map[cgra_IO_pair] += cur_count;
                    found_PE = true;
                    break;
                }else{
                    cur_count -= (count_cgra - cgra_IO_count_map[cgra_IO_pair]);
                    cgra_IO_count_map[cgra_IO_pair] += (count_cgra - cgra_IO_count_map[cgra_IO_pair]);
                    #ifdef DEBUG
                    std::cout << "[DEBUG] Not enough resources in CGRA IO pair (" 
                              << cgra_IO_pair.first << ", " << cgra_IO_pair.second 
                              << "). Remaining count needed: " << cur_count << std::endl;
                    #endif
                }
            }
        }

        if (!found_PE) {
            #ifdef DEBUG
            std::cout << "[DEBUG] No valid PE found for DFG IO pair (" 
                      << dfg_IO_pair.first << ", " << dfg_IO_pair.second << ")" << std::endl;
            #endif
            return false;
        }
    }

    #ifdef DEBUG
    std::cout << "[DEBUG] State is valid!" << std::endl;
    #endif
    return true;
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
);


template <typename StateT, typename DFGT, typename CGRAT, typename... Args>
std::shared_ptr<StateT> create_initial_state(
    std::shared_ptr<DFGT> dfg, std::shared_ptr<CGRAT> cgra, std::tuple<Args...>& args)
{
    return std::apply([&](auto&... unpacked_args) {
        return std::make_shared<StateT>(dfg, cgra, unpacked_args...);
    }, args);
}

template<typename DFGT>
std::unordered_map<std::pair<int,int>, std::vector<std::shared_ptr<DFGT>>> create_node_edge_to_dfgs(
    const std::vector<std::shared_ptr<DFGT>>& dfgs) {
    std::unordered_map<std::pair<int,int>, std::vector<std::shared_ptr<DFGT>>> node_edge_to_dfgs;

    for (const auto& dfg : dfgs) {
        auto node_edge_pair = std::make_pair(dfg->num_vertices(), dfg->get_num_edges());
        node_edge_to_dfgs[node_edge_pair].push_back(dfg);
    }

    return node_edge_to_dfgs;
}

template<typename DFGT>
inline std::vector<std::shared_ptr<DFGT>>
clean_isomorphic_dfgs_curriculum_dfg_augmentation(const std::vector<std::shared_ptr<DFGT>>& dfgs) {
    using KeyT = std::pair<int,int>;
    struct KeyHash {
        std::size_t operator()(const KeyT& p) const noexcept {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    struct Bucket {
        tbb::concurrent_vector<std::shared_ptr<DFGT>> dfgs;
        std::mutex mtx; 
    };

    tbb::concurrent_unordered_map<KeyT, std::shared_ptr<Bucket>, KeyHash> node_edge_to_not_isomorphic_dfgs;

    tbb::parallel_for(size_t(0), dfgs.size(), [&](size_t i) {
        auto& dfg = dfgs[i];
        KeyT node_edge_pair = {dfg->num_vertices(), dfg->get_num_edges()};

#ifdef DEBUG
        std::cout << "[DEBUG] Checking DFG '" << dfg->get_dfg_name()
                  << "' with " << node_edge_pair.first << " nodes and "
                  << node_edge_pair.second << " edges.\n";
#endif

        auto [it, _] = node_edge_to_not_isomorphic_dfgs.insert({node_edge_pair, std::make_shared<Bucket>()});
        auto bucket = it->second;

        bool is_isomorphic = false;
        {
            std::scoped_lock lock(bucket->mtx);
            for (auto& existing_dfg : bucket->dfgs) {
                if (existing_dfg->are_structurally_isomorphic_ptr(dfg) &&
                    existing_dfg->placement_ordering_equals(dfg)) {
#ifdef DEBUG
                    std::cout << "[DEBUG] DFG '" << dfg->get_dfg_name()
                              << "' is structurally isomorphic to '"
                              << existing_dfg->get_dfg_name() << "'. Skipping.\n";
#endif
                    is_isomorphic = true;
                    break;
                }
            }

            if (!is_isomorphic) {
#ifdef DEBUG
                std::cout << "[DEBUG] DFG '" << dfg->get_dfg_name()
                          << "' is not isomorphic to any existing DFG. Keeping it.\n";
#endif
                bucket->dfgs.push_back(dfg);
            }
        }
    });

    std::vector<std::shared_ptr<DFGT>> cleaned_dfgs;
    for (auto& [_, bucket] : node_edge_to_not_isomorphic_dfgs) {
        cleaned_dfgs.insert(cleaned_dfgs.end(), bucket->dfgs.begin(), bucket->dfgs.end());
    }

    return cleaned_dfgs;
}

template<typename DFGT>
inline std::vector<std::shared_ptr<DFGT>> clean_isomorphic_dfgs(
    const std::vector<std::shared_ptr<DFGT>>& dfgs_to_clean,
    const std::vector<std::shared_ptr<DFGT>>& dfgs_to_compare)
{
    using KeyT = std::pair<int,int>;
    struct PairHash {
        std::size_t operator()(const KeyT& p) const noexcept {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    tbb::concurrent_unordered_map<KeyT, tbb::concurrent_vector<std::shared_ptr<DFGT>>, PairHash> compare_map;
    for (auto& dfg : dfgs_to_compare) {
        KeyT node_edge_pair = {dfg->num_vertices(), dfg->get_num_edges()};
        compare_map[node_edge_pair].push_back(dfg);
    }

    tbb::concurrent_vector<std::shared_ptr<DFGT>> cleaned_dfgs;

    tbb::parallel_for(size_t(0), dfgs_to_clean.size(), [&](size_t i) {
        auto& dfg = dfgs_to_clean[i];
        KeyT node_edge_pair = {dfg->num_vertices(), dfg->get_num_edges()};

#ifdef DEBUG
        std::cout << "[DEBUG][Thread " << tbb::this_task_arena::current_thread_index()
                  << "] Processing DFG '" << dfg->get_dfg_name()
                  << "' with " << node_edge_pair.first << " nodes and "
                  << node_edge_pair.second << " edges.\n";
#endif

        bool is_isomorphic = false;
        auto it = compare_map.find(node_edge_pair);
        if (it != compare_map.end()) {
            for (auto& existing_dfg : it->second) {
                if (existing_dfg->are_structurally_isomorphic_ptr(dfg)) {
#ifdef DEBUG
                    std::cout << "[DEBUG][Thread " << tbb::this_task_arena::current_thread_index()
                              << "] DFG '" << dfg->get_dfg_name()
                              << "' is isomorphic to comparison DFG '"
                              << existing_dfg->get_dfg_name() << "'. Skipping.\n";
#endif
                    is_isomorphic = true;
                    break;
                }
            }
        }

        if (!is_isomorphic) {
#ifdef DEBUG
            std::cout << "[DEBUG][Thread " << tbb::this_task_arena::current_thread_index()
                      << "] DFG '" << dfg->get_dfg_name()
                      << "' is not isomorphic to any comparison DFG. Keeping it.\n";
#endif
            cleaned_dfgs.push_back(dfg);
        }
    });

    return std::vector<std::shared_ptr<DFGT>>(cleaned_dfgs.begin(), cleaned_dfgs.end());
}

#include "src/tpp/utils/util_train.tpp"
            
#endif