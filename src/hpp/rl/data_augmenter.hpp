#ifndef DATA_AUGMENTER_HPP
#define DATA_AUGMENTER_HPP

#include "torch/torch.h"
#include <vector>
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/rl/environment.hpp"
#include "src/hpp/enums/enum_training.hpp"
#include "src/hpp/rl/mapping_history.hpp"
#include "src/hpp/utils/mapping_augmentation.hpp"
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <numeric>
#include "src/hpp/utils/util_train.hpp"


constexpr int INF_SHIFT = std::numeric_limits<int>::max();
class DataAugmenter{
    public:
        DataAugmenter(){}

        template<typename ModelConfig, typename StateT>
        static std::vector<std::vector<int>> augment_combinatorial(
            const MappingHistory<StateT>& mapping,
            GlobalConfig<ModelConfig>& global_config)
        {
            const auto& states = mapping.get_states();
            auto last_state = states.back();
            const auto& node_to_PE = last_state->get_node_to_pe_();
            const auto& PE_to_coord = last_state->get_PE_to_coord_();
            const auto& coord_to_PE = last_state->get_coord_to_PE_();
            const auto& dims = last_state->get_cgra_dimensions_();
            const auto& buffer_config = global_config.getBufferConfig();
            const auto& degrees = buffer_config.getDegrees();
            const auto& flips = buffer_config.getFlips();
            const auto& shifts_cfg = buffer_config.getShifts();

            std::vector<std::pair<int, int>> effective_shifts;
            if (shifts_cfg.size() == 1 &&
                shifts_cfg[0].first == INF_SHIFT &&
                shifts_cfg[0].second == INF_SHIFT) {

                int max_up = std::numeric_limits<int>::max();
                int max_down = std::numeric_limits<int>::max();
                int max_left = std::numeric_limits<int>::max();
                int max_right = std::numeric_limits<int>::max();

                const auto& placement_order = last_state->get_placement_order();
                for (const auto& node : placement_order) {
                    if (node == -1 || !last_state->node_was_mapped(node)) continue;
                    auto PE = node_to_PE.at(node);
                    auto [c, y, x] = PE_to_coord.at(PE);
                    max_up = std::min(max_up, dims.first - 1 - y);
                    max_down = std::min(max_down, y);
                    max_left = std::min(max_left, dims.second - 1 - x);
                    max_right = std::min(max_right, x);
                }

                for (int dx = -max_right; dx <= max_left; ++dx) {
                    for (int dy = -max_down; dy <= max_up; ++dy) {
                        if (dx == 0 && dy == 0) continue;
                        effective_shifts.emplace_back(dx, dy);
                    }
                }
            } else {
                effective_shifts = shifts_cfg;
            }

            if (std::find(effective_shifts.begin(), effective_shifts.end(),
                std::make_pair(0, 0)) == effective_shifts.end()) {
                effective_shifts.emplace_back(0, 0);
            }

            std::vector<int> effective_degrees = degrees;
            if (effective_degrees.empty() ||
                std::find(effective_degrees.begin(), effective_degrees.end(), 0) == effective_degrees.end()) {
                effective_degrees.push_back(0);  
            }

            std::vector<std::pair<int, int>> effective_flips = flips;
            if (effective_flips.empty() ||
                std::find(effective_flips.begin(), effective_flips.end(),
                std::make_pair(0, 0)) == effective_flips.end()) {
                effective_flips.emplace_back(0, 0);  
            }

            std::vector<std::vector<int>> all_actions;
            const auto& placement_order = last_state->get_placement_order();

            for (auto& shift_vals : effective_shifts) {
                std::vector<int> shifted_actions;
                bool shift_is_valid = true;
            
                for (const auto& node : placement_order) {
                    if (node == -1 || !last_state->node_was_mapped(node)) continue;
                    auto PE = node_to_PE.at(node);
                    auto shifted_pe = shift(PE, PE_to_coord, coord_to_PE,
                                            dims.first, dims.second, shift_vals, false);
                    if (shifted_pe == -1) {
                        shift_is_valid = false;
                        break;
                    }
                    shifted_actions.emplace_back(shifted_pe);
                }
            
                if (!shift_is_valid) continue;
            
                for (auto& deg : effective_degrees) {
                    std::vector<int> rotated_actions;
                    for (auto& pe : shifted_actions) {
                        rotated_actions.emplace_back(
                            rotate(pe, PE_to_coord, coord_to_PE,
                                   dims.first, dims.second, deg)
                        );
                    }
            
                    for (auto& flip_vals : effective_flips) {
                        std::vector<int> flipped_actions;
                        for (auto& pe : rotated_actions) {
                            flipped_actions.emplace_back(
                                flip(pe, PE_to_coord, coord_to_PE,
                                     dims.first, dims.second, flip_vals)
                            );
                        }
            
                        bool any_transform =
                            !(shift_vals.first == 0 && shift_vals.second == 0 &&
                              deg == 0 &&
                              flip_vals.first == 0 && flip_vals.second == 0);
            
                        if (any_transform) {
                            all_actions.emplace_back(std::move(flipped_actions));
                        }
                    }
                }
            }
            return all_actions;
        }
        
        template<typename ModelConfig, typename StateT>
        static std::vector<std::vector<int>> augment(const MappingHistory<StateT>& mapping, 
                                                    GlobalConfig<ModelConfig>& global_config){
            const auto& states = mapping.get_states();
            const auto& last_state = states.back();
            const auto& node_to_PE = last_state->get_node_to_pe_();
            const auto& PE_to_coord = last_state->get_PE_to_coord_();
            const auto& coord_to_PE = last_state->get_coord_to_PE_();
            const auto& dims = last_state->get_cgra_dimensions_();
             // auto number_of_spatial_PEs = last_state->get_number_of_spatial_PEs();
            const auto& buffer_config = global_config.getBufferConfig();
            const auto& degrees = buffer_config.getDegrees();
            const auto& shifts = buffer_config.getShifts();
            const auto& flips = buffer_config.getFlips();
        
            #ifdef DEBUG
            std::cout << "[DEBUG] Applying rotations..." << std::endl;
            std::cout << "Number of rotations: " << degrees.size() << std::endl;
            #endif
            std::vector<std::vector<int>> action_list;
            const auto& placement_order = last_state->get_placement_order();
            for (auto& deg : degrees) {
                std::vector<int> actions;
            #ifdef DEBUG    
                std::cout << "DEBUG: Applying rotation degree = " << deg << "\n";
            #endif
                for (const auto& node : placement_order) {
                    if(node == -1) continue;
                    if(!last_state->node_was_mapped(node)) {
                        #ifdef DEBUG
                        std::cout << "DEBUG: Node " << node << " was not mapped, skipping shift calculation.\n";
                        #endif
                        continue;
                    }
                    auto PE = node_to_PE.at(node);
                    int result = rotate(PE, PE_to_coord, coord_to_PE, dims.first, dims.second, deg);
            #ifdef DEBUG
                    std::cout << "DEBUG: node " << node << ", original PE " << PE
                              << ", rotated result " << result << ", degree = " << deg << "\n";
            #endif
                    actions.emplace_back(result);
                }
                action_list.emplace_back(std::move(actions));
            }

            std::vector<std::pair<int, int>> effective_shifts;

            if (shifts.size() == 1 && shifts[0].first == INF_SHIFT && shifts[0].second == INF_SHIFT) {
                int max_up = std::numeric_limits<int>::max();
                int max_down = std::numeric_limits<int>::max();
                int max_left = std::numeric_limits<int>::max();
                int max_right = std::numeric_limits<int>::max();

            #ifdef DEBUG
                std::cout << "DEBUG: Calculating maximum allowed shifts in each direction...\n";
            #endif

            for (const auto& node : placement_order) {
                if (node == -1) continue;
            
                if (!last_state->node_was_mapped(node)) {
            #ifdef DEBUG
                    std::cout << "DEBUG: Node " << node << " was not mapped, skipping shift calculation.\n";
            #endif
                    continue;
                }
            
                auto PE = node_to_PE.at(node);
                auto [c, y, x] = PE_to_coord.at(PE);
            
            #ifdef DEBUG
                std::cout << "DEBUG: Processing node " << node
                          << ", PE=" << PE
                          << ", coord=(c=" << c << ", x=" << x << ", y=" << y << ")\n";
                std::cout << "DEBUG: Current limits before update: "
                          << "max_up=" << max_up << ", max_down=" << max_down
                          << ", max_left=" << max_left << ", max_right=" << max_right << "\n";
            #endif
            
                max_up = std::min(max_up, dims.first - 1 - y);
                max_down = std::min(max_down, y);
                max_left = std::min(max_left, dims.second - 1 - x);
                max_right = std::min(max_right, x);
            
            #ifdef DEBUG
                std::cout << "DEBUG: Updated limits after node " << node << ": "
                          << "max_up=" << max_up << ", max_down=" << max_down
                          << ", max_left=" << max_left << ", max_right=" << max_right << "\n";
            #endif
            }
            

            #ifdef DEBUG
                std::cout << "DEBUG: max_up = " << max_up
                        << ", max_down = " << max_down
                        << ", max_left = " << max_left
                        << ", max_right = " << max_right << "\n";
            #endif

                for (int dx = -max_right; dx <= max_left; ++dx) {
                    
                    for (int dy = -max_down; dy <= max_up; ++dy) {
                        if(dx == 0 && dy == 0) continue;
                        effective_shifts.emplace_back(dx, dy);
            #ifdef DEBUG
                        std::cout << "DEBUG: Adding shift (dx = " << dx << ", dy = " << dy << ")\n";
            #endif
                    }
                }
            } else {
                effective_shifts = shifts;
            }

            for (auto& shift_values : effective_shifts) {
            #ifdef DEBUG
                std::cout << "DEBUG: Applying shift (dx = " << shift_values.first 
                        << ", dy = " << shift_values.second << ")\n";
            #endif
                std::vector<int> actions;
                bool shift_is_valid = true;
                for (const auto& node : placement_order) {
                    if(node == -1) continue;
                    if(!last_state->node_was_mapped(node)) {
                        #ifdef DEBUG
                        std::cout << "DEBUG: Node " << node << " was not mapped, skipping shift calculation.\n";
                        #endif
                        continue;
                    }
                    auto PE = node_to_PE.at(node);
                    int result = shift(PE, PE_to_coord, coord_to_PE, dims.first, dims.second, shift_values, false);

                    #ifdef DEBUG
                            std::cout << "DEBUG: node " << node << ", original PE " << PE 
                                    << ", shifted to " << result << "\n";
                    #endif
                    if(result == -1) {
                        #ifdef DEBUG
                        std::cout << "DEBUG: Shifted PE " << PE << " is invalid, skipping node " << node << ".\n";
                        #endif
                        shift_is_valid = false;
                        break;
                    }
                    if(shift_is_valid) {
                        actions.emplace_back(result);
                    }
                }
                if(shift_is_valid){
                    action_list.emplace_back(std::move(actions));
                }
            }
                    
            for (auto& flip_dims : flips) {
                #ifdef DEBUG
                    std::cout << "DEBUG: Applying flip with dimensions (flip_x = " << flip_dims.first
                              << ", flip_y = " << flip_dims.second << ")\n";
                #endif
                    std::vector<int> actions;
                    
                    for (const auto& node : placement_order) {
                        if (node == -1) continue;
                
                        if (!last_state->node_was_mapped(node)) {
                #ifdef DEBUG
                            std::cout << "DEBUG: Node " << node << " was not mapped, skipping flip.\n";
                #endif
                            continue;
                        }
                
                        auto PE = node_to_PE.at(node);
                        
                #ifdef DEBUG
                        auto [c, x, y] = PE_to_coord.at(PE);
                        std::cout << "DEBUG: Flipping node " << node << " at PE " << PE
                                  << " (coord: c=" << c << ", x=" << x << ", y=" << y << ")\n";
                #endif
                
                        int flipped_PE = flip(PE, PE_to_coord, coord_to_PE, dims.first, dims.second, flip_dims);
                
                #ifdef DEBUG
                        auto [fc, fx, fy] = PE_to_coord.at(flipped_PE);
                        std::cout << "DEBUG: Node " << node << " mapped to flipped PE " << flipped_PE
                                  << " (coord: c=" << fc << ", x=" << fx << ", y=" << fy << ")\n";
                #endif
                
                        actions.emplace_back(flipped_PE);
                    }
                
                #ifdef DEBUG
                    std::cout << "DEBUG: Flip action generated with " << actions.size() << " entries.\n";
                #endif
                
                    action_list.emplace_back(std::move(actions));
                }

                return action_list;
        }


        template<typename ModelConfig, typename StateT>
        static std::tuple<std::vector<std::vector<int>>,  std::vector<std::vector<std::vector<int>>> > augment_with_legal_actions(const MappingHistory<StateT>& mapping, 
                                                    GlobalConfig<ModelConfig>& global_config){

            const auto& states = mapping.get_states();
            const auto& last_state = states.back();
            const auto& node_to_PE = last_state->get_node_to_pe_();
            const auto& PE_to_coord = last_state->get_PE_to_coord_();
            const auto& coord_to_PE = last_state->get_coord_to_PE_();
            const auto& dims = last_state->get_cgra_dimensions_();
             // auto number_of_spatial_PEs = last_state->get_number_of_spatial_PEs();
            const auto& buffer_config = global_config.getBufferConfig();
            const auto& degrees = buffer_config.getDegrees();
            const auto& shifts = buffer_config.getShifts();
            const auto& flips = buffer_config.getFlips();
        
            #ifdef DEBUG
            std::cout << "[DEBUG] Applying rotations..." << std::endl;
            std::cout << "Number of rotations: " << degrees.size() << std::endl;
            #endif
            std::vector<std::vector<int>> action_list;
            std::vector<std::vector<std::vector<int>>> augmented_legal_actions;

            const auto& placement_order = last_state->get_placement_order();
            for (auto& deg : degrees) {
                std::vector<int> actions;
            #ifdef DEBUG    
                std::cout << "DEBUG: Applying rotation degree = " << deg << "\n";
            #endif
                for (const auto& node : placement_order) {
                    if(node == -1) continue;
                    if(!last_state->node_was_mapped(node)) {
                        #ifdef DEBUG
                        std::cout << "DEBUG: Node " << node << " was not mapped, skipping shift calculation.\n";
                        #endif
                        continue;
                    }
                    auto PE = node_to_PE.at(node);
                    int result = rotate(PE, PE_to_coord, coord_to_PE, dims.first, dims.second, deg);
            #ifdef DEBUG
                    std::cout << "DEBUG: node " << node << ", original PE " << PE
                              << ", rotated result " << result << ", degree = " << deg << "\n";
            #endif
                    actions.emplace_back(result);
                }
                std::vector<std::vector<int>> legal_actions_per_state;
                for(int i = 0; i < static_cast<int>(states.size() - 1); i++){
                    auto legal_actions = states[i]->get_legal_actions_();
                    std::vector<int> augmented_legal_actions;
                    for(auto& action: legal_actions){
                        int result = rotate(action, PE_to_coord, coord_to_PE, dims.first, dims.second, deg);
                        augmented_legal_actions.emplace_back(result);
                    }
                    legal_actions_per_state.emplace_back(std::move(augmented_legal_actions));
                }
                augmented_legal_actions.emplace_back(std::move(legal_actions_per_state));
                action_list.emplace_back(std::move(actions));
            }


            std::vector<std::pair<int, int>> effective_shifts;

            if (shifts.size() == 1 && shifts[0].first == INF_SHIFT && shifts[0].second == INF_SHIFT) {
                int max_up = std::numeric_limits<int>::max();
                int max_down = std::numeric_limits<int>::max();
                int max_left = std::numeric_limits<int>::max();
                int max_right = std::numeric_limits<int>::max();

            #ifdef DEBUG
                std::cout << "DEBUG: Calculating maximum allowed shifts in each direction...\n";
            #endif

            for (const auto& node : placement_order) {
                if (node == -1) continue;
            
                if (!last_state->node_was_mapped(node)) {
            #ifdef DEBUG
                    std::cout << "DEBUG: Node " << node << " was not mapped, skipping shift calculation.\n";
            #endif
                    continue;
                }
            
                auto PE = node_to_PE.at(node);
                auto [c, y, x] = PE_to_coord.at(PE);
            
            #ifdef DEBUG
                std::cout << "DEBUG: Processing node " << node
                          << ", PE=" << PE
                          << ", coord=(c=" << c << ", x=" << x << ", y=" << y << ")\n";
                std::cout << "DEBUG: Current limits before update: "
                          << "max_up=" << max_up << ", max_down=" << max_down
                          << ", max_left=" << max_left << ", max_right=" << max_right << "\n";
            #endif
            
                max_up = std::min(max_up, dims.first - 1 - y);
                max_down = std::min(max_down, y);
                max_left = std::min(max_left, dims.second - 1 - x);
                max_right = std::min(max_right, x);
            
            #ifdef DEBUG
                std::cout << "DEBUG: Updated limits after node " << node << ": "
                          << "max_up=" << max_up << ", max_down=" << max_down
                          << ", max_left=" << max_left << ", max_right=" << max_right << "\n";
            #endif
            }
            

            #ifdef DEBUG
                std::cout << "DEBUG: max_up = " << max_up
                        << ", max_down = " << max_down
                        << ", max_left = " << max_left
                        << ", max_right = " << max_right << "\n";
            #endif

                for (int dx = -max_right; dx <= max_left; ++dx) {
                    
                    for (int dy = -max_down; dy <= max_up; ++dy) {
                        if(dx == 0 && dy == 0) continue;
                        effective_shifts.emplace_back(dx, dy);
            #ifdef DEBUG
                        std::cout << "DEBUG: Adding shift (dx = " << dx << ", dy = " << dy << ")\n";
            #endif
                    }
                }
            } else {
                effective_shifts = shifts;
            }

            for (auto& shift_values : effective_shifts) {
            #ifdef DEBUG
                std::cout << "DEBUG: Applying shift (dx = " << shift_values.first 
                        << ", dy = " << shift_values.second << ")\n";
            #endif
                std::vector<int> actions;
                bool shift_is_valid = true;
                for (const auto& node : placement_order) {
                    if(node == -1) continue;
                    if(!last_state->node_was_mapped(node)) {
                        #ifdef DEBUG
                        std::cout << "DEBUG: Node " << node << " was not mapped, skipping shift calculation.\n";
                        #endif
                        continue;
                    }
                    auto PE = node_to_PE.at(node);
                    int result = shift(PE, PE_to_coord, coord_to_PE, dims.first, dims.second, shift_values, false);

                    #ifdef DEBUG
                            std::cout << "DEBUG: node " << node << ", original PE " << PE 
                                    << ", shifted to " << result << "\n";
                    #endif
                    if(result == -1) {
                        #ifdef DEBUG
                        std::cout << "DEBUG: Shifted PE " << PE << " is invalid, skipping node " << node << ".\n";
                        #endif
                        shift_is_valid = false;
                        break;
                    }
                    if(shift_is_valid) {
                        actions.emplace_back(result);
                    }
                }
                if(shift_is_valid){
                    std::vector<std::vector<int>> legal_actions_per_state;
                    for(int i = 0; i < static_cast<int>(states.size() - 1); i++){
                        auto legal_actions = states[i]->get_legal_actions_();
                        std::vector<int> augmented_legal_actions_vec;
                        for(auto& action: legal_actions){
                            int result = shift(action, PE_to_coord, coord_to_PE, dims.first, dims.second, shift_values, false);
                            augmented_legal_actions_vec.emplace_back(result);
                        }
                        legal_actions_per_state.emplace_back(std::move(augmented_legal_actions_vec));
                    }
                    augmented_legal_actions.emplace_back(std::move(legal_actions_per_state));
                    action_list.emplace_back(std::move(actions));
                }
            }
                    
            for (auto& flip_dims : flips) {
                #ifdef DEBUG
                    std::cout << "DEBUG: Applying flip with dimensions (flip_x = " << flip_dims.first
                              << ", flip_y = " << flip_dims.second << ")\n";
                #endif
                    std::vector<int> actions;
                    
                    for (const auto& node : placement_order) {
                        if (node == -1) continue;
                
                        if (!last_state->node_was_mapped(node)) {
                #ifdef DEBUG
                            std::cout << "DEBUG: Node " << node << " was not mapped, skipping flip.\n";
                #endif
                            continue;
                        }
                
                        auto PE = node_to_PE.at(node);
                        
                #ifdef DEBUG
                        auto [c, x, y] = PE_to_coord.at(PE);
                        std::cout << "DEBUG: Flipping node " << node << " at PE " << PE
                                  << " (coord: c=" << c << ", x=" << x << ", y=" << y << ")\n";
                #endif
                
                        int flipped_PE = flip(PE, PE_to_coord, coord_to_PE, dims.first, dims.second, flip_dims);
                
                #ifdef DEBUG
                        auto [fc, fx, fy] = PE_to_coord.at(flipped_PE);
                        std::cout << "DEBUG: Node " << node << " mapped to flipped PE " << flipped_PE
                                  << " (coord: c=" << fc << ", x=" << fx << ", y=" << fy << ")\n";
                #endif
                
                        actions.emplace_back(flipped_PE);
                    }
                
                #ifdef DEBUG
                    std::cout << "DEBUG: Flip action generated with " << actions.size() << " entries.\n";
                #endif
                    std::vector<std::vector<int>> legal_actions_per_state;
                    for(int i = 0; i < static_cast<int>(states.size() - 1); i++){
                        auto legal_actions = states[i]->get_legal_actions_();
                        std::vector<int> augmented_legal_actions_vec;
                        for(auto& action: legal_actions){
                            int result = flip(action, PE_to_coord, coord_to_PE, dims.first, dims.second, flip_dims);
                            augmented_legal_actions_vec.emplace_back(result);
                        }
                        legal_actions_per_state.emplace_back(std::move(augmented_legal_actions_vec));
                    }
                    augmented_legal_actions.emplace_back(std::move(legal_actions_per_state));
                
                    action_list.emplace_back(std::move(actions));
                }

                return std::make_tuple(std::move(action_list), std::move(augmented_legal_actions));
        }

        template<typename StateT, typename ModelConfig>
        static std::vector<MappingHistory<StateT>> augment_mapping_ppo(
            MappingHistory<StateT>& mapping,
            GlobalConfig<ModelConfig>& global_config,
            Environment<StateT>& env,
            bool incremental
        ) {
        #ifdef DEBUG
            std::cout << "[DEBUG] Starting augment_mapping_ppo" << std::endl;
            std::cout << "Original mapping history size: " << mapping.get_states().size() << std::endl;
        #endif

            auto actions_list = incremental
                ? DataAugmenter::augment(mapping, global_config)
                : DataAugmenter::augment_combinatorial(mapping, global_config);

            tbb::concurrent_vector<MappingHistory<StateT>> all_mappings;
            all_mappings.push_back(std::move(mapping)); 
            std::reference_wrapper<MappingHistory<StateT>> first_mapping = all_mappings[0];
            const auto& cur_rewards = first_mapping.get().get_rewards();
            const auto& states = first_mapping.get().get_states();
            tbb::parallel_for(size_t(0), actions_list.size(), [&](size_t idx) {
            // for(int idx = 0; idx < static_cast<int>(actions_list.size()); idx++){

                const auto& actions = actions_list[idx];

                MappingHistory<StateT> new_mapping;
                std::vector<std::shared_ptr<StateT>> new_states = {states[0]};
                std::vector<int> new_actions = {-1};
                std::vector<int> new_actions_idxs = {-1};
                new_states.reserve(states.size());

                auto cur_state = states[0];
                bool aug_is_valid = true;

                for (size_t i = 0; i < actions.size(); i++) {
                    int new_action = actions[i];

                    if (!cur_state->is_action_available(new_action)) {
        #ifdef DEBUG
                        std::cout << "PE " << new_action << " is occupied - augmentation invalid" << std::endl;
        #endif
                        aug_is_valid = false;
                        break;
                    }

                    auto [next_state, reward, __] = env.step(*cur_state, new_action);
                    auto old_cur_state = states[i + 1];

        //             if (old_cur_state->get_legal_actions_size() != next_state->get_legal_actions_size()) {
        // #ifdef DEBUG
        //                 std::cout << "Legal actions size mismatch - augmentation invalid" << std::endl;
        // #endif
        //                 aug_is_valid = false;
        //                 assert(false);

        //                 break;
        //             }

                    if (std::abs(reward - static_cast<double>(cur_rewards[i + 1])) >= 1e-6) {
        #ifdef DEBUG
                        std::cout << "Reward mismatch - augmentation invalid" << std::endl;
        #endif
                        aug_is_valid = false;
                        break;
                    }

                    new_states.push_back(next_state);
                    new_actions.push_back(new_action);
                    new_actions_idxs.push_back(cur_state->get_action_idx_by_action(new_action));
                    cur_state = next_state;
                }

                if (aug_is_valid) {
                    new_mapping.set_rewards(first_mapping.get().get_rewards());
                    new_mapping.set_values(first_mapping.get().get_values());
                    new_mapping.set_states(new_states);
                    new_mapping.set_actions_indices(new_actions_idxs);
                    new_mapping.set_actions(new_actions);

                    for (const auto& child_visit : first_mapping.get().get_child_visits()) {
                        new_mapping.append_child_visit(child_visit);
                    }

                    all_mappings.push_back(std::move(new_mapping));
                }
            }
        );

        #ifdef DEBUG
            std::cout << "[DEBUG] augment_mapping_ppo completed with "
                    << all_mappings.size() << " mappings (including original)." << std::endl;
        #endif

            
        std::vector<MappingHistory<StateT>> ret;
        for (const auto& map : all_mappings) {
            ret.push_back(std::move(map));
        }
        return ret;
        }



        template<typename StateT, typename ModelConfig>
        static std::vector<MappingHistory<StateT>> augment_mapping_mcts(
            MappingHistory<StateT>& mapping,
            GlobalConfig<ModelConfig>& global_config,
            Environment<StateT>& env, bool incremental)
        {
        
        
        #ifdef DEBUG
            std::cout << "[DEBUG] Starting augment_mapping_mcts..." << std::endl;
            std::cout << "Original mapping history size: " << mapping.get_states().size() << std::endl;
        #endif
            assert(incremental);
            auto [actions_list, augmented_actions] = DataAugmenter::augment_with_legal_actions(mapping,global_config);
            
            tbb::concurrent_vector<MappingHistory<StateT>> all_mappings;
            all_mappings.push_back(std::move(mapping));  

            std::reference_wrapper<MappingHistory<StateT>> first_mapping = all_mappings[0];
            const auto& cur_rewards = first_mapping.get().get_rewards();
            const auto& states = first_mapping.get().get_states();
            // const auto& dims = states[0]->get_cgra_dimensions_();
        
            tbb::parallel_for(size_t(0), actions_list.size(), [&](size_t j) {
            // for(int j = 0; j < static_cast<int>(actions_list.size()); j++){
                const auto& actions = actions_list[j];
        
                MappingHistory<StateT> new_mapping;
                std::vector<std::shared_ptr<StateT>> new_states = {states[0]};
                std::vector<int> new_actions = {-1};
                std::vector<int> new_actions_idxs = {-1};
        
                auto cur_state = states[0];
                bool aug_is_valid = true;
        
                for (size_t i = 0; i < actions.size(); i++) {
                    int new_action = actions[i];
                    if (!cur_state->is_action_available(new_action)) {
        #ifdef DEBUG
                        std::cout << "PE " << new_action << " is occupied - augmentation invalid" << std::endl;
        #endif
                        aug_is_valid = false;
                        break;
                    }
        
                    auto [next_state, reward, __] = env.step(*cur_state, new_action);
                    auto old_cur_state = states[i + 1];
        
                    if (old_cur_state->get_legal_actions_size() != next_state->get_legal_actions_size()) {
        #ifdef DEBUG
                        std::cout << "Legal actions size mismatch - augmentation invalid" << std::endl;
        #endif
                        aug_is_valid = false;
                        break;
                    }
        
                    if (std::abs(reward - static_cast<double>(cur_rewards[i + 1])) >= 1e-6) {
        #ifdef DEBUG
                        std::cout << "Reward mismatch - augmentation invalid" << std::endl;
        #endif
                        aug_is_valid = false;
                        break;
                    }
        
                    new_states.push_back(next_state);
                    new_actions.push_back(new_action);
                    new_actions_idxs.push_back(cur_state->get_action_idx_by_action(new_action));
                    cur_state = next_state;
                }
        
                if (aug_is_valid) {
                    new_mapping.set_rewards(first_mapping.get().get_rewards());
                    new_mapping.set_values(first_mapping.get().get_values());
                    new_mapping.set_states(new_states);
                    new_mapping.set_actions_indices(new_actions_idxs);
                    new_mapping.set_actions(new_actions);
        
                    for (int z = 0; z < static_cast<int>(first_mapping.get().size()) - 1; z++) {
                        auto& original_legal_actions = states[z]->get_legal_actions_();
                        auto& augmented_legal_actions = augmented_actions[j][z];
                        auto& original_child_visits = first_mapping.get().get_child_visits()[z];
                    
                        std::vector<float> new_child_visits(original_child_visits.size(), 0);
                    
                        bool is_shift = false;
                        if (global_config.getBufferConfig().getShifts().size() > 0) {
                            const auto& current_aug = augmented_legal_actions;
                            // const auto& first_orig  = original_legal_actions;
                            for (int a : current_aug) {
                                if (a == -1) { is_shift = true; break; }
                            }
                        }
                        if (is_shift) {
                            for (int c = 0; c < static_cast<int>(original_legal_actions.size()); c++) {
                                int original_action = original_legal_actions[c];
                                int augmented_action = augmented_legal_actions[c];
                    
                                int original_idx  = states[z]->get_action_idx_by_action(original_action);
                                
                                if (augmented_action == -1) continue;
                                
                                if(new_states[z]->is_action_available(augmented_action) == false) {
                                    // assert(original_child_visits[original_idx] == 0);
                                    continue;
                                };

                                int augmented_idx = new_states[z]->get_action_idx_by_action(augmented_action);
                    
                                new_child_visits[augmented_idx] = original_child_visits[original_idx];
                                #ifdef DEBUG
                                    std::cout << "\n\nc=" << c 
                                    << " orig=" << original_action 
                                    << " aug=" << augmented_action 
                                    << " original_idx=" << original_idx 
                                    << " augmented_idx=" << augmented_idx 
                                    << " original_visit=" << original_child_visits[original_idx]
                                    << " new_visit=" << new_child_visits[augmented_idx]
                                    << std::endl << std::endl;
                                #endif
                            }
                    
                           
                            
                        } else {
                            for (int c = 0; c < static_cast<int>(original_legal_actions.size()); c++) {
                                int original_action = original_legal_actions[c];
                                int augmented_action = augmented_legal_actions[c];
                                int original_idx  = states[z]->get_action_idx_by_action(original_action);
                                if(new_states[z]->is_action_available(augmented_action) == false){
                                    // assert(original_child_visits[original_idx] == 0);
                                    continue;
                                };
                                int augmented_idx = new_states[z]->get_action_idx_by_action(augmented_action);
                                new_child_visits[augmented_idx] = original_child_visits[original_idx];
                            }
                        }

                        double sum_new = std::accumulate(new_child_visits.begin(), new_child_visits.end(), 0.0);
                        // assert(sum_new > 0.0);
                        for (auto& v : new_child_visits) v = static_cast<float>(v / sum_new);
                    
                        new_mapping.append_child_visit(new_child_visits);
                    }
                    new_mapping.append_child_visit({});

                    all_mappings.push_back(std::move(new_mapping));
                } else {
        #ifdef DEBUG
                    std::cout << "Augmentation invalid - skipping" << std::endl;
        #endif
                }
            }
        );
        
        #ifdef DEBUG
            std::cout << "[DEBUG] augment_mapping_mcts completed with "
                      << all_mappings.size() << " mappings (including original)." << std::endl;
        #endif
        
            std::vector<MappingHistory<StateT>> ret;
            for (const auto& map : all_mappings) {
                ret.push_back(std::move(map));
            }
            return ret;
        }



        static std::vector<std::vector<int>> augment_pe_vector(
            const std::vector<int>& pes,
            const std::unordered_map<int, std::tuple<int, int, int>>& PE_to_coord,
            const std::unordered_map<std::tuple<int, int, int>, int>& coord_to_PE,
            const std::pair<int, int>& dims, // dims = {height, width}
            const std::vector<int>& degrees,
            const std::vector<std::pair<int, int>>& shifts,
            const std::vector<std::pair<int, int>>& flips)
        {
            std::vector<std::vector<int>> action_list;
        
        #ifdef DEBUG
            std::cout << "[DEBUG] augment_pe_vector: Start augmentations\n";
        #endif
        
            /* =========================
             * 1. Apply rotations
             * ========================= */
            for (const auto& deg : degrees) {
                std::vector<int> actions;
        
        #ifdef DEBUG
                std::cout << "[DEBUG] Applying rotation: " << deg << " degrees\n";
        #endif
        
                for (const auto& pe : pes) {
                    if (pe == -1) continue;
        
                    int result = rotate(
                        pe,
                        PE_to_coord,
                        coord_to_PE,
                        dims.first,
                        dims.second,
                        deg
                    );
        
        #ifdef DEBUG
                    std::cout << "  PE " << pe << " -> rotated to " << result << "\n";
        #endif
        
                    actions.emplace_back(result);
                }
        
                action_list.emplace_back(std::move(actions));
            }
        
            std::vector<std::pair<int, int>> effective_shifts;
        
            if (shifts.size() == 1 &&
                shifts[0].first == INF_SHIFT &&
                shifts[0].second == INF_SHIFT)
            {
                int max_up    = std::numeric_limits<int>::max();
                int max_down  = std::numeric_limits<int>::max();
                int max_left  = std::numeric_limits<int>::max();
                int max_right = std::numeric_limits<int>::max();
        
        #ifdef DEBUG
                std::cout << "[DEBUG] Calculating maximum allowed shifts\n";
        #endif
        
                for (const auto& pe : pes) {
                    if (pe == -1) continue;
        
                    auto [c, y, x] = PE_to_coord.at(pe);
        
                    max_up    = std::min(max_up,    dims.first  - 1 - y);
                    max_down  = std::min(max_down,  y);
                    max_left  = std::min(max_left,  dims.second - 1 - x);
                    max_right = std::min(max_right, x);
                }
        
        #ifdef DEBUG
                std::cout << "[DEBUG] max_up=" << max_up
                          << " max_down=" << max_down
                          << " max_left=" << max_left
                          << " max_right=" << max_right << "\n";
        #endif
        
                for (int dx = -max_right; dx <= max_left; ++dx) {
                    for (int dy = -max_down; dy <= max_up; ++dy) {
                        if (dx == 0 && dy == 0) continue;
                        effective_shifts.emplace_back(dx, dy);
                    }
                }
            }
            else {
                effective_shifts = shifts;
            }
        
            /* =========================
             * 3. Apply shifts (only valid)
             * ========================= */
            for (const auto& shift_pair : effective_shifts) {
                std::vector<int> actions;
                bool shift_is_valid = true;
        
        #ifdef DEBUG
                std::cout << "[DEBUG] Applying shift dx="
                          << shift_pair.first
                          << " dy=" << shift_pair.second << "\n";
        #endif
        
                for (const auto& pe : pes) {
                    if (pe == -1) continue;
        
                    int result = shift(
                        pe,
                        PE_to_coord,
                        coord_to_PE,
                        dims.first,
                        dims.second,
                        shift_pair,
                        false
                    );
        
        #ifdef DEBUG
                    std::cout << "  PE " << pe << " -> shifted to " << result << "\n";
        #endif
        
                    if (result == -1) {
                        shift_is_valid = false;
        #ifdef DEBUG
                        std::cout << "  Invalid shift detected, discarding this shift\n";
        #endif
                        break;
                    }
        
                    actions.emplace_back(result);
                }
        
                if (shift_is_valid) {
                    action_list.emplace_back(std::move(actions));
                }
            }
        
            /* =========================
             * 4. Apply flips
             * ========================= */
            for (const auto& flip_pair : flips) {
                std::vector<int> actions;
        
        #ifdef DEBUG
                std::cout << "[DEBUG] Applying flip: flip_x="
                          << flip_pair.first
                          << " flip_y=" << flip_pair.second << "\n";
        #endif
        
                for (const auto& pe : pes) {
                    if (pe == -1) continue;
        
                    int result = flip(
                        pe,
                        PE_to_coord,
                        coord_to_PE,
                        dims.first,
                        dims.second,
                        flip_pair
                    );
        
        #ifdef DEBUG
                    std::cout << "  PE " << pe << " -> flipped to " << result << "\n";
        #endif
        
                    actions.emplace_back(result);
                }
        
                action_list.emplace_back(std::move(actions));
            }
        
        #ifdef DEBUG
            std::cout << "[DEBUG] augment_pe_vector: Done. Total augmentations = "
                      << action_list.size() << "\n";
        #endif
        
            return action_list;
        }
        
        



        template<typename ModelConfig>
        static std::vector<std::shared_ptr<OrderedDFG>> augment_dfg(std::shared_ptr<OrderedDFG>& dfg, 
                                                                    GlobalConfig<ModelConfig>& global_config,
                                                                    int min_dfg_size){
                // auto placement_approach = global_config.getDFGConfig().getPlacementApproach();
                // assert(placement_approach != EnumPlacementApproach::TOPOLOGICAL_ORDER && placement_approach != EnumPlacementApproach::RANDOM);

                std::vector<std::shared_ptr<OrderedDFG>> all_dfgs = {dfg};
                
                auto dfg_size = dfg->num_vertices();
                const auto& placement_order = dfg->get_placement_order();
                std::unordered_set<int> nodes_to_ignore;
                int count = 1;
                for(auto it = placement_order.rbegin(); it != placement_order.rend(); it++){
                    auto node = *it;
                    if(node == -1) continue;
                    if (dfg_size - count < min_dfg_size) break;
                    // std::cout << "Node " << node << std::endl;
                    nodes_to_ignore.insert(node);
                    auto dot_string = dfg->to_dot_string(nodes_to_ignore);
                    std::shared_ptr<OrderedDFG> new_dfg = std::make_shared<OrderedDFG>();
                    
                    new_dfg->build_from_string(dot_string, dfg->get_dfg_name() + std::format("_{}", dfg_size - count), global_config.getDFGConfig());
                    // new_dfg->update_placement_order(placement_order, dfg_size - count - 1, dfg->get_node_to_name());
                    all_dfgs.push_back(new_dfg);
                    count++;
                }

                return all_dfgs;    

            }


            template<typename StateT, typename ModelConfig>
            static std::vector<MappingHistory<StateT>> augment_mapping_by_placement_order_cutting(
                MappingHistory<StateT>& mapping,
                GlobalConfig<ModelConfig>& global_config,
                Environment<StateT>& env,
                std::shared_ptr<zmq::socket_t> socket,
                ServerConstants server_k
            ){
                std::vector<MappingHistory<StateT>> all_mappings = {};
            
                auto state = mapping.get_first_state();
                auto dfg = state->get_dfg();
                auto cgra = state->get_cgra();
                auto state_args = global_config.get_state_args(socket, server_k);
            
                const auto& rewards = mapping.get_rewards();
                const auto& placement_order = dfg->get_placement_order();
                auto min_dfg_size = global_config.getTrainConfig().getMinDFGSizeToAugment();
            
                std::unordered_set<int> nodes_to_ignore;
                int dfg_size = dfg->num_vertices();
                int count = 1;
                auto& node_to_pe = mapping.get_last_state()->get_node_to_pe_();
                for (auto it = placement_order.rbegin(); it != placement_order.rend(); it++) {
                    int node = *it;
                    if (node == -1) continue;
                    if (dfg_size - count < min_dfg_size) break;
            
                    nodes_to_ignore.insert(node);
                    auto dot_string = dfg->to_dot_string(nodes_to_ignore);
            
                    auto new_dfg = std::make_shared<OrderedDFG>();
                    new_dfg->build_from_string(dot_string, dfg->get_dfg_name() + std::format("_{}", dfg_size - count), global_config.getDFGConfig());
                    // new_dfg->update_placement_order(placement_order, dfg_size - count - 1, dfg->get_node_to_name());
            
                    auto new_state = create_initial_state<StateT,OrderedDFG,CGRA>(new_dfg, cgra, state_args);
            
                    MappingHistory<StateT> new_mapping;
                    std::vector<std::shared_ptr<StateT>> new_states = {new_state};
                    std::vector<int> new_actions = {-1};
                    std::vector<int> new_actions_idxs = {-1};
                    new_states.reserve(mapping.size());
            
                    auto cur_state = new_state;
                    int step = 0;
                    for (auto node2 : new_dfg->get_placement_order()) {
                        if (node2 == -1) continue;
                        auto node2_name = new_dfg->get_name_by_node(node2);
                        auto original_node = dfg->get_node_by_name(node2_name);
                        int new_action = node_to_pe.at(original_node);
            
                        assert(cur_state->is_action_available(new_action));
                        auto [next_state, reward, __] = env.step(*cur_state, new_action);
                        // assert(std::abs(reward - static_cast<double>(rewards[step + 1])) < 1e-6);
            
                        new_states.push_back(next_state);
                        new_actions.push_back(new_action);
                        new_actions_idxs.push_back(cur_state->get_action_idx_by_action(new_action));
                        cur_state = next_state;
                        step++;
                    }
            
                    new_mapping.set_rewards(rewards);
                    new_mapping.set_values(mapping.get_values());
                    new_mapping.set_states(new_states);
                    new_mapping.set_actions_indices(new_actions_idxs);
                    new_mapping.set_actions(new_actions);
                    assert(new_mapping.get_mapping_is_valid());
                    all_mappings.push_back(std::move(new_mapping));
                    count++;
                }
            
                return all_mappings;
            }
            

            template<typename StateT, typename ModelConfig>
            static std::vector<MappingHistory<StateT>> augment_temporal_mapping(
                MappingHistory<StateT>& mapping,
                GlobalConfig<ModelConfig>& global_config,
                Environment<StateT>& env,
                std::shared_ptr<zmq::socket_t> socket,
                ServerConstants server_k
            ){
                auto state = mapping.get_first_state();
                auto dfg = state->get_dfg();
                auto cgra = state->get_cgra();
                auto cur_II = state->get_II();
                auto unroll_II = global_config.getCGRAConfig().getUnrollII();
                auto state_args = global_config.get_state_args(socket, server_k);
            
                auto dims = cgra->get_cgra_dimensions_();
                auto& pe_to_functionalities = cgra->get_pe_to_functionalities();
                auto& interc_styles = cgra->get_interconnection_styles();
                auto tight_sched = dfg->balanced_schedule();
            
                std::vector<MappingHistory<StateT>> all_mappings = {};
                auto last_state = mapping.get_last_state();
                auto& node_to_pe = last_state->get_node_to_pe_();
                auto& pe_to_coord = last_state->get_PE_to_coord_();
                auto& rewards = mapping.get_rewards();
                auto& placement_order = dfg->get_placement_order();
                auto functionalities = pe_to_functionalities.at(0); 
                for (int ii = cur_II + 1; ii <= cur_II + unroll_II; ii++) {   
                    auto pe_to_functionalities = init_homogeneous_pe_to_functionalities(dims.first * dims.second * ii, functionalities); 

                    auto new_cgra = std::make_shared<CGRA>(ii, dims, pe_to_functionalities, interc_styles, global_config.should_calc_node_dists());

                    auto new_state = create_initial_state<StateT,OrderedDFG,CGRA>(dfg, new_cgra, state_args);
            
                    MappingHistory<StateT> new_mapping;
                    std::vector<std::shared_ptr<StateT>> new_states = {new_state};
                    std::vector<int> new_actions = {-1};
                    std::vector<int> new_actions_idxs = {-1};
                    new_states.reserve(mapping.size());
            
                    auto cur_state = new_state;
                    int step = 0;
                    auto& coord_to_pe = new_state->get_coord_to_PE_();
                    bool aug_is_valid = true;
                    for (auto node : placement_order) {
                        if (node == -1) continue;
            
                        int cycle = tight_sched[node] % ii;
                        int action = node_to_pe.at(node);
                        auto [_, row, col] = pe_to_coord.at(action);
                        auto new_action_coord = std::make_tuple(cycle, row, col);
                        int new_action = coord_to_pe.at(new_action_coord);
            
                        assert(cur_state->is_action_available(new_action));
                        auto [next_state, reward, __] = env.step(*cur_state, new_action);
                        if (!(std::abs(reward - static_cast<double>(rewards[step + 1])) < 1e-6)){
                            aug_is_valid = false;
                            break;
                        }
            
                        new_states.push_back(next_state);
                        new_actions.push_back(new_action);
                        new_actions_idxs.push_back(cur_state->get_action_idx_by_action(new_action));
                        cur_state = next_state;
                        step++;
                    }
                    if (aug_is_valid){
                        new_mapping.set_rewards(rewards);
                        new_mapping.set_values(mapping.get_values());
                        new_mapping.set_states(new_states);
                        new_mapping.set_actions_indices(new_actions_idxs);
                        new_mapping.set_actions(new_actions);
                        all_mappings.push_back(std::move(new_mapping));
                    }else{
                        std::cout << "[INFO] Invalid temporal augmentation \n";
                    }
                }
            
                return all_mappings;
            }
            
};

#endif
