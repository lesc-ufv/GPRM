#include "src/hpp/compilers/mapzero/rl/mapzero_self_mapping.hpp"

template <typename StateT, typename ModelConfig>
MappingHistory<StateT> map_with_mapzero_mcts( GlobalConfig<ModelConfig>& global_config,
                                std::shared_ptr<zmq::socket_t> socket,
                                const ServerConstants& server_k, const std::string& func_name,
                                            const std::shared_ptr<StateT> initial_state, Environment<StateT>& environment,
                                            bool train , double train_ratio,int seed,
                                            std::optional<std::unordered_map<std::string,int>> node_to_action,
                                            std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester
                                        ) {
    MappingHistory<StateT> mapping_history;
    mapping_history.start();
    mapping_history.init_values(initial_state);

    auto cur_state = initial_state;
    bool done = false;
    double reward;
    std::shared_ptr<Node<StateT>> final_root;
    #ifdef DEBUG
    std::vector<std::shared_ptr<Node<StateT>>> roots;
    std::cout << "Initial state set. Starting mapping loop." << std::endl;
    int debug_count = 0;
    #endif
    int first_root = true;
    int action, max_tree_depth;
    std::shared_ptr<Node<StateT>> root = nullptr;
    double predicted_value;
    auto mcts_config = global_config.getMCTSConfig();
    std::vector<std::shared_ptr<Node<StateT>>> solution;
    auto mapping_config = global_config.getMappingConfig();
    bool use_backtracking = mapping_config.getUseBacktracking();
    bool get_first_valid_solution = mapping_config.getFirstValidSolution();

    bool override_root = mcts_config.getRestartRoot();
    while (!done) {
        MapZeroMCTS<StateT,ModelConfig> mcts(global_config, func_name, socket);
        std::optional<std::shared_ptr<Node<StateT>>> root_to_override = (root && override_root)
                  ? std::make_optional(root->get_child(action))
                  : std::nullopt;
        std::tie(root, max_tree_depth, predicted_value, solution) = std::move(mcts.run(
            cur_state,
            root_to_override,
            environment,
            get_first_valid_solution,
            batch_requester
        ));

        if(first_root) first_root = false;

        mapping_history.append_tree_depth(max_tree_depth);
        final_root = root;
        mapping_history.store_search_statistics(root, false);

        #ifdef DEBUG
        roots.push_back(root);

        std::cout << "MCTS run completed (state " << debug_count << "). Max tree depth: " << max_tree_depth 
                  << ", Predicted value: " << predicted_value << ", Solution size: " 
                  << solution.size() << std::endl;
        debug_count += 1;
        #endif

        if (!solution.empty()) {
            auto& trajectory = solution;
            #ifdef DEBUG
                assert(get_first_valid_solution);
                print_line(50);
                for(auto& node: trajectory){
                    for (auto& pair: node->get_children()){
                        std::cout << "\nChild action: " << pair.first << "Mean value: " << pair.second->mean_value() << "\n";
                    }
                }
                print_line(50);
            #endif

            for (auto it = std::next(trajectory.begin()); it != trajectory.end(); ++it) {
                const auto& node = *it;
                mapping_history.append_action(node->get_action());
                mapping_history.append_action_indice(node->get_action_indice());
                // mapping_history.append_action_tree_depth(node->get_action_tree_depth());
                mapping_history.store_search_statistics(node, node->is_end_state());
                mapping_history.append_state(node->get_state());
                mapping_history.append_reward(node->get_reward());

                #ifdef DEBUG
                std::cout << "Appending action: " << node->get_action() 
                          << ", Reward: " << node->get_reward() << std::endl;
                #endif
            }

            #ifdef DEBUG

            std::cout << "Solution found. Exiting mapping loop." << std::endl;
            #endif

            return mapping_history;
        }
        int action_indice = -1;
        auto cur_node_to_be_mapped_name = cur_state->get_node_to_be_mapped_name();
        auto root_children = root->get_children();
        if(node_to_action.has_value() && node_to_action->find(cur_node_to_be_mapped_name) != node_to_action->end()){
            action = node_to_action->at(cur_node_to_be_mapped_name);
            action_indice = cur_state->get_action_idx_by_action(action);
            std::tie(cur_state, reward, done) = std::move(environment.step(*cur_state, action));
            std::cout << "[INFO] Using pre-defined action for node: " << cur_node_to_be_mapped_name 
                          << ", action index: " << action_indice << std::endl;
        }else{
            if (!use_backtracking) {
                action = select_action(root);
                action_indice = cur_state->get_action_idx_by_action(action);
                auto child_node = root_children[action];
                cur_state = child_node->get_state();
                reward = child_node->get_reward();
                done = cur_state->get_is_end_state();
    
                #ifdef DEBUG
                std::cout << "Action selected: " << action << ", Reward: " << reward 
                          << ", Done: " << done << std::endl;
                #endif
            } else {
                action = select_action(root);
                select_action_with_backtracking(root, environment, cur_state, action, action_indice, reward, done);
            }
        }

        mapping_history.append_action(action);
        mapping_history.append_action_indice(action_indice);
        mapping_history.append_state(cur_state);
        mapping_history.append_reward(reward);

        #ifdef DEBUG
        std::cout << "Mapping history updated. Action: " << action 
                  << ", Current reward: " << reward << std::endl;
        #endif
    }
    mapping_history.stop();
    mapping_history.store_search_statistics(final_root, true);

    #ifdef DEBUG
    print_line(50);
    for(auto& node: roots){
        for (auto& pair: node->get_children()){
            std::cout << "\nChild action: " << pair.first << " - Mean value: " << pair.second->mean_value() << "\n";
        }
    print_line(50);
    }
    std::cout << "\nMapping complete. Final tree depths size: " << mapping_history.get_tree_depths().size() << std::endl;
    #endif

    return mapping_history;
}

template <typename StateT>
int select_action(const std::shared_ptr<Node<StateT>>& node) {
    std::vector<int> visit_counts;
    std::vector<int> actions;

    for (const auto& [action, child] : node->get_children()) {
        visit_counts.push_back(child->get_visit_count());
        actions.push_back(action);
    }

    auto max_visit_iter = std::max_element(visit_counts.begin(), visit_counts.end());
    int max_index = std::distance(visit_counts.begin(), max_visit_iter);

    #ifdef DEBUG
    std::cout << "Selecting action with max visits. Action: " << actions[max_index] 
              << ", Visits: " << visit_counts[max_index] << std::endl;
    #endif

    return actions[max_index];
}

template<typename StateT, typename ModelConfig>    
void reanalyze_with_mapzero_mcts(GlobalConfig<ModelConfig>& config,
    MappingHistory<StateT>& mapping_history, std::shared_ptr<zmq::socket_t> socket,
    const ServerConstants& server_k, const std::string& func_name, std::shared_ptr<StateT> state, 
    Environment<StateT>& environment,  std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester){
        MapZeroMCTS<StateT,ModelConfig> mcts(config, func_name, socket);
        auto [root, max_tree_depth, predicted_value, solution] = mcts.run(
            state,
            std::nullopt,
            environment,
            false,
            batch_requester
        );
        mapping_history.store_search_statistics(root, false);
}
