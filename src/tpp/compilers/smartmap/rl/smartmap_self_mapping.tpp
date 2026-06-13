#include "src/hpp/compilers/smartmap/rl/smartmap_self_mapping.hpp"
template<typename StateT, typename ModelConfig>
MappingHistory<StateT> map_with_smartmap_mcts(
                                             GlobalConfig<ModelConfig>& global_config,
                                            std::shared_ptr<zmq::socket_t> socket,
                                            const ServerConstants& server_k, const std::string& func_name,
                                            const std::shared_ptr<StateT> initial_state, Environment<StateT>& environment,
                                            bool train, double train_ratio, int seed, 
                                            std::optional<std::unordered_map<std::string,int>> node_to_action,
                                            std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester
                                                ){
    MappingHistory<StateT> mapping_history;
    mapping_history.start();
    mapping_history.init_values(initial_state);
    uint64_t scaled = static_cast<uint64_t>(train_ratio * (1ULL << 32));

    std::mt19937 gen(seed ^ scaled);
    torch::Generator torch_gen = torch::make_generator<torch::CPUGeneratorImpl>(seed);
    auto cur_state = initial_state;
    bool done = false;
    double reward;
    std::shared_ptr<Node<StateT>> last_root = nullptr;

    std::shared_ptr<Node<StateT>> root;
    int max_tree_depth;
    double predicted_value;
    std::vector<std::shared_ptr<Node<StateT>>> solution;
    

    #ifdef DEBUG
        std::cout << "Starting MCTS with initial state: " << initial_state << std::endl;
    #endif
    auto mcts_config = global_config.getMCTSConfig();
    auto skip_sim_for_one_action = mcts_config.getSkipSimulationsForOneAction();
    SmartMapMCTS<StateT,ModelConfig> mcts(global_config, socket, func_name);
    double temperature;
    if(mcts_config.getUseSoftmaxTemperature() && train){
        if (train_ratio < 0.25){
            temperature = 1.0;
        }else if(train_ratio < 0.5){
            temperature = 0.75;
        }else if(train_ratio < 0.75){
            temperature = 0.5;
        }else{
            temperature = 0.25;
        }
    }else{
        if (!train) {temperature = 0;}
        else {
            temperature = mcts_config.getTemperature();
        }
    }
    auto mapping_config = global_config.getMappingConfig();
    bool use_backtracking = mapping_config.getUseBacktracking();
    bool get_first_valid_solution = mapping_config.getFirstValidSolution();

    while (!done) {
        int backtrackings = 0;

        if(!train && skip_sim_for_one_action){
            if(cur_state->get_legal_actions_size() == 1){
                auto legal_actions = cur_state->get_legal_actions_();
                int action = legal_actions[0];
                int action_indice = cur_state->get_action_idx_by_action(action);
                std::tie(cur_state, reward, done) = std::move(environment.step(*cur_state, action));
                #ifdef DEBUG
                    std::cout << "[INFO] Skipping MCTS for single legal action: " << action 
                    << ", action index: " << action_indice << std::endl;
                #endif
                mapping_history.append_action(action);
                mapping_history.append_action_indice(action_indice);
                mapping_history.append_state(cur_state);
                mapping_history.append_reward(reward);
                mapping_history.append_tree_depth(1);
                mapping_history.append_child_visit({1.0});
                mapping_history.add_predicted_value_mcts(-std::numeric_limits<double>::infinity());
                continue;
            }
        }
        if(!mcts_config.getRestartRoot() || (last_root == nullptr)){
            #ifdef DEBUG
                std::cout << "Running MCTS from root node" << std::endl;
            #endif
            std::tie(root, max_tree_depth, predicted_value, solution) = std::move(mcts.run(
                cur_state,
                std::nullopt,
                environment,
                get_first_valid_solution,
                mcts_config.getAddExplorationNoise() && train,
                batch_requester,
                gen,
                mcts_config.getReturnOptimalMapping()
            ));
        } else {
            #ifdef DEBUG
                std::cout << "Running MCTS replacing last root node" << std::endl;
            #endif
            std::tie(root, max_tree_depth, predicted_value, solution) = std::move(mcts.run(
                cur_state,
                std::make_optional(last_root),
                environment,
                get_first_valid_solution,
                mcts_config.getAddExplorationNoise() && train,
                batch_requester,
                gen,
                mcts_config.getReturnOptimalMapping()
            ));
        }

        mapping_history.append_tree_depth(max_tree_depth);
        mapping_history.add_predicted_value_mcts(predicted_value);
        int action;
        last_root = root;
        mapping_history.store_search_statistics(root, false);
        auto root_children = root->get_children();

        #ifdef DEBUG
            std::cout << "Children of root node: " << root_children.size() << " children" << std::endl;
        #endif

        if (!solution.empty()) {
            auto& trajectory = solution;

            for (auto it = std::next(trajectory.begin()); it != trajectory.end(); ++it) {
                const auto& node = *it;
                mapping_history.append_state(node->get_state());
                mapping_history.append_action(node->get_action());
                mapping_history.append_action_indice(node->get_action_indice());
                mapping_history.append_reward(node->get_reward());
                mapping_history.append_tree_depth(-std::numeric_limits<int>::infinity());
                mapping_history.append_child_visit({-std::numeric_limits<float>::infinity()});
                mapping_history.add_predicted_value_mcts(-std::numeric_limits<double>::infinity());

                // mapping_history.append_action_tree_depth(node->get_action_tree_depth());
                // mapping_history.store_search_statistics(node, node->is_end_state());
                // mapping_history.add_predicted_value_mcts(predicted_value);
            }
            // mapping_history.print();
            #ifdef DEBUG
                std::cout << "Solution found, returning mapping history." << std::endl;
            #endif
            return std::move(mapping_history);
        }

        #ifdef DEBUG
            std::cout << "Selecting action with temperature: " << temperature << std::endl;
        #endif
        int action_indice = -1;
        auto cur_node_to_be_mapped_name = cur_state->get_node_to_be_mapped_name();
        if(node_to_action.has_value() && node_to_action->find(cur_node_to_be_mapped_name) != node_to_action->end()){
            action = node_to_action->at(cur_node_to_be_mapped_name);
            action_indice = cur_state->get_action_idx_by_action(action);
            std::tie(cur_state, reward, done) = std::move(environment.step(*cur_state, action));
            std::cout << "[INFO] Using pre-defined action for node: " << cur_node_to_be_mapped_name 
                          << ", action index: " << action_indice << std::endl;
        }else{
            if (!use_backtracking) {
                action = select_action(root, train ? temperature : 0, gen, torch_gen);
                action_indice = cur_state->get_action_idx_by_action(action);
                auto child_node = root_children[action];
                cur_state = child_node->get_state();
                reward = child_node->get_reward();
                done = cur_state->get_is_end_state();
    
            }else{
                action = select_action(root,0, gen, torch_gen);
                backtrackings = select_action_with_backtracking(root, environment, cur_state, action, action_indice, reward, done);
            }
        }

        #ifdef DEBUG
            std::cout << "Action: " << action << ", Reward: " << reward << ", Done: " << done << std::endl;
        #endif

        mapping_history.append_action(action);
        mapping_history.append_action_indice(action_indice);
        mapping_history.append_state(cur_state);
        mapping_history.append_reward(reward);
        mapping_history.append_num_backtrackings(backtrackings);
    }

    mapping_history.stop();
    mapping_history.store_search_statistics(last_root, true);
    mapping_history.add_predicted_value_mcts(0.);

    return std::move(mapping_history);
}

template<typename StateT>
int select_action(std::shared_ptr<Node<StateT>> node, const double& temperature, std::mt19937& gen,
    torch::Generator& torch_gen) {
    auto children = node->get_children();
    int num_children = children.size();

    #ifdef DEBUG
    std::cout << "[DEBUG] select_action - num_children: " << num_children << ", temperature: " << temperature << std::endl;
    #endif

    if (num_children == 0) {
        #ifdef DEBUG
        std::cout << "[DEBUG] Error: No children available" << std::endl;
        #endif
        throw std::runtime_error("No children available for action selection.");
    }

    torch::Tensor visit_counts = torch::zeros({num_children}, torch::kInt32);
    torch::Tensor actions = torch::zeros({num_children}, torch::kInt32);

    int index = 0;
    for (const auto& child : children) {
        visit_counts[index] = child.second->get_visit_count();
        actions[index] = child.first;
        #ifdef DEBUG
        std::cout << "[DEBUG] Child " << index << ": action=" << child.first << ", visits=" << child.second->get_visit_count() << std::endl;
        #endif
        index++;
    }

    if (temperature == 0) {
        int max_index = torch::argmax(visit_counts).item<int>();
        #ifdef DEBUG
        std::cout << "[DEBUG] Deterministic selection (temp=0), selected index: " << max_index << std::endl;
        #endif
        return actions[max_index].item<int>();
    } else if (temperature == std::numeric_limits<double>::infinity()) {
        std::uniform_int_distribution<> dis(0, num_children - 1);
        int random_index = dis(gen);
        #ifdef DEBUG
        std::cout << "[DEBUG] Random selection (temp=∞), selected index: " << random_index << std::endl;
        #endif
        return actions[random_index].item<int>();
    } else {
        torch::Tensor visit_count_distribution = torch::pow(visit_counts, 1.0 / temperature);
        visit_count_distribution /= visit_count_distribution.sum();
        #ifdef DEBUG
        std::cout << "[DEBUG] Visit count distribution (temp=" << temperature << "): " << visit_count_distribution << std::endl;
        #endif
        torch::Tensor selected_action_index = torch::multinomial(visit_count_distribution, 1, false, torch_gen);
        int selected_index = selected_action_index.item<int>();
        #ifdef DEBUG
        std::cout << "[DEBUG] Sampled selection, selected index: " << selected_index << std::endl;
        #endif
        return actions[selected_index].item<int>();
    }
}
template<typename StateT, typename ModelConfig>    
void reanalyze_with_smartmap_mcts( 
    GlobalConfig<ModelConfig>& config,
    MappingHistory<StateT>& mapping_history, 
    std::shared_ptr<zmq::socket_t> socket,
    const ServerConstants& server_k, 
    const std::string& func_name, 
    std::shared_ptr<StateT> state,
    Environment<StateT>& environment, 
    std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester,
    std::mt19937& gen 
) {
#ifdef DEBUG
    std::cout << "[DEBUG] Starting reanalyze_with_smartmap_mcts..." << std::endl;
    std::cout << "[DEBUG] Function name: " << func_name << std::endl;
#endif

    SmartMapMCTS<StateT,ModelConfig> mcts(config, socket, func_name);
    auto mcts_config = config.getMCTSConfig();
    #ifdef DEBUG
        auto simulations = mcts_config.getNumSimulations();
        std::cout << "[DEBUG] Running SmartMapMCTS with " << simulations << "simulations." << std::endl;
    #endif

    auto [root, max_tree_depth, predicted_value, solution] = mcts.run(
        state,
        std::nullopt,
        environment,
        false,
        mcts_config.getAddExplorationNoise(),
        batch_requester,
         gen,
         false

    );

#ifdef DEBUG
    std::cout << "[DEBUG] MCTS completed." << std::endl;
    std::cout << "[DEBUG] Max tree depth: " << max_tree_depth << std::endl;
    std::cout << "[DEBUG] Predicted value: " << predicted_value << std::endl;
#endif

    mapping_history.add_predicted_value_mcts(predicted_value);
    mapping_history.store_search_statistics(root, false);

#ifdef DEBUG
    std::cout << "[DEBUG] Search statistics stored in mapping history." << std::endl;
#endif
}
