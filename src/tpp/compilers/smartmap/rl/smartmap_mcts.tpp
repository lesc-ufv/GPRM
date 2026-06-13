#include "src/hpp/compilers/smartmap/rl/smartmap_mcts.hpp"

template<typename StateT, typename ModelConfig>
SmartMapMCTS<StateT,ModelConfig>::SmartMapMCTS(GlobalConfig<ModelConfig>& global_config,  std::shared_ptr<zmq::socket_t> socket,  const std::string& func_name): 
global_config(global_config), socket(socket), func_name(func_name)
{
    auto rl_config = this->global_config.getRLConfig();
    this->discount = rl_config.getDiscount();
}

template<typename StateT, typename ModelConfig>
std::tuple<std::shared_ptr<Node<StateT>>, int, float, std::vector<std::shared_ptr<Node<StateT>>>>
SmartMapMCTS<StateT,ModelConfig>::run(const std::shared_ptr<StateT> root_state, std::optional<std::shared_ptr<Node<StateT>>> root_to_start,
     Environment<StateT>& environment, bool get_first_valid_solution, bool add_exploration_noise, 
     std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester,  std::mt19937& gen, bool return_optimal_solution) 
{
    std::shared_ptr<Node<StateT>> root_node;
    torch::Tensor root_predicted_value;
    torch::Tensor value, policy_logits;
    std::vector<c10::IValue> model_inputs;
    std::vector<torch::Tensor> returns;
    int traversed_path_max_len;
    auto mcts_config = this->global_config.getMCTSConfig();
    auto max_num_expansion = mcts_config.getMaxNumExpansion();
    auto device = torch::kCPU;
    // double cumulative_sum_predicted_values = 0.0;
    // doubl = 0.0;
    if(root_to_start){
        root_node = root_to_start.value();
        root_predicted_value = torch::tensor(std::numeric_limits<double>::infinity());
        traversed_path_max_len = root_node->get_state_().get_dfg_size() + 1;
    }else{
        root_node = std::make_shared<Node<StateT>>(0.0);

        std::tie(policy_logits, root_predicted_value) = std::move(infer(socket, server_k, batch_requester, func_name, root_state));

        traversed_path_max_len = root_state->get_dfg_size() + 1;

        #ifdef DEBUG
        std::cout << "Root node initialized. Predicted value: " << root_predicted_value << ", Policy logits: " << policy_logits << std::endl;
        #endif

        double initial_reward = 0.0;
        root_node->expand(root_state,
                        root_state->get_legal_actions_(),
                        initial_reward,
                        policy_logits,
                        max_num_expansion
                        );
        root_node->increment_visit_count();
    }

    if (add_exploration_noise){
        root_node->add_exploration_noise(
            mcts_config.getDirichletAlpha(),
            mcts_config.getExplorationFactor(),
            gen
        );
    }

    int max_tree_depth = -1;
    int action;
    std::vector<std::shared_ptr<Node<StateT>>> traversed_path;
    traversed_path.reserve(traversed_path_max_len);
    auto min_max_stats = MinMaxStats();
    auto num_simulations = mcts_config.getNumSimulations(); 
    for (int i = 0; i < num_simulations; i++) {

        auto node = root_node;
        traversed_path = {node};
        int current_tree_depth = 0;
        action = -1;

        #ifdef DEBUG
        int num_lines = 50;
        print_line(num_lines);
        std::cout << "Starting simulation " << i + 1 << " of " << num_simulations << std::endl;
        #endif

        while (node->expanded()) {
            current_tree_depth += 1;
            #ifdef DEBUG
                auto father_visits = node->get_visit_count();
            #endif 
            std::tie(action, node) = this->select_child(*node,min_max_stats);
            traversed_path.emplace_back(node);

            #ifdef DEBUG
                std::cout << "Current tree depth: " << current_tree_depth <<" - Father visits: " << father_visits << " - Child visits: " << node->get_visit_count() << " - Selected action: " << action  << "\n\n";
                print_line(num_lines);
            #endif
        }

        auto parent = traversed_path[traversed_path.size() - 2];
        auto [next_state, reward, next_state_is_end_state] = environment.step(StateT(parent->get_state_()), action);

        #ifdef DEBUG
            print_line(num_lines);
            std::cout << "Stepped into next state. Reward: " << reward << ", Is end state: " << next_state_is_end_state << std::endl;
        #endif

        if (!next_state_is_end_state) {
            model_inputs = next_state->get_model_inputs(device);
            std::tie(policy_logits, value) = std::move(infer(socket, server_k, batch_requester, func_name, next_state));
        
            #ifdef DEBUG
                print_line(50);

                std::cout << "Next state expanded. Value: " << value << ", Policy logits: " << policy_logits << std::endl;
            #endif
            // cumulative_sum_predicted_values += value.item<double>();
        } else {
            value = torch::tensor(0);
            policy_logits = torch::tensor({});
        }

        node->expand(next_state,
                    next_state->get_legal_actions_(),
                    reward,
                    policy_logits,
                    max_num_expansion
                    );

        this->backpropagate(traversed_path, value.item().toDouble(), this->discount, min_max_stats);

        max_tree_depth = std::max(max_tree_depth, current_tree_depth);

        #ifdef DEBUG
        std::cout << "Backpropagation complete. Current max tree depth: " << max_tree_depth << std::endl;
        #endif

        if (next_state_is_end_state && return_optimal_solution && next_state->get_mapping_is_valid_() && next_state->get_num_routing_nodes() == 0) {
            std::cout << "Optimal solution found." << std::endl;

            #ifdef DEBUG
            std::cout << "Returning early with a valid solution." << std::endl;
            #endif

            return std::make_tuple(root_node, max_tree_depth, root_predicted_value.item().toFloat(), traversed_path);
        }

        if (get_first_valid_solution && next_state_is_end_state && next_state->get_mapping_is_valid_()) {
            std::cout << "Solution found before finishing the search." << std::endl;

            #ifdef DEBUG
            std::cout << "Returning early with a valid solution." << std::endl;
            #endif

            return std::make_tuple(root_node, max_tree_depth, root_predicted_value.item().toFloat(), traversed_path);
        }

    }

    // std::cout << "MCTS search complete. Total simulations: " << num_simulations << std::endl;
    // std::cout << "Cumulative sum of predicted values: " << cumulative_sum_predicted_values << std::endl;
    // std::cout << "Cumulative sum of UCB scores: " << cumulative_sum_ucb_scores << std::endl;

    return std::make_tuple(root_node, max_tree_depth, root_predicted_value.item().toFloat(), std::vector<std::shared_ptr<Node<StateT>>>());
}

template<typename StateT,typename ModelConfig>
std::pair<int, std::shared_ptr<Node<StateT>>> SmartMapMCTS<StateT,ModelConfig>::select_child(const Node<StateT>& node, MinMaxStats& min_max_stats)
{
    const auto children = node.get_children();
    int best_action = -1;
    double max_ucb_score = -std::numeric_limits<double>::infinity();
    auto mcts_config = this->global_config.getMCTSConfig();
    auto c1 = mcts_config.getC1();
    auto c2 = mcts_config.getC2();
    #ifdef DEBUG
    std::cout << "Selecting child node. Number of children: " << children.size() << std::endl << std::flush;
    print_line(50);
    #endif
    for (const auto& pair : children) {
        const auto& child = *(pair.second);
        auto cur_ucb_score = smartmap_ucb_score(node.get_visit_count(),child.get_visit_count(),c1,c2,child.get_prior(), this->discount,child.get_reward(),
        child.mean_value(), min_max_stats);
        // sum += cur_ucb_score;
        #ifdef DEBUG
        std::cout << "Child action: " << pair.first << ", ucb score: " << cur_ucb_score << std::endl << std::flush;
        std::cout << "Mean value: " << child.mean_value() << " - Prior: " << child.get_prior() <<" - Father visit count: " << node.get_visit_count()  << " - Child visit count:" << child.get_visit_count() << std::endl << std::flush;
        std::cout << "Sum value: " << child.get_value_sum() << std::endl << std::flush;
        std::cout << "Reward value: " << child.get_reward() << std::endl << std::flush;
        print_line(50);
        #endif

        if (cur_ucb_score > max_ucb_score) {
            max_ucb_score = cur_ucb_score;
            best_action = pair.first;
        }
    }

    #ifdef DEBUG
    std::cout << "Best action selected: " << best_action << ", Max ucb score: " << max_ucb_score << std::endl << std::flush;
    #endif

    return std::make_pair(best_action, children.at(best_action));
}


template<typename StateT,typename ModelConfig>
void SmartMapMCTS<StateT,ModelConfig>::backpropagate(std::vector<std::shared_ptr<Node<StateT>>>& traversed_path, double last_value, const double& discount, MinMaxStats& min_max_stats)
{
    #ifdef DEBUG
    print_line(50);
    std::cout << "\nStarting function: smartmap_backpropagation" << std::endl;
    std::cout << "Initial value: " << last_value << ", Discount factor: " << discount << std::endl;
    print_line(50);
    int i =0;
    #endif
    for (auto it = traversed_path.rbegin(); it != traversed_path.rend(); ++it) {
        auto& node = *it;
        #ifdef DEBUG
        std::cout << "Iteration " << i << std::endl;
        std::cout << "Node's value sum before update: " << node->get_value_sum() << std::endl;
        std::cout << "Node's visit count before update: " << node->get_visit_count() << std::endl;
        std::cout << "Node's reward: " << node->get_reward() << std::endl;
        #endif

        node->increment_value_sum(last_value);
        node->increment_visit_count();

        #ifdef DEBUG
        std::cout << "Node's value sum after update: " << node->get_value_sum() << std::endl;
        std::cout << "Node's visit count after update: " << node->get_visit_count() << std::endl;
        #endif

        min_max_stats.update(node->get_reward() + discount * node->mean_value());       


        last_value = node->get_reward() + discount * last_value;
        #ifdef DEBUG
        std::cout << "Updated last_value: " << last_value << std::endl;
        print_line(50);
        i+=1;
        #endif
    }

    #ifdef DEBUG
    std::cout << "Ending function: smartmap_backpropagation" << std::endl;
    print_line(50);
    #endif
}

