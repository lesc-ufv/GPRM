#include "src/hpp/rl/policy_logits_self_mapping.hpp"

template<typename StateT, typename ModelConfig>
MappingHistory<StateT> map_with_policy_logits( GlobalConfig<ModelConfig>& config, std::shared_ptr<zmq::socket_t> socket,
    const ServerConstants& server_k, const std::string& func_name,const std::shared_ptr<StateT> initial_state, 
    Environment<StateT>& environment, bool train,  double train_ratio, int seed,
    std::optional<std::unordered_map<std::string,int>> node_to_action,
    std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester)
{
    torch::Generator gen = torch::make_generator<torch::CPUGeneratorImpl>(seed);
    
    MappingHistory<StateT> mapping_history;
    mapping_history.start();
    mapping_history.init_values(initial_state);
    bool end_mapping = false;
    auto cur_state = initial_state;
    auto mapping_config = config.getMappingConfig();
    auto use_backtracking = mapping_config.getUseBacktracking();

    #ifdef DEBUG
        std::cout << "Initial state: \n";
        cur_state->print();
    #endif
    while (!end_mapping){
        auto [policy_logits, value] = infer(socket, server_k, batch_requester, func_name, cur_state);

        mapping_history.append_value(value.item().toDouble());
        
        #ifdef DEBUG
            std::cout << "Policy logits: " << policy_logits << std::endl;
            std::cout << "Value: " << value.item().toDouble() << std::endl;
        #endif

        torch::Tensor probabilities = torch::softmax(policy_logits, -1);
        int action, action_indice;
        std::shared_ptr<StateT> next_state;
        double reward;
        bool is_final_state;
        auto cur_node_to_be_mapped_name = cur_state->get_node_to_be_mapped_name();
        if(node_to_action.has_value() && node_to_action->find(cur_node_to_be_mapped_name) != node_to_action->end()){
            action = node_to_action->at(cur_node_to_be_mapped_name);
            action_indice = cur_state->get_action_idx_by_action(action);
            std::tie(next_state, reward, is_final_state) = std::move(environment.step(*cur_state, action));
            std::cout << "[INFO] Using pre-defined action for node: " << cur_node_to_be_mapped_name 
                          << ", action index: " << action_indice << std::endl;
        }else{
            if(!use_backtracking){
                std::tie(action, action_indice) = std::move(get_action_and_action_indice<StateT>(cur_state, probabilities, train, gen));
                std::tie(next_state, reward, is_final_state) = std::move(environment.step(*cur_state, action));
            }else{
                int num_backtrackings = 0;
                int num_valid_probabilities = cur_state->get_legal_actions_size();
                auto [probs, top_k_indices] = torch::topk(probabilities, num_valid_probabilities);
                top_k_indices = top_k_indices.squeeze(0);
                for(int i = 0; i < num_valid_probabilities; i++){
                    action = cur_state->get_action_by_action_idx(top_k_indices[i].item().toInt());
                    action_indice = top_k_indices[i].item().toInt();
                    auto data_start = std::chrono::steady_clock::now();

                std::tie(next_state, reward, is_final_state) = std::move(environment.step(*cur_state, action));
                auto data_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> data_duration = data_end - data_start;
                std::cout << "[INFO] Environment step " << data_duration.count() << " seconds." << std::endl;
                
                    if(next_state->do_backtracking()){
                        num_backtrackings++;
                        continue;
                    }else{
                        break;
                    }
                }
                mapping_history.append_num_backtrackings(num_backtrackings);
            }
    }
        
        #ifdef DEBUG
            std::cout << "Probabilities: " << probabilities << std::endl;
            std::cout << "Action index: " << action_indice << std::endl;
        #endif
        
        mapping_history.append_action_indice(action_indice);
        mapping_history.append_action(action);
        mapping_history.append_state(next_state);
    
        mapping_history.append_reward(reward);
        auto* ptr = probabilities[0].data_ptr<float>();
        auto probs = std::vector<float>(ptr, ptr + probabilities[0].numel());
        mapping_history.append_probabilities(probs);
        
        #ifdef DEBUG
            std::cout << "Next state: \n";
            next_state->print();
            std::cout << "Reward: " << reward << ", is final state: " << is_final_state << std::endl;
            std::stringstream ss;
            // next_state->draw_mapping(ss);
            std::cout << "Mapping: \n" << ss.str() << std::endl;
        #endif
        
        cur_state = next_state;
        end_mapping = is_final_state;
    }
    mapping_history.stop();
    mapping_history.append_probabilities({0});
    mapping_history.append_value(0.0);
    return std::move(mapping_history);
}

template<typename StateT>    
std::tuple<int, int> 
get_action_and_action_indice(const std::shared_ptr<StateT> cur_state, torch::Tensor policy_probs, bool train, torch::Generator& gen)
{
    torch::Tensor action_idx;
    
    #ifdef DEBUG
    auto& legal_actions = cur_state->get_legal_actions_();
    std::cout << "Legal actions: " << legal_actions << std::endl;
    std::cout << "Policy probabilities: " << policy_probs << std::endl;
    std::cout << "Train: " <<( train ? "Yes" : "No") << std::endl;
    #endif
    
    if (train){
        // try
        // {
            action_idx = torch::multinomial(policy_probs, 1, false, gen);
        // }
        // catch(const std::exception& e)
        // // {
        //     std::stringstream ss;
        //     cur_state->print_base_state(ss);
            
        //     std::cerr << e.what() << "\n\n"<< ss.str()  ;
        //     throw std::runtime_error("Error");
        // }
        
    }else{
        action_idx = torch::argmax(policy_probs);
    }
    
    auto int_action_idx = action_idx.item().toInt();
    auto action = cur_state->get_action_by_action_idx(int_action_idx);
    #ifdef DEBUG
        std::cout << "Chosen action: " << action << ", action index: " << int_action_idx << std::endl;
    #endif

    return std::make_tuple(action, int_action_idx);
}

template<typename StateT, typename ModelConfig>    
void reanalyze_with_policy_logits(GlobalConfig<ModelConfig>& config, MappingHistory<StateT>& mapping_history, std::shared_ptr<zmq::socket_t> socket,
    const ServerConstants& server_k, const std::string& func_name, std::shared_ptr<StateT> state, Environment<StateT>& environment,
    std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester){
    auto model_inputs = state->get_model_inputs(torch::kCPU);
    auto [policy_logits, value] = infer(socket, server_k, batch_requester, func_name, state);
    mapping_history.append_value(value.item().toDouble());
    torch::Tensor probabilities = torch::softmax(policy_logits, -1);
    auto* ptr = probabilities[0].data_ptr<float>();
    auto probs = std::vector<float>(ptr, ptr + probabilities[0].numel());
    mapping_history.append_probabilities(probs);

}