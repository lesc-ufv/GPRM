template<typename StateT, typename ModelConfig>
MappingHistory<StateT> map_with_beam_search(
        GlobalConfig<ModelConfig>& config, std::shared_ptr<zmq::socket_t> socket,
        const ServerConstants& server_k, const std::string& func_name, 
        const std::shared_ptr<StateT> initial_state, 
        Environment<StateT>& environment, bool train, double train_ratio, int seed,
        std::optional<std::unordered_map<std::string,int>> node_to_action = std::nullopt,
        std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester = std::nullopt)
{
    torch::Generator gen = torch::make_generator<torch::CPUGeneratorImpl>(seed);
    
    struct BeamEntry {
        std::shared_ptr<StateT> state;
        std::shared_ptr<BeamEntry> parent;
        int action = -1;
        int action_idx = -1;
        double score = 0.0; 
        double reward = 0.0;
        double value = 0.0;
        std::vector<float> probabilities;
        bool finished = false;
    };
    
    auto beam_width = config.getTrainConfig().getBeamWidth();
    auto return_valid_mapping = config.getMappingConfig().getFirstValidSolution();
    
    auto root = std::make_shared<BeamEntry>();
    root->state = initial_state;

    std::vector<std::shared_ptr<BeamEntry>> beam = {root};
    
    bool found_valid = false;
    std::shared_ptr<BeamEntry> first_valid_entry = nullptr;

    MappingHistory<StateT> mapping_history;
    mapping_history.start();

    while (true) {
        std::vector<std::shared_ptr<BeamEntry>> new_beam;

        for (std::size_t j = 0; j < beam.size(); ++j) {

            if (return_valid_mapping && found_valid)
                break;

            auto& entry = beam[j];

            if (entry->finished) {
                new_beam.push_back(entry);
                continue;
            }

            auto [policy_logits, value] =
                infer(socket, server_k, batch_requester, func_name, entry->state);

            torch::Tensor probabilities = torch::softmax(policy_logits, -1);
            auto* ptr = probabilities[0].data_ptr<float>();
            entry->probabilities =
                    std::vector<float>(ptr, ptr + probabilities[0].numel());
            entry->value = value.squeeze().template item<double>();
            std::vector<int> candidate_indices;

            torch::Tensor top_probs, top_indices;
            std::tie(top_probs, top_indices) =
                torch::topk(policy_logits,
                    std::min(beam_width,
                        std::min((int)(policy_logits.size(-1)),
                                 entry->state->get_legal_actions_size())));

            top_indices = top_indices.squeeze(0);
            candidate_indices.resize(top_indices.size(0));

            for (int64_t i = 0; i < top_indices.size(0); ++i)
                candidate_indices[i] = top_indices[i].template item<int>();

            for (std::size_t i = 0; i < candidate_indices.size(); ++i) {

                if (return_valid_mapping && found_valid)
                    break;

                int idx = candidate_indices[i];
                int action = entry->state->get_action_by_action_idx(idx);

                auto [next_state, reward, is_final] =
                    environment.step(*entry->state, action);

                auto child = std::make_shared<BeamEntry>();
                child->state = next_state;
                child->parent = entry;
                child->action = action;
                child->action_idx = idx;
                child->finished = is_final;
                child->reward = reward;
            

                if (child->state->get_is_end_state() &&
                    child->state->get_mapping_is_valid_())
                {
                    child->score = 100000 - child->state->calc_num_rout_nodes();
                    if (return_valid_mapping) {
                        found_valid = true;
                        first_valid_entry = child;
                        first_valid_entry->value = 0.;
                        first_valid_entry->probabilities = {0.};
                        break;
                    }

                }
                else {
                    child->score = entry->score + std::log(entry->probabilities[idx] + 1e-9);
                }

                new_beam.push_back(child);
            }
        }

        if (return_valid_mapping && found_valid) {
            std::vector<std::shared_ptr<BeamEntry>> path;

            for (auto cur = first_valid_entry; cur; cur = cur->parent)
                path.push_back(cur);

            std::reverse(path.begin(), path.end());

            mapping_history.init_values(initial_state);
            mapping_history.append_value(path[0]->value);
            mapping_history.append_probabilities(path[0]->probabilities);
            for (size_t i = 1; i < path.size(); i++) {
                auto& node = path[i];
                mapping_history.append_action(node->action);
                mapping_history.append_action_indice(node->action_idx);
                mapping_history.append_state(node->state);
                mapping_history.append_reward(node->reward);
                mapping_history.append_value(node->value);
                mapping_history.append_probabilities(node->probabilities);
            }

            
            mapping_history.stop();

            return mapping_history;
        }

        std::sort(new_beam.begin(), new_beam.end(),
                  [](const auto& a, const auto& b) {
                      return a->score > b->score;
                  });

        if ((int)new_beam.size() > beam_width)
            new_beam.resize(beam_width);

        beam = std::move(new_beam);

        bool all_finished = std::all_of(
            beam.begin(), beam.end(),
            [](auto& e){ return e->finished; });

        if (all_finished)
            break;
    }

    auto best = *std::max_element(
        beam.begin(), beam.end(),
        [](const auto& a, const auto& b){
            return a->score < b->score;
        });

    std::vector<std::shared_ptr<BeamEntry>> path;
    for (auto cur = best; cur; cur = cur->parent)
        path.push_back(cur);

    std::reverse(path.begin(), path.end());

    mapping_history.init_values(initial_state);
    mapping_history.append_value(path[0]->value);
    mapping_history.append_probabilities(path[0]->probabilities);

    for (size_t i = 1; i < path.size(); i++) {
        auto& node = path[i];
        mapping_history.append_action(node->action);
        mapping_history.append_action_indice(node->action_idx);
        mapping_history.append_state(node->state);
        mapping_history.append_reward(node->reward);
        mapping_history.append_value(node->value);
        mapping_history.append_probabilities(node->probabilities);
    }

    
    mapping_history.stop();

    return mapping_history;
}
