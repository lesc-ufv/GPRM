#include "src/hpp/rl/mapping_history.hpp"

template <typename StateT>
MappingHistory<StateT>::MappingHistory() {}

template <typename StateT>
MappingHistory<StateT>::MappingHistory(const MappingHistory<StateT>& other) 
: states(other.states), actions(other.actions), actions_indices(other.actions_indices),
  tree_depths(other.tree_depths), rewards(other.rewards),
  probabilities(other.probabilities), values(other.values), num_backtrackings(other.num_backtrackings), mapping_time(other.mapping_time),
  predicted_values_mcts(other.predicted_values_mcts) {}


template <typename StateT>
MappingHistory<StateT>::MappingHistory(MappingHistory<StateT>&& other) noexcept
  : states(std::move(other.states)), actions(std::move(other.actions)), 
    actions_indices(std::move(other.actions_indices)), tree_depths(std::move(other.tree_depths)), 
    rewards(std::move(other.rewards)), probabilities(std::move(other.probabilities)), 
    values(std::move(other.values)), num_backtrackings(std::move(other.num_backtrackings)),
    mapping_time(std::move(other.mapping_time)), predicted_values_mcts(std::move(other.predicted_values_mcts)) {}

template <typename StateT>
MappingHistory<StateT>& MappingHistory<StateT>::operator=(MappingHistory<StateT>&& other) noexcept {
    if (this != &other) { 
        states = std::move(other.states);
        actions = std::move(other.actions);
        actions_indices = std::move(other.actions_indices);
        tree_depths = std::move(other.tree_depths);
        rewards = std::move(other.rewards);
        probabilities = std::move(other.probabilities);
        values = std::move(other.values);
        num_backtrackings = std::move(other.num_backtrackings);
        mapping_time = std::move(other.mapping_time);
        predicted_values_mcts = std::move(other.predicted_values_mcts);
    }
    return *this;
}

template <typename StateT>
const std::vector<std::shared_ptr<StateT>>& MappingHistory<StateT>::get_states() const {
    return this->states;
}

template <typename StateT>
const std::vector<int>& MappingHistory<StateT>::get_actions() const {
    return this->actions;
}

template <typename StateT>
const std::vector<int>& MappingHistory<StateT>::get_actions_indices() const {
    return this->actions_indices;
}

template <typename StateT>
const std::vector<int>& MappingHistory<StateT>::get_tree_depths() const{
    return this->tree_depths;
}
template <typename StateT>
const std::vector<float>& MappingHistory<StateT>::get_rewards() const {
    return this->rewards;
}

template <typename StateT>
void MappingHistory<StateT>::set_rewards( std::vector<float> rewards)  {
    this->rewards = rewards;
}

template <typename StateT>
void MappingHistory<StateT>::set_values( std::vector<float> values) {
    this->values = values;
}

template <typename StateT>
void MappingHistory<StateT>::set_states( std::vector<std::shared_ptr<StateT>> states)  {
    this->states = states;
}

template <typename StateT>
void MappingHistory<StateT>::set_actions_indices( std::vector<int> actions) {
    this->actions_indices = actions;
}

template <typename StateT>
void MappingHistory<StateT>::set_policies( std::vector<std::vector<float>> policies)  {
    this->probabilities = policies;
}

template <typename StateT>
void MappingHistory<StateT>::set_actions( std::vector<int> actions) {
    this->actions = actions;
}


template <typename StateT>
const std::shared_ptr<StateT> MappingHistory<StateT>::get_state_by_idx(int idx) const{
    return this->states[idx];
}

template <typename StateT>
const std::vector<std::vector<float>>& MappingHistory<StateT>::get_child_visits() const {
    return this->probabilities;
}

template <typename StateT>
const std::vector<std::vector<float>>& MappingHistory<StateT>::get_probabilities() const {
    return this->probabilities;
}

template <typename StateT>
const std::vector<float>& MappingHistory<StateT>::get_values() const {
    return this->values;
}

template <typename StateT>
void MappingHistory<StateT>::append_state(const std::shared_ptr<StateT>& state) {
    states.push_back(state);
}

template <typename StateT>
void MappingHistory<StateT>::append_action(int action) {
    actions.push_back(action);
}

template <typename StateT>
void MappingHistory<StateT>::append_action_indice(int action_indice) {
    actions_indices.push_back(action_indice);
}

template <typename StateT>
void MappingHistory<StateT>::append_tree_depth(int tree_depth) {
    tree_depths.push_back(tree_depth);
}
template <typename StateT>
void MappingHistory<StateT>::append_reward(float reward) {
    rewards.push_back(reward);
}

template <typename StateT>
void MappingHistory<StateT>::append_child_visit(const std::vector<float>& visit) {
    probabilities.push_back(visit);
}

template <typename StateT>
void MappingHistory<StateT>::append_value(float value) {
    values.push_back(value);
}

template <typename StateT>
int MappingHistory<StateT>::get_episode_length() {
    return this->states.size();
}

template <typename StateT>
bool MappingHistory<StateT>::get_mapping_is_valid(){
    return this->states[this->states.size() - 1]->get_mapping_is_valid_();
}

template<typename StateT>
void MappingHistory<StateT>::store_search_statistics(const std::shared_ptr<Node<StateT>> root,  bool end_state){
    auto state = root->get_state();
    auto actions = state->get_legal_actions_();
    auto is_model_general = state->model_is_general(); 
    int num_actions_cur_state = actions.size();
    auto dims = state->get_cgra_dimensions_();
    auto cgra_spatial_size = dims.first * dims.second;
    std::vector<float> cur_probabilities(is_model_general ?  num_actions_cur_state :  cgra_spatial_size , 0.0);
    std::unordered_map<int,int> action_to_idx_logits;
    if (!end_state){
        float sum_visits = 0.0;
            auto& children = root->get_children();
            for (const auto& child : children ) {
                sum_visits += child.second->get_visit_count();
            }
            
            for (auto& pair: children) {
                int relative_action = state->get_action_idx_by_action(pair.first);
                cur_probabilities[relative_action] = pair.second->get_visit_count() / sum_visits;
            }
            this->values.push_back(root->mean_value());  
    }else{
        this->values.push_back(0.0);
    }

    // if(!end_state){
    //     assert(num_actions_cur_state != 0);
    // }

    this->probabilities.push_back(cur_probabilities);  
}


template<typename StateT>
void MappingHistory<StateT>::init_values(const std::shared_ptr<StateT>& initial_state){
    this->append_action(-1);
    this->append_action_indice(-1);
    this->append_state(initial_state);
    this->append_reward(std::numeric_limits<double>::infinity());

}

template<typename StateT>
void MappingHistory<StateT>::write_mapping(std::stringstream& ss) const {

    this->print_md(ss);
}

template<typename StateT>
int MappingHistory<StateT>::size() const {
    return this->states.size();
}

template <typename StateT>
void MappingHistory<StateT>::print() const {
    std::cout << "===== MappingHistory Debug Print =====" << std::endl;
    std::cout << "DFG Name: " << get_dfg_name() << std::endl;
    std::cout << "Arch Name: " << get_arch_name() << std::endl;
    std::cout << "Arch dims: " << get_arch_dims_str() << std::endl;
    std::cout << "II: " << get_II() << std::endl;
    std::cout << "Total states: " << states.size() << std::endl;
    for (size_t i = 0; i < states.size(); ++i) {
        std::cout << "State " << i << ": " << states[i] << std::endl;
    }

    std::cout << "Actions (" << actions.size() << "): ";
    for (const auto& action : actions) {
        std::cout << action << " ";
    }
    std::cout << std::endl;

    std::cout << "Action Indices (" << actions_indices.size() << "): ";
    for (const auto& idx : actions_indices) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    std::cout << "Tree Depths (" << tree_depths.size() << "): ";
    for (const auto& depth : tree_depths) {
        std::cout << depth << " ";
    }
    std::cout << std::endl;

    std::cout << "Rewards (" << rewards.size() << "): ";
    for (const auto& reward : rewards) {
        std::cout << reward << " ";
    }
    std::cout << std::endl;

    std::cout << "Probabilities (" << probabilities.size() << "):" << std::endl;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        std::cout << "Probabilities state " << i << ": ";
        for (const auto& visit : probabilities[i]) {
            std::cout << visit << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Values (" << values.size() << "): ";
    for (const auto& value : values) {
        std::cout << value << " ";
    }

    std::cout << "\nPredicted Values MCTS (" << predicted_values_mcts.size() << "): ";
    for (const auto& value : predicted_values_mcts) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    std::cout << "  Return: " << calc_sum_reward() << std::endl;
    std::cout << "======================================" << std::endl;
}


template <typename StateT>
void MappingHistory<StateT>::print(std::stringstream& os) const {
    os << "========== Mapping Info ===========" << std::endl;
    os << "DFG Name: " << get_dfg_name() << std::endl;
    os << "Arch Name: " << get_arch_name() << std::endl;
    os << "Arch dims: " << get_arch_dims_str() << std::endl;
    os << "II: " << get_II() << std::endl;
    os << "Actions (" << actions.size() << "): ";
    for (const auto& action : actions) {
        os << action << " ";
    }
    os << std::endl;

    os << "Action Indices (" << actions_indices.size() << "): ";
    for (const auto& idx : actions_indices) {
        os << idx << " ";
    }
    os << std::endl;

    os << "Tree Depths (" << tree_depths.size() << "): ";
    for (const auto& depth : tree_depths) {
        os << depth << " ";
    }
    os << std::endl;

    os << "Rewards (" << rewards.size() << "): ";
    for (const auto& reward : rewards) {
        os << reward << " ";
    }
    os << std::endl;

    os << "Probabilities (" << probabilities.size() << "):" << std::endl;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        os << "Probabilities state " << i << ": ";
        for (const auto& visit : probabilities[i]) {
            os << visit << " ";
        }
        os << std::endl;
    }

    os << "Values (" << values.size() << "): ";
    for (const auto& value : values) {
        os << value << " ";
    }
    os << std::endl;
    os << "  Return: " << calc_sum_reward() << std::endl;
    os << "========== End Mapping Info ===========" << std::endl;
}


