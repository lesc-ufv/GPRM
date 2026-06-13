#include "src/hpp/rl/node.hpp"

template <typename StateT>
Node<StateT>::Node(double prior)
    : prior(prior), visit_count(0), value_sum(0), reward(0), state(nullptr), action(-1) {}


template <typename StateT>
void Node<StateT>::set_prior(double prior){
    this->prior = prior;
}

template <typename StateT>
Node<StateT>::Node()
    : prior(0), visit_count(0), value_sum(0), reward(0), state(nullptr), action(-1) {}

template <typename StateT>
bool Node<StateT>::expanded() const {
    return !children.empty();
}

template <typename StateT>
double Node<StateT>::mean_value() const {
    return visit_count == 0 ? 0.0 : value_sum / visit_count;
}

template <typename StateT>
const std::unordered_map<int, std::shared_ptr<Node<StateT>>>& Node<StateT>::get_children() const {
    return children;
}

template <typename StateT>
const std::shared_ptr<Node<StateT>> Node<StateT>::get_child(const int& action){
    return this->children.at(action);
}


template <typename StateT>
void Node<StateT>::increment_value_sum(double value) {
    value_sum += value;
}

template <typename StateT>
void Node<StateT>::increment_visit_count() {
    ++visit_count;
}

template <typename StateT>
double Node<StateT>::get_reward() const {
    return reward;
}
template <typename StateT>
double Node<StateT>::get_prior(){
    return this->prior;
}
template <typename StateT>
int Node<StateT>::get_visit_count() const {
    return static_cast<int>(visit_count);
}

template <typename StateT>
std::shared_ptr<StateT> Node<StateT>::get_state() const {
    return state;
}

template <typename StateT>
const StateT& Node<StateT>::get_state_() const {
    return *state;
}

template <typename StateT>
void Node<StateT>::set_action(int action) {
    this->action = action;
}

template <typename StateT>
void Node<StateT>::print() const {
    std::cout << "Node Information:" << std::endl;
    std::cout << "  Prior: " << prior << std::endl;
    std::cout << "  Visit Count: " << visit_count << std::endl;
    std::cout << "  Value Sum: " << value_sum << std::endl;
    std::cout << "  Reward: " << reward << std::endl;
    std::cout << "  Action: " << action << std::endl;
    std::cout << "  Number of Children: " << children.size() << std::endl;
}

template <typename StateT>
const int& Node<StateT>::get_action() const {
    return action;
}

template <typename StateT>
bool Node<StateT>::is_end_state() {
    return state->get_is_end_state();
}

template <typename StateT>
const double& Node<StateT>::get_prior() const {
    return prior;
}

template <typename StateT>
double Node<StateT>::get_value_sum() const {
    return value_sum;
}

template<typename StateT>
void Node<StateT>::add_exploration_noise(const double &dirichlet_alpha, const double &exploration_fraction,  std::mt19937& gen)
{
    std::vector<int> actions;
    for (const auto& pair : children) {
        actions.push_back(pair.first);
    }

    std::vector<double> noise(actions.size());
    generateDirichletNoise(noise, dirichlet_alpha,  gen);

    #ifdef DEBUG
        std::cout << "Generated Dirichlet noise: ";
        for (const auto& n : noise) {
            std::cout << n << " ";
        }
        std::cout << std::endl;
    #endif

    double frac = exploration_fraction;
    for (size_t i = 0; i < actions.size(); ++i) {
        int action = actions[i];
        double n = noise[i];
        children[action]->set_prior(children[action]->get_prior() * (1.0 - frac) + n * frac);

        #ifdef DEBUG
            std::cout << "Updated prior for action " << action << " to " << children[action]->get_prior() << std::endl;
        #endif
    }
}

template<typename StateT>
void Node<StateT>::generateDirichletNoise(std::vector<double>& noise, double alpha, std::mt19937& gen) {

    std::gamma_distribution<> gamma(alpha, 1.0);

    for (auto& n : noise) {
        n = gamma(gen);
    }

    double sum = std::accumulate(noise.begin(), noise.end(), 0.0);
    for (auto& n : noise) {
        n /= sum;
    }

    #ifdef DEBUG
        std::cout << "Generated Dirichlet Noise: ";
        for (const auto& n : noise) {
            std::cout << n << " ";
        }
        std::cout << std::endl;
    #endif
}

template<typename StateT>
void Node<StateT>::expand(std::shared_ptr<StateT> state, const std::vector<int> &actions, double reward, 
torch::Tensor &policy_logits, const int &max_num_expansion)
{
    this->reward = reward;
    this->state = state;
    
        #ifdef DEBUG
        
        std::cout <<"Expanding Node \n\n" << "Reward: " << reward << std::endl;
        #endif

    if (!state->get_is_end_state() && !actions.empty()) {
        #ifdef DEBUG
            std::cout << "Expanding: State is not an end state, and actions are available." << std::endl;
        #endif

        torch::Tensor policy_probs = torch::softmax(policy_logits[0], 0);
        #ifdef DEBUG
            std::cout << "Policy Probabilities (first element): " << policy_probs << std::endl;
        #endif

        size_t max_num_children_to_expand = std::min(static_cast<size_t>(max_num_expansion), actions.size());
        #ifdef DEBUG
            std::cout << "Max number of children to expand: " << max_num_children_to_expand << std::endl;
        #endif

        torch::Tensor indexes, selected_policy_probs;
        policy_probs = torch::where(
            policy_logits[0] == -std::numeric_limits<float>::infinity(),
            torch::full_like(policy_probs, std::numeric_limits<float>::lowest()),
            policy_probs
        );
        std::tie(selected_policy_probs, indexes) = torch::topk(policy_probs, max_num_children_to_expand);
        #ifdef DEBUG
            std::cout << "Selected Policy Probabilities: " << selected_policy_probs << std::endl;
            std::cout << "Indexes of selected actions: " << indexes << std::endl;
        #endif

        children.reserve(max_num_children_to_expand);
        
        // int cur_modulo = actions[0]/state->get_number_of_spatial_PEs(); /// used only if first condition is true
        // int initial_pe = cur_modulo*state->get_number_of_spatial_PEs();
        
        for (size_t i = 0; i < max_num_children_to_expand; ++i) {
            int action_idx = indexes[i].item<int>();
            int action = state->get_action_by_action_idx(action_idx);
            // if(!state->model_is_general()){
            //     action = initial_pe + indexes[i].item<int>();
            // }else{
            //     action = actions[indexes[i].item<int>()];
            // }
            // int action_idx = state->get_action_idx_by_action(action);
            auto probability = policy_probs[action_idx].item<double>();
            assert(probability >= 0.0 && probability <= 1.0);
            
            #ifdef DEBUG
                std::cout << "Action: " << action << " - action_idx " << action_idx
                          << ", Probability: " << probability << std::endl;
            #endif

            auto node = std::make_shared<Node<StateT>>(probability);
            node->set_action(action);
            node->set_action_idx(action_idx);

            #ifdef DEBUG
                std::cout << "Created child node with action " << action 
                          << " and probability: " << node->get_prior() << std::endl;
            #endif

            children.emplace(action, std::move(node));
        }
    } else {
        #ifdef DEBUG
            std::cout << "No expansion, either end state or no actions available." << std::endl;
        #endif
    }
}

template<typename StateT>
int Node<StateT>::get_action_indice() const {
    return this->action_idx;
}

