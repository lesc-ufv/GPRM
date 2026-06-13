#include "src/hpp/utils/util_mcts.hpp"

template<typename StateT>
int select_action_with_backtracking(const std::shared_ptr<Node<StateT>> root, Environment<StateT>& environment,
     std::shared_ptr<StateT>& cur_state, int& action, int& action_indice, double& reward, bool& done ){
    auto& root_children = root->get_children();
    auto child_node = root_children.at(action);
    action_indice = cur_state->get_action_idx_by_action(action);
    cur_state = child_node->get_state();
    reward = child_node->get_reward();
    if (!cur_state->do_backtracking()) {
        return 0;
    }

    int count = 1;
    auto num_children = root_children.size();
    std::vector<std::pair<int, int>> action_visit_pairs;
    action_visit_pairs.reserve(num_children);
    for (const auto& [action, child] : root_children) {
        action_visit_pairs.emplace_back(action, child->get_visit_count());
    }

    std::sort(action_visit_pairs.begin(), action_visit_pairs.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second > b.second;  
            });

    auto root_state = root->get_state();

    while (count < static_cast<int>(num_children)) {
        action = action_visit_pairs[count].first;
        child_node = root_children.at(action);
        if (!child_node->expanded()){
            std::tie(cur_state, reward, done) = std::move(environment.step(*root_state, action));
        }else{
            cur_state = child_node->get_state();
            reward = child_node->get_reward();
            done = cur_state->get_is_end_state();
        }
        #ifdef DEBUG
            std::cout << "Action: " << action << ", Reward: " << reward 
                    << ", Done: " << done << ", Backtrackings: " << count << std::endl;
        #endif

        if (!root_state->do_backtracking()) {
            break;
        }
        count++;
    }

    return count;
}