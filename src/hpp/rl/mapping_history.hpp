#ifndef MAPPING_HISTORY_HPP
#define MAPPING_HISTORY_HPP

#include <iostream>
#include <vector>
#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"
#include <sstream>
#include "src/hpp/utils/util_print.hpp"
#include "src/hpp/rl/node.hpp"
#include <time.h>
template <typename StateT>
class MappingHistory{
protected:
    std::vector<std::shared_ptr<StateT>> states;
    std::vector<int> actions;
    std::vector<int> actions_indices;
    std::vector<int> tree_depths;
    std::vector<float> rewards;
    std::vector<std::vector<float>> probabilities;
    std::vector<float> values;
    std::vector<int> num_backtrackings;
    std::chrono::steady_clock::time_point start_time;
    double mapping_time = 0.0;
    std::vector<float> predicted_values_mcts;

public:
    MappingHistory();

    bool is_mcts_data() const {
        std::cout << "Predicted values MCTS size: " << predicted_values_mcts.size() << std::endl;
        return !predicted_values_mcts.empty();
    }

    void add_predicted_value_mcts(double value){
        this->predicted_values_mcts.push_back(value);
    }
    const std::vector<float>& get_predicted_values_mcts() const {
        return this->predicted_values_mcts;
    }
    void start() {
        start_time = std::chrono::steady_clock::now();
    }

    void stop() {
        auto end_time = std::chrono::steady_clock::now();
        mapping_time = std::chrono::duration<double>(end_time - start_time).count();
    }

    std::shared_ptr<StateT> get_last_state() const {
        if (states.empty()) {
            return nullptr;
        }
        return states.back();
    }

    std::shared_ptr<StateT> get_first_state() const {
        if (states.empty()) {
            return nullptr;
        }
        return states.front();
    }

    double get_mapping_time() const {
        return mapping_time;
    }
    std::string get_dfg_name() const {
        return this->states[0]->get_dfg_name();
    }
    std::string get_arch_name() const {
        return this->states[0]->get_arch_name();
    }

    std::string get_arch_dims_str() const {
        return this->states[0]->get_str_dims();
    }

    int get_II() const {
        return this->states[0]->get_II();
    }

    double calc_mean_tree_depth() const {
        if(tree_depths.size() == 0) {
            return 0.0;
        }
        double sum = 0;
        
        for (int i = 1; i < static_cast<int>(tree_depths.size()); i++) {
            sum += tree_depths[i];
        }

        return  sum / static_cast<double>(tree_depths.size());
    }
    double calc_pct_mapped_nodes() const {
        return static_cast<double>(this->states[this->states.size() - 1]->get_num_mapped_nodes()) / static_cast<double>(this->states[0]->get_num_dfg_nodes());
    }
    double calc_pct_mapped_edges() const {
        return static_cast<double>(this->states[this->states.size() - 1]->get_num_mapped_edges()) / static_cast<double>(this->states[0]->get_num_dfg_edges());
    }
    double calc_pct_valid_node_convergences() const {
        return static_cast<double>(this->states[this->states.size() - 1]->get_num_valid_convergent_nodes()) / static_cast<double>(this->states[0]->get_num_convergent_nodes());
    }
    double calc_pct_valid_edge_convergences() const {
        return static_cast<double>(this->states[this->states.size() - 1]->get_num_valid_convergent_edges()) / static_cast<double>(this->states[0]->get_num_convergent_edges());
    }
    int get_total_backtrackings() const {
        int total = 0;
        for (int i = 0; i < static_cast<int>(num_backtrackings.size()); i++) {
            total += num_backtrackings[i];
        }
        return total;
    }


    MappingHistory(MappingHistory<StateT>&& other) noexcept;
    MappingHistory(const MappingHistory<StateT>& other);
    MappingHistory<StateT>& operator=(MappingHistory<StateT>&& other) noexcept;
    void set_mapping_time(double time) {
        this->mapping_time = time;
    }
    double calc_sum_reward() const {
        double sum = 0;
        for (int i = 1; i  < static_cast<int>(rewards.size());  i++) {
            sum += rewards[i];
        }
        return sum;
    }
    void init_values(const std::shared_ptr<StateT>& initial_state);
    const std::vector<std::shared_ptr<StateT>> & get_states() const;
    const std::vector<int>& get_actions() const;
    const std::vector<int>& get_actions_indices() const;
    void set_skip_initial_state(bool value){
        this->states[0]->set_skip_state(value);
    }
    
    const std::vector<int>& get_tree_depths() const;
    const std::vector<float>& get_rewards() const;
    const std::vector<std::vector<float>>& get_child_visits() const;
    const std::vector<std::vector<float>>& get_probabilities() const;

    const std::vector<float>& get_values() const;    
    void append_state(const std::shared_ptr<StateT> & state);
    void append_action(int action);
    void append_num_backtrackings(int num_backtrackings){
        this->num_backtrackings.push_back(num_backtrackings);
    };

    void append_action_indice(int action_indice);

    void append_tree_depth(int tree_depth);
    void append_reward(float reward);
    void append_child_visit(const std::vector<float>& visit);
    void append_probabilities(const std::vector<float>& probs) { 
        this->probabilities.push_back(probs); 
    };

    void append_value(float value);
    int get_episode_length();
    bool get_mapping_is_valid();
    void store_search_statistics(const std::shared_ptr<Node<StateT>> root, bool end_state);
    void set_rewards( std::vector<float> rewards);
    void set_values( std::vector<float> values);
    const std::shared_ptr<StateT> get_state_by_idx(int idx) const ;
    void write_mapping(std::stringstream& ss) const;
    int size() const;
    
    void set_states( std::vector<std::shared_ptr<StateT>> states);
    
    void set_actions_indices( std::vector<int> actions);
    
    void set_policies( std::vector<std::vector<float>> policies);
    
    void set_actions( std::vector<int> actions);

    void print() const;

    int get_num_dfg_nodes() const{
        return this->states[0]->get_num_dfg_nodes();
    }
    int get_num_dfg_edges() const{
        return this->states[0]->get_num_dfg_edges();
    }
    int get_num_mapped_dfg_nodes() const{
        return this->states[this->states.size() - 1]->get_num_mapped_nodes();
    }

    int get_num_mapped_dfg_edges() const{
        return this->states[this->states.size() - 1]->get_num_mapped_edges();
    }

    int get_num_dfg_node_convergences() const{
        return this->states[0]->get_num_convergent_nodes();
    }

    int get_num_dfg_edge_convergences() const{
        return this->states[0]->get_num_convergent_edges();
    }
    int get_num_valid_dfg_node_convergences() const{
        return this->states[this->states.size() - 1]->get_num_valid_convergent_nodes();
    }
    int get_num_valid_dfg_edge_convergences() const{
        return this->states[this->states.size() - 1]->get_num_valid_convergent_edges();
    }   
 
    int get_total_routing_nodes() const {
        return this->states[this->states.size() - 1]->get_num_routing_nodes();
    }   
    void print(std::stringstream& os) const;

    std::string get_invalid_mapping_reason() const {
        return this->states[this->states.size() - 1]->get_invalid_mapping_reason_();
    }
    void print_md(std::stringstream& os) const {
        auto final_state = this->states[this->states.size() - 1];

        os << "# Mapping\n\n";
        os << "## DFG Information\n\n";
        this->states[0]->print_dfg_info_md(os);
        os << "\n## CGRA Information\n\n";
        this->states[0]->print_cgra_info_md(os);

        os << "\n## Final Mapping Information\n\n";
        final_state->write_final_mapping_information_md(os);
        final_state->draw_mapping_md(os, true);
        os << "### RL Summary\n";
        os << "| Step | Action | Action Index | Reward | Value (v(s)) | Probabilities (π(s)) | Num Backtrackings |\n";
        os << "|------|--------|---------------|--------|--------|---------------|-------------------|\n";
    
        for (size_t i = 0; i < actions_indices.size(); ++i) {
            os << "| " << i;
            os << " | " << actions[i];
            os << " | " << actions_indices[i];
            os << " | " << rewards[i] << " | " << values[i] << " | ";
    
            if (i == actions_indices.size() - 1) {
                os << "-";
            } else {
                for (size_t j = 0; j < probabilities[i].size(); ++j) {
                    if (j == static_cast<size_t>(actions_indices[i + 1])) {
                        os << "**" << probabilities[i][j] << "** ";
                    } else {
                        os << probabilities[i][j] << " ";
                    }
                }
            }
    
            os << " | ";
            if (!num_backtrackings.empty() && i < num_backtrackings.size()) {
                os << num_backtrackings[i];
            } else {
                os << "-";
            }
    
            os << " |\n";
        }

    
        os << "\n### Return: **" << calc_sum_reward() << "**\n";
    
        os << "\n### Mapping Statistics\n";
        os << "- DFG: **" << get_dfg_name() << "**\n";
        os << "- Architecture: **" << get_arch_name() << "** (" << get_arch_dims_str() << ")\n";
        os << "- Initiation Interval (II): **" << get_II() << "**\n";
        os << "- Mapping Time: **" << get_mapping_time() << "s**\n";
        os << "- Total Routing Nodes: **" << get_total_routing_nodes() << "**\n";
        os << "- Mean Tree Depth: **" << calc_mean_tree_depth() << "**\n";
    
        os << "\n### Mapping Quality\n";
        os << "- Mapped Nodes: **" << get_num_mapped_dfg_nodes() << "/" << get_num_dfg_nodes()
           << "** (" << calc_pct_mapped_nodes() * 100 << "%)\n";
        os << "- Mapped Edges: **" << get_num_mapped_dfg_edges() << "/" << get_num_dfg_edges()
           << "** (" << calc_pct_mapped_edges() * 100 << "%)\n";
        os << "- Valid Node Convergences: **" << get_num_valid_dfg_node_convergences() << "/" << get_num_dfg_node_convergences()
           << "** (" << calc_pct_valid_node_convergences() * 100 << "%)\n";
        os << "- Valid Edge Convergences: **" << get_num_valid_dfg_edge_convergences() << "/" << get_num_dfg_edge_convergences()
           << "** (" << calc_pct_valid_edge_convergences() * 100 << "%)\n";
    
        os << "\n---\n";


        os << "\n## Mapping Information Over Time\n\n";

        for (int i = 0; i < static_cast<int>(this->states.size() - 1); i++) {
            os << "### State " << i << "\n";
            os << "**Node to place:** " << this->states[i]->get_node_to_be_mapped_name() << "\n\n";
            if(static_cast<int>(this->tree_depths.size()) > i)
                os << "**Tree Depth**: " << this->tree_depths[i] << "\n\n";
            os << "**Expected Return**: " << this->values[i] << "\n\n";
            os << "**Reward**: " << this->rewards[i+1] << "\n\n";
            // os << "**Action Probabilities:**\n\n";
            // os << "| Action | Action Index | Probability |\n";
            // os << "|--------|--------------|-------------|\n";
        
            auto cur_state = this->states[i];
            // auto& legal_actions = cur_state->get_legal_actions_();
        
            // for (size_t j = 0; j < legal_actions.size(); ++j) {
            //     auto cur_action_idx = cur_state->get_action_idx_by_action(legal_actions[j]);
        
            //     os << "| " << legal_actions[j] << " | " << cur_action_idx << " | ";
        
                
            //     if (j == static_cast<size_t>(actions_indices[i + 1])) {
            //         os << "**" << probabilities[i][j] << "** ";
            //     } else {
            //         os << probabilities[i][j] << " ";
            //     }
                
        
            //     os << " |\n";
            // }
        
            os << "\n";
            cur_state->draw_mapping_md(os, false, &probabilities[i]);        
        }
        
 
    }

    int get_MII() const {
        return this->states[0]->get_MII();
    }
    
};
    
#include "src/tpp/rl/mapping_history.tpp"
#endif