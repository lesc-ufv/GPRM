#ifndef NODE_HPP
#define NODE_HPP

#include <iostream>
#include <map>
#include <limits>
#include <unordered_map>
#include <optional>
#include <thread>
#include <mutex>
#include <memory>
#include <algorithm>
#include <execution>
#include <random>
#include <torch/torch.h>
#include "src/hpp/rng_singleton.hpp"

template <typename StateT>
class Node {
protected:    
    double prior;
    double visit_count;
    double value_sum;
    double reward;
    std::shared_ptr<StateT> state;
    std::unordered_map<int, std::shared_ptr<Node<StateT>>> children;
    int action, action_idx;

public:
    Node(double prior);  
    Node();
    virtual ~Node() = default;
    const int& get_action() const;
    bool expanded() const; 
    double mean_value() const;  
    const std::unordered_map<int, std::shared_ptr<Node<StateT>>>& get_children() const;
    // virtual void expand(std::shared_ptr<StateT> state, const std::vector<int>& actions, double reward,
    //                       torch::Tensor& policy_logits, const int& max_num_expansion) = 0;
    void increment_value_sum(double value);
    void increment_visit_count();
    double get_reward() const;  
    int get_visit_count() const; 
    std::shared_ptr<StateT> get_state() const; 
    const StateT& get_state_() const;
    void set_action(int action);
    void set_action_idx(int action_idx){
        this->action_idx = action_idx;
    }
    void print() const;
    const std::shared_ptr<Node<StateT>> get_child(const int& action);
    bool is_end_state();
    const double& get_prior() const ;
    double get_value_sum() const;
    void set_prior(double prior);
    double get_prior();
    void expand(std::shared_ptr<StateT> state, const std::vector<int> &actions, double reward, 
        torch::Tensor &policy_logits, const int &max_num_expansion);
    void generateDirichletNoise(std::vector<double>& noise, double alpha, std::mt19937& gen);
    void add_exploration_noise(const double &dirichlet_alpha, const double &exploration_fraction, std::mt19937& gen);
    int get_action_indice() const;
    
};

#include "src/tpp/rl/node.tpp"

#endif
