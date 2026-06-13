#ifndef MCTS_CONFIG_HPP
#define MCTS_CONFIG_HPP

#include <iostream>
#include <sstream>

class MCTSConfig {
private:
    int max_num_expansion;
    int num_simulations;
    double c1;
    double c2;
    bool add_exploration_noise;
    double exploration_factor;
    double dirichlet_alpha;
    bool restart_root;
    double uct_constant;
    double temperature;
    bool use_softmax_temperature;
    bool skip_simulation_for_one_action = false;
    bool return_optimal_mapping = false;
public:
    MCTSConfig()
        : max_num_expansion(0),
          num_simulations(0),
          c1(0.0),
          c2(0.0),
          add_exploration_noise(false),
          exploration_factor(0.0),
          dirichlet_alpha(0.0),
          restart_root(false),
          uct_constant(0.0),
          temperature(1.0),
          use_softmax_temperature(false) {}

    bool getReturnOptimalMapping() const { return return_optimal_mapping; }
    void setReturnOptimalMapping(bool value) { return_optimal_mapping = value; }
    bool getSkipSimulationsForOneAction() const { return skip_simulation_for_one_action; }
    void setSkipSimulationsForOneAction(bool value) { skip_simulation_for_one_action = value; }
    int getMaxNumExpansion() const { return max_num_expansion; }
    int getNumSimulations() const { return num_simulations; }
    double getC1() const { return c1; }
    double getC2() const { return c2; }
    bool getAddExplorationNoise() const { return add_exploration_noise; }
    double getExplorationFactor() const { return exploration_factor; }
    double getDirichletAlpha() const { return dirichlet_alpha; }
    bool getRestartRoot() const { return restart_root; }
    bool getUseSoftmaxTemperature() const { return use_softmax_temperature; }
    double getUCTConstant() const { return uct_constant; }
    double getTemperature() const { return temperature; }

    void setMaxNumExpansion(int value) { max_num_expansion = value; }
    void setNumSimulations(int value) { num_simulations = value; }
    void setC1(double value) { c1 = value; }
    void setC2(double value) { c2 = value; }
    void setAddExplorationNoise(bool value) { add_exploration_noise = value; }
    void setUseSoftmaxTemperature(bool value) { use_softmax_temperature = value; }
    void setExplorationFactor(double value) { exploration_factor = value; }
    void setDirichletAlpha(double value) { dirichlet_alpha = value; }
    void setRestartRoot(bool value) { restart_root = value; }
    void setUCTConstant(double value) { uct_constant = value; }
    void setTemperature(double value) { temperature = value; }

    void print_ss(std::stringstream& ss) const {
        ss << "\nMCTS Configuration:\n"
           << "\tMax Num Expansion: " << max_num_expansion << "\n"
           << "\tNum Simulations: " << num_simulations << "\n"
           << "\tC1: " << c1 << "\n"
           << "\tC2: " << c2 << "\n"
           << "\tAdd Exploration Noise: " << (add_exploration_noise ? "true" : "false") << "\n"
           << "\tExploration Factor: " << exploration_factor << "\n"
           << "\tDirichlet Alpha: " << dirichlet_alpha << "\n"
           << "\tRestart Root: " << (restart_root ? "true" : "false") << "\n"
           << "\tUCT Constant: " << uct_constant << "\n"
           << "\tTemperature: " << temperature << "\n"
           << "\tUse Softmax Temperature: " << (use_softmax_temperature ? "true" : "false") << "\n";
    }
};

#endif
