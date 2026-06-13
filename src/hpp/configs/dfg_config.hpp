#ifndef DFG_CONFIG_HPP
#define DFG_CONFIG_HPP

#include "src/hpp/enums/enum_placement_approach.hpp"
#include "src/hpp/enums/enum_greedy_type.hpp"
#include "src/hpp/enums/enum_scheduling.hpp"
#include <sstream>
#include <iostream>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
class DFGConfig {
private:
    EnumPlacementApproach placement_approach;
    EnumGreedyType greedy_type;
    EnumScheduling scheduling;
    int max_depth;
    bool start_traversal_from_output;
    int seed;
    double alpha_complexity, beta_complexity, gamma_complexity;
    bool generate_dfg_images = false;

public:
    DFGConfig()
        : placement_approach(EnumPlacementApproach::TOPOLOGICAL_ORDER),
          greedy_type(EnumGreedyType::GREEDY),
          scheduling(EnumScheduling::TIGHT),
          max_depth(0),
          start_traversal_from_output(false),
          seed(0), alpha_complexity(1), beta_complexity(1), gamma_complexity(1) {}

    bool getGenerateDFGImages() const {
        return generate_dfg_images;
    }

    void setGenerateDFGImages(bool value) {
        generate_dfg_images = value;
    }
    void set_config_from_json(const json& js) {
        placement_approach = static_cast<EnumPlacementApproach>(js["placement_id"].get<int>());
        greedy_type = static_cast<EnumGreedyType>(js["greedy_type_id"].get<int>());
        scheduling = static_cast<EnumScheduling>(js["scheduling_id"].get<int>());
        start_traversal_from_output = js["start_traversal_from_output"].get<bool>();
        max_depth = js["max_depth"].get<int>();
    }


    EnumPlacementApproach getPlacementApproach() const { return placement_approach; }
    EnumScheduling getScheduling() const { return scheduling; }
    EnumGreedyType getGreedyType() const { return greedy_type; }
    int getMaxDepth() const { return max_depth; }
    int getSeed() const { return seed; }
    bool getIsGreedy() const { return greedy_type != EnumGreedyType::DEFAULT; }
    bool getStartTraversalFromOutput() const { return start_traversal_from_output; }

    double getAlphaComplexity() const { return alpha_complexity; }
    void setAlphaComplexity(double value) { alpha_complexity = value; }
    double getBetaComplexity() const { return beta_complexity; }
    void setBetaComplexity(double value) { beta_complexity = value; }
    double getGammaComplexity() const { return gamma_complexity; }
    void setGammaComplexity(double value) { gamma_complexity= value; }

    void setPlacementApproach(EnumPlacementApproach value) { placement_approach = value; }
    void setScheduling(EnumScheduling value) { scheduling = value; }
    void setGreedyType(EnumGreedyType value) { greedy_type = value; }
    void setMaxDepth(int value) { max_depth = value; }
    void setSeed(int value) { seed = value; }
    void setStartTraversalFromOutput(bool value) { start_traversal_from_output = value; }

    void print_ss(std::stringstream& ss) const {
        ss << "\nDFG Configuration:\n"
           << "\tPlacement Approach: " << enumPlacementToString(placement_approach) << "\n"
           << "\tGreedy Type: " << enumToString(greedy_type) << "\n"
           << "\tScheduling: " << enumSchedulingToString(scheduling) << "\n"
           << "\tMax Depth: " << max_depth << "\n"
           << "\tStart Traversal From Output: " << (start_traversal_from_output ? "true" : "false") << "\n"
           << "\tSeed: " << seed << "\n"
           << "\tAlpha Complexity: " << alpha_complexity << "\n"
              << "\tBeta Complexity: " << beta_complexity << "\n"
              << "\tGamma Complexity: " << gamma_complexity << "\n";
    }

    void print() const {
        std::stringstream ss;
        print_ss(ss);
        std::cout << ss.str();
    }
    
};

#endif
