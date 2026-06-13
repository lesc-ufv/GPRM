#ifndef SMARTMAP_CONFIG_HPP
#define SMARTMAP_CONFIG_HPP

#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include "src/hpp/enums/enum_interconnection_styles.hpp"
#include "src/hpp/enums/enum_functionalities.hpp"
#include "src/hpp/utils/cgra/util_cgra.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
class SmartMapConfig {
private:
    int out_dim;
    int n_heads;   
    double negative_slope; 

public:
    static constexpr int dfg_in_feat_dim = 9;
    static constexpr int cgra_in_feat_dim = 7;

    SmartMapConfig();

    void set_config_from_json(const json& js){
        out_dim = js.at("out_dim").get<int>();
        n_heads = js.at("n_heads").get<int>();
        negative_slope = js.at("negative_slope").get<double>();
    };

    int getOutDim() const;
    int getNHeads() const;
    double getNegativeSlope() const {return negative_slope;};


    void setOutDim(int value);
    void setNegativeSlope(double value) {negative_slope = value;};
    void setNHeads(int value);

    void print() const;
    void print_ss(std::stringstream& ss) const;

};

#endif 
