#ifndef MAPZERO_CONFIG_HPP
#define MAPZERO_CONFIG_HPP

#include <iostream>
#include <sstream>

using json = nlohmann::json;
class MapZeroConfig {
private:
    int out_dim;
    int n_heads;
    int len_action_space;
    double negative_slope;

public: 
    static constexpr int dfg_feat_dim = 10;
    static constexpr int cgra_feat_dim = 7;

    MapZeroConfig();

    void set_config_from_json(const json& js) {
        out_dim = js.at("out_dim").get<int>();
        n_heads = js.at("n_heads").get<int>();
        len_action_space = js.at("number_of_spatial_PEs").get<int>();
        negative_slope = js.at("negative_slope").get<double>();
    };

    int getOutDim() const;
    int getNHeads() const;
    int getLenActionSpace() const;
    double getNegativeSlope() const;

    void setOutDim(int value);
    void setNHeads(int value);
    void setLenActionSpace(int value);
    void setNegativeSlope(double value);

    void print_ss(std::stringstream& ss) const;

};

#endif 
