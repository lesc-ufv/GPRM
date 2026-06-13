#include "src/hpp/compilers/mapzero/configs/mapzero_config.hpp"

MapZeroConfig::MapZeroConfig() 
    : out_dim(-1), n_heads(-1), len_action_space(-1), negative_slope(-1) {}

int MapZeroConfig::getOutDim() const { return out_dim; }
int MapZeroConfig::getNHeads() const { return n_heads; }
int MapZeroConfig::getLenActionSpace() const { return len_action_space; }
double MapZeroConfig::getNegativeSlope() const { return negative_slope; }

void MapZeroConfig::setOutDim(int value) { out_dim = value; }
void MapZeroConfig::setNHeads(int value) { n_heads = value; }
void MapZeroConfig::setLenActionSpace(int value) { len_action_space = value; }
void MapZeroConfig::setNegativeSlope(double value) { negative_slope = value; }


void MapZeroConfig::print_ss(std::stringstream& ss) const {
    ss << "\nMapZeroConfig:" << std::endl;
    ss << "\tOut Dim: " << out_dim << std::endl;
    ss << "\tNumber of Heads: " << n_heads << std::endl;
    ss << "\tLength of Action Space: " << len_action_space << std::endl;
    ss << "\tNegative Slope: " << negative_slope << std::endl;
    ss << "\tDFG Dim Feat (constant): " << dfg_feat_dim << std::endl;
    ss << "\tDim Feat CGRA (constant): " << cgra_feat_dim << std::endl;
}
