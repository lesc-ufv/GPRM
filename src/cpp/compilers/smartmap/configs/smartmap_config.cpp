#include "src/hpp/compilers/smartmap/configs/smartmap_config.hpp"

SmartMapConfig::SmartMapConfig() {}

int SmartMapConfig::getOutDim() const { return out_dim; }
int SmartMapConfig::getNHeads() const { return n_heads; }


void SmartMapConfig::setOutDim(int value) { out_dim = value; }
void SmartMapConfig::setNHeads(int value) { n_heads = value; }


void SmartMapConfig::print() const {
    std::cout << "=================================" << std::endl;
    std::cout << "       SmartMap Configuration    " << std::endl;
    std::cout << "=================================" << std::endl;
    std::cout << "  Out Dimension: " << out_dim << std::endl;
    std::cout << "  Number of Heads: " << n_heads << std::endl;
    std::cout << "---------------------------------" << std::endl;

    std::cout << "=================================" << std::endl;
}

void SmartMapConfig::print_ss(std::stringstream& ss) const {
    ss << "=================================" << std::endl;
    ss << "       SmartMap Configuration    " << std::endl;
    ss << "=================================" << std::endl;
    ss << "  Out Dimension: " << out_dim << std::endl;
    ss << "  Number of Heads: " << n_heads << std::endl;
    ss << "---------------------------------" << std::endl;
    ss << "=================================" << std::endl;
}

