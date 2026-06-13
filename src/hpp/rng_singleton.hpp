#ifndef RNG_SINGLETON_HPP
#define RNG_SINGLETON_HPP
#include <random>
#include <iostream>

class RNG {
public:
    static void set_seed(int seed) {
        if (seed < 0) {
            std::cerr << "[ERROR] Seed must be non-negative. Using default seed 0." << std::endl;
            seed = 0;
        }
        std::cout << "[INFO] Setting RNG seed to: " << seed << std::endl;
        instance() = std::mt19937(seed);
    }

    static std::mt19937& get() {
        return instance();
    }

private:
    static std::mt19937& instance() {
        static std::mt19937 rng;  
        return rng;
    }
};

#endif 