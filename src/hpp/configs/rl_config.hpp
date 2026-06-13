#ifndef RL_CONFIG_HPP
#define RL_CONFIG_HPP

#include <iostream>
#include <sstream>

class RLConfig {
private:
    double discount;
    double lambda;
    bool override_base_state;
    bool use_mean_norm;
    bool use_min_0_max_1_norm;
    bool use_min_minus_1_max_1_norm;
    bool use_GAE;
    int td_steps;
    int unroll_steps;
    bool mask_invalid_PEs;

public:
    RLConfig()
        : discount(0.99),
          lambda(0.95),
          override_base_state(false),
          use_mean_norm(false),
          use_min_0_max_1_norm(false),
          use_min_minus_1_max_1_norm(false),
          use_GAE(true),
          td_steps(5),
          unroll_steps(5),
          mask_invalid_PEs(false) {}

    double getDiscount() const { return discount; }
    double getLambda() const { return lambda; }
    bool getOverrideBaseState() const { return override_base_state; }
    bool getUseMeanNorm() const { return use_mean_norm; }
    bool getUseMinMinus1Max1Norm() const { return use_min_minus_1_max_1_norm; }
    bool getUseMin0Max1Norm() const { return use_min_0_max_1_norm; }
    bool getUseGAE() const { return use_GAE; }
    int getTDSteps() const { return td_steps; }
    int getUnrollSteps() const { return unroll_steps; }
    bool getMaskInvalidPEs() const { return mask_invalid_PEs; }

    void setDiscount(double value) { discount = value; }
    void setLambda(double value) { lambda = value; }
    void setOverrideBaseState(bool value) { override_base_state = value; }
    void setUseMeanNorm(bool value) { use_mean_norm = value; }
    void setUseMinMinus1Max1Norm(bool value) { use_min_minus_1_max_1_norm = value; }
    void setUseMin0Max1Norm(bool value) { use_min_0_max_1_norm = value; }
    void setUseGAE(bool value) { use_GAE = value; }
    void setTDSteps(int value) { td_steps = value; }
    void setUnrollSteps(int value) { unroll_steps = value; }
    void setMaskInvalidPEs(bool value) { mask_invalid_PEs = value; }

    void print_ss(std::stringstream& ss) const {
        ss << "\nRL Configuration:\n"
           << "\tDiscount: " << discount << "\n"
           << "\tLambda: " << lambda << "\n"
           << "\tOverride Base State: " << (override_base_state ? "true" : "false") << "\n"
           << "\tUse GAE: " << (use_GAE ? "true" : "false") << "\n"
           << "\tTD Steps: " << td_steps << "\n"
           << "\tUnroll Steps: " << unroll_steps << "\n"
           << "\tUse Mean Normalization: " << (use_mean_norm ? "true" : "false") << "\n"
           << "\tUse Min 0 Max 1 Normalization: " << (use_min_0_max_1_norm ? "true" : "false") << "\n"
           << "\tUse Min -1 Max 1 Normalization: " << (use_min_minus_1_max_1_norm ? "true" : "false") << "\n"
           << "\tMask Invalid PEs: " << (mask_invalid_PEs ? "true" : "false") << "\n";
    }
};

#endif
