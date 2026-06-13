#ifndef PPO_CONFIG_HPP
#define PPO_CONFIG_HPP

#include <iostream>
#include <sstream>

class PPOConfig {
private:
    double clip_param;
    double value_loss_coef;
    double initial_entropy_coef;
    double target_entropy_coef;
    double kl_target;
    double kl_coefficient;
    double weight_decay_entropy;
    int step_to_start_decay_entropy;

public:
    PPOConfig()
        : clip_param(0.2),
          value_loss_coef(0.5),
          initial_entropy_coef(0.01),
          target_entropy_coef(0.01),
          kl_target(0.01),
          kl_coefficient(1.0),
          weight_decay_entropy(1.0),
          step_to_start_decay_entropy(0) {}

    double getClipParam() const { return clip_param; }
    double getValueLossCoef() const { return value_loss_coef; }
    double getInitialEntropyCoef() const { return initial_entropy_coef; }
    double getTargetEntropyCoef() const { return target_entropy_coef; }
    double getKlTarget() const { return kl_target; }
    double getKlCoefficient() const { return kl_coefficient; }
    double getWeightDecayEntropy() const { return weight_decay_entropy; }
    int getStepToStartDecayEntropy() const { return step_to_start_decay_entropy; }

    void setClipParam(double value) { clip_param = value; }
    void setValueLossCoef(double value) { value_loss_coef = value; }
    void setInitialEntropyCoef(double value) { initial_entropy_coef = value; }
    void setTargetEntropyCoef(double value) { target_entropy_coef = value; }
    void setKlTarget(double value) { kl_target = value; }
    void setKlCoefficient(double value) { kl_coefficient = value; }
    void setWeightDecayEntropy(double value) { weight_decay_entropy = value; }
    void setStepToStartDecayEntropy(int value) { step_to_start_decay_entropy = value; }

    void print_ss(std::stringstream& ss) const {
        ss << "\nPPO Configuration:\n"
           << "\tClip Param: " << clip_param << "\n"
           << "\tValue Loss Coefficient: " << value_loss_coef << "\n"
           << "\tInitial Entropy Coefficient: " << initial_entropy_coef << "\n"
           << "\tTarget Entropy Coefficient: " << target_entropy_coef << "\n"
           << "\tKL Target: " << kl_target << "\n"
           << "\tKL Coefficient: " << kl_coefficient << "\n";
        ss << "\tWeight Decay Entropy: " << weight_decay_entropy << "\n";
        ss << "\tStep to Start Decay Entropy: " << step_to_start_decay_entropy << "\n";
    }
};

#endif
