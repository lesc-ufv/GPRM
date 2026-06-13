#ifndef MCTS_LOSS_HPP
#define MCTS_LOSS_HPP
#include <torch/torch.h>
class MapZeroMCTSLoss : public torch::nn::Module {
public:
    MapZeroMCTSLoss() {};
    torch::Tensor forward(const torch::Tensor& predicted_logits,
                          const torch::Tensor&  target_policies) {
        auto mask = predicted_logits == -std::numeric_limits<double>::infinity();
        auto log_probs = torch::log_softmax(predicted_logits,-1);
        auto masked_log_probs = torch::where(mask, torch::tensor(0),log_probs);
        return target_policies * masked_log_probs;
        }
};


#endif