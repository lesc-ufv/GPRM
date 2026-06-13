#ifndef POLICY_GRADIENT_LOSS_HPP
#define POLICY_GRADIENT_LOSS_HPP
#include <torch/torch.h>

class PolicyGradientLoss : public torch::nn::Module {
public:
    PolicyGradientLoss() {};

    torch::Tensor forward(const torch::Tensor& predicted_logits,
                          const torch::Tensor& actions,
                          const torch::Tensor& advantages) {
        auto probs = torch::softmax(predicted_logits,-1);
        auto batch_indices = torch::arange(actions.size(0));
        auto actions_probs = probs.index({batch_indices, actions});
        return actions_probs.log()*advantages;
    }
};


#endif