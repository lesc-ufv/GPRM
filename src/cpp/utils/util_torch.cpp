#include "src/hpp/utils/util_torch.hpp"

torch::Tensor mean_std_norm(torch::Tensor tensor, double target_mean, double target_std) {
    assert (tensor.dim() == 1);
    auto current_mean = tensor.mean();
    auto current_std = tensor.std(false);

    const double eps = 1e-8;

    torch::Tensor normalized_tensor = ((tensor - current_mean) / (current_std + eps)) * target_std + target_mean;

    return normalized_tensor;
}

torch::Tensor min_max_norm(torch::Tensor tensor, double target_min, double target_max) {
    auto current_min = tensor.min();
    auto current_max = tensor.max();

    #ifdef DEBUG
    std::cout << "[DEBUG] Tensor minimum value: " << current_min.item<double>() << std::endl;
    std::cout << "[DEBUG] Tensor maximum value: " << current_max.item<double>() << std::endl;
    #endif

    if (current_max.item<double>() == current_min.item<double>()) {
        #ifdef DEBUG
        std::cout << "[DEBUG] The tensor has constant values, returning the original tensor." << std::endl;
        #endif
        return tensor;
    }

    torch::Tensor normalized_tensor = ((tensor - current_min) / (current_max - current_min)) * (target_max - target_min) + target_min;

    #ifdef DEBUG
    std::cout << "[DEBUG] Normalized tensor: " << normalized_tensor << std::endl;
    #endif

    return normalized_tensor;
}

torch::Tensor normalize_features_by_column(const torch::Tensor& input_tensor) {
    TORCH_CHECK(input_tensor.dim() == 2, "Input must be a 2D tensor.");

    auto mean = input_tensor.mean(0);
    auto std = input_tensor.std(0,false);

    const double eps = 1e-8;
    auto normalized = (input_tensor - mean) / (std + eps);

    return normalized;
}
