#ifndef UTIL_TORCH_HPP
#define UTIL_TORCH_HPP
#include <torch/torch.h>
#include <unordered_map>
#include <typeindex>

torch::Tensor mean_std_norm(torch::Tensor tensor, double target_mean, double target_std);
torch::Tensor min_max_norm(torch::Tensor tensor, double target_min, double target_max);

template <typename T, typename TargetType>
torch::Tensor matrix_to_tensor(const std::vector<std::vector<T>>& matrix);

template <typename T, typename TargetType>
torch::Tensor pad_matrix_to_tensor(const std::vector<std::vector<T>>& matrix, TargetType pad_value);

template <typename TargetType>
torch::Tensor tensor_vector_to_tensor_with_padding(const std::vector<torch::Tensor>& tensor_vector, TargetType pad_value);

torch::Tensor normalize_features_by_column(const torch::Tensor& input_tensor);

#include "src/tpp/utils/util_torch.tpp"

#endif