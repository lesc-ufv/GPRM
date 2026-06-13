#include "src/hpp/utils/util_torch.hpp"

template <typename T, typename TargetType>
torch::Tensor matrix_to_tensor(const std::vector<std::vector<T>>& matrix) {
    std::vector<TargetType> flattened_matrix;
    
    static const std::unordered_map<std::type_index, torch::ScalarType> type_to_torch_map = {
        {typeid(int), torch::kInt32},
        {typeid(long), torch::kInt64},
        {typeid(float), torch::kFloat32},
        {typeid(double), torch::kFloat64},
        {typeid(bool), torch::kBool},
        {typeid(uint8_t), torch::kByte},
        {typeid(int8_t), torch::kInt8},
        {typeid(int16_t), torch::kInt16},
        {typeid(uint16_t), torch::kShort},  
        {typeid(uint32_t), torch::kInt},    
        {typeid(uint64_t), torch::kLong},   
    };

    auto tensor_type = type_to_torch_map.at(typeid(TargetType));

    for (const auto& vec : matrix) {
        flattened_matrix.insert(flattened_matrix.end(), vec.begin(), vec.end());
    }

    auto tensor = torch::from_blob(
        flattened_matrix.data(),
        {static_cast<long>(matrix.size()), static_cast<long>(matrix[0].size())},
        tensor_type
    ).clone();

    return tensor;
}

template <typename T, typename TargetType>
torch::Tensor pad_matrix_to_tensor(const std::vector<std::vector<T>>& matrix, TargetType pad_value) {

    long unsigned int max_cols = 0;
    for (const auto& row : matrix) {
        if (row.size() > max_cols) {
            max_cols = row.size();
        }
    }

    std::unordered_map<std::type_index, torch::ScalarType> type_to_torch_map = {
        {typeid(float), torch::kFloat32},
        {typeid(double), torch::kFloat64},
        {typeid(int), torch::kInt32},
        {typeid(long), torch::kInt64},
        {typeid(bool), torch::kBool}
    };

    auto tensor_type = type_to_torch_map.at(typeid(TargetType));

    std::vector<torch::Tensor> padded_rows;
    for (const auto& row : matrix) {
        torch::Tensor padded_row = torch::tensor(row, torch::dtype(tensor_type));

        if (padded_row.size(0) < static_cast<long int>(max_cols)) {
            padded_row = torch::cat({padded_row, torch::full({static_cast<long int>(max_cols) - padded_row.size(0)}, pad_value, tensor_type)});
        }
        padded_rows.push_back(padded_row);
    }

    return torch::stack(padded_rows);
}


template <typename TargetType>
torch::Tensor tensor_vector_to_tensor_with_padding(const std::vector<torch::Tensor>& tensor_vector, TargetType pad_value) {
    #ifdef DEBUG
    std::cout << "\n=== Starting tensor_vector_to_tensor_with_padding ===" << std::endl;
    std::cout << "Input vector size: " << tensor_vector.size() << std::endl;
    #endif

    int max_size = 0;
    for (size_t i = 0; i < tensor_vector.size(); i++) {
        const auto& tensor = tensor_vector[i];
        if (tensor.dim() == 0) {
            throw std::runtime_error("Received scalar tensor instead of vector at index " + std::to_string(i));
        }
        max_size = std::max(max_size, static_cast<int>(tensor.size(1)));
        
        #ifdef DEBUG
        std::cout << "Tensor " << i << " shape: " << tensor.sizes() 
                 << ", dtype: " << tensor.dtype() 
                 << ", device: " << tensor.device() << std::endl;
        #endif
    }

    #ifdef DEBUG
    std::cout << "Max size: " << max_size << std::endl;
    #endif

    std::unordered_map<std::type_index, torch::ScalarType> type_to_torch_map = {
        {typeid(float), torch::kFloat32},
        {typeid(double), torch::kFloat64},
        {typeid(int), torch::kInt32},
        {typeid(int64_t), torch::kInt64},
        {typeid(bool), torch::kBool}
    };

    auto type_it = type_to_torch_map.find(typeid(TargetType));
    if (type_it == type_to_torch_map.end()) {
        std::string type_name = typeid(TargetType).name();
        throw std::runtime_error("Unsupported target type: " + type_name);
    }
    auto tensor_type = type_it->second;

    #ifdef DEBUG
    std::cout << "Target type: " << tensor_type << std::endl;
    #endif

    std::vector<torch::Tensor> padded_tensors;
    for (size_t i = 0; i < tensor_vector.size(); i++) {
        const auto& tensor = tensor_vector[i];
        
        #ifdef DEBUG
        std::cout << "\nProcessing tensor " << i << std::endl;
        std::cout << "Original shape: " << tensor.sizes() << std::endl;
        #endif

        torch::Tensor padded_tensor = tensor;

        if (padded_tensor.size(1) < max_size) {
            int pad_size = max_size - padded_tensor.size(1);
            auto padding = torch::full({padded_tensor.size(0),pad_size}, pad_value, tensor_type);
            
            #ifdef DEBUG
            std::cout << "Padding size: " << pad_size << std::endl;
            std::cout << "Padding tensor shape: " << padding.sizes() << std::endl;
            #endif
            
            if (padding.device() != tensor.device()) {
                padding = padding.to(tensor.device());
            }
            
            padded_tensor = torch::cat({padded_tensor, padding},1);
            
            #ifdef DEBUG
            std::cout << "Padded tensor shape: " << padded_tensor.sizes() << std::endl;
            #endif
        }

        if (padded_tensor.dtype() != tensor_type) {
            #ifdef DEBUG
            std::cout << "Converting tensor dtype from " << padded_tensor.dtype() 
                     << " to " << tensor_type << std::endl;
            #endif
            padded_tensor = padded_tensor.to(tensor_type);
        }

        padded_tensors.push_back(padded_tensor);
    }

    #ifdef DEBUG
    std::cout << "\nStacking " << padded_tensors.size() << " tensors" << std::endl;
    if (!padded_tensors.empty()) {
        std::cout << "First tensor shape: " << padded_tensors[0].sizes() 
                 << ", dtype: " << padded_tensors[0].dtype()
                 << ", device: " << padded_tensors[0].device() << std::endl;
    }
    #endif

    auto result = torch::cat(padded_tensors,0);
    
    #ifdef DEBUG
    std::cout << "Result shape: " << result.sizes() << std::endl;
    std::cout << "=== Finished tensor_vector_to_tensor_with_padding ===" << std::endl;
    #endif

    return result;
}


