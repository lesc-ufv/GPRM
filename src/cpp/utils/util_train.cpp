#include "src/hpp/utils/util_train.hpp"

torch::Tensor create_batch_edge_indices(const int& max_nodes, const std::vector<torch::Tensor>& edge_indices) {
    int total_edges = 0;

    for (const auto& edges : edge_indices) {
        total_edges += edges.size(1); 
    }

    torch::Tensor batched_edge_indices = torch::empty({2, total_edges}, torch::kInt64);

    int current_offset = 0; 
    int cum_sum = 0;
    for (size_t i = 0; i < edge_indices.size(); ++i) {
        const auto& edges = edge_indices[i];
        int num_edges = edges.size(1);
        batched_edge_indices[0].slice(0, current_offset, current_offset + num_edges) = edges[0] + cum_sum; 

        batched_edge_indices[1].slice(0, current_offset, current_offset + num_edges) = edges[1] + cum_sum;

        current_offset += num_edges;
        cum_sum += max_nodes ;
    }

    return batched_edge_indices;
}

torch::Tensor create_batch_edge_indices(const std::vector<int>& node_sizes, const std::vector<torch::Tensor>& edge_indices) {
    int total_edges = 0;
    // #define DEBUG
    for (const auto& edges : edge_indices) {
        if (edges.numel() == 0) continue;
        if (edges.dim() != 2 || edges.size(0) != 2) {
#ifdef DEBUG
            std::cerr << "Edge index tensor has invalid shape: " << edges.sizes() << std::endl;
#endif
            throw std::runtime_error("Invalid edge_indices tensor shape. Expected [2, num_edges]");
        }
        total_edges += edges.size(1);
    }

#ifdef DEBUG
    std::cout << "Total edges after batching: " << total_edges << std::endl;
#endif

    torch::Tensor batched_edge_indices = torch::zeros({2, total_edges}, torch::kInt64);

    int current_offset = 0;
    int cum_sum = 0;

    for (size_t i = 0; i < edge_indices.size(); ++i) {
        const auto& edges = edge_indices[i].contiguous();
        int num_edges = 0;
        if (edges.numel() > 0) {
            num_edges = edges.size(1);

#ifdef DEBUG
            std::cout << "Writing batch " << i
                      << " from offset " << current_offset
                      << " to " << (current_offset + num_edges)
                      << ", edges[0].sizes(): " << edges[0].sizes()
                      << std::endl;
#endif

            batched_edge_indices[0].slice(0, current_offset, current_offset + num_edges) = edges[0] + cum_sum;
            batched_edge_indices[1].slice(0, current_offset, current_offset + num_edges) = edges[1] + cum_sum;

            current_offset += num_edges;

            auto max_src = edges[0].max().item<int64_t>();
            auto min_src = edges[0].min().item<int64_t>();
            auto max_dst = edges[1].max().item<int64_t>();
            auto min_dst = edges[1].min().item<int64_t>();

            if (max_src >= node_sizes[i] || min_src < 0 || max_dst >= node_sizes[i] || min_dst < 0) {
                std::cerr << "❌ Invalid edge index in batch " << i << std::endl;
                std::cerr << "  ➤ node_sizes[i]: " << node_sizes[i] << std::endl;
                std::cerr << "  ➤ edges[0].min(): " << min_src << ", edges[0].max(): " << max_src << std::endl;
                std::cerr << "  ➤ edges[1].min(): " << min_dst << ", edges[1].max(): " << max_dst << std::endl;
                std::cerr << "  ➤ edge_indices[i]: " << edges << std::endl;
                throw std::runtime_error("Edge index out of bounds in create_batch_edge_indices");
            }
        }
        cum_sum += node_sizes[i];
    }

#ifdef DEBUG
    std::cout << "✅ Batched edge indices shape: " << batched_edge_indices.sizes() << std::endl;
#endif
    // #undef DEBUG
    return batched_edge_indices;
}


void add_and_pad_graph(torch::Tensor& graph_features, torch::Tensor& graph_pad_mask, 
                    const int& max_graph_size, torch::Tensor& batch_graph_features,
                     torch::Tensor& batch_graph_pad_mask)
{
        int graph_size = graph_features.size(0);
        if(graph_size < max_graph_size){
            auto diff = max_graph_size - graph_size;
            auto temp_graph_features = torch::cat({graph_features, torch::zeros({diff,graph_features.size(1)})},0);
            batch_graph_features = torch::cat({batch_graph_features, temp_graph_features}, 0);
    
            auto temp_graph_pad_mask = torch::cat({graph_pad_mask, torch::zeros({1,diff})}, 1);
            batch_graph_pad_mask = torch::cat({batch_graph_pad_mask,temp_graph_pad_mask}, 0);
        }else{
            #ifdef DEBUG
                assert(graph_size == max_graph_size);
            #endif
            batch_graph_features = torch::cat({batch_graph_features, graph_features}, 0);
            batch_graph_pad_mask = torch::cat({batch_graph_pad_mask, graph_pad_mask}, 0);
        }
}

torch::Tensor make_advantages(const torch::Tensor& rewards, const torch::Tensor& values, const double& discount ){
    auto ret = torch::tensor(0, rewards.options()); 
    torch::Tensor advantages = torch::zeros({rewards.size(0) - 1}, rewards.options()); 
    for (int i = rewards.size(0) - 1; i > 0; i--){
        ret =  rewards[i] + discount*ret;
        advantages[i - 1] = ret - values[i - 1];
    }

    return advantages;
}

torch::Tensor compute_n_step_target_value(const torch::Tensor& rewards, const torch::Tensor& values, const int& td_steps, const double& discount, const int& index) {
    int factor = 0;
    torch::Tensor target_value = torch::tensor(0, torch::kFloat32);  
    int len_rewards = rewards.size(0);
    int temp_td_steps;

    if (td_steps == -1) {
        temp_td_steps = len_rewards - index - 1;
    } else {
        temp_td_steps = std::min(td_steps, len_rewards - index - 1); 
    }

    #ifdef DEBUG
    std::cout << "Computing N-step target value for index: " << index << std::endl;
    std::cout << "  temp_td_steps: " << temp_td_steps << std::endl;
    #endif

    for (int i = index + 1; i <= index + temp_td_steps; i++) {
        target_value += std::pow(discount, factor) * rewards[i];
        factor++;

        #ifdef DEBUG
        std::cout << "  Step " << i << ":" << std::endl;
        std::cout << "    Discount factor: " << std::pow(discount, factor) << std::endl;
        std::cout << "    Reward: " << rewards[i].item<float>() << std::endl;
        std::cout << "    Target value (accumulated): " << target_value.item<float>() << std::endl;
        #endif
    }

    if (temp_td_steps != len_rewards - index - 1) {
        target_value += std::pow(discount, factor) * values[index + temp_td_steps];

        #ifdef DEBUG
        std::cout << "  Adding discounted value[" << index + temp_td_steps << "]: " << values[index + temp_td_steps].item<float>() << std::endl;
        std::cout << "  Final target value: " << target_value.item<float>() << std::endl;
        #endif
    }

    return target_value;
}

torch::Tensor compute_n_step_target_values(const torch::Tensor& rewards, const torch::Tensor& values, const int& td_steps, const double& discount){
    torch::Tensor target_values = torch::zeros({rewards.size(0) - 1}, rewards.options());

    #ifdef DEBUG
    std::cout << "Starting N-step target values computation..." << std::endl;
    std::cout << "Rewards: " << rewards << std::endl;
    std::cout << "Values: " << values << std::endl;
    std::cout << "TD Steps: " << td_steps << ", Discount: " << discount << std::endl;
    #endif

    for (int i = 0; i < rewards.size(0) - 1; i++) {
        target_values[i] = compute_n_step_target_value(rewards, values, td_steps, discount, i);

        #ifdef DEBUG
        std::cout << "Computed target_values[" << i << "]: " << target_values[i].item<float>() << std::endl;
        #endif
    }

    #ifdef DEBUG
    std::cout << "Final N-step target values: " << target_values << std::endl;
    std::cout << "N-step target values computation complete." << std::endl;
    #endif

    return target_values;
}

torch::Tensor compute_gae(const torch::Tensor& rewards, const torch::Tensor& values, const double& lambda, const double& discount) {
    torch::Tensor last_gae = torch::tensor(0.0);
    int len_rewards = rewards.size(0);
    auto advantages = torch::zeros({rewards.size(0) - 1}, rewards.options());

    #ifdef DEBUG
    std::cout << "Starting GAE computation..." << std::endl;
    std::cout << "Rewards: " << rewards << std::endl;
    std::cout << "Values: " << values << std::endl;
    std::cout << "Lambda: " << lambda << ", Discount: " << discount << std::endl;
    #endif

    for (int i = len_rewards - 2; i >= 0; i--) {
        auto delta = rewards[i + 1] + discount * values[i + 1] - values[i];
        last_gae = delta + discount * lambda * last_gae;
        advantages[i] = last_gae;

        #ifdef DEBUG
        std::cout << "Step " << i << ":" << std::endl;
        std::cout << "  Delta: " << delta.item<double>() << std::endl;
        std::cout << "  Last GAE: " << last_gae.item<double>() << std::endl;
        std::cout << "  Advantage[" << i << "]: " << advantages[i].item<double>() << std::endl;
        #endif
    }


    #ifdef DEBUG
    std::cout << "Final Advantages: " << advantages << std::endl;
    std::cout << "Target Values: " << target_values << std::endl;
    std::cout << "GAE computation complete." << std::endl;
    #endif

    return advantages;
}



torch::Tensor compute_target_values_with_gae(const torch::Tensor& rewards, const torch::Tensor& values, const double& lambda, const double& discount) {
    torch::Tensor last_gae = torch::tensor(0.0);
    int len_rewards = rewards.size(0);
    auto advantages = torch::zeros({rewards.size(0) - 1}, rewards.options());

    #ifdef DEBUG
    std::cout << "Starting GAE computation..." << std::endl;
    std::cout << "Rewards: " << rewards << std::endl;
    std::cout << "Values: " << values << std::endl;
    std::cout << "Lambda: " << lambda << ", Discount: " << discount << std::endl;
    #endif

    for (int i = len_rewards - 2; i >= 0; i--) {
        auto delta = rewards[i + 1] + discount * values[i + 1] - values[i];
        last_gae = delta + discount * lambda * last_gae;
        advantages[i] = last_gae;

        #ifdef DEBUG
        std::cout << "Step " << i << ":" << std::endl;
        std::cout << "  Delta: " << delta.item<double>() << std::endl;
        std::cout << "  Last GAE: " << last_gae.item<double>() << std::endl;
        std::cout << "  Advantage[" << i << "]: " << advantages[i].item<double>() << std::endl;
        #endif
    }

    auto target_values = values.slice(0, 0, values.size(0) - 1) + advantages;

    #ifdef DEBUG
    std::cout << "Final Advantages: " << advantages << std::endl;
    std::cout << "Target Values: " << target_values << std::endl;
    std::cout << "GAE computation complete." << std::endl;
    #endif

    return target_values;
}



torch::Tensor create_indices(int total_size, bool shuffle, torch::Dtype type, const torch::Generator& generator) {
    if (shuffle) return torch::randperm(total_size, generator, type);
    return torch::arange(total_size,type);
    
}

std::vector<std::vector<int>> create_batch_size_idxs(int total_size, int batch_size, bool shuffle,  const torch::Generator& generator){
    auto indices = create_indices(total_size, shuffle, torch::kInt32, generator);
    std::vector<std::vector<int>> groups;
    if (batch_size == - 1){
        groups.push_back(std::vector<int>(indices.data_ptr<int>(), indices.data_ptr<int>() + indices.size(0)));
        return groups;
    }

    for(int i =0; i < indices.size(0); i+= batch_size){
        auto end = std::min(i + batch_size, static_cast<int>(indices.size(0)));
        auto cur_indices = indices.slice(0,i,end);
        std::vector<int> cur_indices_vec(cur_indices.data_ptr<int>(), cur_indices.data_ptr<int>() + cur_indices.size(0));
        groups.push_back(std::move(cur_indices_vec));
    }
    
    return groups;
}