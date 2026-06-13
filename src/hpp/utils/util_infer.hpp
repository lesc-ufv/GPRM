#ifndef UTIL_INFER_HPP
#define UTIL_INFER_HPP

#include <zmq.hpp>
#include <torch/torch.h>
#include <msgpack.hpp>
#include <memory>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <string>
#include <optional>
#include "src/hpp/utils/constants/server_constants.hpp"
#include "src/hpp/utils/util_server.hpp"
#include "src/hpp/utils/util_torch.hpp"
#include "src/hpp/utils/batch_requester.hpp"
#include <future>
#include <chrono>

template<typename StateT>
std::tuple<torch::Tensor, torch::Tensor> infer(
    std::shared_ptr<zmq::socket_t> socket,
    const ServerConstants& server_k,
    std::optional<std::shared_ptr<BatchRequester<StateT>>> batch_requester,
    std::string func_name,
    const std::shared_ptr<StateT>& state,
    int polling_interval_ms = 10  
) {
    torch::Tensor policy_logits;
    torch::Tensor value;
    std::vector<torch::Tensor> returns;
    #ifdef DEBUG
        std::cout << "[DEBUG] Starting infer with function: " << func_name << std::endl;
    #endif

    if (batch_requester.has_value()) {
        auto batch_requester_value = batch_requester.value();
        auto future = batch_requester_value->request_inference(state);

        while (true) {
            auto status = future.wait_for(std::chrono::milliseconds(polling_interval_ms));
            
            if (status == std::future_status::ready) {
                returns = future.get();
                returns[0] = returns[0].unsqueeze(0);
                break;
            } else if (status == std::future_status::timeout) {
                continue;
            }
        }
    } else {
        auto model_inputs = state->get_model_inputs(torch::kCPU);
        returns = infer(socket, model_inputs, func_name, server_k);
    }

    policy_logits = returns[0];
    value = returns[1];

    #ifdef DEBUG
        std::cout << "Policy logits: " << policy_logits << std::endl;
        std::cout << "Value: " << value.item().toDouble() << std::endl;
    #endif

    return std::make_tuple(policy_logits, value);
}

#endif
