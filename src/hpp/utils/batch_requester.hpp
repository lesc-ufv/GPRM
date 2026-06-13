#ifndef BATCH_INFERENCER_HPP
#define BATCH_INFERENCER_HPP

#include <vector>
#include <thread>
#include <condition_variable>
#include <future>
#include <functional>
#include <chrono>
#include <numeric>
#include <memory>
#include <atomic>
#include <iostream>

#include <tbb/concurrent_queue.h>

#include "src/hpp/rl/collate.hpp"
#include "src/hpp/utils/util_server.hpp"

template <typename StateT>
class BatchRequester {
public:
    using StatePtr = std::shared_ptr<StateT>;
    using Result = std::vector<torch::Tensor>;
    using Request = std::pair<StatePtr, std::promise<Result>>;

    BatchRequester(
        std::shared_ptr<zmq::socket_t> socket,
        int batch_size,
        const Collate<StateT>& collate,
        const ServerConstants& server_k
    )
    : batch_size_(batch_size),
      stop_(false),
      collate(collate),
      server_k(server_k),
      socket(socket)
    {
        worker_ = std::thread(&BatchRequester::worker_thread, this);
    }

    ~BatchRequester() {
        stop_ = true;
        cond_var_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        socket->close();
    }

    std::future<Result> request_inference(const StatePtr& state) {
        std::promise<Result> promise;
        auto future = promise.get_future();

        requests_.push(std::make_pair(state, std::move(promise)));
        cond_var_.notify_one();

        return future;
    }

private:
    void worker_thread() {
        while (!stop_) {
            std::vector<StatePtr> batch;
            std::vector<std::promise<Result>> promises;

            std::unique_lock<std::mutex> lock(cond_mutex_);
            cond_var_.wait_for(lock, std::chrono::milliseconds(5), [this] {
                return !requests_.empty() || stop_;
            });

            while (!stop_) {
                Request req;
                bool ok = requests_.try_pop(req);
                if (!ok)
                    break;

                batch.push_back(req.first);
                promises.push_back(std::move(req.second));

                if (batch.size() >= batch_size_)
                    break;
            }

            if (!batch.empty()) {
#ifdef DEBUG
                std::cout << "[DEBUG] Processing batch of size " << batch.size() << std::endl;
#endif
                std::vector<int> indices(batch.size());
                std::iota(indices.begin(), indices.end(), 0);

                auto batch_collated = collate.collate(batch, indices);

                auto ret = infer(socket, batch_collated, "infer", server_k);
                auto policies = ret[0];
                auto values   = ret[1];

                for (int i = 0; i < static_cast<int>(policies.size(0)); ++i) {
                    std::vector<torch::Tensor> result = {
                        policies[i],
                        values[i]
                    };
                    promises[i].set_value(result);
                }
            }
        }
    }

    size_t batch_size_;
    tbb::concurrent_queue<Request> requests_;
    std::mutex cond_mutex_;
    std::condition_variable cond_var_;
    std::thread worker_;
    std::atomic<bool> stop_;
    std::string func_name;
    Collate<StateT> collate;
    ServerConstants server_k;
    std::shared_ptr<zmq::socket_t> socket;
};

#endif
