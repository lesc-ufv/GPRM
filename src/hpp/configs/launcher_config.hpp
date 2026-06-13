#ifndef LAUNCHER_CONFIG_HPP
#define LAUNCHER_CONFIG_HPP

#include <iostream>
#include <sstream>
#include <unordered_set>
#include "src/hpp/configs/rl_config.hpp"
#include "src/hpp/enums/enum_task_type.hpp"
#include "src/hpp/enums/enum_model.hpp"
#include "src/hpp/enums/enum_self_mapping.hpp"
#include "src/hpp/enums/enum_environment.hpp"
#include "src/hpp/enums/enum_replay_buffer.hpp"
#include "src/hpp/enums/enum_training.hpp"
#include "nlohmann/json.hpp"
using json = nlohmann::json;
class LauncherConfig {
private:
    std::string server_port;
    EnumTaskType task_type;
    EnumModel model;
    EnumSelfMapping self_mapping;
    EnumEnvironment environment;
    EnumReplayBuffer enum_buffer;
    EnumTraining enum_training;
    bool save_model;
    int n_threads;
    bool use_batch_requester;
    int batch_size_requester;
    double max_exp_time = std::numeric_limits<double>::infinity();
    bool aug_test_only = false;

public:
    LauncherConfig() :
        server_port("8000"),
        task_type(EnumTaskType::DEFAULT),
        model(EnumModel::DEFAULT),
        self_mapping(EnumSelfMapping::DEFAULT),
        environment(EnumEnvironment::DEFAULT),
        enum_buffer(EnumReplayBuffer::DEFAULT),
        enum_training(EnumTraining::DEFAULT),
        save_model(false),
        n_threads(1),
        use_batch_requester(false),
        batch_size_requester(0) {}

    void set_config_from_json(const json& js) {
        model = js["model_id"].get<EnumModel>();
        environment = js["environment_id"].get<EnumEnvironment>();
    }

  
    bool getAugTestOnly() const { return aug_test_only; }
    void setAugTestOnly(bool value) { aug_test_only = value; }
    double getMaxExpTime() const { return max_exp_time; }
    void setMaxExpTime(double value) { max_exp_time = value; }
    const EnumTaskType& getEnumTaskType() const { return task_type; }
    const std::string& getServerPort() const { return server_port; }
    const EnumModel& getModel() const { return model; }
    const EnumSelfMapping& getSelfMapping() const { return self_mapping; }
    const EnumEnvironment& getEnvironment() const { return environment; }
    const EnumReplayBuffer& getReplayBuffer() const { return enum_buffer; }
    const EnumTraining& getTraining() const { return enum_training; }
    const bool& getSaveModel() const { return save_model; }
    const int& getNumThreads() const { return n_threads; }
    const bool& getUseBatchRequester() const {return use_batch_requester;}
    const int& getBatchSizeRequester() const { return batch_size_requester; }

    void setUseBatchRequester(bool value) { use_batch_requester = value; }
    void setEnumTaskType(const EnumTaskType& new_task_type) { task_type = new_task_type; }
    void setServerPort(const std::string& new_server_port) { server_port = new_server_port; }
    void setModel(const EnumModel& new_model) { model = new_model; }
    void setSelfMapping(const EnumSelfMapping& new_self_mapping) { self_mapping = new_self_mapping; }
    void setEnvironment(const EnumEnvironment& new_environment) { environment = new_environment; }
    void setReplayBuffer(const EnumReplayBuffer& new_enum_buffer) { enum_buffer = new_enum_buffer; }
    void setTraining(const EnumTraining& new_enum_training) { enum_training = new_enum_training; }
    void setSaveModel(bool value) { save_model = value; }
    void setNumThreads(int num_threads) { n_threads = num_threads; }
    void setBatchSizeRequester(int batch_size) { batch_size_requester = batch_size; }

    void print_ss(std::stringstream& ss) const {
        ss << "Launcher Configuration:\n"
           << "Server Port: " << server_port << "\n"
           << "Task Type: " << enumToString(task_type) << "\n"
           << "Model: " << get_model_name_by_enum(model) << "\n"
           << "Self Mapping: " << enumSelfMappingToString(self_mapping) << "\n"
           << "Environment: " << enumEnvironmentToString(environment) << "\n"
           << "Replay Buffer: " << enum_replay_buffer_to_string(enum_buffer) << "\n"
           << "Training Strategy: " << enumToString(enum_training) << "\n"
           << "Save Model: " << (save_model ? "true" : "false") << "\n"
           << "Threads: " << n_threads << "\n";
        ss << "Use Batch Requester: " << (use_batch_requester ? "true" : "false") << "\n"
              << "Batch Size Requester: " << batch_size_requester << "\n";
    }
};

#endif
