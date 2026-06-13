#ifndef PATH_CONFIG_HPP
#define PATH_CONFIG_HPP

#include <string>
#include <iostream>
#include <sstream>

class PathConfig {
private:
    std::string save_checkpoint_path;
    std::string load_checkpoint_path;
    std::string checkpoint_name;
    std::string tensorboard_log_path;
    std::string train_dfgs_dot_path;
    std::string eval_dfgs_dot_path;
    std::string load_model_path;

public:
    PathConfig()
        : save_checkpoint_path(""),
          load_checkpoint_path(""),
          checkpoint_name(""),
          tensorboard_log_path(""),
          train_dfgs_dot_path(""),
          eval_dfgs_dot_path(""),
          load_model_path("") {}

    std::string getSaveCheckpointPath() const { return save_checkpoint_path; }
    std::string getLoadCheckpointPath() const { return load_checkpoint_path; }
    std::string getCheckpointName() const { return checkpoint_name; }
    std::string getTensorboardLogPath() const { return tensorboard_log_path; }
    std::string getTrainDfgsDotPath() const { return train_dfgs_dot_path; }
    std::string getEvalDfgsDotPath() const { return eval_dfgs_dot_path; }
    
    std::string get_load_model_path() const {
        return load_model_path;
    }

    void set_load_model_path(std::string path) {
        load_model_path = path;
    }

    void setSaveCheckpointPath(const std::string& value) { save_checkpoint_path = value; }
    void setLoadCheckpointPath(const std::string& value) { load_checkpoint_path = value; }
    void setCheckpointName(const std::string& value) { checkpoint_name = value; }
    void setTensorboardLogPath(const std::string& value) { tensorboard_log_path = value; }
    void setTrainDfgsDotPath(const std::string& value) { train_dfgs_dot_path = value; }
    void setEvalDfgsDotPath(const std::string& value) { eval_dfgs_dot_path = value; }

    void print_ss(std::stringstream& ss) const {
        ss << "\nPath Configuration:\n"
           << "\tSave Checkpoint Path: " << save_checkpoint_path << "\n"
           << "\tLoad Checkpoint Path: " << load_checkpoint_path << "\n"
           << "\tCheckpoint Name: " << checkpoint_name << "\n"
           << "\tTensorboard Log Path: " << tensorboard_log_path << "\n"
           << "\tTrain DFGs Dot Path: " << train_dfgs_dot_path << "\n"
           << "\tEval DFGs Dot Path: " << eval_dfgs_dot_path << "\n"
           << "\tLoad Model Path: " << load_model_path << "\n"
           ;
    }
};

#endif
