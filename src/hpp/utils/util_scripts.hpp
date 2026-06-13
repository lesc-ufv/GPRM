#ifndef UTIL_SCRIPTS_HPP
#define UTIL_SCRIPTS_HPP
#include "src/hpp/configs/path_config.hpp"
#include "src/hpp/configs/train_config.hpp"
#include <string>
#include "src/hpp/configs/ppo_config.hpp"
#include <filesystem>
#include "src/hpp/enums/enum_end_state.hpp"
using json = nlohmann::json;

json get_json_from_file(const std::string& json_path){
    if(fs::exists(json_path)) {
        json js;
        std::ifstream file(json_path);
        if (file.is_open()) {
            file >> js;
        }else {
            std::cerr << "[Error] Error opening file: " << json_path << "\n";
            throw;
        }
        file.close();
        return js;
    }
    else{
        std::cerr << "[Error] " << json_path << " doesn't exists" << "\n";
        throw;
    }
}

inline void set_configs_from_json(json js_config, DFGConfig& dfg_config, StateConfig& state_config, LauncherConfig& launcher_config){

        state_config.setOverrideBaseState(js_config["override_base_state"]);
        state_config.setUseCongestionEdgeMask(js_config["use_edge_mask"]);
        state_config.setUseSymmetricMask(js_config["use_sym_mask"]);
        state_config.setUseUsedPeMask(js_config["use_pe_mask"]);
        state_config.setEndState(static_cast<EnumEndState>(js_config["end_state_id"]));
        state_config.setUseRTScheduler(js_config["use_rt_scheduler"]);

        launcher_config.setEnvironment(static_cast<EnumEnvironment>(js_config["environment_id"]));
        launcher_config.setModel(static_cast<EnumModel>(js_config["model_id"]));



}
#endif