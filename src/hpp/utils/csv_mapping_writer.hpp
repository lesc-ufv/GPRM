#ifndef CSV_MAPPING_WRITER_HPP
#define CSV_MAPPING_WRITER_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
class CSVMappingWriter{
private:
    std::fstream file;
    std::string key;

    void write_header(){
        std::string header = "key, model_name, dfg_name, arch_name, arch_dims, MII, II, return, routing_nodes, mapping_time, pct_mapped_nodes,\
        pct_mapped_edges, pct_valid_node_convergences, pct_valid_edge_convergences, num_backtrackings, mean_tree_depth, mapping_is_valid, reason, mapping_approach,\
        mapping_type, checkpoint_path, dfg_nodes, dfg_edges\n";
        file << header;
    };
public:
    std::string get_key() const {
        return key;
    }
    CSVMappingWriter(std::string file_path, std::string key):  key(key) {
        std::filesystem::path path(file_path);
        if (path.has_parent_path() && !std::filesystem::exists(path.parent_path())) {
            std::cout << "[INFO] Creating parent directories for: " << path.parent_path() << std::endl;
            std::filesystem::create_directories(path.parent_path());
        }
    
        file.open(file_path, std::ios::out | std::ios::app);
        if(!file.is_open()){
            std::cerr << "[ERROR] Could not open file: " << file_path << std::endl;
            return;
        }

        if (std::filesystem::file_size(file_path) == 0) {
            write_header();
        }
    }

    template<typename ModelConfig, typename StateT>
    void write_data(
        GlobalConfig<ModelConfig> global_config,
        MappingHistory<StateT> mapping_history,
        bool zero_shot_mode
    ) {
        auto path_config = global_config.getPathConfig();
        auto launcher_config = global_config.getLauncherConfig();
        std::string model_name = global_config.get_model_name();
        std::string dfg_name = mapping_history.get_dfg_name();
        std::string arch_name = mapping_history.get_arch_name();
        std::string arch_dims = mapping_history.get_arch_dims_str();
        int II = mapping_history.get_II();
        double ret = mapping_history.calc_sum_reward();
        double routing_nodes = mapping_history.get_total_routing_nodes();
        double mapping_time = mapping_history.get_mapping_time();
        double pct_mapped_nodes = mapping_history.calc_pct_mapped_nodes();
        double pct_mapped_edges = mapping_history.calc_pct_mapped_edges();
        double pct_valid_node_convergences = mapping_history.calc_pct_valid_node_convergences();
        double pct_valid_edge_convergences = mapping_history.calc_pct_valid_edge_convergences();
        int total_backtrackings = mapping_history.get_total_backtrackings();
        double mean_tree_depth = mapping_history.calc_mean_tree_depth();
        bool mapping_is_valid = mapping_history.get_mapping_is_valid();
        std::string checkpoint_path = path_config.getLoadCheckpointPath();
        int MII = mapping_history.get_MII(); 
        int num_dfg_nodes = mapping_history.get_num_dfg_nodes();
        int num_dfg_edges = mapping_history.get_num_dfg_edges();

        file << this-> key << ", "<< model_name << ", "  << dfg_name << ", "<< arch_name << ", " << arch_dims << ", " << MII <<","<< II << ", " << ret << ", " 
             << routing_nodes << ", " << mapping_time << ", " 
             << pct_mapped_nodes << ", " << pct_mapped_edges << ", "
             << pct_valid_node_convergences << ", " << pct_valid_edge_convergences << ", "
             << total_backtrackings << ", "  << mean_tree_depth  << ", "
             << mapping_is_valid  << ","
             << mapping_history.get_invalid_mapping_reason() << ", "
             << enumSelfMappingToString(launcher_config.getSelfMapping()) << ", "
             << (zero_shot_mode ? "zero_shot" : "finetune") << ", "
             << checkpoint_path << ", " << num_dfg_nodes << ", " << num_dfg_edges <<"\n";
        
        file.flush();
    };
    

    void close(){
        file.close();
    };

    ~CSVMappingWriter() {
        if (file.is_open()) {
            file.close();
        }
    }

};

#endif 
