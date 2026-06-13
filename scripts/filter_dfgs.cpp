#include "src/hpp/utils/util_train.hpp"
#include <iostream>

int main(int argc, char* argv[]){
    #define DEBUG
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int min_dfg_size = std::stoi(argv[3]);
    int max_dfg_size = std::stoi(argv[4]);
    if(!std::filesystem::exists(input_path)){
        throw std::runtime_error("Input path does not exist: " + input_path);
    }

    if(!std::filesystem::exists(output_path)){
        std::filesystem::create_directories(output_path);
    }

    auto dfg_config = DFGConfig();
    auto [dfgs, _] = initialize_dfgs<DFG>(input_path, false, dfg_config);
    auto cleaned_dfgs = clean_isomorphic_dfgs(dfgs);
    int count = 0;
    for(auto& dfg: cleaned_dfgs){
        int num_nodes = dfg->num_vertices();
        if (dfg->is_disconnected()) continue;
        if(num_nodes >= min_dfg_size && num_nodes <= max_dfg_size){
            std::string save_path = output_path + "/" + std::format("{}",count) + ".dot";
            auto dot_string = dfg->to_dot_string_complete();
            std::ofstream ofs(save_path);
            ofs << dot_string;
            ofs.close();
            std::cout << "[INFO] Saved DFG '" << dfg->get_dfg_name() << "' with " << num_nodes << " nodes to " << save_path << std::endl;
            count++;
        }
    }
    #undef DEBUG
}