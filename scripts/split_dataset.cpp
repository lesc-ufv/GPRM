#include "src/hpp/data_structures/dfgs/dfg.hpp"
#include "src/hpp/configs/dfg_config.hpp"
#include "src/hpp/utils/util_dfg.hpp"
#include "src/hpp/utils/util_train.hpp"
#include <iostream>
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <random>  
#include <unordered_map>
#include <unordered_set>

int main(int argc, char* argv[]){
    if(argc < 6){
        std::cerr << "Usage: " << argv[0] << " <input_path> <output_path> <n_bins> <test_split> <seed>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int n_bins = std::stoi(argv[3]);
    double test_split = std::stod(argv[4]);
    unsigned int seed = static_cast<unsigned int>(std::stoul(argv[5]));

    if(!std::filesystem::exists(input_path)){
        throw std::runtime_error("Input path does not exist: " + input_path);
    }

    if(!std::filesystem::exists(output_path)){
        std::filesystem::create_directories(output_path);
    }

    if(!std::filesystem::exists(output_path + "/train")){
        std::filesystem::create_directories(output_path + "/train");
    }
    if(!std::filesystem::exists(output_path + "/test")){
        std::filesystem::create_directories(output_path + "/test");
    }

    auto dfg_config = DFGConfig();
    dfg_config.setAlphaComplexity(1);
    dfg_config.setBetaComplexity(1);

    auto [dfgs, _] = initialize_dfgs<DFG>(input_path, false, dfg_config);

    std::vector<std::pair<int,double>> idx_complexity_pair;
    int count = 0;
    double max_complexity = 0.0;
    for(auto& dfg: dfgs){
        auto complexity = dfg->get_dfg_value();
        idx_complexity_pair.push_back(std::make_pair(count, complexity));
        if(complexity > max_complexity){
            max_complexity = complexity;
        }
        count++;
    }

    // Ordena por complexidade crescente
    std::sort(idx_complexity_pair.begin(), idx_complexity_pair.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::unordered_map<int, std::vector<std::shared_ptr<DFG>>> binned_dfgs;
    double bin_size = 1.0 / static_cast<double>(n_bins);
    double cur_bin = 0.0;
    count = 0;

    for(auto& pair : idx_complexity_pair){
        auto normalized_val = pair.second / max_complexity;
        if(normalized_val <= cur_bin + bin_size || count == n_bins - 1){
            binned_dfgs[count].push_back(dfgs[pair.first]);
        } else {
            cur_bin += bin_size;
            count++;
            binned_dfgs[count].push_back(dfgs[pair.first]);
        }
    }

    std::vector<std::shared_ptr<DFG>> train_dfgs;
    std::vector<std::shared_ptr<DFG>> test_dfgs;

    std::mt19937 gen(seed);

    for(auto& pair : binned_dfgs){
        auto& vec = pair.second;
        int num_dfgs = static_cast<int>(vec.size());
        int num_test = static_cast<int>(std::ceil(num_dfgs * test_split));
        std::unordered_set<int> test_idxs;

        std::uniform_int_distribution<> dis(0, num_dfgs - 1);

        while(static_cast<int>(test_idxs.size()) < num_test){
            test_idxs.insert(dis(gen));
        }

        for(int i = 0; i < num_dfgs; i++){
            if(test_idxs.find(i) != test_idxs.end())
                test_dfgs.push_back(vec[i]);
            else
                train_dfgs.push_back(vec[i]);
        }
    }

    count = 0;
    for(auto& dfg: train_dfgs){
        std::string save_path = output_path + "/train/" + std::format("{}.dot", count);
        auto dot_string = dfg->to_dot_string_complete();
        std::ofstream ofs(save_path);
        ofs << dot_string;
        ofs.close();
        count++;
    }

    count = 0;
    for(auto& dfg: test_dfgs){
        std::string save_path = output_path + "/test/" + std::format("{}.dot", count);
        auto dot_string = dfg->to_dot_string_complete();
        std::ofstream ofs(save_path);
        ofs << dot_string;
        ofs.close();
        count++;
    }

    nlohmann::json complexities_json;
    complexities_json["train"] = nlohmann::json::array();
    complexities_json["test"] = nlohmann::json::array();

    for(auto& dfg: train_dfgs){
        complexities_json["train"].push_back(dfg->get_dfg_value());
    }
    for(auto& dfg: test_dfgs){
        complexities_json["test"].push_back(dfg->get_dfg_value());
    }

    std::ofstream ofs(output_path + "/complexities.json");
    ofs << complexities_json.dump(4);
    ofs.close();

    std::cout << "[INFO] Split dataset into train and test sets with "
              << train_dfgs.size() << " train DFGs and "
              << test_dfgs.size() << " test DFGs." << std::endl;

    return 0;
}
