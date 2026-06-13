#include "src/hpp/configs/ppo_config.hpp"
#include "src/hpp/configs/train_config.hpp"
#include "src/hpp/configs/cgra_config.hpp"
#include "src/hpp/configs/rl_config.hpp"
#include "src/hpp/configs/dfg_config.hpp"
#include "src/hpp/configs/launcher_config.hpp"
#include "src/hpp/configs/mapping_config.hpp"
#include "src/hpp/configs/buffer_config.hpp"
#include "src/hpp/configs/path_config.hpp"
#include "src/hpp/configs/mcts_config.hpp"
#include "src/hpp/configs/state_config.hpp"
#include "src/hpp/compilers/smartmap/configs/smartmap_config.hpp"
#include "src/hpp/launcher.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/utils/util_args.hpp"
#include "src/hpp/rl/data_augmenter.hpp"
#include "experiments/sota/commom_configs.hpp"
#include <algorithm>  
#include <random>     
#include <vector>
#include <string>
#include <iostream>

using json = nlohmann::json;
namespace fs = std::filesystem;

bool is_file_empty_or_invalid(const fs::path& p) {
    if (!fs::exists(p) || !fs::is_regular_file(p)) {
        return true;
    }

    if (fs::file_size(p) == 0) {
        return true;
    }

    std::ifstream f(p);
    return !f.good();
}

int main(int argc, char* argv[]) {

    std::string path_to_train_data = argv[1];
    std::string path_to_eval_data = argv[2];
    std::string path_to_save_mappings = argv[3];
    std::string port = argv[4];
    int batch_size = std::stoi(argv[5]);
    int batches_per_block = std::stoi(argv[6]);
    int num_threads = std::stoi(argv[7]);

    PPOConfig ppo_config;

    TrainConfig train_config;
    train_config.setShuffleTrainData(true);
    train_config.setBatchSizeStates(batch_size);

    BufferConfig buffer_config;
    int buffer_size = 100000;
    buffer_config.setBufferSize(buffer_size);
    buffer_config.setDegrees({});
    buffer_config.setShifts({});
    buffer_config.setFlips({});
    buffer_config.setAugmentData(false);

    CGRAConfig cgra_config = get_general_cgra_config();

    PathConfig path_config;
    path_config.setTrainDfgsDotPath(path_to_train_data);
    path_config.setEvalDfgsDotPath(path_to_eval_data);

    RLConfig rl_config;
    MCTSConfig mcts_config;

    StateConfig cur_state_config = get_state_config_gprm();
    DFGConfig dfg_config = get_dfg_config_gprm();
    dfg_config.setScheduling(EnumScheduling::TIGHT);

    GPRMConfig model_config;
    model_config.setLpeDim(32);

    MappingConfig mapping_config;

    LauncherConfig launcher_config;
    launcher_config.setEnumTaskType(EnumTaskType::TRAIN);
    launcher_config.setServerPort(port);
    launcher_config.setSaveModel(true);
    launcher_config.setEnvironment(EnumEnvironment::GPRM);
    launcher_config.setModel(EnumModel::GPRM);
    launcher_config.setReplayBuffer(EnumReplayBuffer::SUPERVISED_BUFFER);
    launcher_config.setTraining(EnumTraining::SUPERVISED);
    launcher_config.setSelfMapping(EnumSelfMapping::POLICY_LOGITS);
    launcher_config.setNumThreads(num_threads);
    launcher_config.setUseBatchRequester(true);
    launcher_config.setBatchSizeRequester(8192);

    auto global_config = GlobalConfig<GPRMConfig>(
        std::ref(ppo_config),
        std::ref(train_config),
        std::ref(rl_config),
        std::ref(path_config),
        std::ref(cgra_config),
        std::ref(dfg_config),
        std::ref(mcts_config),
        std::ref(buffer_config),
        std::ref(launcher_config),
        std::ref(model_config),
        std::ref(mapping_config),
        std::ref(cur_state_config)
    );

    json js;
    auto launcher = Launcher<GPRMState, GPRMConfig>(global_config);


    if (fs::exists("file_list.json")) {
        std::ifstream i("file_list.json");
        i >> js;
    } else {
        std::vector<std::string> dot_files;
        int count = 0;

        for (const auto& entry : fs::recursive_directory_iterator(path_to_train_data)) {
            if (entry.is_regular_file() && entry.path().extension() == ".dot") {
                dot_files.push_back(entry.path().string());
            }
        }

        std::random_device rd;
        std::mt19937 g(rd());    

        std::shuffle(dot_files.begin(), dot_files.end(), g);
        for (const auto& entry : dot_files) {
            js[std::to_string(count++)] = entry;
        }

        std::ofstream o("file_list.json");
        o << std::setw(4) << js << std::endl;
    }

    int last_file = 0;
    if (fs::exists("last_file.txt")) {
        std::ifstream in("last_file.txt");
        in >> last_file;
    }

    int total_samples_in_block = batch_size * batches_per_block;
    auto eval_initial_states = launcher.get_eval_states();
    auto test_dfgs = launcher.get_unique_dfgs_from_states(eval_initial_states);
    auto node_edge_to_dfgs = create_node_edge_to_dfgs<OrderedDFG>(test_dfgs);

    fs::create_directories(path_to_save_mappings);

    int num_samples = 0;
    int total_batches = 0;
    std::vector<std::string> dot_files;

    int empty_files_count = 0;

    while (last_file < static_cast<int>(js.size())) {

        std::string source_path = js[std::to_string(last_file)];
        fs::path file_path(source_path);

        if (is_file_empty_or_invalid(file_path)) {
            std::cout << "[WARN] Skipping empty/invalid file: "
                      << source_path << "\n";
            empty_files_count++;
            last_file++;
            continue;
        }

        std::cout << "[INFO] Processing file " << source_path
                  << " (" << last_file + 1 << "/" << js.size() << ")\n";

        bool save = false;
        auto parent_folder_name = file_path.parent_path().filename();
        int samples_in_file = std::stoi(parent_folder_name.string());

        if (num_samples + samples_in_file < total_samples_in_block) {
            dot_files.push_back(source_path);
            num_samples += samples_in_file;
            last_file++;
        } else {
            save = true;
        }

        if (!save && last_file == static_cast<int>(js.size())) {
            save = true;
        }

        if (save) {
            auto mappings = launcher.gen_supervised_data_by_files(
                dot_files,
                test_dfgs,
                node_edge_to_dfgs,
                std::make_optional(path_to_train_data)
            );

            auto batches = launcher.generate_batches(
                mappings,
                train_config.getBatchSizeStates()
            );

            launcher.send_data_to_server(batches);
            launcher.save_block_of_supervised_data(path_to_save_mappings);

            std::ofstream out("last_file.txt");
            out << last_file;

            std::cout << "[INFO] Saved block with "
                      << num_samples << " samples and "
                      << batches.size() << " mini-batches\n";

            num_samples = 0;
            total_batches += static_cast<int>(batches.size());
            dot_files.clear();
        }
    }

    std::cout << "[INFO] Total empty/invalid files skipped: "
              << empty_files_count << "\n";

    launcher.finish();
}
