#include <iostream>
#include <vector>
#include <unordered_set>
#include <filesystem>
#include "src/hpp/enums/enum_functionalities.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_self_mapping.hpp"
#include "src/hpp/utils/constants/server_constants.hpp"
#include <zmq.hpp>
#include "src/hpp/enums/enum_model.hpp"
#include <msgpack.hpp> 
#include "src/hpp/utils/constants/path_constants.hpp"
#include "src/hpp/utils/util_dfg.hpp"
#include "src/hpp/utils/util_server.hpp"
#include "src/hpp/utils/util_torch.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_pre_training_buffer.hpp"
#include "src/hpp/utils/util_args.hpp"
#include "src/hpp/utils/util_train.hpp"
#include "src/hpp/data_structures/dfgs/ordered_dfg.hpp"
#include "src/hpp/data_structures/cgras/cgra.hpp"
#include "src/hpp/compilers/smartmap/configs/smartmap_config.hpp"
#include "src/hpp/compilers/gprm/configs/gprm_config.hpp"
#include "src/hpp/configs/ppo_config.hpp"
#include "src/hpp/configs/train_config.hpp"
#include "src/hpp/configs/rl_config.hpp"
#include "src/hpp/configs/path_config.hpp"
#include "src/hpp/configs/cgra_config.hpp"
#include "src/hpp/compilers/smartmap/configs/smartmap_config.hpp"
#include "src/hpp/compilers/mapzero/configs/mapzero_config.hpp"
#include "src/hpp/configs/dfg_config.hpp"
#include "src/hpp/configs/mcts_config.hpp"
#include "src/hpp/configs/buffer_config.hpp"
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/configs/mapping_config.hpp"
#include "src/hpp/launcher.hpp"
#include "src/hpp/rl/buffers/buffer.hpp"
#include "src/hpp/compilers/smartmap/rl/smartmap_self_mapping.hpp"
#include "experiments/configs/configs.hpp"
#include <filesystem>
#include <tbb/global_control.h>
#include "experiments/utils/exp_utils.hpp"
#include "src/hpp/utils/util_scripts.hpp"
#include "src/hpp/variants/global_config_variant.hpp"
#include "src/hpp/enums/enum_interconnection_styles.hpp"
#include <fstream>
#include <src/externals/cxxopts.hpp>
#include "src/hpp/utils/csv_mapping_writer.hpp"

using json = nlohmann::json;

template<typename StateT, typename ModelConfig>
void zero_shot_mapping(
                        std::string dot_path,
                       const std::vector<std::pair<int,int>>& dimensions_list,
                       const std::vector<std::unordered_set<EnumInterconnectionStyles>>& connections_list,
                       int max_II,
                       const std::string& save_mapping_path,
                       GlobalConfig<ModelConfig> global_config,
                       CSVMappingWriter& csv_writer) {
    auto launcher = Launcher<StateT,ModelConfig>(global_config);
    launcher.run_config_server();
    launcher.map_with_zero_shot(csv_writer);
    launcher.finish_socket();
    // launcher.finish();
}

int main(int argc, char* argv[]){
    cxxopts::Options options("Zero-shot", "Zero-shot CGRA mapping with RL-based compilers");
    //     /SmartMapMapper \
    // --checkpoint results/reprodutibility/run_0/iter_20 \
    // --save ./results/mappings/ \
    // --dot benchmarks/toy/eval/ \
    // --csv unused.csv \
    // --port 8000 \
    // --max_expansion 0 \
    // --simulations 0 \
    // --c1 0 \
    // --c2 0 \
    // --restart_root 0 \
    // --uct 0 \
    // --threads 16 \
    // --dims 4x4 \
    // --conns 1000-1111 \
    // --backtrack 0 \
    // --first_valid 0 \
    // --self_map 0 \
    // --max_unroll_ii 2
    
    options.add_options()
        ("key", "", cxxopts::value<std::string>())
        ("checkpoint", "Path to checkpoint folder", cxxopts::value<std::string>())
        ("save", "Path to save generated mappings", cxxopts::value<std::string>())
        ("dot", "Path to .dot DFG file", cxxopts::value<std::string>())
        ("csv", "CSV file", cxxopts::value<std::string>())
        ("port", "Port for server", cxxopts::value<std::string>())
        ("max_expansion", "Max MCTS expansions", cxxopts::value<int>()->default_value("0"))
        ("simulations", "Number of MCTS simulations", cxxopts::value<int>()->default_value("0"))
        ("c1", "MCTS parameter C1", cxxopts::value<double>()->default_value("0"))
        ("c2", "MCTS parameter C2", cxxopts::value<double>()->default_value("0"))
        ("restart_root", "Restart root node (1/0)", cxxopts::value<bool>()->default_value("false"))
        ("uct", "UCT constant", cxxopts::value<double>())
        ("threads", "Number of threads", cxxopts::value<int>())
        ("dims", "Dimensions list e.g. 2x2-4x4-...", cxxopts::value<std::string>())
        ("conns", "List of 4-bit binary strings (e.g., 1000-1001-...) representing interconnection styles. "
            "Each bit indicates whether to enable (1) or disable (0): "
            "mesh,one-hop, toroidal, and diagonal (left to right).",
            cxxopts::value<std::string>())
          ("backtrack", "Use backtracking", cxxopts::value<bool>()->default_value("false"))
        ("first_valid", "Get first valid solution (1/0)",cxxopts::value<bool>()->default_value("false"))
        ("self_map", "Self mapping ID", cxxopts::value<int>())
        ("max_unroll_ii", "Max II (initiation interval)", cxxopts::value<int>())
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string checkpoint_path = result["checkpoint"].as<std::string>();
    std::string save_mapping_path = result["save"].as<std::string>();
    std::string dot_path = result["dot"].as<std::string>();
    std::string csv_file = result["csv"].as<std::string>();
    std::string port = result["port"].as<std::string>();
    int max_num_expansion = result["max_expansion"].as<int>();
    int num_simulations = result["simulations"].as<int>();
    double c1 = result["c1"].as<double>();
    double c2 = result["c2"].as<double>();
    bool restart_root = result["restart_root"].as<bool>();
    double uct_constant = result["uct"].as<double>();
    int num_threads = result["threads"].as<int>();
    std::string dims = result["dims"].as<std::string>();
    std::string connections = result["conns"].as<std::string>();
    std::string key = result["key"].as<std::string>();
    bool use_backtracking = result["backtrack"].as<bool>();
    bool get_first_valid_solution = result["first_valid"].as<bool>();
    bool self_mapping_id = result["self_map"].as<int>();
    int max_II = result["max_unroll_ii"].as<int>();

    auto dims_list = process_dimensions(dims);
    auto connections_list = process_connections(connections);

    std::string base_tensorboard_dir;

    PathConfig path_config;
    path_config.setLoadCheckpointPath(checkpoint_path);
    path_config.setEvalDfgsDotPath(dot_path);
    path_config.setSaveCheckpointPath(save_mapping_path);
    
    CGRAConfig cgra_config;
    cgra_config.setUnrollII(max_II);
    cgra_config.setDimensionsList(dims_list);
    cgra_config.setConnectionsList(connections_list);
    cgra_config.setFunctionalitiesList(process_functionalities("111"));

    MCTSConfig mcts_config;
    mcts_config.setC1(c1);
    mcts_config.setC2(c2);
    mcts_config.setAddExplorationNoise(false);
    mcts_config.setDirichletAlpha(0.0);
    mcts_config.setExplorationFactor(0.);
    mcts_config.setMaxNumExpansion(max_num_expansion);
    mcts_config.setNumSimulations(num_simulations);
    mcts_config.setRestartRoot(restart_root);
    mcts_config.setTemperature(0.);
    mcts_config.setUCTConstant(uct_constant);
    mcts_config.setUseSoftmaxTemperature(false);
        
    MappingConfig mapping_config;
    mapping_config.setFirstValidSolution(get_first_valid_solution);
    mapping_config.setUseBacktracking(use_backtracking);

    
    std::vector<std::shared_ptr<CGRA>> cgras = {};
    
    LauncherConfig launcher_config;
   
    
    DFGConfig dfg_config;
    StateConfig state_config;

    std::cout << "[INFO] Loading configs \n";
    std::filesystem::path checkpoint(checkpoint_path);

    std::string config_path = checkpoint.parent_path().string() + "/config.json";


    json js_config = get_json_from_file(config_path);
    
    
    PPOConfig ppo_config;
    TrainConfig train_config;
    BufferConfig buffer_config;
    RLConfig rl_config;

    launcher_config.set_config_from_json(js_config);
    dfg_config.set_config_from_json(js_config);
    dfg_config.setGenerateDFGImages(true);
    state_config.set_config_from_json(js_config);
    launcher_config.setUseBatchRequester(true);
    launcher_config.setBatchSizeRequester(8192);
    launcher_config.setEnumTaskType(EnumTaskType::ZERO_SHOT_MAP);
    launcher_config.setSelfMapping(static_cast<EnumSelfMapping>(self_mapping_id));
    launcher_config.setNumThreads(num_threads);
    launcher_config.setServerPort(port);

    GlobalConfigVariant global_config = GlobalConfigVariant(
        ppo_config,
        train_config,
        rl_config,
        path_config,
        cgra_config,
        dfg_config,
        mcts_config,
        buffer_config,
        launcher_config,
        mapping_config,
        state_config
    );
    global_config.set_model_config_from_json(js_config);

    auto model = global_config.getModel();

    CSVMappingWriter csv_writer = CSVMappingWriter(csv_file,key);
    switch(model){
        case EnumModel::SMARTMAP:
            zero_shot_mapping<SmartMapState,SmartMapConfig>(dot_path, dims_list, connections_list, max_II, save_mapping_path, 
                global_config.getGlobalConfigAs<SmartMapConfig>(),csv_writer);
            break;
        case EnumModel::GPRM:
            zero_shot_mapping<GPRMState,GPRMConfig>(dot_path, dims_list, connections_list, max_II, save_mapping_path,
                global_config.getGlobalConfigAs<GPRMConfig>(),csv_writer);
            break;
        case EnumModel::MAPZERO:
            zero_shot_mapping<MapZeroState,MapZeroConfig>(dot_path, dims_list, connections_list, max_II, save_mapping_path,
                global_config.getGlobalConfigAs<MapZeroConfig>(),csv_writer);
            break;
        default:
            std::cerr << "[ERROR] Unknown model type\n";
            throw std::runtime_error("Unknown model type");
    }
    
        
}