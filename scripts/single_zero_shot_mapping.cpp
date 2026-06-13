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
#include "src/hpp/utils/util_print.hpp"
using json = nlohmann::json;

template<typename StateT, typename ModelConfig>
void zero_shot_mapping(
                       std::string dot_filename,
                       int II,
                        std::string& file_to_write_mapping,
                       GlobalConfig<ModelConfig>& global_config,
                       std::optional<std::unordered_map<std::string,int>>& node_to_actions) {
    auto launcher = Launcher<StateT,ModelConfig>(global_config);
    launcher.run_config_server();
    launcher.single_zero_shot_map(dot_filename, file_to_write_mapping, II, node_to_actions);
    launcher.finish();
}

int main(int argc, char* argv[]){
    cxxopts::Options options("Single zero shot mapping mapping", "Single zero-shot CGRA mapping with RL-based compilers");
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
        ("checkpoint", "Path to checkpoint folder", cxxopts::value<std::string>())
        ("file_to_write_map", "File to write mapping (.md extension)", cxxopts::value<std::string>())
        ("dot", "Path to .dot DFG file", cxxopts::value<std::string>())
        ("port", "Port for server", cxxopts::value<std::string>())
        ("max_expansion", "Max MCTS expansions", cxxopts::value<int>()->default_value("0"))
        ("simulations", "Number of MCTS simulations", cxxopts::value<int>()->default_value("0"))
        ("c1", "MCTS parameter C1", cxxopts::value<double>()->default_value("0"))
        ("c2", "MCTS parameter C2", cxxopts::value<double>()->default_value("0"))
        ("restart_root", "Restart root node (1/0)", cxxopts::value<bool>())
        ("uct", "UCT constant", cxxopts::value<double>())
        ("threads", "Number of threads", cxxopts::value<int>())
        ("dims", "Dimensions e.g. 2x2 or 4x4...", cxxopts::value<std::string>())
        ("conn", "4-bit binary string (e.g., 1000 or 1001 or ...) representing interconnection style. "
            "Each bit indicates whether to enable (1) or disable (0): "
            "one-hop, diagonal, toroidal, and mesh (left to right).",
            cxxopts::value<std::string>())
          ("backtrack", "Use backtracking (1/0)", cxxopts::value<bool>())
        ("first_valid", "Get first valid solution (1/0)", cxxopts::value<bool>())
        ("self_map", "Self mapping ID", cxxopts::value<int>())
        ("II", "II (initiation interval)", cxxopts::value<int>())
        ("node_to_actions", "DFG node-to-action pairs (predefined mapping). E.g., add0->10,add2->9", cxxopts::value<std::string>()->default_value(""))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string checkpoint_path = result["checkpoint"].as<std::string>();
    std::string file_to_write_mapping = result["file_to_write_map"].as<std::string>();
    std::string dot_filename = result["dot"].as<std::string>();
    std::string port = result["port"].as<std::string>();
    int max_num_expansion = result["max_expansion"].as<int>();
    int num_simulations = result["simulations"].as<int>();
    double c1 = result["c1"].as<double>();
    double c2 = result["c2"].as<double>();
    bool restart_root = result["restart_root"].as<bool>();
    double uct_constant = result["uct"].as<double>();
    int num_threads = result["threads"].as<int>();
    std::string dims = result["dims"].as<std::string>();
    std::string connections = result["conn"].as<std::string>();
    bool use_backtracking = result["backtrack"].as<bool>();
    bool get_first_valid_solution = result["first_valid"].as<bool>();
    bool self_mapping_id = result["self_map"].as<int>();
    int II = result["II"].as<int>();
    std::string node_to_actions_str = result["node_to_actions"].as<std::string>();
    auto dims_list = process_dimensions(dims);
    auto connections_list = process_connections(connections);
    auto node_to_actions = parse_node_to_action_map(node_to_actions_str);
    std::string base_tensorboard_dir;

    PathConfig path_config;
    path_config.setLoadCheckpointPath(checkpoint_path);
    path_config.setEvalDfgsDotPath(dot_filename);
    
    for(auto& [dfg_name, pe]: node_to_actions.value()){
        std::cout << dfg_name << " " << pe << std::endl;
    }
    
    CGRAConfig cgra_config;
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
    launcher_config.setSelfMapping(static_cast<EnumSelfMapping>(self_mapping_id));
    launcher_config.setNumThreads(num_threads);
    launcher_config.setServerPort(port);
    
    DFGConfig dfg_config;
    StateConfig state_config;

    std::cout << "[INFO] Loading configs \n";
    std::string config_path = checkpoint_path + "config.json";
    json js_config = get_json_from_file(config_path);
    
    PPOConfig ppo_config;
    TrainConfig train_config;
    BufferConfig buffer_config;
    RLConfig rl_config;

    launcher_config.set_config_from_json(js_config);
    dfg_config.set_config_from_json(js_config);
    state_config.set_config_from_json(js_config);
    
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
    for (const auto& [node, action] : node_to_actions.value_or(std::unordered_map<std::string,int>{})) {
        std::cout << "Node: " << node << ", Action: " << action << std::endl;
    }
    switch(model){
        case EnumModel::SMARTMAP:
            zero_shot_mapping<SmartMapState,SmartMapConfig>(dot_filename, II, file_to_write_mapping, 
                global_config.getGlobalConfigAs<SmartMapConfig>(),node_to_actions);
            break;
        case EnumModel::GPRM:
            zero_shot_mapping<GPRMState,GPRMConfig>(dot_filename, II, file_to_write_mapping,
                global_config.getGlobalConfigAs<GPRMConfig>(),node_to_actions);
            break;
        case EnumModel::MAPZERO:
            zero_shot_mapping<MapZeroState,MapZeroConfig>(dot_filename, II, file_to_write_mapping,
                global_config.getGlobalConfigAs<MapZeroConfig>(),node_to_actions);
            break;
        default:
            std::cerr << "[ERROR] Unknown model type\n";
            throw std::runtime_error("Unknown model type");
    }
    
        
}