#include "src/hpp/utils/util_args.hpp"

std::vector<std::unordered_set<EnumInterconnectionStyles>> process_connections(std::string connections){
    int num_bit_connections = 4;
    std::vector<std::unordered_set<EnumInterconnectionStyles>> interconnection_styles_list;

    for(int i = 0; i< static_cast<int>(connections.size()); i += num_bit_connections + 1){
        std::string cur_connection = connections.substr(i,num_bit_connections);
        std::unordered_set<EnumInterconnectionStyles> interconnection_styles = {};
        for(int j = 0; j < static_cast<int>(cur_connection.size()); j++){
            if (cur_connection[j] == '1') interconnection_styles.emplace(static_cast<EnumInterconnectionStyles>(j));
        }
        interconnection_styles_list.emplace_back(interconnection_styles);

    }
    return interconnection_styles_list;
}




std::optional<std::unordered_map<std::string, int>> parse_node_to_action_map(const std::string& input) {
    // #define DEBUG
    if (input.empty()) {
#ifdef DEBUG
        std::cerr << "[DEBUG] Input string is empty.\n";
#endif
        return std::nullopt;
    }

    std::unordered_map<std::string, int> node_to_action;
    std::stringstream ss(input);
    std::string token;

#ifdef DEBUG
    std::cerr << "[DEBUG] Starting to parse input: " << input << "\n";
#endif

    while (std::getline(ss, token, ',')) {
        auto minus_pos = token.rfind("-");
        if (minus_pos != std::string::npos && minus_pos > 0 && minus_pos + 1 < token.size()) {
            std::string node_str = token.substr(0, minus_pos);
            std::string action_str = token.substr(minus_pos + 1);

#ifdef DEBUG
            std::cerr << "[DEBUG] Parsed pair: node = '" << node_str << "', action = '" << action_str << "'\n";
#endif

            try {
                int action = std::stoi(action_str);
                node_to_action[node_str] = action;

#ifdef DEBUG
                std::cerr << "[DEBUG] Added to map: " << node_str << " -> " << action << "\n";
#endif
            } catch (...) {
                std::cerr << "[ERROR] Failed to parse action integer from: " << token << "\n";
            }
        } else {
#ifdef DEBUG
            std::cerr << "[DEBUG] Skipping invalid token (no valid '-'): " << token << "\n";
#endif
        }
    }

#ifdef DEBUG
    std::cerr << "[DEBUG] Finished parsing. Total entries: " << node_to_action.size() << "\n";
#endif

    return node_to_action;
    // #undef DEBUG
}
std::vector<std::pair<int,int>> process_dimensions(std::string dimensions){
    std::vector<std::pair<int,int>> dimensions_list;
    std::string cur_dim = "";
    for(int i = 0; i < static_cast<int>(dimensions.size()); i++){
        if (dimensions[i] == 'x'){
            int n_rows = std::stoi(cur_dim);
            cur_dim = "";
            i++;
            while(i < static_cast<int>(dimensions.size()) && dimensions[i] != '-'){
                cur_dim += dimensions[i];
                i++;
            }
            int n_cols = std::stoi(cur_dim);
            dimensions_list.emplace_back(n_rows,n_cols);
            cur_dim = "";
        }else{
            cur_dim += dimensions[i];
        }
    }
    return dimensions_list;
}

std::vector<std::unordered_set<EnumFunctionalities>> process_functionalities(std::string functionalities){
    int num_bit_functionalities = 8;
    std::vector<std::unordered_set<EnumFunctionalities>> functionalities_list;

    for(int i =0; i< static_cast<int>(functionalities.size());i += num_bit_functionalities){
        std::string cur_functionalities = functionalities.substr(i,num_bit_functionalities);
        std::unordered_set<EnumFunctionalities> functionalities_set = {};
        for(int j = 0; j < static_cast<int>(cur_functionalities.size()); j++){
            if (cur_functionalities[j] == '1') functionalities_set.emplace(static_cast<EnumFunctionalities>(j));
        }
        functionalities_list.emplace_back(functionalities_set);

    }
    return functionalities_list;
}

std::unordered_set<EnumTaskType> process_tasks(std::string tasks){
    std::unordered_set<EnumTaskType> tasks_set = {};
    for(int i =0; i< static_cast<int>(tasks.size());i ++){
        char cur_tasks = tasks[i];
        if (cur_tasks == '1') tasks_set.emplace(static_cast<EnumTaskType>(i));
        
    }
    return tasks_set;
}