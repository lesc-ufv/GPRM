#ifndef UTIL_ARGS_HPP
#define UTIL_ARGS_HPP

#include <vector>
#include <unordered_set>
#include <string>
#include "src/hpp/enums/enum_interconnection_styles.hpp"
#include "src/hpp/enums/enum_functionalities.hpp"
#include "src/hpp/enums/enum_task_type.hpp"
#include <optional>
#include <unordered_map>
#include <sstream>
std::vector<std::unordered_set<EnumInterconnectionStyles>> process_connections(std::string connections);

std::vector<std::pair<int,int>> process_dimensions(std::string dimensions);

std::vector<std::unordered_set<EnumFunctionalities>> process_functionalities(std::string functionalities);

std::unordered_set<EnumTaskType> process_tasks(std::string tasks);


std::optional<std::unordered_map<std::string, int>> parse_node_to_action_map(const std::string& input);



#endif