
#include "src/hpp/utils/cgra/scheduler.hpp"
bool Scheduler::scheduling_is_valid( const std::vector<int>& vertices,
                                            const std::unordered_map<int, int> &node_to_sched_time_slice,
                                            const std::unordered_map<int, std::unordered_set<int>> &node_to_in_vertices,
                                            const std::unordered_map<int, int> &node_to_pe, 
                                            const std::unordered_map<std::pair<int, int>,std::vector<int>> &pes_to_routing)
{
    for (const auto& node : vertices) {
        int cur_time_slice = node_to_sched_time_slice.at(node);
        int child_pe = node_to_pe.at(node);

        for (const auto& father : node_to_in_vertices.at(node)) {
            int father_pe = node_to_pe.at(father);

            if (static_cast<int>((node_to_sched_time_slice.at(father)) + (pes_to_routing.at(std::make_pair(father_pe, child_pe)).size() - 1)) != cur_time_slice) {
                return false;  
            }
        }
    }
    return true; 
}

// bool Scheduler::down_top_aux_schedule(
//     const int& node, int time_to_sched_node,
//     std::unordered_map<int,int>& node_to_sched_time, 
//     const std::unordered_map<int, std::unordered_set<int>> &in_vertices, 
//     const std::unordered_map<std::pair<int, int>, std::vector<int>> &pes_to_routing, 
//     std::queue<int>& fifo, 
//     const std::unordered_map<int, int> &node_to_pe,
//     bool is_last_root
// ) {
//     if (time_to_sched_node < 0) {
//         node_to_sched_time[node] = 0;
//         #ifdef DEBUG
//         std::cout << "[DEBUG] Scheduling failed: Node " << node << " has negative time (" << time_to_sched_node << ")\n";
//         #endif
        
//         return false;
//     }else{
//         node_to_sched_time[node] = time_to_sched_node;
//     }
//     fifo.push(node);
//     auto child_pe = node_to_pe.at(node);
    

// #ifdef DEBUG
//     std::cout << "[DEBUG] Node " << node << " scheduled at time " << time_to_sched_node << " on PE " << child_pe << "\n";
// #endif

//     for (const auto& father : in_vertices.at(node)) {
//         if (node_to_sched_time.find(father) == node_to_sched_time.end()) {
//             auto father_pe = node_to_pe.at(father);
//             auto delay = static_cast<int>(pes_to_routing.at(std::make_pair(father_pe, child_pe)).size()) - 1;
            
// #ifdef DEBUG
//             std::cout << "[DEBUG] Trying to schedule parent node " << father << " from PE " << father_pe 
//                       << " to PE " << child_pe << " with delay " << delay << "\n";
// #endif
            
//             auto mapping_is_valid =  down_top_aux_schedule(father, time_to_sched_node - delay, 
//                                                           node_to_sched_time, in_vertices, pes_to_routing, fifo, node_to_pe);
//             if (!mapping_is_valid) {
// #ifdef DEBUG
//                 std::cout << "[DEBUG] Mapping is invalid for parent node " << father << "\n";
// #endif
//                 return false;
//             }
//         }
//     }

//     return true;
// }

// std::tuple<std::unordered_map<int,int>, bool> Scheduler::schedule(
//     const std::vector<int>& root_dfg_nodes,
//     const std::unordered_map<int, int> &node_to_pe, 
//     const std::unordered_map<int, std::unordered_set<int>> &in_vertices,
//     const std::unordered_map<int, std::unordered_set<int>> &out_vertices,
//     const std::unordered_map<std::pair<int, int>, std::vector<int>> &pes_to_routing
// ) {
//     int count = 0;
//     std::unordered_map<int, int> node_to_sched_time;

// #ifdef DEBUG
//     std::cout << "\n[DEBUG] Starting scheduling process\n";
//     std::cout << "[DEBUG] Number of root nodes: " << root_dfg_nodes.size() << "\n";
// #endif

//     for (const auto& root : root_dfg_nodes) {
//         node_to_sched_time.clear();
//         auto last_root_node = count == static_cast<int>(root_dfg_nodes.size()) - 1;
//         node_to_sched_time = {{root, 0}};
//         bool mapping_is_valid = true;
//         std::queue<int> fifo;
//         fifo.push(root);

// #ifdef DEBUG
//         std::cout << "[DEBUG] Processing root node: " << root << "\n";
// #endif

//         while (!fifo.empty()) {
//             auto cur_node = fifo.front();
//             fifo.pop();
//             auto father_pe = node_to_pe.at(cur_node);

// #ifdef DEBUG
//             std::cout << "[DEBUG] Processing node " << cur_node << " assigned to PE " << father_pe << "\n";
// #endif

//             for (const auto& out_node : out_vertices.at(cur_node)) {
//                 auto child_pe = node_to_pe.at(out_node);
//                 auto time_to_sched_child = (static_cast<int>(pes_to_routing.at(std::make_pair(father_pe, child_pe)).size()) - 1) + node_to_sched_time.at(cur_node);

// #ifdef DEBUG
//                 std::cout << "[DEBUG] Scheduling child node " << out_node << " from PE " << father_pe 
//                           << " to PE " << child_pe << " at time " << time_to_sched_child << "\n";
// #endif

//                 if (node_to_sched_time.find(out_node) == node_to_sched_time.end()) {
//                     node_to_sched_time[out_node] = time_to_sched_child;
//                     fifo.push(out_node);

//                     for (const auto& in_out_node : in_vertices.at(out_node)) {
//                         auto aux_father_pe = node_to_pe.at(in_out_node);
//                         auto time_to_sched_in_out_node = node_to_sched_time[out_node] - 
//                             (static_cast<int>(pes_to_routing.at(std::make_pair(aux_father_pe, child_pe)).size()) - 1);

// #ifdef DEBUG
//                         std::cout << "[DEBUG] Verifying in-edge node " << in_out_node 
//                                   << " from PE " << aux_father_pe 
//                                   << " scheduled at time " << time_to_sched_in_out_node << "\n";
// #endif

//                         if (node_to_sched_time.find(in_out_node) == node_to_sched_time.end()) {
//                             mapping_is_valid = Scheduler::down_top_aux_schedule(in_out_node, time_to_sched_in_out_node, 
//                                                                                 node_to_sched_time, in_vertices, pes_to_routing, fifo, node_to_pe)
//                                                                                 && mapping_is_valid;
//                             if (!mapping_is_valid && !last_root_node) {
// #ifdef DEBUG
//                                 std::cout << "[DEBUG] Mapping invalid at in-edge node " << in_out_node << "\n";
// #endif
//                                 break;
//                             }
//                         }

//                         if (time_to_sched_in_out_node != node_to_sched_time.at(in_out_node)) {
//                             mapping_is_valid = false;
// #ifdef DEBUG
//                             std::cout << "[DEBUG] Scheduling inconsistency detected at node " << in_out_node << "\n";
// #endif
//                             break;
//                         }
//                     }

//                     if (!mapping_is_valid && !last_root_node) break;

//                 } else if (time_to_sched_child != node_to_sched_time.at(out_node)) {
//                     mapping_is_valid = false;
// #ifdef DEBUG
//                     std::cout << "[DEBUG] Conflict detected in scheduling node " << out_node << "\n";
// #endif
//                     if (!last_root_node) break;
//                 }
//             }

//             if (!mapping_is_valid && !last_root_node) {
//                 break;
//             }
//         }

//         if (mapping_is_valid) {
// #ifdef DEBUG
//             std::cout << "[DEBUG] Successfully scheduled mapping!\n";
// #endif
//             return std::make_tuple(node_to_sched_time, true);
//         }

// #ifdef DEBUG
//         std::cout << "[DEBUG] Mapping failed for root node " << root << "\n";
// #endif

//         count += 1;
//     }

// #ifdef DEBUG
//     std::cout << "[DEBUG] Scheduling process failed. No valid mappings found.\n";
// #endif

//     return std::make_tuple(node_to_sched_time, false);
// }


std::tuple<std::unordered_map<int,int>, bool> Scheduler::schedule(
    const std::vector<int>& root_dfg_nodes,
    const std::unordered_map<int, int> &node_to_pe, 
    const std::unordered_map<int, std::unordered_set<int>> &in_vertices,
    const std::unordered_map<int, std::unordered_set<int>> &out_vertices,
    const std::unordered_map<std::pair<int, int>, std::vector<int>> &pes_to_routing
) {
    std::unordered_map<int, int> node_to_sched_time;
    std::queue<int> fifo;
    bool mapping_is_valid = true;
    int min_value = 0;

#ifdef DEBUG
    std::cout << "\n[DEBUG] Starting scheduling process\n";
    std::cout << "[DEBUG] Number of root nodes: " << root_dfg_nodes.size() << "\n";
#endif

    auto root = root_dfg_nodes[0]; 
    node_to_sched_time[root] = 0;
    fifo.push(root);


    while (!fifo.empty()) {
        int cur_node = fifo.front();
        fifo.pop();
        int cur_pe = node_to_pe.at(cur_node);

#ifdef DEBUG
        std::cout << "[DEBUG] Processing node " << cur_node << " assigned to PE " << cur_pe << "\n";
#endif

        for (const auto& out_node : out_vertices.at(cur_node)) {
            int child_pe = node_to_pe.at(out_node);
            int time_to_sched_child = node_to_sched_time.at(cur_node) + 
                static_cast<int>(pes_to_routing.at({cur_pe, child_pe}).size()) - 1;

#ifdef DEBUG
            std::cout << "[DEBUG] Scheduling child node " << out_node << " from PE " << cur_pe 
                      << " to PE " << child_pe << " at time " << time_to_sched_child << "\n";
#endif

            if (node_to_sched_time.find(out_node) == node_to_sched_time.end()) {
                min_value = std::min(min_value, time_to_sched_child);
                node_to_sched_time[out_node] = time_to_sched_child;
                fifo.push(out_node);
            } else if (node_to_sched_time.at(out_node) != time_to_sched_child) {
                mapping_is_valid = false;
#ifdef DEBUG
                std::cout << "[DEBUG] Conflict detected in scheduling node " << out_node << "\n";
#endif
            }
        }

        for (const auto& in_node : in_vertices.at(cur_node)) {
            int father_pe = node_to_pe.at(in_node);
            int time_to_sched_father = node_to_sched_time.at(cur_node) - 
                static_cast<int>(pes_to_routing.at({father_pe, cur_pe}).size()) + 1;

#ifdef DEBUG
            std::cout << "[DEBUG] Scheduling parent node " << in_node << " from PE " << father_pe 
                      << " to PE " << cur_pe << " at time " << time_to_sched_father << "\n";
#endif

            if (node_to_sched_time.find(in_node) == node_to_sched_time.end()) {
                min_value = std::min(min_value, time_to_sched_father);
                node_to_sched_time[in_node] = time_to_sched_father;
                fifo.push(in_node);
            } else if (node_to_sched_time.at(in_node) != time_to_sched_father) {
                mapping_is_valid = false;
#ifdef DEBUG
                std::cout << "[DEBUG] Conflict detected in scheduling node " << in_node << "\n";
#endif
            }
        }
    }

    if (min_value < 0) {
        for (auto& pair : node_to_sched_time) {
            node_to_sched_time[pair.first] = pair.second - min_value;
        }
    }

    return std::make_tuple(node_to_sched_time, mapping_is_valid);
}
