#include "src/hpp/utils/cgra/router.hpp"

std::tuple<
        std::vector<int>, 
        int
> Router::route(
    int source_pe, std::pair<int, int> arch_dims, int target_pe,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_map<int, std::unordered_set<int>>& occupied_pes_per_module, 
    const std::unordered_map<std::pair<int, int>, std::vector<int>>& pes_to_routing,
    const std::unordered_map<int, std::unordered_set<int>>& free_interconnections, 
    const std::unordered_map<int, std::unordered_set<int>>& out_vertices) {
    
    std::vector<int> routing_path = a_star_routing(source_pe, arch_dims, target_pe, occupied_pes_per_module, out_vertices, free_interconnections);
    int cost = routing_path.size() - 1;
    return std::make_tuple(routing_path, cost);
}


std::tuple<
        std::vector<int>, 
        int
> Router::route(
    int source_pe, std::pair<int, int> arch_dims, int target_pe,
    const std::unordered_map<int, std::tuple<int, int, int>>& pe_to_coord,
    const std::unordered_set<int>& occupied_pes, 
    const std::unordered_map<std::pair<int, int>, std::vector<int>>& pes_to_routing,
    const std::unordered_map<int, std::unordered_set<int>>& free_interconnections, 
    const std::unordered_map<int, std::unordered_set<int>>& out_vertices) {
    
    std::vector<int> routing_path = a_star_routing(source_pe, arch_dims, target_pe, occupied_pes, out_vertices, free_interconnections);

    int cost = routing_path.size() - 1;
    return std::make_tuple(routing_path, cost);
}

bool Router::routing_is_valid(const std::unordered_map<std::pair<int, int>, std::vector<int>> &pes_to_routing)
{
    for (const auto& pair: pes_to_routing) {
        if (pair.second.empty()){
            return false;
        }
    }

    return true;
}

std::vector<int> Router::a_star_routing(
    int source_pe, std::pair<int, int> arch_dims, int target_pe, 
    const std::unordered_map<int, std::unordered_set<int>>& occupied_pes_per_module, 
    const std::unordered_map<int, std::unordered_set<int>>& out_vertices, 
    const std::unordered_map<int, std::unordered_set<int>>& free_interconnections) {

    auto pe_pos = [](int id, int row, int col) {
        return std::make_pair(id / col, id % col);
    };

    auto manhattan_distance = [](std::pair<int, int> pe1, std::pair<int, int> pe2) {
        return std::abs(pe1.first - pe2.first) + std::abs(pe1.second - pe2.second);
    };

    using PQElement = std::pair<int, int>; 
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> heap;

    std::unordered_map<int, int> g_costs;  
    std::unordered_map<int, int> f_costs;  
    std::unordered_map<int, int> parent;   

    g_costs[source_pe] = 0;
    f_costs[source_pe] = manhattan_distance(pe_pos(source_pe, arch_dims.first, arch_dims.second), 
                                            pe_pos(target_pe, arch_dims.first, arch_dims.second));

    heap.push({f_costs[source_pe], source_pe});

    std::unordered_map<int, bool> visited; 
    for (const auto& occupied_pes: occupied_pes_per_module) {
        for (const auto& pe: occupied_pes.second) {
            if (pe != target_pe)
            visited[pe] = true;
        }
    }

    while (!heap.empty()) {
        auto [_, current_pe] = heap.top();
        heap.pop();

        if (current_pe == target_pe) {
            return reconstruct_routing_path(parent, target_pe, source_pe);
        }

        for (int neighbor : out_vertices.at(current_pe)) {
            if (!visited[neighbor] && free_interconnections.at(current_pe).count(neighbor)) {
                visited[neighbor] = true;
                parent[neighbor] = current_pe;

                int g_new = g_costs[current_pe] + 1;  
                int h_new = manhattan_distance(pe_pos(neighbor, arch_dims.first, arch_dims.second),
                                                pe_pos(target_pe, arch_dims.first, arch_dims.second));  
                int f_new = g_new + h_new; 

                if (f_costs.find(neighbor) == f_costs.end() || f_new < f_costs[neighbor]) {
                    f_costs[neighbor] = f_new;
                    g_costs[neighbor] = g_new;
                    heap.push({f_new, neighbor});
                }
            }
        }
    }

    return {};  
}


std::vector<int> Router::a_star_routing(
    int source_pe, std::pair<int, int> arch_dims, int target_pe, 
    const std::unordered_set<int>& occupied_pes, 
    const std::unordered_map<int, std::unordered_set<int>>& out_vertices, 
    const std::unordered_map<int, std::unordered_set<int>>& free_interconnections) {

    #ifdef DEBUG
    std::cout << "\n[DEBUG] A* Routing Start\n";
    std::cout << "[DEBUG] Source PE: " << source_pe << ", Target PE: " << target_pe << "\n";
    std::cout << "[DEBUG] Architecture Dimensions: " << arch_dims.first << "x" << arch_dims.second << "\n";
    std::cout << "[DEBUG] Occupied PEs: ";
    for (const auto& pe : occupied_pes) std::cout << pe << " ";
    std::cout << "\n";
    #endif

    auto pe_pos = [](int id, int row, int col) {
        return std::make_pair(id / col, id % col);
    };

    auto manhattan_distance = [](std::pair<int, int> pe1, std::pair<int, int> pe2) {
        return std::abs(pe1.first - pe2.first) + std::abs(pe1.second - pe2.second);
    };

    using PQElement = std::pair<int, int>; 
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> heap;

    std::unordered_map<int, int> g_costs;  
    std::unordered_map<int, int> f_costs;  
    std::unordered_map<int, int> parent;   

    g_costs[source_pe] = 0;
    f_costs[source_pe] = manhattan_distance(pe_pos(source_pe, arch_dims.first, arch_dims.second), 
                                            pe_pos(target_pe, arch_dims.first, arch_dims.second));

    heap.push({f_costs[source_pe], source_pe});

    std::unordered_map<int, bool> visited; 
    for (const auto& pe: occupied_pes) {
        if (pe != target_pe) visited[pe] = true;
    }

    #ifdef DEBUG
    std::cout << "[DEBUG] Initial heap size: " << heap.size() << "\n";
    #endif

    while (!heap.empty()) {
        auto [current_f, current_pe] = heap.top();
        heap.pop();

        #ifdef DEBUG
        std::cout << "\n[DEBUG] Processing PE: " << current_pe << " (f=" << current_f << ")\n";
        #endif

        if (current_pe == target_pe) {
            #ifdef DEBUG
            std::cout << "[DEBUG] Target PE reached!\n";
            auto path = reconstruct_routing_path(parent, target_pe, source_pe);
            std::cout << "[DEBUG] Path: ";
            for (int pe : path) std::cout << pe << " ";
            std::cout << "\n[DEBUG] A* Routing Complete\n";
            #endif
            return reconstruct_routing_path(parent, target_pe, source_pe);
        }

        #ifdef DEBUG
        std::cout << "[DEBUG] Neighbors: ";
        #endif

        for (int neighbor : out_vertices.at(current_pe)) {
            #ifdef DEBUG
            std::cout << "\n  [DEBUG] Analyzing neighbor " << neighbor 
                      << " | Free: " << (free_interconnections.at(current_pe).count(neighbor) ? "YES" : "NO")
                      << " | Visited: " << (visited.count(neighbor) ? "YES" : "NO");
            #endif
        
            #ifdef DEBUG
            bool condition1 = !visited[neighbor];
            bool condition2 = free_interconnections.at(current_pe).count(neighbor);
            std::cout << "\n    [DEBUG] IF conditions: "
                      << "!visited[" << neighbor << "]=" << (condition1 ? "TRUE" : "FALSE") << ", "
                      << "free_interconnections[" << current_pe << "].count(" << neighbor << ")=" 
                      << (condition2 ? "TRUE" : "FALSE");
            #endif
        
            if (!visited[neighbor] && free_interconnections.at(current_pe).count(neighbor)) {
                #ifdef DEBUG
                std::cout << "\n    [DEBUG] Both conditions met - processing neighbor";
                #endif
        
                visited[neighbor] = true;
                parent[neighbor] = current_pe;
        
                int g_new = g_costs[current_pe] + 1;
                auto neighbor_pos = pe_pos(neighbor, arch_dims.first, arch_dims.second);
                auto target_pos = pe_pos(target_pe, arch_dims.first, arch_dims.second);
                int h_new = manhattan_distance(neighbor_pos, target_pos);
                int f_new = g_new + h_new;
        
                #ifdef DEBUG
                std::cout << "\n    [DEBUG] Cost calculations:"
                         << "\n      g(" << neighbor << ") = " << g_new << " (previous: " 
                         << (g_costs.count(neighbor) ? std::to_string(g_costs[neighbor]) : "∞") << ")"
                         << "\n      h(" << neighbor << ") = " << h_new
                         << "\n      f(" << neighbor << ") = " << g_new << " + " << h_new << " = " << f_new;
                #endif
        
                bool update_condition = (f_costs.find(neighbor) == f_costs.end()) || (f_new < f_costs[neighbor]);
                #ifdef DEBUG
                std::cout << "\n    [DEBUG] Update condition: "
                          << "f_costs.exists=" << (f_costs.find(neighbor) != f_costs.end() ? "YES" : "NO");
                if (f_costs.find(neighbor) != f_costs.end()) {
                    std::cout << ", f_current=" << f_costs[neighbor] << " vs f_new=" << f_new 
                              << " => " << (f_new < f_costs[neighbor] ? "UPDATE" : "KEEP");
                }
                #endif
        
                if (update_condition) {
                    f_costs[neighbor] = f_new;
                    g_costs[neighbor] = g_new;
                    heap.push({f_new, neighbor});
        
                    #ifdef DEBUG
                    std::cout << "\n    [DEBUG] !! Better path found !!" 
                              << " Added to heap (f=" << f_new << ")";
                    #endif
                }
            }
            #ifdef DEBUG
            else {
                std::cout << "\n    [DEBUG] Neighbor skipped - ";
                if (visited.count(neighbor)) std::cout << "already visited";
                else if (!free_interconnections.at(current_pe).count(neighbor)) std::cout << "interconnection busy";
            }
            #endif
        }

        #ifdef DEBUG
        std::cout << "\n[DEBUG] Heap size: " << heap.size();
        std::cout << "\n[DEBUG] Visited PEs: " << visited.size() << "\n";
        #endif
    }

    #ifdef DEBUG
    std::cout << "[DEBUG] No path found!\n";
    std::cout << "[DEBUG] A* Routing Complete\n";
    #endif

    return {};  
}
std::vector<int> Router::reconstruct_routing_path(const std::unordered_map<int, int>& parent, int target_pe, int source_pe) {
    std::vector<int> routing_path;
    int current_pe = target_pe;
    while (current_pe != source_pe) {
        routing_path.push_back(current_pe);
        current_pe = parent.at(current_pe);
    }
    routing_path.push_back(source_pe);
    std::reverse(routing_path.begin(), routing_path.end());
    return routing_path;
}
