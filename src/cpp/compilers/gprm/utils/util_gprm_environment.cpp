#include "src/hpp/compilers/gprm/utils/util_gprm_environment.hpp"

double num_mapped_edges_by_neigh(const BaseState& state, const int& node, const int& action, bool in_vertices){
    const auto& neigh_vertices = in_vertices ? state.get_in_vertices_dfg_by_node_id(node) : 
                        state.get_out_vertices_dfg_by_node_id(node);
    const auto& PEs_to_routing = state.get_PEs_to_routing_();
    int num_mapped_edges = 0;
    for(auto& node: neigh_vertices){
        if(state.node_was_mapped(node)){
            auto neigh_PE = state.get_assigned_PE_by_node_id(node);
            auto pe_routing_pair = in_vertices ? std::make_pair(neigh_PE, action) : std::make_pair(action, neigh_PE);
            num_mapped_edges += PEs_to_routing.at(pe_routing_pair).empty() ? 0 : 1;
        }
    }   
    return num_mapped_edges;
}


std::tuple<double, double> calc_routing_node_reward_by_neigh(const BaseState& state, const int& node, const int& action, bool in_vertices){
    const auto& neigh_vertices = in_vertices ? state.get_in_vertices_dfg_by_node_id(node) : 
                                        state.get_out_vertices_dfg_by_node_id(node);
    const auto& PEs_to_routing = state.get_PEs_to_routing_();
    double rout_node_reward = 0;
    double num_valid_edges = 0;

    #ifdef DEBUG
    std::cout << "[DEBUG] Calculating routing reward for node " << node << " with action " << action << std::endl;
    std::cout << "[DEBUG] Number of neighbor vertices: " << neigh_vertices.size() << std::endl;
    #endif
    
    if (neigh_vertices.size() == 0) return std::make_tuple(0.0, 0.0);

    for (auto& neigh : neigh_vertices) {
        if (state.node_was_mapped(neigh)) {
            auto neigh_PE = state.get_assigned_PE_by_node_id(neigh);
            auto pe_routing_pair = in_vertices ? std::make_pair(neigh_PE, action) 
                                               : std::make_pair(action, neigh_PE);

            if (!PEs_to_routing.at(pe_routing_pair).empty()) {
                double routing_size = PEs_to_routing.at(pe_routing_pair).size() - 1;
                rout_node_reward += 1.0 / routing_size;

                #ifdef DEBUG
                std::cout << "[DEBUG] Routing path found for neighbor " << neigh 
                          << " with PE " << neigh_PE 
                          << ". Routing size: " << routing_size
                          << ", Incrementing reward: " << 1.0 / routing_size << std::endl;
                #endif
            }
            num_valid_edges += 1;
        }
    }

    #ifdef DEBUG
    std::cout << "[DEBUG] Total valid edges: " << num_valid_edges << std::endl;
    std::cout << "[DEBUG] Final routing node reward: " << rout_node_reward << std::endl;
    #endif

    return std::make_tuple(rout_node_reward, num_valid_edges);
}



double routing_node_reward(const BaseState& state, const int& node, const int& action){
    auto [out_reward,out_valid_rewards] =  calc_routing_node_reward_by_neigh(state,node,action, false);
    auto [in_reward,in_valid_rewards] =  calc_routing_node_reward_by_neigh(state,node,action,true);
    auto denom = out_valid_rewards + in_valid_rewards;
    if(denom == 0) return 1;
    return (out_reward + in_reward)/denom;
}

double num_mapped_edges(const BaseState& state, const int& node, const int& action, bool in_vertices){
    return num_mapped_edges_by_neigh(state, node, action, true) + num_mapped_edges_by_neigh(state, node, action, false);
}

double num_placed_neigh_nodes_by_dir(const BaseState& state, const int& node,bool in_vertices){
    int count = 0;
    const auto& neighs = in_vertices ? state.get_in_vertices_dfg_by_node_id(node) : state.get_out_vertices_dfg_by_node_id(node);
    for (auto& neigh: neighs){
        if(state.node_was_mapped(neigh)){
            count+=1;
        }
    }
    return count;
}

double num_placed_neigh_nodes(const BaseState& state, const int& node, const int& action){
    return num_placed_neigh_nodes_by_dir(state,node, true) + num_placed_neigh_nodes_by_dir(state,node, false);
}
