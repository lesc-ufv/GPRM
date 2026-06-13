#include "src/hpp/utils/util_placement.hpp"

int get_dir_to_follow(const int& dir, const int& node, bool is_zizag, const std::unordered_map<int, std::unordered_set<int>>& fan_in,
 const std::unordered_map<int, std::unordered_set<int>>& fan_out){
    if ((is_zizag) &&
     ((dir && static_cast<int>(fan_out.at(node).size()) > 1) || (!dir && static_cast<int>(fan_in.at(node).size()) > 1))) return !dir;
    return dir;
}

std::unordered_set<int> get_neighbors(const int& dir, const int& node, 
                                  const std::unordered_map<int, std::unordered_set<int>>& fan_in, 
                                  const std::unordered_map<int, std::unordered_set<int>>& fan_out) {
    if (dir) return fan_in.find(node) != fan_in.end() ? fan_in.at(node) : std::unordered_set<int>();
    return fan_out.find(node) != fan_out.end() ? fan_out.at(node) : std::unordered_set<int>();

}



void note(
    std::unordered_map<std::pair<int,int>, std::vector<Annotation>>& edge_to_annotations,  
    const std::unordered_map<int, int>& dst_to_src, const int& src, const int& dst, const int& conv_node, Annotation& annotation,
    const std::unordered_map<int, std::unordered_set<int>>& fan_in, int cur_depth, int max_depth) {


    if(max_depth != -1 && cur_depth == max_depth) return;
    if (src == - 1 || dst == -1){
        return ;
    }
    
    bool note_condition;
    
  
        note_condition = dst_to_src.at(src) != conv_node;
        
        if (note_condition) {
            auto calc_edge_dir = [](const int& src, const int& dst, 
                                    const std::unordered_map<int, std::unordered_set<int>>& fan_in) {
                return fan_in.at(dst).find(src) != fan_in.at(dst).end();
            };
            #ifdef DEBUG
            std::cout << "[DEBUG] Processing note_condition, fetching new_src from dst_to_src.at(" 
                    << src << ")\n";
            #endif
            auto& annotation_dirs = std::get<1>(annotation); 
            auto dir = calc_edge_dir(src, dst, fan_in);
            if(annotation_dirs.empty()){
                annotation_dirs.push_back(std::make_pair(dir, 1));
            }else{
            auto& last_annotation = annotation_dirs.back();
            if (last_annotation.first == dir) {
                last_annotation.second += 1;
            } else {
                annotation_dirs.push_back(std::make_pair(dir, 1));
            }
    
            }
    
            auto new_src = dst_to_src.at(src);
            auto new_dst = src;
    
            #ifdef DEBUG
            std::cout << "[DEBUG] new_src=" << new_src << ", new_dst=" << new_dst << "\n";
            #endif
            
            if (new_src == -1 || new_dst == - 1) return;
    
            auto edge = std::pair<int,int>({new_src, src});
            auto cur_annot = annotation;
            if (edge_to_annotations.find(edge) == edge_to_annotations.end()){
                edge_to_annotations[edge] = {cur_annot};
            }else{
                edge_to_annotations.at(edge).emplace_back(cur_annot);
            }
            note(edge_to_annotations, dst_to_src, new_src, new_dst, conv_node, annotation, fan_in, cur_depth + 1 , max_depth);
        }    
}

std::pair<std::vector<std::tuple<int, int, int>>, std::unordered_map<std::pair<int,int>, std::vector<Annotation>>> do_traversal( 
    const std::vector<int>& nodes, 
    const std::vector<std::pair<int,int>>& edges,
    const std::unordered_map<int, std::unordered_set<int>>& fan_in, 
    const std::unordered_map<int, std::unordered_set<int>>& fan_out,
    int max_depth, bool use_stack, bool start_from_outputs, bool get_annotations, bool make_shuffle, bool is_zigzag)
{
    // #define DEBUG
    std::vector<std::tuple<int, int, int, int>> initial_nodes;
    #ifdef DEBUG
    std::cout << "[DEBUG] Starting traversal with " << nodes.size() << " nodes.\n Starting from output: " << start_from_outputs << "\n";
    #endif

    auto in_or_out = start_from_outputs ? fan_out : fan_in;
    for (auto& node : nodes) {
        if (in_or_out.at(node).empty()) {
            initial_nodes.emplace_back(-1, node, start_from_outputs ? 1 : 0, start_from_outputs ? 1 : 0);

            #ifdef DEBUG
            std::string fan_in_out_str =  (start_from_outputs ? "fan_out" : "fan_in") ; 
            std::cout << "[DEBUG] Node " << node << " has no " << fan_in_out_str << ". Adding to initial_nodes.\n";
            #endif
            
            break;//add only the first //comment to run failed tests
        }
    }

    std::unordered_set<std::pair<int,int>> visited_edges;
    std::unordered_set<int> visited_nodes;

    std::deque<std::tuple<int, int, int, int>> list(initial_nodes.begin(), initial_nodes.end());
    std::vector<std::tuple<int, int, int>> edges_result;
    std::unordered_map<int, int> dst_to_src;
    std::unordered_map<std::pair<int, int>, std::vector<Annotation>> edge_to_annotations;
    std::unordered_map<std::pair<int, int>, std::vector<Annotation>> cleaned_edge_to_annotations;
    std::unordered_set<int> node_in_edges;
    std::unordered_map<int,int> node_to_edge_idx;
    #ifdef DEBUG
    std::cout << "[DEBUG] Initial list size: " << list.size() << "\n";
    #endif
    int idx = 0;
    while (!list.empty()) {
        auto [src, dst, followed_dir, dir_to_follow] = use_stack ? list.back() : list.front();
        if (use_stack) list.pop_back();
        else list.pop_front();

        #ifdef DEBUG
        std::cout << "[DEBUG] Processing node: src=" << src << ", dst=" << dst 
                  << ", followed_dir=" << followed_dir << ", dir_to_follow=" << dir_to_follow << "\n";
        #endif

        if (visited_nodes.find(dst) != visited_nodes.end()) {
            
            if ((node_in_edges.find(src) == node_in_edges.end())){
                assert(node_in_edges.find(dst) != node_in_edges.end());
                if(dst != -1 && src != -1){
                    edges_result.emplace_back(dst, src, !followed_dir);
                    #ifdef DEBUG
                    std::cout << "[DEBUG] Edge added to result: (" << src << ", " << dst << ")\n";
                    #endif
                    node_in_edges.insert(src);
                    dst_to_src[src] = dst;
    
                    node_to_edge_idx[src] = idx;
                    idx+= 1;
                    #ifdef DEBUG
                    std::cout << "[DEBUG] node_to_edge_idx dst_to_src[" << dst << "] = " << idx << "\n";
                    #endif
    
                    #ifdef DEBUG
                    std::cout << "[DEBUG] Updated dst_to_src[" << src << "] = " << dst << "\n";
                    #endif
                }

            }else if (get_annotations) {
                #ifdef DEBUG
                std::cout << "[DEBUG] Calling note() for src=" << src << ", dst=" << dst << "\n";
                #endif
                auto annot = Annotation({dst, {}});
                note(edge_to_annotations, dst_to_src, src, dst, dst, annot, fan_in, 0, max_depth);
            }
            #ifdef DEBUG
            std::cout << "[DEBUG] Node " << dst << " already visited Skipping.\n";
            #endif

            // if(nodes.size() - 1 == edges_result.size()) break;
            continue;

        }
        #ifdef DEBUG
        std::cout << "Edges result size " << edges_result.size() << ")\n";
        #endif
        
        // if(nodes.size() - 1 == edges_result.size()) break;
        
        if (src != -1 && dst != -1) {
            edges_result.emplace_back(src, dst, followed_dir);
            if(idx == 0){
                node_in_edges.insert(src);
                node_to_edge_idx[src] = idx;
                #ifdef DEBUG
                std::cout << "[DEBUG] node_to_edge_idx dst_to_src[" << src << "] = " << idx << "\n";
                #endif
            }
            node_in_edges.insert(dst);  
            node_to_edge_idx[dst] = idx;
            #ifdef DEBUG
            std::cout << "[DEBUG] node_to_edge_idx dst_to_src[" << dst << "] = " << idx << "\n";
            #endif
            idx += 1;

            #ifdef DEBUG
            std::cout << "[DEBUG] Edge added to result: (" << src << ", " << dst << ")\n";
            #endif
        }

        
        std::pair<int,int> cur_edge = followed_dir ? std::make_pair(dst,src) : std::make_pair(src, dst);

        dst_to_src[dst] = src;
        #ifdef DEBUG
        std::cout << "[DEBUG] Updated dst_to_src[" << dst << "] = " << src << "\n";
        #endif

       

        visited_edges.insert(cur_edge);
        visited_nodes.insert(dst);

        auto neighbors_in_dir = get_neighbors(dir_to_follow, dst, fan_in, fan_out);
        auto neighbors_not_in_dir = get_neighbors(!dir_to_follow, dst, fan_in, fan_out);

        #ifdef DEBUG
        std::cout << "[DEBUG] Neighbors in direction " << dir_to_follow << ": ";
        for (auto neigh : neighbors_in_dir) std::cout << neigh << " ";
        std::cout << "\n";

        std::cout << "[DEBUG] Neighbors in opposite direction: ";
        for (auto neigh : neighbors_not_in_dir) std::cout << neigh << " ";
        std::cout << "\n";
        #endif

        for (auto& neigh_node : neighbors_not_in_dir) {
            cur_edge = !dir_to_follow ? std::make_pair(neigh_node, dst) : std::make_pair(dst, neigh_node);
            if (visited_edges.find(cur_edge) == visited_edges.end()) {
                auto new_dir = get_dir_to_follow(!dir_to_follow, neigh_node, is_zigzag, fan_in, fan_out);
                list.emplace_back(dst, neigh_node, !dir_to_follow, new_dir);

                #ifdef DEBUG
                std::cout << "[DEBUG] Checking neighbor (opposite dir): " << neigh_node 
                          << " with new_dir=" << new_dir << "\n";
                std::cout << "[DEBUG] Added to list: dst=" << dst 
                          << ", neigh_node=" << neigh_node 
                          << ", new_dir=" << new_dir << "\n";
                #endif
            }
            
        }

        for (auto& neigh_node : neighbors_in_dir) {
            cur_edge = dir_to_follow ? std::make_pair(neigh_node, dst) : std::make_pair(dst, neigh_node);
            if (visited_edges.find(cur_edge) == visited_edges.end()) {
                auto new_dir = get_dir_to_follow(dir_to_follow, neigh_node, is_zigzag, fan_in, fan_out);
                list.emplace_back(dst, neigh_node, dir_to_follow, new_dir);

                #ifdef DEBUG
                std::cout << "[DEBUG] Checking neighbor (in dir): " << neigh_node 
                          << " with new_dir=" << new_dir << "\n";
                std::cout << "[DEBUG] Added to list: dst=" << dst 
                          << ", neigh_node=" << neigh_node 
                          << ", new_dir=" << new_dir << "\n";
                #endif
            }
            
        }
        
        
    }
    
    
    for (auto& pair: edge_to_annotations){
        auto edge = pair.first;
        cleaned_edge_to_annotations[edge] = std::vector<Annotation>();
        for (auto& annot: pair.second){
            auto& [annot_node,_] = annot;
            if (node_to_edge_idx.at(annot_node) < node_to_edge_idx.at(edge.second) ) {
                cleaned_edge_to_annotations[pair.first].push_back(std::move(annot));
            }
            if(cleaned_edge_to_annotations[edge].empty()) cleaned_edge_to_annotations.erase(edge);
        }
    }

        #ifdef DEBUG
        std::cout << "[DEBUG] Traversal complete. Edges result size: " << edges_result.size() << "\n";
        std::cout << "[DEBUG] Final edges: ";
        for (const auto& edge : edges_result) {
            std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge) << ", " << std::get<2>(edge) << ") ";
        }
        std::cout << "\n";
        
        std::cout << "[DEBUG] Edge annotations (edge_to_annotations) size: " << edge_to_annotations.size() << "\n";
        for (const auto& entry : edge_to_annotations) {
            std::cout << "[DEBUG] Edge (" << entry.first.first << ", " << entry.first.second << ") -> Annotations: ";
            for (const auto& annotation : entry.second) {
                std::cout << "(" << std::get<0>(annotation) << ", " 
                        << std::get<1>(annotation) << ") ";
            }
            std::cout << "\n";
        }

        std::cout << "[DEBUG] Cleaned Edge annotation size: " << cleaned_edge_to_annotations.size() << "\n";
        for (const auto& entry : cleaned_edge_to_annotations) {
            std::cout << "[DEBUG] Edge (" << entry.first.first << ", " << entry.first.second << ") -> Annotations: ";
            for (const auto& annotation : entry.second) {
                std::cout << "(" << std::get<0>(annotation) << ", " 
                        << std::get<1>(annotation) << ") ";
            }
            std::cout << "\n";
        }
    #endif


    return std::make_pair(
        std::move(edges_result),   
        std::move(cleaned_edge_to_annotations)
    );
    // #undef DEBUG
    
}




std::vector<int> gen_placement_order(const std::vector<std::tuple<int,int, int>>& traversal_edges){
    auto first_edge = traversal_edges[0];
    std::vector<int> placement_order = {};
    placement_order.emplace_back(std::get<0>(first_edge));
    placement_order.emplace_back(std::get<1>(first_edge));
    for(auto tuple_ptr = traversal_edges.begin() + 1 ; tuple_ptr != traversal_edges.end(); tuple_ptr++){
        auto edge = *tuple_ptr;
        placement_order.emplace_back(std::get<1>(edge));
    }
    return placement_order;
}

std::unordered_map<int,std::tuple<int, int, int>> gen_node_to_traversal_edge(std::vector<std::tuple<int,int, int>>& tarversal_edges){
    auto first_edge = tarversal_edges[0];
    std::unordered_map<int,std::tuple<int, int, int>> node_to_edge;

    node_to_edge[std::get<0>(first_edge)] = first_edge;
    node_to_edge[std::get<1>(first_edge)] = first_edge;
    for(auto tuple_ptr = tarversal_edges.begin() + 1 ; tuple_ptr != tarversal_edges.end(); tuple_ptr++){
        auto edge = *tuple_ptr;
        node_to_edge[std::get<1>(edge)] = edge;
    }
    return node_to_edge;
}