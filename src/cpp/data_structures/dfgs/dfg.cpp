#include "src/hpp/data_structures/dfgs/dfg.hpp"

DFG::DFG(const std::string& dot_filename, DFGConfig& dfg_config){
    std::ifstream file(dot_filename);
    this->path_to_dot = dot_filename;
    std::string path_to_png = dot_filename.substr(0, dot_filename.find_last_of('.')) + ".png"; 
    this->path_to_png = path_to_png;
    bool generate_images = dfg_config.getGenerateDFGImages();
    if(generate_images){
        std::ifstream png_file(path_to_png);
        if (!png_file.good()) {
            std::cout << "[INFO] PNG file does not exist. Generating: " << path_to_png << std::endl;
            std::string command = "dot -Tpng " + dot_filename + " -o " + path_to_png;
            system(command.c_str());
        } 
        else {
            std::cout << "[INFO] PNG file already exists: " << path_to_png << std::endl;
        }
        png_file.close();
    }

    if (!file.is_open()) {
        throw std::runtime_error("Error opening the file: " + dot_filename);
    }

    this->dfg_name = std::filesystem::path(dot_filename).stem().string();
    
    boost::dynamic_properties dp;
    node_to_operation = get(vertex_opcode_t(), graph);
    node_to_name = get(vertex_name_t(), graph);
    dp.property("opcode", node_to_operation);
    dp.property("name", node_to_name); 
    boost::read_graphviz(file, graph, dp, "name");

   this->build_graph(dfg_config);
   
}



const std::string& DFG::get_operation_by_node(const int& node) const {
    return this->node_to_operation[node];
}

const std::string& DFG::get_name_by_node(const int& node) const {
    const auto& name  = this->node_to_name[node];
    if(name == "") throw std::runtime_error("Invalid node.");
    return name;
}
const std::vector<int>& DFG::get_vertices() const{
    return this->vertices;
}
const std::vector<std::pair<int,int>>& DFG::get_edges() const{
    return this->edges;
}

const std::vector<int>& DFG::get_topological_sort()   {
    if (!this->topo_order.empty()) return this->topo_order;

    std::vector<int> in_degree(boost::num_vertices(this->graph), 0);
    std::vector<int> cur_topo_order; 
    std::queue<int> zero_in_degree_queue;

    boost::graph_traits<DFGType>::vertex_iterator vi, vi_end;
    boost::graph_traits<DFGType>::adjacency_iterator ai, ai_end;

    for (std::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
        for (std::tie(ai, ai_end) = boost::adjacent_vertices(*vi, graph); ai != ai_end; ++ai) {
            ++in_degree[*ai];
        }
    }

    for (std::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
        if (in_degree[*vi] == 0) {
            zero_in_degree_queue.push(*vi);
        }
    }

    while (!zero_in_degree_queue.empty()) {
        int v = zero_in_degree_queue.front();
        zero_in_degree_queue.pop();
        cur_topo_order.emplace_back(v); 

        for (std::tie(ai, ai_end) = boost::adjacent_vertices(v, graph); ai != ai_end; ++ai) {
            --in_degree[*ai]; 
            if (in_degree[*ai] == 0) {
                zero_in_degree_queue.push(*ai); 
            }
        }
    }

    if (cur_topo_order.size() != boost::num_vertices(graph)) {
        throw std::runtime_error("Graph contains cycle.");
    }
    this->topo_order = std::move(cur_topo_order);

    return this->topo_order;  

}

std::unordered_map<int,int> DFG::get_node_levels_with_topo_order(const std::vector<int>& topological_order) const {
    std::unordered_map<int,int> levels;

    for (const int& v : topological_order) {
        for (const auto& neighbor : boost::make_iterator_range(boost::adjacent_vertices(v, graph))){
            levels[neighbor] = std::max(levels[neighbor], levels[v] + 1);
        }
    }

    return levels; 
}

int DFG::num_vertices() const {
    return this->vertices.size();
}

const std::vector<std::vector<int>>& DFG::get_edge_indices() const{
    return this->edge_indices;
}

const std::unordered_map<int,std::unordered_set<int>>& DFG::get_node_to_in_vertices() const{
    return this->in_vertices;
}

const std::unordered_map<int,std::unordered_set<int>>& DFG::get_node_to_out_vertices() const{
    return this->out_vertices;
}

const std::unordered_set<int>& DFG::get_in_vertices_by_node_id(const int& node) const{
    return this->in_vertices.at(node);
}
const std::unordered_set<int>& DFG::get_out_vertices_by_node_id(const int& node) const{
    return this->out_vertices.at(node);
}

const std::vector<int>& DFG::get_root_nodes(){
    return this->root_nodes;
}

const std::unordered_map<std::pair<int,int>,int>& DFG::get_edge_to_id(){
    return this->edge_to_id;
}

const DFGType& DFG::get_graph() const{
    return this->graph;
}

void DFG::print() const {
    std::cout << "\nDFG State Dump (" << dfg_name << ")\n";
    std::cout << "================================\n";
    
    std::cout << "- Vertices: " << num_vertices() << "\n";
    std::cout << "- Edges: " << edges.size() << "\n";
    std::cout << "- Root nodes: " << root_nodes.size() << "\n";
    
    std::cout << "\nVertices:\n";
    for (const auto& v : vertices) {
        std::cout << "  " << v << " [op: " << get_operation_by_node(v) 
                    << ", name: " << get_name_by_node(v) << "]\n";
    }
    
    std::cout << "\nEdges:\n";
    for (const auto& e : edges) {
        std::cout << "  " << e.first << " -> " << e.second;
        if (edge_to_id.count(e)) {
            std::cout << " (id: " << edge_to_id.at(e) << ")";
        }
        std::cout << "\n";
    }
    
    if (!topo_order.empty()) {
        std::cout << "\nTopological Order:\n  ";
        for (const auto& n : topo_order) {
            std::cout << n << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nAdjacency Info:\n";
    for (const auto& v : vertices) {
        std::cout << "  Node " << v << ":\n";
        std::cout << "    Inputs: ";
        for (const auto& in : in_vertices.at(v)) {
            std::cout << in << " ";
        }
        std::cout << "\n    Outputs: ";
        for (const auto& out : out_vertices.at(v)) {
            std::cout << out << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "================================\n\n";
}
