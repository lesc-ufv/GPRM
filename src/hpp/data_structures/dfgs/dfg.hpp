#ifndef DFG_GRAPH_HPP
#define DFG_GRAPH_HPP

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/property_map.hpp>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include "src/hpp/utils/util_dfg.hpp"
#include "src/hpp/utils/util_graph.hpp"
#include "src/hpp/utils/hashes.hpp"
#include <filesystem>
#include "src/externals/cereal/cereal.hpp"
#include "src/externals/cereal/archives/binary.hpp"
#include "src/externals/cereal/types/vector.hpp"
#include "src/externals/cereal/types/unordered_map.hpp"
#include "src/externals/cereal/types/unordered_set.hpp"
#include "src/externals/cereal/types/string.hpp"
#include <cereal/archives/binary.hpp>
#include <cereal/details/traits.hpp>
#include "src/externals/cereal/types/utility.hpp" 
#include <boost/graph/reverse_graph.hpp>
#include <boost/graph/dominator_tree.hpp>
#include <bitset>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/set.hpp>
#include "src/hpp/configs/dfg_config.hpp"
#include <boost/graph/isomorphism.hpp>
#include <boost/graph/connected_components.hpp>
struct VertexData {
    int id;
    std::string opcode;
    std::string name;

    template<class Archive>
    void serialize(Archive & ar) {
        ar(id, opcode, name);
    }
};

struct EdgeData {
    int source;
    int target;
    int weight;

    template<class Archive>
    void serialize(Archive & ar) {
        ar(source, target, weight);
    }
};

struct vertex_opcode_t { typedef boost::vertex_property_tag kind; };
struct vertex_name_t { typedef boost::vertex_property_tag kind; };

typedef boost::adjacency_list<
    boost::listS, boost::vecS, boost::bidirectionalS,
    boost::property<vertex_opcode_t, std::string,
        boost::property<vertex_name_t, std::string>
    >,
    boost::property<boost::edge_weight_t, int>
> DFGType;

class DFG{
protected:
    DFGType graph;
    std::string path_to_dot;
    std::string path_to_png; 
    std::string dfg_name;
    boost::property_map<DFGType, vertex_opcode_t>::type node_to_operation;
    boost::property_map<DFGType, vertex_name_t>::type node_to_name;
    std::vector<int> vertices;
    std::vector<std::pair<int,int>> edges;
    std::vector<std::vector<int>> edge_indices;
    std::unordered_map<int,std::unordered_set<int>> in_vertices;
    std::unordered_map<int,std::unordered_set<int>> out_vertices;
    std::vector<int> root_nodes;
    std::unordered_map<std::pair<int,int>,int> edge_to_id;
    int num_convergent_edges;
    int num_convergent_nodes;
    int total_rec_paths;
    std::vector<int> topo_order;
    double dfg_value, alpha_complexity,beta_complexity, gamma_complexity;
    std::unordered_set<int> divergent_nodes;
    std::unordered_set<int> convergent_nodes;
    std::unordered_map<std::pair<int, int>, std::pair<int,bool>> rec_to_num_rec_path_is_dom_pair; 
    std::unordered_map<int, std::unordered_set<int>> node_to_convergence;
    std::unordered_map<std::pair<int,int>, int> IO_to_count; 
    int num_connected_components = -1;
    std::unordered_map<std::string, int> name_to_node;
    
    std::vector<VertexData> export_vertices() const {
        std::vector<VertexData> verts;
        for(auto v : vertices) {
            VertexData vd;
            vd.id = v;
            vd.opcode = get_operation_by_node(v);
            vd.name = get_name_by_node(v);
            verts.push_back(vd);
        }
        return verts;
    }

    
    std::vector<EdgeData> export_edges() const {
        std::vector<EdgeData> eds;
        for(const auto& e : edges) {
            EdgeData ed;
            ed.source = e.first;
            ed.target = e.second;
    
            auto edge_descriptor = boost::edge(e.first, e.second, graph);
            if(edge_descriptor.second) {
                ed.weight = boost::get(boost::edge_weight, graph, edge_descriptor.first);
            } else {
                ed.weight = 0; 
            }
            eds.push_back(ed);
        }
        return eds;
    }


    void rebuild_graph(const std::vector<VertexData>& verts,
                       const std::vector<EdgeData>& eds) {
        graph.clear();
    
        std::unordered_map<int, DFGType::vertex_descriptor> id_to_vd;
    
        for(const auto& vd : verts) {
            auto v = boost::add_vertex(graph);
            node_to_operation[v] = vd.opcode;
            node_to_name[v] = vd.name;
            name_to_node[vd.name] = v;
            id_to_vd[vd.id] = v;
        }
    
        for(const auto& ed : eds) {
            auto src = id_to_vd.at(ed.source);
            auto tgt = id_to_vd.at(ed.target);
            auto e = boost::add_edge(src, tgt, graph);
            if(e.second) {
                boost::put(boost::edge_weight, graph, e.first, ed.weight);
            }
        }
    }

public:
    DFG() : graph(),
    path_to_dot(),
    path_to_png(),
    dfg_name(),
    node_to_operation(boost::get(vertex_opcode_t(), graph)),
    node_to_name(boost::get(vertex_name_t(), graph)),
    vertices(),
    edges(),
    edge_indices(),
    in_vertices(),
    out_vertices(),
    root_nodes(),
    edge_to_id(),
    num_convergent_edges(0),
    num_convergent_nodes(0),
    topo_order(),
    node_to_convergence(),
    name_to_node() {};


    std::vector<int> balanced_schedule()
    {
        int n = boost::num_vertices(graph);
        std::vector<int> level(n, std::numeric_limits<int>::max());
        
        int start = topo_order[0];
        if (start == -1) throw std::runtime_error("");
        level[start] = 0;
    
        std::queue<int> q;
        q.push(start);
        int min_value = 0;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
    
            auto [oe, oe_end] = boost::out_edges(u, graph);
            for (; oe != oe_end; ++oe) {
                int v = boost::target(*oe, graph);
                int new_level = level[u] + 1;
                if (level[v] == std::numeric_limits<int>::max()) {
                    level[v] = new_level;
                    q.push(v);
                } else if (new_level > level[v]) {
                    level[v] = new_level;
                    q.push(v);
                }
            }
    
            auto [ie, ie_end] = boost::in_edges(u, graph);
            for (; ie != ie_end; ++ie) {
                int p = boost::source(*ie, graph);
                int new_level = level[u] - 1;
                min_value = std::min(min_value,new_level);
                if (level[p] == std::numeric_limits<int>::max()) {
                    level[p] = new_level;
                    q.push(p);
                } else if (new_level < level[p]) {
                    level[p] = new_level;
                    q.push(p);
                }
            }
        }
        for(int i = 0; i < static_cast<int>(level.size()); i++){
            level[i] -= min_value;
        }
        return level;
    }

    DFG(const std::string& dot_filename, DFGConfig& dfg_config );

    bool are_structurally_isomorphic(const DFG& dfg) {
        return boost::isomorphism(this->graph, dfg.get_graph());
    }

    bool are_structurally_isomorphic_ptr(const std::shared_ptr<DFG>& dfg) {
        return boost::isomorphism(this->graph, dfg->get_graph());
    }

    const std::unordered_map<int, std::unordered_set<int>>& get_node_to_convergence() const {
        return node_to_convergence;
    }
   
    const boost::property_map<DFGType, vertex_name_t>::type& get_node_to_name() {
        return node_to_name;
    }


    

    void calc_IO_to_count(){
        IO_to_count.clear();
        for(const auto& node: this->vertices){
            auto in_edges_size = this->in_vertices.at(node).size();
            auto out_edges_size = this->out_vertices.at(node).size();
            IO_to_count[std::make_pair(in_edges_size,out_edges_size)] += 1;
        }
    }

    bool is_disconnected() {
        if(this->num_connected_components != -1){
            return this->num_connected_components > 1;
        }
        // using DirectedGraph = decltype(graph);
        using UndirectedGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
    
        UndirectedGraph undirected_graph(boost::num_vertices(graph));
    
        auto edges_pair = boost::edges(graph);
        for (auto it = edges_pair.first; it != edges_pair.second; ++it) {
            auto src = boost::source(*it, graph);
            auto tgt = boost::target(*it, graph);
            boost::add_edge(src, tgt, undirected_graph);
        }
    
        std::vector<int> component(boost::num_vertices(undirected_graph));
        int num = boost::connected_components(undirected_graph, &component[0]);
    
        this->num_connected_components = num;
        return num > 1; 
    }
    

    std::unordered_set<int> reachable_nodes(int start) {
        std::unordered_set<int> visited;
        std::queue<int> q;
        q.push(start);
        visited.insert(start);

        while (!q.empty()) {
            int u = q.front(); q.pop();

            auto out_edges = boost::out_edges(u, graph);
            for (auto it = out_edges.first; it != out_edges.second; ++it) {
                int v = boost::target(*it, graph);
                if (!visited.count(v)) {
                    visited.insert(v);
                    q.push(v);
                }
            }
        }
        return visited;
    }

    void compute_divergent_and_convergent_nodes() {
        auto [v_begin, v_end] = boost::vertices(graph);
        for (auto it = v_begin; it != v_end; ++it) {
            int v = *it;
            size_t in_deg = boost::in_degree(v, graph);
            size_t out_deg = boost::out_degree(v, graph);

            if (out_deg > 1) divergent_nodes.insert(v);
            if (in_deg > 1) convergent_nodes.insert(v);
        }
    }

    void compute_convergent_nodes_reached_by_divergent_paths() {
        compute_divergent_and_convergent_nodes();
    
        for (int div_node : divergent_nodes) {
            std::vector<int> neighbors;
            auto out_edges = boost::out_edges(div_node, graph);
            for (auto it = out_edges.first; it != out_edges.second; ++it) {
                neighbors.push_back(boost::target(*it, graph));
            }
    
            size_t n = neighbors.size();
            if (n < 2) continue;
    
            std::vector<std::unordered_set<int>> reachable_sets(n);
            for (size_t i = 0; i < n; ++i)
                reachable_sets[i] = reachable_nodes(neighbors[i]);
    
            for (int k = 2; k <= static_cast<int>(n); ++k) {
                std::vector<bool> bitmask(k, true);
                bitmask.resize(n, false); 
    
                do {
                    std::unordered_set<int> intersection;
                    bool first = true;
    
                    for (size_t i = 0; i < n; ++i) {
                        if (!bitmask[i]) continue;
    
                        if (first) {
                            intersection = reachable_sets[i];
                            first = false;
                        } else {
                            std::unordered_set<int> temp;
                            for (int node : intersection) {
                                if (reachable_sets[i].count(node)) {
                                    temp.insert(node);
                                }
                            }
                            intersection = std::move(temp);
                            if (intersection.empty()) break;
                        }
                    }
    
                    for (int node : intersection) {
                        if (convergent_nodes.count(node)) {
                            if (rec_to_num_rec_path_is_dom_pair.find({div_node, node}) == rec_to_num_rec_path_is_dom_pair.end()) {
                                rec_to_num_rec_path_is_dom_pair[{div_node, node}] = std::make_pair(k, k == static_cast<int>(n));
                                
                                #ifdef DEBUG
                                std::cout << "Inserted new pair into rec_to_num_rec_path_is_dom_pair for key ("
                                          << div_node << ", " << node << ") -> (0, false)\n";
                                #endif
                            } else {
                                auto& pair = rec_to_num_rec_path_is_dom_pair[{div_node, node}];
                                
                                #ifdef DEBUG
                                    int old_first = pair.first;
                                    bool old_second = pair.second;
                                #endif

                                pair.first = std::max(pair.first, k);
                                pair.second |= k == static_cast<int>(n);
                            
                                #ifdef DEBUG
                                std::cout << "Updated pair in rec_to_num_rec_path_is_dom_pair for key ("
                                          << div_node << ", " << node << ") from ("
                                          << old_first << ", " << std::boolalpha << old_second
                                          << ") to (" << pair.first << ", " << pair.second << ")\n";
                                #endif
                            }
                            
                            node_to_convergence[div_node].insert(node);
                        }
                    }
    
                } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
            }
        }

        #ifdef DEBUG
            // for(auto& [src, dst]: this->rec_to_is_postdominant){
            //     std::cout << "[DEBUG] Divergent node: " << this->get_name_by_node(src.first) 
            //               << " reconverges to node: " << this->get_name_by_node(src.second) 
            //               << ", is postdominant: " << dst << std::endl;
            // }
            // for(auto& [rec, pair]: this->rec_to_num_rec_path_is_dom_pair){
            //     std::cout << "[DEBUG] Divergent node: " << this->get_name_by_node(rec.first) 
            //               << " reconverges to node: " << this->get_name_by_node(rec.second) 
            //               << ", num paths: " << pair.first 
            //               << ", is postdominant: " << pair.second << std::endl;
            // }
        #endif

    }

    double get_dfg_value() {
        return this->dfg_value;
    }

    int get_total_rec_paths() const {
        return this->total_rec_paths;
    }

    double calc_dfg_value()  {
        double partial_sum = this->alpha_complexity * static_cast<double>(this->get_num_edges());
    #ifdef DEBUG
        std::cout << "[DEBUG] Initial partial_sum: " << partial_sum << std::endl;
    #endif
    
        std::unordered_set<int> dst_rec_nodes;
        auto node_to_levels = this->get_node_levels_with_topo_order(this->topo_order);
       
        std::unordered_set<std::pair<int,int>> rec_to_ignore;

        #ifdef DEBUG
        std::cout << "Starting to process rec_to_num_rec_path_is_dom_pair...\n";
        #endif

        for (auto& [rec, pair] : this->rec_to_num_rec_path_is_dom_pair) {
            auto [_, is_postdom] = pair;
            if (is_postdom) {
                if (this->node_to_convergence.find(rec.second) == this->node_to_convergence.end()) {
                    #ifdef DEBUG
                    std::cout << "No convergence nodes found for " << rec.second << "\n";
                    #endif
                    continue;
                }
                for (auto& dst : this->node_to_convergence.at(rec.second)) {
                    rec_to_ignore.insert(std::make_pair(rec.first, dst));
                    #ifdef DEBUG
                    std::cout << "Inserting into rec_to_ignore: (" << rec.first << ", " << dst << ")\n";
                    #endif
                }
            }
        }

        double sum_rec_paths = 0;
        for (auto& [rec, pair] : this->rec_to_num_rec_path_is_dom_pair) {
            if (rec_to_ignore.find(rec) != rec_to_ignore.end()) {
                #ifdef DEBUG
                std::cout << "Skipping rec because in rec_to_ignore: (" << this->get_name_by_node(rec.first) << ", " << this->get_name_by_node(rec.second) << ")\n";
                #endif
                continue;
            }

            auto [max_rec_path, _] = pair;
            sum_rec_paths += max_rec_path;

            #ifdef DEBUG
            std::cout << "Adding max_rec_path " << max_rec_path << " for rec (" << this->get_name_by_node(rec.first) << ", " << this->get_name_by_node(rec.second) << ")\n";
            #endif
        }

        #ifdef DEBUG
        std::cout << "Total sum of relevant rec paths: " << sum_rec_paths << "\n";
        std::cout << "Convergent edges: " << num_convergent_edges << "\n";
        #endif
        this->total_rec_paths = static_cast<int>(sum_rec_paths);
        partial_sum += this->beta_complexity*sum_rec_paths;
    
        return partial_sum;
    }
     
    void write_dfg_md(std::stringstream& ss) const {
        ss << "DFG " << this->dfg_name << "\n\n";
        ss << get_dfg_image_md();
        ss << "\n";
        ss << "### DFG Summary\n";
        ss << "- Number of nodes: " << vertices.size() << "\n";
        ss << "- Number of edges: " << edges.size() << "\n";
        ss << "- Number of convergent edges: " << num_convergent_edges << "\n";
        ss << "- Number of convergent nodes: " << num_convergent_nodes << "\n";

    }

    std::string get_dfg_image_md() const {
        namespace fs = std::filesystem;
    
        fs::path p = fs::absolute(this->path_to_png);   // converte para caminho completo
        return "![DFG](" + p.string() + ")\n";
    }

 
    
    const std::string& get_path_to_png() const { return path_to_png;};
    const DFGType& get_graph() const;
    int get_num_convergent_edges() const { return this->num_convergent_edges;};
    int get_num_convergent_nodes() const { return this->num_convergent_nodes;};

    bool is_root(const int& node) const{
        return this->get_in_vertices_by_node_id(node).empty();
    }
    
    bool is_leaf(const int& node) const{
        return this->get_out_vertices_by_node_id(node).empty();
    }
    
    const std::string& get_operation_by_node(const int& node) const ;
    const std::string& get_name_by_node(const int& node) const;
    std::string get_dfg_name() const { return this->dfg_name;};
    const std::vector<int>& get_vertices() const;
    const std::vector<std::pair<int,int>>& get_edges() const;
    const std::vector<int>& get_topological_sort() ;
    std::unordered_map<int,int> get_node_levels_with_topo_order(const std::vector<int>& topo_order) const;
    int num_vertices() const ;
    const std::vector<std::vector<int>>& get_edge_indices() const;
    const std::unordered_set<int>& get_in_vertices_by_node_id(const int& node) const;
    const std::unordered_set<int>& get_out_vertices_by_node_id(const int& node) const;
    const std::vector<int>& get_root_nodes();
    const std::unordered_map<int,std::unordered_set<int>>& get_node_to_in_vertices() const;
    const std::unordered_map<int,std::unordered_set<int>>& get_node_to_out_vertices() const;
    const std::unordered_map<std::pair<int,int>,int>& get_edge_to_id();
    void print() const;

    template<class Archive>
    void serialize(Archive & ar) {
        std::vector<VertexData> verts;
        std::vector<EdgeData> eds;
    
        if constexpr (std::is_same<Archive, cereal::BinaryInputArchive>::value){
            ar(path_to_dot,
               path_to_png,
               dfg_name,
               vertices,
               edges,
               edge_indices,
               in_vertices,
               out_vertices,
               root_nodes,
               edge_to_id,
               num_convergent_edges,
               num_convergent_nodes,
               topo_order,
               verts,
               eds,
               convergent_nodes,
               divergent_nodes,
               node_to_convergence,
                IO_to_count,
                num_connected_components);
    
            rebuild_graph(verts, eds);
    
            node_to_operation = boost::get(vertex_opcode_t(), graph);
            node_to_name = boost::get(vertex_name_t(), graph);
    
        } else {
            verts = export_vertices();
            eds = export_edges();
    
            ar(path_to_dot,
               path_to_png,
               dfg_name,
               vertices,
               edges,
               edge_indices,
               in_vertices,
               out_vertices,
               root_nodes,
               edge_to_id,
               num_convergent_edges,
               num_convergent_nodes,
               topo_order,
               verts,
               eds,
               convergent_nodes,
               divergent_nodes,
               node_to_convergence,
            IO_to_count,
            num_connected_components);
        }
    }

    static std::string sanitize_dot_label(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            if (c == '"' || c == '\\') {
                out.push_back('\\');
            }
            out.push_back(c);
        }
        return out;
    }

    std::string to_dot_string(const std::unordered_set<int>& ignore_nodes = {}) const {
        std::ostringstream oss;
        oss << "strict digraph {\n";
    
        for (int v : vertices) {
            if (ignore_nodes.contains(v)) continue;
    
            std::string name;
            std::string opcode;
            try {
                name = node_to_name[boost::vertex(v, graph)];
            } catch (...) {
                name = "v" + std::to_string(v);
            }
            try {
                opcode = node_to_operation[boost::vertex(v, graph)];
            } catch (...) {
                opcode = "unknown";
            }
    
            std::string safe_name = sanitize_dot_label(name);
            std::string safe_opcode = sanitize_dot_label(opcode);
    
            oss << "    " << safe_name << "[name=" << safe_name << "," <<"opcode=" << safe_opcode << "];\n";
        }
    
        for (const auto& [src, dst] : edges) {
            if (ignore_nodes.contains(src) || ignore_nodes.contains(dst)) continue;
    
            std::string src_name;
            std::string dst_name;
            try {
                src_name = node_to_name[boost::vertex(src, graph)];
            } catch (...) {
                src_name = "v" + std::to_string(src);
            }
            try {
                dst_name = node_to_name[boost::vertex(dst, graph)];
            } catch (...) {
                dst_name = "v" + std::to_string(dst);
            }
    
            std::string safe_src = sanitize_dot_label(src_name);
            std::string safe_dst = sanitize_dot_label(dst_name);
            oss << "    " << safe_src << " -> " << safe_dst << ";\n";
        }
    
        oss << "}\n";
        return oss.str();
    }
    
    
    std::string to_dot_string_complete() const {
        std::ostringstream oss;
        oss << "strict digraph {\n";
            for (int v : vertices) {
            std::string name;
            std::string opcode;
            try {
                auto name_prop = node_to_name[boost::vertex(v, graph)];
                name = name_prop;
            } catch (...) {
                name = "v" + std::to_string(v);
            }
            try {
                auto op_prop = node_to_operation[boost::vertex(v, graph)];
                opcode = op_prop;
            } catch (...) {
                opcode = "unknown";
            }
    
            std::string safe_name = sanitize_dot_label(name);
            std::string safe_opcode = sanitize_dot_label(opcode);
    
            oss << "    " << safe_name << "[name=" << safe_name << "," <<"opcode=" << safe_opcode << "];\n";
        }
        for (const auto& [src, dst] : edges) {
            std::string src_name;
            std::string dst_name;
            try {
                src_name = node_to_name[boost::vertex(src, graph)];
            } catch (...) {
                src_name = "v" + std::to_string(src);
            }
            try {
                dst_name = node_to_name[boost::vertex(dst, graph)];
            } catch (...) {
                dst_name = "v" + std::to_string(dst);
            }
            std::string safe_src = sanitize_dot_label(src_name);
            std::string safe_dst = sanitize_dot_label(dst_name);
            oss << "    " << safe_src << " -> " << safe_dst << ";\n";
        }
        oss << "}\n";
        return oss.str();
    }

    int get_num_nodes() const {
        return static_cast<int>(vertices.size());
    }

    int get_num_edges() const {
        return static_cast<int>(edges.size());
    }
    
    void build_dfg_from_string(std::string& dot_string, std::string dfg_name, DFGConfig& dfg_config){
            this->dfg_name = dfg_name;
            this->path_to_dot = dfg_name + ".dot";
            this->path_to_png = dfg_name + ".png";

            std::istringstream dot_stream(dot_string);

        #ifdef DEBUG
            std::cout << "DEBUG: DOT content:\n" << dot_string << "\n";
        #endif
            boost::dynamic_properties dp;
            node_to_operation = get(vertex_opcode_t(), graph);
            node_to_name = get(vertex_name_t(), graph);
            dp.property("opcode", node_to_operation);
            dp.property("name", node_to_name);

            boost::read_graphviz(dot_stream, graph, dp, "name");
            this->build_graph(dfg_config);
    }
    const std::unordered_map<std::pair<int,int>, int>& get_IO_to_count() const {
        return IO_to_count;
    }

    int get_node_by_name(const std::string& name) const {
        auto it = name_to_node.find(name);
        if (it != name_to_node.end()) {
            return it->second;
        } else {
            throw std::runtime_error("Node name not found: " + name);
        }
    }
   void build_graph(DFGConfig& dfg_config){
        boost::graph_traits<DFGType>::vertex_iterator vi, vi_end;
        for (boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
            vertices.emplace_back(*vi);
            auto name = boost::get(this->node_to_name, *vi);
            name_to_node[name] = *vi;
        }

        auto weight_map = get(boost::edge_weight, graph);
        
        boost::graph_traits<DFGType>::edge_iterator ei, ei_end;
        for (boost::tie(ei, ei_end) = boost::edges(graph); ei != ei_end; ++ei) {
            auto source_vertice = boost::source(*ei, graph);
            auto target_vertice = boost::target(*ei, graph);
            weight_map[*ei] = 1;  
            edges.emplace_back(source_vertice, target_vertice);    
        }

        this->edge_indices = create_edges_indices(this->edges);


        this->in_vertices = generate_in_vertices_dict(this->edges);
        for(auto& vertice: vertices){
            if(this->in_vertices.find(vertice) == this->in_vertices.end()){
                this->in_vertices[vertice] = {};
            }
        }
        for(const auto& pair: this->in_vertices){
            if (pair.second.size() == 0) this->root_nodes.push_back(pair.first);
        }

        this->out_vertices = generate_out_vertices_dict(this->edges);
        for(auto& vertice: vertices){
            if(this->out_vertices.find(vertice) == this->out_vertices.end()){
                this->out_vertices[vertice] = {};
            }
        }
        this->topo_order = this->get_topological_sort();
        this->num_convergent_nodes = 0;
        this->num_convergent_edges = 0;
        for (const auto& node : vertices) {
            auto neighs = this->in_vertices.at(node);
            if (neighs.size() > 1) {
                num_convergent_edges += neighs.size();
                num_convergent_nodes += 1;
            }   
        }

        // this->reconvergent_nodes = this->find_reconvergent_nodes();
        //keep the order
        compute_convergent_nodes_reached_by_divergent_paths();
        this->alpha_complexity = dfg_config.getAlphaComplexity();
        this->beta_complexity = dfg_config.getBetaComplexity();
        this->gamma_complexity = dfg_config.getBetaComplexity();
        this->dfg_value = this->calc_dfg_value();
        // // #ifdef DEBUG
        // std::cout << "Reconvergent nodes:\n" << std::flush;
        // for(auto& [src,dst]: this->reconvergent_nodes){
        //     std::cout << "( " <<this->get_name_by_node(src) << ", " << this->get_name_by_node(dst) << " )\n";
        // }
        // print_pair_vector(this->reconvergent_nodes);
        // std::cout << "DFG value: " << this->dfg_value << std::flush;
        // // #endif
        this->calc_IO_to_count();
        this->edge_to_id = create_edge_to_id(this->edge_indices);
   }
};

#endif