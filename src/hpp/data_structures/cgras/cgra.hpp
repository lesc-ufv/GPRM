#ifndef CGRA_HPP
#define CGRA_HPP
#include "src/hpp/utils/hashes.hpp"
#include "src/hpp/enums/enum_functionalities.hpp"
#include "src/hpp/enums/enum_interconnection_styles.hpp"
#include <map>
#include <functional>
#include <vector>
#include <iostream>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include "src/hpp/utils/util_graph.hpp"
#include "src/hpp/utils/util_print.hpp"
#include "src/hpp/utils/cgra/util_cgra.hpp"
#include "src/hpp/utils/cgra/util_interconnections.hpp"
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include "src/externals/cereal/types/utility.hpp"       
#include "src/externals/cereal/types/tuple.hpp"         
#include "src/externals/cereal/types/unordered_map.hpp" 
#include "src/externals/cereal/types/unordered_set.hpp" 
#include "src/externals/cereal/types/vector.hpp"        
#include "src/externals/cereal/types/string.hpp"
#include <cereal/types/memory.hpp>
#include <bitset>
#include "nlohmann/json.hpp"
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, 
                              boost::no_property, boost::property<boost::edge_weight_t, int>> CGRAGraph;
typedef boost::graph_traits<CGRAGraph>::vertex_descriptor Vertex;
typedef boost::property_map<CGRAGraph, boost::edge_weight_t>::type WeightMap;

using json = nlohmann::json;

class CGRA {
private:
    unsigned int II;
    std::pair<int,int> dimensions;
    std::unordered_map<int,std::unordered_set<EnumFunctionalities>> pe_to_functionalities;
    std::unordered_set<EnumInterconnectionStyles> interconnection_styles;
    std::vector<int> PEs; //Vertices
    std::vector<std::pair<int,int>> connections; // Edges
    std::unordered_map<int, std::tuple<int, int, int>> pe_to_coord;
    std::unordered_map<std::tuple<int, int, int>, int> coord_to_pe;
    std::unordered_map<std::pair<int,int>, std::vector<int>> PEs_to_routing;    
    std::unordered_map<int,std::unordered_set<int>> free_connections;
    std::unordered_map<int,std::unordered_set<int>> in_vertices;
    std::unordered_map<int,std::unordered_set<int>> out_vertices;
    CGRAGraph graph;
    int number_of_spatial_PEs;
    std::unordered_map<std::pair<int,int>,int> edge_to_id;
    std::vector<std::vector<int>> edge_indices;
    std::unordered_map<int, std::unordered_map<int,std::vector<int>>> node_to_out_dist_to_nodes;
    std::unordered_map<int, std::unordered_map<int,std::vector<int>>> node_to_in_dist_to_nodes;
    int num_spatial_edges = -1;
    std::unordered_map<std::pair<int,int>, int> IO_to_count; 

public:
    CGRA( const unsigned int& II,
         const std::pair<int,int>& dimensions, 
         const std::unordered_map<int,std::unordered_set<EnumFunctionalities>>& pe_to_functionalities,
         const std::unordered_set<EnumInterconnectionStyles>& interconnection_styles, bool calc_node_dists);

    CGRA();

    const std::unordered_map<std::pair<int,int>, int>& get_IO_to_count() const {
        return IO_to_count;
    }
    void calc_IO_to_count(){
        IO_to_count.clear();
        for(const auto& node: this->PEs){
            auto in_edges_size = this->in_vertices.at(node).size();
            auto out_edges_size = this->out_vertices.at(node).size();
            IO_to_count[std::make_pair(in_edges_size,out_edges_size)] += 1;
        }
    }

    nlohmann::json get_interconnection_styles_json(){
        nlohmann::json js;
        std::unordered_map<std::string, std::string> style_to_key = {
            {enum_connection_style_to_string(EnumInterconnectionStyles::MESH), "mesh"},
            {enum_connection_style_to_string(EnumInterconnectionStyles::DIAGONAL), "diagonal"},
            {enum_connection_style_to_string(EnumInterconnectionStyles::ONE_HOP), "one_hop"},
            {enum_connection_style_to_string(EnumInterconnectionStyles::TOROIDAL), "toroidal"}
        };
        for(auto style: {EnumInterconnectionStyles::MESH, EnumInterconnectionStyles::DIAGONAL,
                         EnumInterconnectionStyles::ONE_HOP, EnumInterconnectionStyles::TOROIDAL}){
            js[style_to_key.at(enum_connection_style_to_string(style))] = false;
        }
        for(auto style: interconnection_styles){
            js[style_to_key.at(enum_connection_style_to_string(style))] = true;
        }
        return js;
    }
    std::string get_arch_name() const {
        return get_interconnection_styles_str();
    }
    int get_num_spatial_edges(){
        if(this->II == 1) return this->get_num_edges();
        if(this->num_spatial_edges != -1) return this->num_spatial_edges;
        this->num_spatial_edges = 0;
        for(int pe = 0; pe < this->number_of_spatial_PEs; pe++){
            int num_conn = this->out_vertices.at(pe).size();
            this->num_spatial_edges += num_conn - 1;
        }
        return this->num_spatial_edges;
    }
    
    void write_cgra_md(std::stringstream& ss) const {
        auto [rows, cols] = this->get_cgra_dimensions_();
        ss << "### CGRA Summary\n";
        ss << "- **CGRA Dimensions:** " << rows << "x" << cols << "\n";
        ss << "- **Cycles (II):** " << this->II << "\n";
        ss << "- **Connections Style:** " << this->get_interconnection_styles_str() << "\n";
    }

    std::string get_interconnection_styles_str() const {
        std::string str = "";
        for(auto conn_style: interconnection_styles){
            str+= " " + enum_connection_style_to_string(conn_style) + "+";
        }

        if(!str.empty()){
            str.erase(0, 1);
        }

        str.pop_back();
        return str;
    }


    CGRA(const CGRA& other);
    // std::vector<std::pair<int, int>> get_edges() const;
    // std::vector<int> get_vertices() const;

    const std::vector<std::pair<int, int>>& get_edges_() const;
    const std::vector<int>& get_vertices_() const;

    // std::unordered_set<int> get_in_vertices_by_id(const int& id) const;
    // std::unordered_set<int> get_out_vertices_by_id(const int& id) const;
    const std::unordered_map<int,std::unordered_set<EnumFunctionalities>>& get_pe_to_functionalities() const;

    const std::unordered_set<int>& get_in_vertices_by_id_(const int& id) const;
    const std::unordered_set<int>& get_out_vertices_by_id_(const int& id) const;
    const std::pair<int,int>& get_cgra_dimensions_() const;
    const std::unordered_map<int,std::tuple<int, int, int>>& get_PE_to_coord_() const;
    const std::unordered_map< std::tuple<int, int, int>, int>& get_coord_to_PE_() const;
    const std::tuple<int, int, int>& get_coord_by_PE_(const int& PE);
    const std::unordered_map<std::pair<int, int>, std::vector<int>>& get_PEs_to_routing_() const;
    const std::unordered_map<int,std::unordered_set<int>>& get_free_connections_() const;
    const std::unordered_map<int, std::unordered_set<int>>& get_out_vertices_() const;
    const std::unordered_map<int, std::unordered_set<int>>& get_in_vertices_() const;
    void remove_connections(const int& father_node, const int& child_node);
    std::pair<int,int> get_pe_pos(const int& PE) const;
    const int& get_PE_by_coord(const std::tuple<int,int,int>& coord) {
        return this->coord_to_pe.at(coord);
    };
    const int& number_of_spatial_PEs_() const;
    void add_routing(const std::pair<int, int> tgt_routing_PEs, std::vector<int> routing_path); 
    int get_II() const;
    void print() const;
    std::unordered_map<int,std::unordered_set<int>> get_free_connections() const ;
    int get_n_rows();
    int get_n_cols();
    int total_PEs() const;
    void add_self_edges();
    const std::unordered_map<std::pair<int,int>,int>& get_edge_to_id();

    std::bitset<4> get_connection_bitset(){
        std::bitset<4> bitset_conn;
        for(auto style: interconnection_styles){
            if(style == EnumInterconnectionStyles::MESH) bitset_conn.set(static_cast<int>(EnumInterconnectionStyles::MESH));
            if(style == EnumInterconnectionStyles::ONE_HOP) bitset_conn.set(static_cast<int>(EnumInterconnectionStyles::ONE_HOP));
            if(style == EnumInterconnectionStyles::DIAGONAL) bitset_conn.set(static_cast<int>(EnumInterconnectionStyles::DIAGONAL));
            if(style == EnumInterconnectionStyles::TOROIDAL) bitset_conn.set(static_cast<int>(EnumInterconnectionStyles::TOROIDAL));
        }
        return bitset_conn;
        
    }
    std::string get_supported_operations_string_by_node(const int& node) const;
    int get_num_edges() const;
    const std::unordered_set<int>& get_in_vertices_by_node_id(const int& node) const;
    const std::unordered_set<int>& get_out_vertices_by_node_id(const int& node) const;
    const std::vector<std::vector<int>>& get_edge_indices() const;
    const std::unordered_map<int, std::unordered_map<int,std::vector<int>>>& get_node_to_in_dist_to_nodes() const;
    const std::unordered_map<int, std::unordered_map<int,std::vector<int>>>& get_node_to_out_dist_to_nodes() const;
    const std::unordered_set<EnumInterconnectionStyles>& get_interconnection_styles() const {
        return interconnection_styles;
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(II,
           dimensions);
        serialize_map_of_set_of_enum(ar, pe_to_functionalities);
        serialize_set_of_enum(ar, interconnection_styles);
        ar(PEs,
           connections,
           pe_to_coord,
           coord_to_pe,
           PEs_to_routing,
           free_connections,
           in_vertices,
           out_vertices,
           number_of_spatial_PEs,
           edge_to_id,
           edge_indices,
           node_to_out_dist_to_nodes,
           node_to_in_dist_to_nodes,
           IO_to_count);
    }
    

};  
#endif