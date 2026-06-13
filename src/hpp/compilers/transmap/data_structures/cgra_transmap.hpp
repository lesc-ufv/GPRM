// #ifndef _CGRA_TRANSMAP_HPP
// #define _CGRA_TRANSMAP_HPP

// #include "src/hpp/data_structures/graphs/cgra.hpp"
// #include "src/hpp/utils/util_dfg.hpp"
// #include "src/hpp/utils/util_graph.hpp"
// #include <vector>
// #include <unordered_set>
// #include <map>
// #include <torch/torch.h>
// #include "src/cpp/utils/util_server.cpp"

// class CGRATransMap{
// private:
//     std::unordered_map<int, std::unordered_set<int>> occupied_PEs;
//     int fill_value;
//     std::shared_ptr<CGRA> cgra;
//     std::vector<std::vector<int>> edge_indices;
//     std::vector<std::array<int,10>> node_features;
//     std::vector<std::array<int,2>> edge_features;

//     std::unordered_map<std::pair<int,int>, int> edge_to_id;
//     std::unordered_map<int, std::pair<int,int>> id_to_edge;
//     int laplacian_emb_dims;
//     torch::Tensor laplacian_embeddings;
    

 
// public:
//     

//     CGRATransMap(const int& fill_value,
//     const unsigned int& II,
//     const std::pair<int, int>& dimensions,
//     const std::unordered_map<int, std::unordered_set<EnumFunctionalities>>& pe_to_functionalities,
//     const std::unordered_set<EnumInterconnectionStyles>& interconnection_styles, const int& laplacian_emb_dims);
//     CGRATransMap();
//     CGRATransMap(const CGRATransMap& other);

//     std::vector<std::array<int,7>> get_features_by_modulo(const int& modulo);
//     std::vector<std::vector<int>> get_edge_indices();
//     void assign_node_to_PE(int node, const int& PE);
//     const std::unordered_map<std::pair<int, int>, std::vector<int>>& get_PEs_to_routing_() const;
//     const std::unordered_map<int, std::unordered_set<int>>& get_free_connections_() const;
//     const std::unordered_map<int, std::unordered_set<int>>& get_out_vertices_() const;
//     void remove_connections(const int& father_node, const int& child_node);
//     void add_occupied_PE(int PE);
//     const std::pair<int,int> get_pe_pos(const int& PE) const;
//     bool all_PEs_in_modulo_occupied(const int& modulo) const;
//     std::unordered_set<int> get_not_occupied_PEs_by_modulo(const int& modulo);
//     int calc_relative_PE(const int& PE);
//     const std::vector<int>& get_vertices_() const;
//     const int& number_of_spatial_PEs_() const;
//     const std::pair<int, int>& get_cgra_dimensions_() const;
//     const std::unordered_map<int,std::tuple<int, int, int>>& get_PE_to_coord_() const;
//     const std::unordered_map<std::tuple<int, int, int>, int>& get_coord_to_PE_() const;

    
//     const std::unordered_set<int>& get_occupied_PEs_by_modulo_(const int& modulo) const;
//     void add_routing(const std::pair<int, int>& tgt_routing_PEs, std::vector<int>& routing_path);
//     int get_II() const;
//     void print() const;
//     const std::unordered_map<int, std::unordered_set<int>> get_free_connections() const;
//     int get_n_rows();
//     int get_n_cols();

// };

// #endif