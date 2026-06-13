// #ifndef TRANSMAP_DFG_H
// #define TRANSMAP_DFG_H

// #include "src/hpp/data_structures/dfgs/dfg.hpp"
// #include "src/hpp/enums/enum_operation.hpp"
// #include "src/hpp/utils/util_dfg.hpp"
// #include "src/hpp/utils/util_print.hpp"
// #include "src/hpp/utils/util_graph.hpp"
// #include "src/hpp/utils/cgra/util_cgra.hpp"
// #include <iostream>
// #include <vector>
// #include <tuple>
// #include <map>
// #include <fstream>
// #include <sstream>
// #include <unordered_set>
// #include <unordered_map>
// #include <cmath>
// #include <array>
// #include <string>
// #include <stdexcept>
// class DFGTransMap{
// private:
//     int II;
//     int fill_value;
//     std::vector<std::array<int, 7>> node_features;
//     std::vector<std::array<int, 4>> edge_features;
//     std::unordered_map<int,int> node_to_level;
//     std::vector<int> placement_order;
//     std::shared_ptr<DFG> dfg;
//     std::unordered_map<int,int> node_to_planned_modulo_time_slice;
//     std::unordered_map<int,int> node_to_scheduled_time_slice;    
//     int laplacian_emb_dims;
//     torch::Tensor laplacian_embeddings;
//     std::unordered_map<std::pair<int,int>, int> edge_to_id;
//     std::unordered_map<int, std::pair<int,int>> id_to_edge;

// public:
//    
    

//     DFGTransMap(const std::string& dot_filename,
//                const int II,
//                const int num_PEs,
//                const int fill_value,
//                int laplacian_emb_dims );

//     DFGTransMap();


//     DFGTransMap(const DFGTransMap& other);

//     std::vector<std::array<int,7>> get_node_features() const;
//     std::vector<std::array<int,4>> get_edge_features() const;
//     std::array<int,10> get_features_by_node(const int& node_id) const;
//     const std::vector<std::vector<int>>& get_edge_indices() const;
//     const std::string get_real_node(const int& node_id) const;
//     void assign_pe_to_node(int PE, int node);
//     void assign_sched_modulo_time_slice_to_node(int II, int node);
//     void assign_sched_time_slice_to_node(int time, const int& node);
//     int number_of_nodes() const;
//     const std::vector<int>& get_vertices_() const ;
//     const int& get_node_in_placement_order_by_idx_(const int& idx) const;
//     const int& get_node_level_by_node_(const int& node) const;
//     const int& get_planned_modulo_time_slice_(const int& node) const;
//     int get_planned_modulo_time_slice(const int& node) const;
//     const std::unordered_set<int>& get_in_vertices_by_node_(const int& node) const; 
    
//     int get_assigned_PE_by_node(const int& node) const;
//     int get_node_scheduled_time_slice(const int& node) const;
//     const std::unordered_map<int, int>& get_node_to_scheduled_time_slice_() const;
//     const std::unordered_map<int, std::unordered_set<int>>& get_node_to_in_vertices_() const;
//     const std::unordered_map<int, std::unordered_set<int>>& get_node_to_out_vertices_() const;

//     const std::unordered_map<int,int>& get_node_to_PE_() const;
//     const int& get_II() const;
//     void print() const;
//     const std::vector<int>& get_root_nodes() const;

//     };  

// #endif
