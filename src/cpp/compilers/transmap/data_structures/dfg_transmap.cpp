// #include "src/hpp/compilers/transmap/data_structures/dfg_transmap.hpp"

// DFGTransMap::DFGTransMap(const DFGTransMap& other)
//     : II(other.II),  
//       fill_value(other.fill_value),
//       node_features(other.node_features),
//       edge_features(other.edge_features),
//       node_to_level(other.node_to_level),
//       placement_order(other.placement_order),
//       dfg(other.dfg), 
//       node_to_planned_modulo_time_slice(other.node_to_planned_modulo_time_slice),
//       node_to_scheduled_time_slice(other.node_to_scheduled_time_slice)
// {

// }



// DFGTransMap::DFGTransMap(){}

// std::vector<std::array<int,10>> DFGTransMap::get_features() const {
//     return this->features;
// }

// std::array<int,10> DFGTransMap::get_features_by_node(const int& node_id) const {
//     if (node_id < 0 ||node_id >=  static_cast<int>(features.size())) {
//         throw std::out_of_range("Node ID is out of range");
//     }
//     return this->features[node_id];
// }

// const std::vector<std::vector<int>>& DFGTransMap::get_edge_indices() const {
//     return this->dfg->get_edge_indices();
// }

// void DFGTransMap::assign_pe_to_node(int PE, int node) {
//     if (node < 0 || node >= static_cast<int>(this->features.size())) {
//         throw std::out_of_range("Node ID is out of range");
//     }
//     this->features[node][this->assigned_pe_idx] = PE;
//     this->node_to_PE[node] = PE;
// }

// void DFGTransMap::assign_sched_modulo_time_slice_to_node(int II, int node) {
//     if (node < 0 || node >= static_cast<int>(this->features.size())) {
//         throw std::out_of_range("Node ID is out of range");
//     }
//     this->features[node][sched_mod_time_slice_idx] = II;
// }

// void DFGTransMap::assign_sched_time_slice_to_node(int time, const int& node) {
//     if (node < 0 || node >= static_cast<int>(this->features.size())) {
//         throw std::out_of_range("Node ID is out of range");
//     }
//     this->features[node][sched_time_slice_idx] = time;
//     this->node_to_scheduled_time_slice[node] = time;
// }

// int DFGTransMap::number_of_nodes()
// const{
//     return this->dfg->num_vertices();
// }


// const std::vector<int>& DFGTransMap::get_vertices_() const {
//     return this->dfg->get_vertices();
// }

// const int& DFGTransMap::get_node_in_placement_order_by_idx_(const int& idx) const{
//     return this->placement_order[idx];
// }

// const int &DFGTransMap::get_node_level_by_node_(const int &node) const
// {
//     return this->node_to_level.at(node);
// }

// const int& DFGTransMap:: get_planned_modulo_time_slice_(const int& node) const {
//     return this->node_to_planned_modulo_time_slice.at(node);
// }

// int DFGTransMap::get_planned_modulo_time_slice(const int& node) const {
//     return this->node_to_planned_modulo_time_slice.at(node);
// }

// const std::unordered_set<int>& DFGTransMap::get_in_vertices_by_node_(const int& node) const {
//     return this->dfg->get_node_to_in_vertices().at(node);
// }

// int DFGTransMap::get_assigned_PE_by_node(const int& node) const {
//     return this->features[node].at(this->assigned_pe_idx);
// }

// int DFGTransMap::get_node_scheduled_time_slice(const int &node) const
// {
//     return this->features[node][this->sched_time_slice_idx];
// }

// const std::unordered_map<int, std::unordered_set<int>> &DFGTransMap::get_node_to_in_vertices_() const
// {
//     return this->dfg->get_node_to_in_vertices();
// }

// const std::unordered_map<int, std::unordered_set<int>> &DFGTransMap::get_node_to_out_vertices_() const
// {
//     return this->dfg->get_node_to_out_vertices();
// }


// const std::unordered_map<int, int>& DFGTransMap::get_node_to_scheduled_time_slice_() const{
//     return this->node_to_scheduled_time_slice;
// }

// const std::string DFGTransMap::get_real_node(const int& node_id) const{
//     return this->dfg->get_name_by_node(node_id);
// }


// void DFGTransMap::print() const {
//     std::cout << "=== DFGTransMap ===\n";
    
//     std::cout << "II: " << II << "\n";
//     std::cout << "Fill Value: " << fill_value << "\n";
    
//     std::cout << "\nVertices:\n";
//     auto vertices = this->dfg->get_vertices();
//     print_vector(vertices);
    
//     std::cout << "\nEdges:\n";
//     print_pair_vector(this->dfg->get_edges());
    
//     std::cout << "\nNode Features:\n";
//     print_matrix(node_features);

//     std::cout << "\nEdge Features:\n";
//     print_matrix(edge_features);
    
//     std::cout << "\nEdge Indices:\n";
//     print_matrix(this->dfg->get_edge_indices());
    
//     std::cout << "\nNode to Level:\n";
//     print_element_key_element_value(node_to_level);
    
//     std::cout << "\n\nPlacement Order:\n";
//     print_vector(placement_order);
    
//     std::cout << "\n\nNode to Planned Modulo Time Slice:\n";
//     print_element_key_element_value(node_to_planned_modulo_time_slice);
    
//     std::cout << "\n\nIn Vertices:\n";
//     auto in_vertices = this->dfg->get_node_to_in_vertices();
//     print_element_key_list_value(in_vertices);
    
//     std::cout << "\nOut Vertices:\n";
//     auto out_vertices = this->dfg->get_node_to_out_vertices();
//     print_element_key_list_value(out_vertices);
    
//     std::cout << "\nNode to Scheduled Time Slice:\n";
//     print_element_key_element_value(node_to_scheduled_time_slice);
    
//     std::cout << "\nNode to PE:\n";
//     print_element_key_element_value(node_to_PE);

//     std::cout << "\nGraph Properties:\n";
//     std::cout << "Node to Operation:\n";
//     for (const auto& node : vertices) {
//         std::cout << "Node " << node << " -> " << this->dfg->get_operation_by_node(node) << "\n";
//     }

//     std::cout << "\nNode to Name:\n";
//     for (const auto& node : vertices ) {
//         std::cout << "Node " << node << " -> " << this->dfg->get_name_by_node(node) << "\n";
//     }
    
//     std::cout << "=== End of DFGTransMap ===\n\n";
// }


// const int& DFGTransMap::get_II() const{
//     return this->II;
// }

// const std::vector<int>& DFGTransMap::get_root_nodes() const {
//     return this->dfg->get_root_nodes();
// }