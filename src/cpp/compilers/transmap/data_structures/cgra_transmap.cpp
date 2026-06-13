// #include "src/hpp/compilers/transmap/data_structures/cgra_transmap.hpp"

// CGRATransMap::CGRATransMap(
//     const int& fill_value,
//     const unsigned int& II,
//     const std::pair<int, int>& dimensions,
//     const std::unordered_map<int, std::unordered_set<EnumFunctionalities>>& pe_to_functionalities,
//     const std::unordered_set<EnumInterconnectionStyles>& interconnection_styles,
//     const int& laplacian_emb_dims,
//     torch::Tensor laplacian_emb
//    ) : fill_value(fill_value), laplacian_embeddings(laplacian_emb), laplacian_emb_dims(laplacian_emb_dims){ 

//     cgra = std::make_shared<CGRA>(II, dimensions, pe_to_functionalities, interconnection_styles);
//     edge_indices = create_edges_indices(cgra->get_edges_());

//     auto num_total_PEs = cgra->total_PEs();
//     auto total_edges = cgra->get_edges_().size();


//     node_features.resize(num_total_PEs, std::array<int,this->num_node_features>());
//     edge_features.resize(total_edges, std::array<int,this->num_edge_features>());

//     std::unordered_map<EnumFunctionalities, int> functionality_to_idx = {
//         {EnumFunctionalities::ADD_AND_SUB, 0},
//         {EnumFunctionalities::MULT_AND_DIV, 1},
//         {EnumFunctionalities::LOGICAL, 2},
//         {EnumFunctionalities::COMPARISON, 3},
//         {EnumFunctionalities::SHIFT, 4},
//         {EnumFunctionalities::MEMORY_ACCESS, 5}
//     };


//     for (auto& vertice : cgra->get_vertices()){
//         node_features[vertice][id_idx] = vertice;
//         for (const auto& functionality: pe_to_functionalities.at(vertice)){
//             node_features[vertice][functionalities_initial_idx + functionality_to_idx.at(functionality)] = 1;
//         }
//         node_features[vertice][in_degree_idx] = cgra->get_in_vertices_by_id(vertice).size();
//         node_features[vertice][out_degree_idx] =  cgra->get_out_vertices_by_id(vertice).size();
//         node_features[vertice][id_mapped_dfg_node_idx] =  fill_value;
//         }

//     int count = 0;
//     for (auto& edge : cgra->get_edges_()){
//         edge_to_id[edge] = count;
//         id_to_edge[count] = edge;
//         edge_features[count] = {count, fill_value};
//         count += 1;
//     }
// };

// CGRATransMap::CGRATransMap(const CGRATransMap& other)
//     : occupied_PEs(other.occupied_PEs),
//       fill_value(other.fill_value),
//       cgra(other.cgra),
//       edge_indices(other.edge_indices),
//       node_features(other.features),
//       edge_features(other.cgra_features), {
// }


// CGRATransMap::CGRATransMap(){}

// std::vector<std::array<int,10>> CGRATransMap::get_node_features(){
//     return this->node_features;
// }

// std::vector<std::array<int,2>> CGRATransMap::get_node_features(){
//     return this->node_features;
// }

// std::array<int, 10> CGRATransMap::get_features_by_PE(const int& PE){
//     return this->node_features[PE];
// }

// std::array<int, 2> CGRATransMap::get_features_by_edge(const std::pair<int,int>& edge){
//     return this->edge_features[PE];
// }


// const std::vector<std::vector<int>>& CGRATransMap::get_edge_indices()
// {
//     return this->edge_indices;
// }


// const std::unordered_map<std::pair<int, int>, std::vector<int>> &CGRATransMap::get_PEs_to_routing_() const
// {
//     return this->cgra->get_PEs_to_routing_();
// }


// const std::unordered_map<int, std::unordered_set<int>> &CGRATransMap::get_out_vertices_() const {
//     return this->cgra->get_out_vertices_();
// }

// void CGRATransMap::add_occupied_PE(int PE){
//     this->occupied_PEs.insert(PE);
// }



// const std::pair<int, int> CGRATransMap::get_pe_pos(const int &PE) const
// {
//     return this->cgra->get_pe_pos(PE);
// }

// const std::vector<int>& CGRATransMap::get_vertices_() const{
//     return this->cgra->get_vertices_();
// }

// const int& CGRATransMap::number_of_spatial_PEs_() const{
//     return this->cgra->number_of_spatial_PEs_();
// }

// const std::pair<int, int>& CGRATransMap::get_cgra_dimensions_() const{
//     return this->cgra->get_cgra_dimensions_();
// }

// const std::unordered_map<int,std::tuple<int, int, int>>& CGRATransMap::get_PE_to_coord_() const {
//     return this->cgra->get_PE_to_coord_();
// }
// const std::unordered_map<std::tuple<int, int, int>, int>& CGRATransMap::get_coord_to_PE_() const{
//     return this->cgra->get_coord_to_PE_();
// }


// const std::tuple<int, int, int>& CGRATransMap::get_coord_by_PE_(const int& PE) const{
//     return this->cgra->get_PE_to_coord_().at(PE);
// }

// void CGRATransMap::add_routing(const std::pair<int, int> &tgt_routing_PEs, std::vector<int> &routing_path)
// {
//     this->cgra->add_routing(tgt_routing_PEs, routing_path);
// }

// int CGRATransMap::get_II() const
// {
//     return this->cgra->get_II();
// }
// const std::unordered_map<int, std::unordered_set<int>> &CGRATransMap::get_free_connections_() const
// {
//     return this->cgra->get_free_connections_();
// }

// const std::unordered_map<int, std::unordered_set<int>> CGRATransMap::get_free_connections() const
// {
//     return this->cgra->get_free_connections();
// }

// void CGRATransMap::print() const {
//         std::cout << "=== CGRATransMap ===\n";
        
//         this->cgra->print();

//         std::cout << "\nFill Value: " << this->fill_value << "\n";

//         std::cout << "\nOccupied PEs:\n";
//         for (const auto& PE : this->occupied_PEs) std::cout << PE << " ";

//         std::cout << "\nEdge Indices:\n";
//         for (const auto& edge : edge_indices) {
//             std::cout << "[ ";
//             for (int index : edge) {
//                 std::cout << index << " ";
//             }
//             std::cout << "]\n";
//         }

//         std::cout << "\n Node Features:\n";
//         print_matrix(this->node_features);
        
//         std::cout << "\n Edge Features:\n";
//         print_matrix(this->edge_features);

//         std::cout << "=== End of CGRATransMap ===\n";
//     }

// int CGRATransMap::get_n_rows(){
//     return this->cgra->get_n_cols();
// }

// int CGRATransMap::get_n_cols(){
//     return this->cgra->get_n_cols();
// }