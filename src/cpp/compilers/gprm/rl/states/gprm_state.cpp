#include "src/hpp/compilers/gprm/rl/states/gprm_state.hpp"


GPRMState::GPRMState(const GPRMState& other) : 
         BaseState(other), 
          placed_id(other.placed_id),
          not_placed_id(other.not_placed_id),
          to_place_id(other.to_place_id),
          to_not_place_id(other.to_not_place_id),
          mapping_node_to_id(other.mapping_node_to_id),
          lpe_dfg(other.lpe_dfg),
          dfg_placement_order_ids(other.dfg_placement_order_ids),
          dfg_node_scheduling_ids(other.dfg_node_scheduling_ids),
          dfg_edge_indices(other.dfg_edge_indices),
          dfg_operation_ids(other.dfg_operation_ids),
          lpe_adg(other.lpe_adg),
          adg_pos(other.adg_pos),
          adg_edge_indices(other.adg_edge_indices),
          adg_operation_ids(other.adg_operation_ids),
        //   dfg_routing_type_ids(other.dfg_routing_type_ids.clone()),
        //   dfg_placement_type_ids(other.dfg_placement_type_ids.clone()),
        //   adg_placement_type_ids(other.adg_placement_type_ids.clone()),
        //   adg_routing_type_ids(other.adg_routing_type_ids.clone()),
          mapping_edge_indices(other.mapping_edge_indices.clone()),
          placements(other.placements.clone()),
          routings(other.routings.clone()),
          action_batch_idx(torch::empty({0})),
          batch_idx(torch::empty({0})),
          seq_idx(torch::empty({0})),
          contextual_action_seq_idx(torch::empty({0})),
          action_seq_idx(torch::empty({0})),
          state_seq_idx(torch::empty({0})),
          max_range_values(other.max_range_values),
          max_zeros_values(other.max_zeros_values)
        //   max_seq_len(),
        //   max_legal_actions()
{} 

GPRMState::GPRMState(GPRMState&& other) noexcept : BaseState(std::move(other)) {
    this->placed_id = other.placed_id;
    this->not_placed_id = other.not_placed_id;
    this->to_place_id = other.to_place_id;
    this->to_not_place_id = other.to_not_place_id;
    this->mapping_node_to_id = std::move(other.mapping_node_to_id);
    this->lpe_dfg = std::move(other.lpe_dfg);
    this->dfg_placement_order_ids = std::move(other.dfg_placement_order_ids);
    this->dfg_node_scheduling_ids = std::move(other.dfg_node_scheduling_ids);
    this->dfg_edge_indices = std::move(other.dfg_edge_indices);
    this->dfg_operation_ids = std::move(other.dfg_operation_ids);
    this->lpe_adg = std::move(other.lpe_adg);
    this->adg_pos = std::move(other.adg_pos);
    this->adg_edge_indices = std::move(other.adg_edge_indices);
    this->adg_operation_ids = std::move(other.adg_operation_ids);
    this->mapping_edge_indices = std::move(other.mapping_edge_indices);
    this->placements = std::move(other.placements);
    this->routings = std::move(other.routings);
    this->action_batch_idx = std::move(other.action_batch_idx);
    this->batch_idx = std::move(other.batch_idx);
    this->seq_idx = std::move(other.seq_idx);
    this->contextual_action_seq_idx = std::move(other.contextual_action_seq_idx);
    this->action_seq_idx = std::move(other.action_seq_idx);
    this->state_seq_idx = std::move(other.state_seq_idx);
    this->max_seq_len = other.max_seq_len;
    this->max_legal_actions = other.max_legal_actions;    
    this->max_range_values = std::move(other.max_range_values);
    this->max_zeros_values = std::move(other.max_zeros_values);
}
GPRMState::GPRMState(std::shared_ptr<OrderedDFG> dfg, std::shared_ptr<CGRA> cgra,
    std::shared_ptr<zmq::socket_t> socket, const ServerConstants& server_k, 
    const GPRMConfig& gprm_config, StateConfig& state_config) 
: BaseState(dfg, cgra, state_config) 
{
    auto lpe_dim = gprm_config.getLpeDim();
    // this->placed_id = gprm_config.getPlacedId();
    // this->to_place_id = gprm_config.getPlacementOptionId();
    // this->not_placed_id = gprm_config.getNotPlacedId();
    // this->to_not_place_id = gprm_config.getNotPlacementOptionId();

    auto vocab = gprm_config.getVocab();

    // WARN: Replace with correct operations if necessary 
    auto adg_sup_operation_id = vocab.at("Any");

    #ifdef DEBUG
    std::cout << "LPE Dimension: " << lpe_dim << std::endl;
    std::cout << "Placed ID: " << this->placed_id << std::endl;
    std::cout << "To Place ID: " << this->to_place_id << std::endl;
    std::cout << "Not Placed ID: " << this->not_placed_id << std::endl;
    std::cout << "ADG Sup Operation ID: " << adg_sup_operation_id << std::endl;
    #endif

    auto adg_edge_idx = create_edges_indices(this->cgra->get_edges_());
    auto dfg_edge_idx = create_edges_indices(this->dfg->get_edges());
    this->adg_edge_indices = matrix_to_tensor<int, int>(adg_edge_idx).to(torch::kInt64);
    this->dfg_edge_indices = matrix_to_tensor<int, int>(dfg_edge_idx).to(torch::kInt64);


    auto max_possible_seq_len = this->calc_max_possible_seq_len();
    this->max_range_values = torch::arange(0,max_possible_seq_len, torch::kInt32);
    this->max_zeros_values = torch::zeros({max_possible_seq_len}, torch::kInt32);

    #ifdef DEBUG
    std::cout << "ADG Edge Indices: " << adg_edge_idx.size() << " edges." << std::endl;
    std::cout << "DFG Edge Indices: " << dfg_edge_idx.size() << " edges." << std::endl;
    #endif

    this->lpe_adg = get_laplacian_embeddings(socket, "gen_laplacian_embeddings", adg_edge_idx,
                                lpe_dim, cgra->total_PEs(), false, server_k);
    this->lpe_dfg = get_laplacian_embeddings(socket, "gen_laplacian_embeddings", dfg_edge_idx,
                                lpe_dim, dfg->num_vertices(), false, server_k);

    #ifdef DEBUG
    std::cout << "LPE ADG: " << this->lpe_adg.sizes() << std::endl;
    std::cout << "LPE DFG: " << this->lpe_dfg.sizes() << std::endl;
    #endif

    this->adg_operation_ids = torch::full({this->cgra->total_PEs()}, adg_sup_operation_id, torch::kInt32);
    this->dfg_operation_ids = torch::full({this->dfg->num_vertices()}, adg_sup_operation_id, torch::kInt32);

    #ifdef DEBUG
    std::cout << "ADG Operation IDs Tensor: " << this->adg_operation_ids.sizes() << std::endl;
    std::cout << "DFG Operation IDs Tensor: " << this->dfg_operation_ids.sizes() << std::endl;
    #endif

    for (const auto& v : this->dfg->get_vertices()) {
    this->dfg_operation_ids[v] = vocab.at(this->dfg->get_operation_by_node(v));
    }

    for (const auto& v : this->cgra->get_vertices_()) {
    this->adg_operation_ids[v] = vocab.at(this->cgra->get_supported_operations_string_by_node(v));
    }

    #ifdef DEBUG
    std::cout << "Updated DFG Operation IDs:  " <<  this->dfg_operation_ids << std::endl;

    std::cout << "Updated ADG Operation IDs: " << this->adg_operation_ids  << std::endl;
    #endif

    const auto& topological_order = this->dfg->get_topological_sort();

    #ifdef DEBUG
    std::cout << "Topological Order Size: " << topological_order.size() << std::endl;
    #endif

    auto asap_scheduling = torch::tensor(
    calc_scheduling(topological_order, this->dfg->get_node_to_in_vertices(),
        this->dfg->get_node_to_out_vertices(), this->dfg->get_root_nodes(), true), 
        torch::kInt32);

    auto alap_scheduling = torch::tensor(
    calc_scheduling(topological_order, this->dfg->get_node_to_in_vertices(),
        this->dfg->get_node_to_out_vertices(), this->dfg->get_root_nodes(), false), 
        torch::kInt32);

    #ifdef DEBUG
    std::cout << "ASAP Scheduling: " << asap_scheduling.sizes() << std::endl;
    std::cout << "ALAP Scheduling: " << alap_scheduling.sizes() << std::endl;
    #endif
    std::vector<int> dfg_placement_order_vec = std::vector<int>(this->dfg->num_vertices(), -1);
    int order = 0;
    for(const auto& v: this->dfg->get_placement_order()) {
        if(v != -1){
            dfg_placement_order_vec[v] = order;
            order++;
        }
    }
    this->dfg_placement_order_ids = torch::tensor(dfg_placement_order_vec, torch::kInt32);
    this->dfg_node_scheduling_ids = torch::cat({asap_scheduling, alap_scheduling}, -1).reshape({-1,this->dfg->num_vertices()}).to(torch::kInt32);
    // this->dfg_placement_type_ids = torch::empty({2, this->dfg->num_vertices()}, torch::kInt32);
    // this->dfg_placement_type_ids[0] = torch::full({this->dfg->num_vertices()}, not_placed_id, torch::kInt32);
    // this->dfg_placement_type_ids[1] = torch::full({this->dfg->num_vertices()}, to_not_place_id, torch::kInt32);
    
    #ifdef DEBUG
    std::cout << "DFG Node Scheduling IDs: " << this->dfg_node_scheduling_ids.sizes() << std::endl;
    #endif

    std::vector<std::vector<int>> adg_pos_vec;
    for (int i = 0; i < this->cgra->total_PEs(); i++) {
    const auto& coord = this->cgra->get_coord_by_PE_(i);
    std::vector<int> cur_pos = {std::get<0>(coord), std::get<1>(coord), std::get<2>(coord)};
    adg_pos_vec.push_back(std::move(cur_pos));
    }

    #ifdef DEBUG
    std::cout << "ADG Positions Vector Size: " << adg_pos_vec.size() << std::endl;
    #endif

    adg_pos = matrix_to_tensor<int, int>(adg_pos_vec);

    #ifdef DEBUG
    std::cout << "ADG Positions Tensor Size: " << adg_pos.sizes() << std::endl;
    #endif

    // auto node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);

    // #ifdef DEBUG
    // std::cout << "Node to be Mapped ID: " << node_to_be_mapped << std::endl;
    // #endif
    // this->dfg_routing_type_ids = torch::empty({2, this->dfg->get_num_edges()}, torch::kInt32);
    // this->dfg_routing_type_ids[0] = torch::full({this->dfg->get_num_edges()}, this->not_placed_id, torch::kInt32);
    // this->dfg_routing_type_ids[1] = torch::full({this->dfg->get_num_edges()}, this->to_not_place_id, torch::kInt32);

    // this->adg_routing_type_ids = torch::empty({2, this->cgra->get_num_edges()}, torch::kInt32);

    // this->adg_routing_type_ids[0] = torch::full({this->cgra->get_num_edges()}, this->not_placed_id, torch::kInt32);
    // this->adg_routing_type_ids[1] = torch::full({this->cgra->get_num_edges()}, this->to_not_place_id, torch::kInt32);

    // update_dfg_routing_type_by_id(node_to_be_mapped, this->to_place_id);
    // update_adg_routing_type_by_id(node_to_be_mapped, this->to_place_id);

    // this->dfg_placement_type_ids[1][node_to_be_mapped] = to_place_id;
    
    // this->adg_placement_type_ids = torch::empty({2,this->cgra->total_PEs()},torch::kInt32);
    // this->adg_placement_type_ids[0] = torch::full({this->cgra->total_PEs()}, this->not_placed_id, torch::kInt32);
    // this->adg_placement_type_ids[1] = torch::full({this->cgra->total_PEs()}, this->to_not_place_id, torch::kInt32);
    

    // #ifdef DEBUG
    // std::cout << "DFG Routing Type IDs: " << this->dfg_routing_type_ids.sizes() << std::endl;
    // std::cout << "ADG Routing Type IDs: " << this->adg_routing_type_ids.sizes() << std::endl;
    // #endif

    // for (auto& action : legal_actions) {
    //     this->adg_placement_type_ids[1][action] = to_place_id;
    // }

    #ifdef DEBUG
    std::cout << "Updated ADG Placement Type IDs: ";
    for (auto& action : legal_actions) {
    std::cout << action << " ";
    }
    std::cout << std::endl;
    #endif

    this->placements = torch::tensor({}, torch::kInt32);
    this->routings = torch::tensor({}, torch::kInt32);
    this->mapping_edge_indices = torch::tensor({}, torch::kInt64);

    #ifdef DEBUG
    std::cout << "Placements Tensor Size: " << this->placements.sizes() << std::endl;
    std::cout << "Routings Tensor Size: " << this->routings.sizes() << std::endl;
    std::cout << "Mapping Edge Indices Size: " << this->mapping_edge_indices.sizes() << std::endl;
    #endif

    this->update_idx_and_mask();

    #ifdef DEBUG
    std::cout << "GPRMState initialization complete." << std::endl;
    #endif
    this->initialize();
}

// void GPRMState::update_dfg_routing_type_by_id(const int& node_to_be_mapped, const int& id){
//     auto dfg_edge_to_id = this->dfg->get_edge_to_id();
//     auto in_vertices = this->dfg->get_in_vertices_by_node_id(node_to_be_mapped);
//     auto out_vertices = this->dfg->get_out_vertices_by_node_id(node_to_be_mapped);

//     #ifdef DEBUG
//         std::cout << "Node to be Mapped: " << node_to_be_mapped << std::endl;
//         std::cout << "In Vertices for Node " << node_to_be_mapped << ": ";
//         for (auto& node : in_vertices) {
//             std::cout << node << " ";
//         }
//         std::cout << std::endl;
    
//         std::cout << "Out Vertices for Node " << node_to_be_mapped << ": ";
//         for (auto& node : out_vertices) {
//             std::cout << node << " ";
//         }
//         std::cout << std::endl;
    
//         std::cout << "DFG Edge to ID mapping size: " << dfg_edge_to_id.size() << std::endl;
//     #endif
    
//     for (auto& node : in_vertices) {
//         if (this->node_was_mapped(node)) {
//             auto edge_id = dfg_edge_to_id.at(std::make_pair(node, node_to_be_mapped));
            
//         #ifdef DEBUG
//                 std::cout << "In Vertex " << node << " was mapped. Setting edge_id: " << edge_id << std::endl;
//                 assert(this->dfg_routing_type_ids[0][edge_id].item<int>() != (this->placed_id));
//         #endif
    
//             this->dfg_routing_type_ids[1][edge_id] = id;
//         }
//     }
    
//     for (auto& node : out_vertices) {
//         if (this->node_was_mapped(node)) {
//             auto edge_id = dfg_edge_to_id.at(std::make_pair(node_to_be_mapped, node));
            
//     #ifdef DEBUG
//             std::cout << "Out Vertex " << node << " was mapped. Setting edge_id: " << edge_id << std::endl;
//             assert(this->dfg_routing_type_ids[0][edge_id].item<int>() != (this->placed_id));

//     #endif
    
//             this->dfg_routing_type_ids[1][edge_id] = id;
//         }
//     }
    
// }

// void GPRMState::update_adg_routing_type_by_id(const int& node_to_be_mapped, const int& id){
//     auto cgra_edge_to_id = this->cgra->get_edge_to_id();
//     auto in_vertices = this->dfg->get_in_vertices_by_node_id(node_to_be_mapped);
//     auto out_vertices = this->dfg->get_out_vertices_by_node_id(node_to_be_mapped);

// #ifdef DEBUG
//     std::cout << "Node to be Mapped: " << node_to_be_mapped << std::endl;
//     std::cout << "In Vertices for Node " << node_to_be_mapped << ": ";
//     for (auto& node : in_vertices) {
//         std::cout << node << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "Out Vertices for Node " << node_to_be_mapped << ": ";
//     for (auto& node : out_vertices) {
//         std::cout << node << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "CGRA Edge to ID size: " << cgra_edge_to_id.size() << std::endl;
// #endif

//     for (auto& node : in_vertices){
//         if (this->node_was_mapped(node)){
//             auto pe = this->node_to_pe.at(node);
            
// #ifdef DEBUG
//             std::cout << "In Vertex " << node << " was mapped. PE: " << pe << std::endl;
// #endif

//             for (auto& out_pe : this->cgra->get_out_vertices_by_node_id(pe)){
//                 auto edge_id = cgra_edge_to_id.at(std::make_pair(pe, out_pe));
//                 if(pe_to_node.find(out_pe) != pe_to_node.end()){
//                     continue;
//                 }
//                 if(this->adg_routing_type_ids[0][edge_id].item<int>() != (this->placed_id)){
//                 #ifdef DEBUG
//                 std::cout << "Mapping In Vertex " << node << " to PE " << pe << " with Edge ID: " << edge_id << " (Out PE: " << out_pe << ")" << std::endl;
//     #endif
//                     this->adg_routing_type_ids[1][edge_id] = id;
//             }
//         }
//         }
//     }

//     for (auto& node : out_vertices){
//         if (this->node_was_mapped(node)){
//             auto pe = this->node_to_pe.at(node);

// #ifdef DEBUG
//             std::cout << "Out Vertex " << node << " was mapped. PE: " << pe << std::endl;
// #endif
//         for (auto& in_pe : this->cgra->get_in_vertices_by_node_id(pe)){
//             auto edge_id = cgra_edge_to_id.at(std::make_pair(in_pe, pe));
            
//             if(pe_to_node.find(in_pe) != pe_to_node.end()){
//                 continue;
//             }
    
//                 if(this->adg_routing_type_ids[0][edge_id].item<int>() != (this->placed_id)){
//                     #ifdef DEBUG
//                         std::cout << "Mapping Out Vertex " << node << " to PE " << pe << " with Edge ID: " << edge_id << " (In PE: " << in_pe << ")" << std::endl;
//                     #endif

//                     this->adg_routing_type_ids[1][edge_id] = id;
//             }
//             }
//         }
//     }
// }

void GPRMState::update_idx_and_mask() {
    this->max_seq_len = this->calc_max_seq_len();

#ifdef DEBUG
    std::cout << "Max sequence length: " << this->max_seq_len << std::endl;
#endif

    this->max_legal_actions = this->legal_actions.size();

#ifdef DEBUG
    std::cout << "Legal actions size: " << max_legal_actions << std::endl;
#endif

    // this->batch_idx = torch::zeros({this->max_seq_len},torch::kInt32);
    this->batch_idx = this->max_zeros_values.slice(0, 0, this->max_seq_len);

#ifdef DEBUG
    std::cout << "Batch Index shape: " << this->batch_idx.sizes() << std::endl;
#endif

    // this->seq_idx = torch::arange(0, max_seq_len,torch::kInt32);
    this->seq_idx = this->max_range_values.slice(0, 0, this->max_seq_len);

#ifdef DEBUG
    std::cout << "Sequence Index: ";
    for (int i = 0; i < this->seq_idx.size(0); i++) {
        std::cout << this->seq_idx[i].item<int>() << " ";
    }
    std::cout << std::endl;
#endif

    this->contextual_action_seq_idx = torch::tensor(this->legal_actions,torch::kInt32) + this->dfg->num_vertices() + this->dfg->get_num_edges();

#ifdef DEBUG
    std::cout << "Contextual Action Sequence Index: ";
    for (int i = 0; i < this->contextual_action_seq_idx.size(0); i++) {
        std::cout << this->contextual_action_seq_idx[i].item<int>() << " ";
    }
    std::cout << std::endl;
#endif

     // this->action_seq_idx = torch::arange(0, this->legal_actions.size(),torch::kInt32);Add commentMore actions
     this->action_seq_idx = this->max_range_values.slice(0, 0, this->legal_actions.size());
     // this->action_batch_idx= torch::zeros(this->legal_actions.size(),torch::kInt32);
     this->action_batch_idx = this->max_zeros_values.slice(0,0, this->legal_actions.size());

#ifdef DEBUG
    std::cout << "Action Sequence Index: ";
    for (int i = 0; i < this->action_seq_idx.size(0); i++) {
        std::cout << this->action_seq_idx[i].item<int>() << " ";
    }
    std::cout << std::endl;
#endif

    this->state_seq_idx = torch::full({1, 1}, this->max_seq_len - 1,torch::kInt32);

#ifdef DEBUG
    std::cout << "State Sequence Index: " << this->state_seq_idx << std::endl;
#endif
}


int GPRMState::calc_max_seq_len() const {
    #ifdef DEBUG
    std::cout << "Calculating max sequence length...\n";
    std::cout << "Object address: " << this << std::endl;
    std::cout << "DFG num vertices: " << this->dfg->num_vertices() << std::endl;
    std::cout << "CGRA total PEs: " << this->cgra->total_PEs() << std::endl;
    std::cout << "DFG num edges: " << this->dfg->get_num_edges() << std::endl;
    std::cout << "CGRA num edges: " << this->cgra->get_num_edges() << std::endl;
    std::cout << "Placements size(1): " << this->placements << std::endl;
    std::cout << "Routings size(1): " << this->routings << std::endl;
    #endif

    return this->dfg->num_vertices() +
            this->dfg->get_num_edges() +
            this->cgra->total_PEs() +
        this->cgra->get_num_edges() +
        (this->placements.numel() == 0 ? 0 : this->placements.size(1)) +
        (this->routings.numel() == 0 ?  0 : this->routings.size(1)) + 1; // +1 for state
}


void GPRMState::update_features_before_state_step(const int& action) {
    auto id_node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);
    
#ifdef DEBUG
    std::cout << "Node to be mapped (ID): " << id_node_to_be_mapped << std::endl;
#endif

    // this->dfg_placement_type_ids[0][id_node_to_be_mapped] = this->placed_id;

#ifdef DEBUG
    std::cout << "Updated DFG placement type for node " << id_node_to_be_mapped << " to placed_id: " << this->placed_id << std::endl;
#endif

    // auto legal_actions_tensor = torch::tensor(this->legal_actions, torch::kInt32);
    // this->adg_placement_type_ids[0].index_put_({legal_actions_tensor}, this->not_placed_id);

#ifdef DEBUG
    std::cout << "ADG placement type IDs updated with not_placed_id: " << this->not_placed_id << std::endl;
#endif

    // auto next_node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped + 1);
    // if (next_node_to_be_mapped != -1) {
        // this->dfg_placement_type_ids[1][id_node_to_be_mapped] = this->to_not_place_id;
        // this->dfg_placement_type_ids[1][next_node_to_be_mapped] = this->to_place_id;

    // #ifdef DEBUG
    //         std::cout << "Next node to be mapped is -1, updating DFG placement type for node " << id_node_to_be_mapped << " to to_place_id: " << this->to_place_id << std::endl;
    // #endif
    // }

    std::vector<std::vector<int>> cur_placements = {{id_node_to_be_mapped}, {action}};
    
#ifdef DEBUG
    std::cout << "Current placements: Node " << id_node_to_be_mapped << " mapped to action " << action << std::endl;
#endif

    add_routing_nodes_placement(cur_placements, id_node_to_be_mapped, action);
    // update_dfg_routing_type_by_id(id_node_to_be_mapped, this->placed_id);
    // this->adg_routing_type_ids[1].masked_fill_(
    //     this->adg_routing_type_ids[1] == this->to_place_id,
    //     this->to_not_place_id
    // );
        // update_adg_routing_type_by_id(id_node_to_be_mapped, this->not_placed_id);

    auto initial_id = this->placements.numel() == 0 ? 0 : this->placements.size(1);
    for (int i = 0; i < static_cast<int>(cur_placements[0].size()); i++) {
        auto dfg_node = cur_placements[0][i];
        auto cgra_node = cur_placements[1][i];
        this->pe_to_node[cgra_node] = dfg_node;

        mapping_node_to_id[{dfg_node, cgra_node}] = initial_id;

#ifdef DEBUG
        std::cout << "Mapping DFG node " << dfg_node << " to CGRA node " << cgra_node << " with initial ID " << initial_id << std::endl;
#endif

        initial_id += 1;
    }

    auto cur_placements_tensor = matrix_to_tensor<int, int>(cur_placements);
    // this->adg_placement_type_ids[0].index_put_({cur_placements_tensor[1]}, this->placed_id);
    this->placements = torch::cat({this->placements, cur_placements_tensor}, 1);

#ifdef DEBUG
    std::cout << "Updated ADG placement type IDs and placements tensor size: " << this->placements.sizes() << std::endl;
#endif

    std::vector<std::vector<int>> cur_routings = {{}, {}};
    this->add_routing(cur_routings, id_node_to_be_mapped, action);

#ifdef DEBUG
    std::cout << "Cur routings after add_routing: \n" << cur_routings << std::endl;
#endif

    if (cur_routings.size() != 0) {
        auto cur_routings_tensor = matrix_to_tensor<int, int>(cur_routings);
        this->routings = torch::cat({this->routings, cur_routings_tensor}, 1);

#ifdef DEBUG
        std::cout << "Updated routings tensor size: \n" << this->routings << std::endl;
#endif

        std::vector<std::vector<int>> cur_mapp_edge_indices = {{}, {}};
        const auto& cgra_edges = this->cgra->get_edges_();
        // const auto& dfg_edge_to_id = this->dfg->get_edge_to_id();
        // const auto& dfg_edges = this->dfg->get_edges();

        for (int i = 0; i < static_cast<int>(cur_routings[0].size()); i++) {
            auto cgra_edge = cgra_edges[cur_routings[1][i]];
            // auto dfg_edge = dfg_edges[cur_routings[0][i]];

            auto src_cgra_node = cgra_edge.first;
            auto dst_cgra_node = cgra_edge.second;
            // auto src_dfg_node = dfg_edge.first;
            // auto dst_dfg_node = dfg_edge.second;
            auto id_mapped_dfg_edge = mapping_node_to_id.at({pe_to_node.at(src_cgra_node), src_cgra_node});
            auto id_mapped_adg_edge = mapping_node_to_id.at({pe_to_node.at( dst_cgra_node), dst_cgra_node});

            cur_mapp_edge_indices[0].emplace_back(id_mapped_dfg_edge);
            cur_mapp_edge_indices[1].emplace_back(id_mapped_adg_edge);
            // auto dfg_edge_id = dfg_edge_to_id.at(std::make_pair(src_dfg_node, dst_dfg_node));
            // this->dfg_routing_type_ids[0][dfg_edge_id] = this->placed_id;
            // this->adg_routing_type_ids[0][cur_routings[1][i]] = this->placed_id;

// #ifdef DEBUG
//             std::cout << "Mapped DFG edge " << dfg_edge_id << " and CGRA edge " << cur_routings[1][i] << " with placed_id: " << this->placed_id << std::endl;
// #endif
        }

        this->mapping_edge_indices = torch::cat({this->mapping_edge_indices, matrix_to_tensor<int, long>(cur_mapp_edge_indices)}, 1);


    }
}


void GPRMState::update_features_after_state_step(const int& action)  {
    auto node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);

        
    if(node_to_be_mapped != -1 && !this->is_end_state){
        auto legal_actions_tensor = torch::tensor(this->legal_actions, torch::kInt32);
    
        // this->adg_placement_type_ids[1].masked_fill_(
        //     this->adg_placement_type_ids[1] == this->to_place_id,
        //     this->to_not_place_id
        // );
        
        // this->adg_placement_type_ids[1].index_put_({legal_actions_tensor}, this->to_place_id); 
        // auto prev_mapped_node = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped - 1);
        // this->dfg_placement_type_ids[1][prev_mapped_node] = this->to_not_place_id;
        // this->dfg_placement_type_ids[1][node_to_be_mapped] = this->to_place_id;
        // this->dfg_routing_type_ids[1].masked_fill_(
        //     this->dfg_routing_type_ids[1] == this->to_place_id,
        //     this->to_not_place_id
        // );
    
        // update_dfg_routing_type_by_id(node_to_be_mapped, this->to_place_id);
    
        // this->adg_routing_type_ids[1].masked_fill_(
        //     this->adg_routing_type_ids[1] == this->to_place_id,
        //     this->to_not_place_id
        // );
    
        // update_adg_routing_type_by_id(node_to_be_mapped, this->to_place_id);
        this->update_idx_and_mask();
     
        #ifdef DEBUG
    
        // assert(this->dfg_placement_type_ids[1].eq(this->to_place_id).sum().item<int>() == 1);
        // assert(this->dfg_placement_type_ids[1].eq(this->to_not_place_id).sum().item<int>() == this->dfg->num_vertices() - 1);
        // assert(this->dfg_placement_type_ids[0].eq(this->placed_id).sum().item<int>() == this->node_to_pe.size());
        // assert(this->dfg_placement_type_ids[0].eq(this->not_placed_id).sum().item<int>() ==  this->dfg->num_vertices() - this->node_to_pe.size());
    
        // assert(this->dfg_routing_type_ids[0].eq(this->placed_id).sum().item<int>() == this->num_mapped_edges);
        // assert(this->dfg_routing_type_ids[0].eq(this->not_placed_id).sum().item<int>() == this->dfg->get_num_edges() - this->num_mapped_edges);
        int num_edges_to_be_placed = 0;
        for(auto node : this->dfg->get_node_to_in_vertices().at(node_to_be_mapped)){
                if(this->node_was_mapped(node)){
                    num_edges_to_be_placed++;
            }
        }
    
        for(const auto& node : this->dfg->get_node_to_out_vertices().at(node_to_be_mapped)){
                if(this->node_was_mapped(node)){
                    num_edges_to_be_placed++;
            }
        }
        // assert(this->dfg_routing_type_ids[1].eq(this->to_place_id).sum().item<int>() == num_edges_to_be_placed);
        // assert(this->dfg_routing_type_ids[1].eq(this->to_not_place_id).sum().item<int>() == this->dfg->get_num_edges() - num_edges_to_be_placed);
    
        // assert(this->adg_placement_type_ids[1].eq(this->to_place_id).sum().item<int>() == legal_actions.size());
        // assert(this->adg_placement_type_ids[1].eq(this->to_not_place_id).sum().item<int>() == this->cgra->total_PEs() - legal_actions.size());
        // assert(this->adg_placement_type_ids[0].eq(this->placed_id).sum().item<int>() == this->pe_to_node.size());
        // assert(this->adg_placement_type_ids[0].eq(this->not_placed_id).sum().item<int>() == this->cgra->total_PEs() - this->pe_to_node.size());
        
        int num_placed_adg_edges = 0;
    
        for(auto& pair: this->PEs_to_routing){
            auto routing = pair.second;
            num_placed_adg_edges += routing.size() - 1;
        }
        // assert(this->adg_routing_type_ids[0].eq(this->placed_id).sum().item<int>() == num_placed_adg_edges);
        // assert(this->adg_routing_type_ids[0].eq(this->not_placed_id).sum().item<int>() == this->cgra->get_num_edges() - num_placed_adg_edges);
        int edges_to_place = 0;
        const auto& in_vertices = this->dfg->get_in_vertices_by_node_id(node_to_be_mapped);
        const auto& out_vertices = this->dfg->get_out_vertices_by_node_id(node_to_be_mapped);
        // const auto& cgra_edge_to_id = this->cgra->get_edge_to_id();
    
        for (auto& node : in_vertices){
            if (this->node_was_mapped(node)){
                auto pe = this->node_to_pe.at(node);
                
                std::cout << "In Vertex " << node << " was mapped. PE: " << pe << std::endl;
    
                for (const auto& out_pe : this->cgra->get_out_vertices_by_node_id(pe)){
                    if(pe_to_node.find(out_pe) != pe_to_node.end()){
                        continue;
                    }
                    edges_to_place++;
            }
            }
        }
    
        for (auto& node : out_vertices){
            if (this->node_was_mapped(node)){
                auto pe = this->node_to_pe.at(node);
    
                std::cout << "Out Vertex " << node << " was mapped. PE: " << pe << std::endl;
            for (const auto& in_pe : this->cgra->get_in_vertices_by_node_id(pe)){
                
                if(pe_to_node.find(in_pe) != pe_to_node.end()){
                    continue;
                }
        
                edges_to_place++;
                }
            }
        }
        // assert(this->adg_routing_type_ids[1].eq(this->to_place_id).sum().item<int>() == edges_to_place);
        // assert(this->adg_routing_type_ids[1].eq(this->to_not_place_id).sum().item<int>() == this->cgra->get_num_edges() - edges_to_place);
        assert(this->mapping_edge_indices.size(1) == this->routings.size(1));
        #endif
      
    }

}




void GPRMState::add_routing_nodes_placement_by_neigh(std::vector<std::vector<int>>& placements, const int& id_node_to_be_mapped,
                                const int& action, bool in_vertices){
    const auto& neigh_vertices = in_vertices ? BaseState::get_in_vertices_dfg_by_node_id(id_node_to_be_mapped) : 
                        BaseState::get_out_vertices_dfg_by_node_id(id_node_to_be_mapped);
    
    for (const auto& neigh : neigh_vertices) {
        if (BaseState::node_was_mapped(neigh)){
            auto neigh_PE = BaseState::get_assigned_PE_by_node_id(neigh);
            auto pe_routing_pair = in_vertices ? std::make_pair(neigh_PE, action) : std::make_pair(action, neigh_PE);
            auto routing = this->PEs_to_routing.at(pe_routing_pair);
            auto dfg_node = in_vertices ? neigh: id_node_to_be_mapped;
            for(int i = 1; i < static_cast<int>(routing.size()) - 1; i++){
                placements[0].emplace_back(dfg_node);
                placements[1].emplace_back(routing[i]);
           }
        }
    }
}

void GPRMState::add_routing_nodes_placement(std::vector<std::vector<int>>& placements, const int& id_node_to_be_mapped,
                                const int& action){
    add_routing_nodes_placement_by_neigh(placements,id_node_to_be_mapped,action,true);
    add_routing_nodes_placement_by_neigh(placements,id_node_to_be_mapped,action,false);
}

void GPRMState::add_routing_by_neigh(std::vector<std::vector<int>>& routings, const int& id_node_to_be_mapped,
    const int& action, bool in_vertices){
    #ifdef DEBUG
    std::cout << "[DEBUG] Adding routing by neigh for node: " << id_node_to_be_mapped 
    << ", action: " << action << ", in_vertices: " << in_vertices << std::endl;
    #endif

    auto neigh_vertices = in_vertices ? BaseState::get_in_vertices_dfg_by_node_id(id_node_to_be_mapped) : 
    BaseState::get_out_vertices_dfg_by_node_id(id_node_to_be_mapped);

    #ifdef DEBUG
        std::cout << "[DEBUG] Neigh vertices: ";
        for (const auto& neigh : neigh_vertices) {
            std::cout << neigh << " ";
        }
        std::cout << std::endl;
    #endif

    const auto& dfg_edge_to_id = this->dfg->get_edge_to_id();
    const auto& cgra_edge_to_id = this->cgra->get_edge_to_id();

    for (const auto& neigh : neigh_vertices) {
    if(this->node_was_mapped(neigh)){
    #ifdef DEBUG
    std::cout << "[DEBUG] Neighbour " << neigh << " was mapped, getting assigned PE." << std::endl;
    #endif

    auto neigh_PE = BaseState::get_assigned_PE_by_node_id(neigh);
    auto pe_routing_pair = in_vertices ? std::make_pair(neigh_PE, action) : std::make_pair(action, neigh_PE);
    auto dfg_edge = in_vertices ? std::make_pair(neigh, id_node_to_be_mapped) : std::make_pair(id_node_to_be_mapped, neigh);

    #ifdef DEBUG
    std::cout << "[DEBUG] PE routing pair: (" << pe_routing_pair.first << ", " << pe_routing_pair.second << ")" << std::endl;
    std::cout << "[DEBUG] DFG edge: (" << dfg_edge.first << ", " << dfg_edge.second << ")" << std::endl;
    #endif

    auto routing = this->PEs_to_routing.at(pe_routing_pair);

    for(int i = 0; i < static_cast<int>(routing.size()) - 1; i++){
    auto cgra_edge = std::make_pair(routing[i],routing[i+1]); 

    #ifdef DEBUG
    std::cout << "[DEBUG] Adding routing: DFG edge ID: " 
    << dfg_edge_to_id.at(dfg_edge) << ", CGRA edge ID: " 
    << cgra_edge_to_id.at(cgra_edge) << std::endl;
    #endif
    auto dfg_edge_id = dfg_edge_to_id.at(dfg_edge);
    routings[0].push_back(dfg_edge_id);

    auto cgra_edge_id =cgra_edge_to_id.at(cgra_edge);
    routings[1].emplace_back(cgra_edge_id);
    }
    } else {
    #ifdef DEBUG
    std::cout << "[DEBUG] Neighbour " << neigh << " was not mapped, skipping." << std::endl;
    #endif
    }
    }
}


void GPRMState::add_routing(std::vector<std::vector<int>>& routings, const int& id_node_to_be_mapped,
    const int& action) {
    #ifdef DEBUG
    std::cout << "Adding routing for node " << id_node_to_be_mapped << " with action " << action << std::endl;
    #endif

    add_routing_by_neigh(routings, id_node_to_be_mapped, action, true);

    #ifdef DEBUG
    std::cout << "Routing added by neighbor (true direction) for node " << id_node_to_be_mapped << std::endl;
    #endif

    add_routing_by_neigh(routings, id_node_to_be_mapped, action, false);

    #ifdef DEBUG
    std::cout << "Routing added by neighbor (false direction) for node " << id_node_to_be_mapped << std::endl;
    #endif
}





std::vector<c10::IValue> GPRMState::get_model_inputs(c10::DeviceType device) const
{
    std::vector<c10::IValue> model_input;
    model_input.push_back(lpe_dfg);
    model_input.push_back(dfg_edge_indices);
    model_input.push_back(dfg_placement_order_ids);

    // model_input.push_back(dfg_placement_type_ids);
    // model_input.push_back(dfg_routing_type_ids);
    model_input.push_back(dfg_node_scheduling_ids);
    model_input.push_back(dfg_operation_ids);

    model_input.push_back(lpe_adg);
    model_input.push_back(adg_edge_indices);
    // model_input.push_back(adg_placement_type_ids);
    // model_input.push_back(adg_routing_type_ids);
    model_input.push_back(adg_operation_ids);
    model_input.push_back(adg_pos);


    model_input.push_back(mapping_edge_indices);
    model_input.push_back(placements);
    model_input.push_back(routings);

    model_input.push_back(batch_idx);
    model_input.push_back(seq_idx);
    model_input.push_back(contextual_action_seq_idx);
    model_input.push_back(action_batch_idx);
    model_input.push_back(action_seq_idx);
    model_input.push_back(state_seq_idx);   
    auto max_seq_len = this->calc_max_seq_len();
    model_input.push_back(torch::tensor(max_seq_len));
    model_input.push_back(torch::tensor(static_cast<int>(this->legal_actions.size())));

    return model_input;
};

const torch::Tensor& GPRMState::getLpeDfg() const { return lpe_dfg; }
const torch::Tensor& GPRMState::getDfgNodeSchedulingIds() const { return dfg_node_scheduling_ids; }
const torch::Tensor& GPRMState::getDfgEdgeIndices() const { return dfg_edge_indices; }
const torch::Tensor& GPRMState::getDfgOperationIds() const { return dfg_operation_ids; }

const torch::Tensor& GPRMState::getLpeAdg() const { return lpe_adg; }
const torch::Tensor& GPRMState::getAdgEdgeIndices() const { return adg_edge_indices; }
const torch::Tensor& GPRMState::getAdgOperationIds() const { return adg_operation_ids; }

// const torch::Tensor& GPRMState::getDfgRoutingTypeIds() const { return dfg_routing_type_ids; }
// const torch::Tensor& GPRMState::getDfgPlacementTypeIds() const { return dfg_placement_type_ids; }
// const torch::Tensor& GPRMState::getAdgPlacementTypeIds() const { return adg_placement_type_ids; }
// const torch::Tensor& GPRMState::getAdgRoutingTypeIds() const { return adg_routing_type_ids; }

const torch::Tensor& GPRMState::getMappingEdgeIndices() const { return mapping_edge_indices; }
const torch::Tensor& GPRMState::getPlacements() const { return placements; }
const torch::Tensor& GPRMState::getRoutings() const { return routings; }

const torch::Tensor& GPRMState::getActionBatchIdx() const { return action_batch_idx; }

const torch::Tensor& GPRMState::getBatchIdx() const { return batch_idx; }
const torch::Tensor& GPRMState::getSeqIdx() const { return seq_idx; }
const torch::Tensor& GPRMState::getContextualActionSeqIdx() const { return contextual_action_seq_idx; }
const torch::Tensor& GPRMState::getActionSeqIdx() const { return action_seq_idx; }

const torch::Tensor& GPRMState::getStateSeqIdx() const { return state_seq_idx; }
const torch::Tensor& GPRMState::getADGPos() const { return adg_pos; }


void GPRMState::print() const {
    #ifdef DEBUG
        std::cout << "===== GPRMState Variables =====" << std::endl;
        std::cout << "placed_id: " << placed_id << std::endl;
        std::cout << "not_placed_id: " << not_placed_id << std::endl;
        std::cout << "to_place_id: " << to_place_id << std::endl;
        std::cout << "to_not_place_id: " << to_not_place_id << std::endl;
        
        std::cout << "mapping_node_to_id size: " << mapping_node_to_id.size() << std::endl;

        auto print_tensor = [](const std::string& name, const torch::Tensor& tensor) {
            if (tensor.defined()) {
                std::cout << name << " shape: " << tensor.sizes() << std::endl;
                std::cout << name << " values: " << tensor << std::endl;
            } else {
                std::cout << name << " is undefined." << std::endl;
            }
        };

        print_tensor("lpe_dfg", lpe_dfg);
        print_tensor("dfg_node_scheduling_ids", dfg_node_scheduling_ids);
        print_tensor("dfg_placement_order_ids", dfg_placement_order_ids);

        print_tensor("dfg_edge_indices", dfg_edge_indices);
        print_tensor("dfg_operation_ids", dfg_operation_ids);
        
        print_tensor("lpe_adg", lpe_adg);
        print_tensor("adg_pos", adg_pos);
        print_tensor("adg_edge_indices", adg_edge_indices);
        print_tensor("adg_operation_ids", adg_operation_ids);
        
        // print_tensor("dfg_routing_type_ids", dfg_routing_type_ids);
        // print_tensor("dfg_placement_type_ids", dfg_placement_type_ids);
        // print_tensor("adg_placement_type_ids", adg_placement_type_ids);
        // print_tensor("adg_routing_type_ids", adg_routing_type_ids);
        
        print_tensor("mapping_edge_indices", mapping_edge_indices);
        
        print_tensor("placements", placements);
        print_tensor("routings", routings);
        

        print_tensor("action_batch_idx", action_batch_idx);
        
        print_tensor("batch_idx", batch_idx);
        print_tensor("seq_idx", seq_idx);
        print_tensor("contextual_action_seq_idx", contextual_action_seq_idx);
        print_tensor("action_seq_idx", action_seq_idx);
        
        print_tensor("state_seq_idx", state_seq_idx);
        
        std::cout << "max_seq_len: " << max_seq_len << std::endl;
        std::cout << "max_legal_actions_size: " << max_legal_actions << std::endl;

        std::cout << "================================" << std::endl;
        #endif
        std::stringstream ss;
        BaseState::print_base_state(ss);
        std::cout << ss.str();}

int GPRMState::get_num_mapping_nodes() const{
    return this->placements.numel() > 0 ? this->placements.size(1) : 0 ;
}
int GPRMState::get_num_mapping_edges() const{
    return this->routings.numel() > 0 ? this->routings.size(1) : 0 ;

}

int GPRMState::get_max_seq_len() const{
    return this->max_seq_len;
}

bool GPRMState::do_backtracking() const{
    return this->is_end_state && !this->mapping_is_valid;
}


void GPRMState::update_action_to_idx(){
    action_to_idx.clear();
    idx_to_action.clear();
    int idx = 0;
    for(auto& action: legal_actions){
        action_to_idx[action] = idx; 
        this->idx_to_action[idx] = action;
        idx++;
    }
}