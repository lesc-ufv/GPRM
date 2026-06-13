#ifndef TRANSMAP_STATE_HPP
#define TRANSMAP_STATE_HPP

#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <string>
#include "src/hpp/utils/cgra/router.hpp"
#include <execution>
#include "src/hpp/rl/base_state.hpp"
#include "src/hpp/configs/state_config.hpp"
#include "src/hpp/enums/enum_nomalization.hpp"
#include "src/hpp/compilers/transmap/rl/configs/transmap_config.hpp"
#include "src/hpp/compilers/transmap/rl/transmap_environment.hpp"
class TransMapState: public BaseState {
private:
    torch::Tensor dfg_node_features;
    torch::Tensor dfg_edge_features;

    torch::Tensor adg_node_features;
    torch::Tensor adg_edge_features;

    torch::Tensor mask;
    torch::Tensor dfg_pad_mask;
    torch::Tensor adg_pad_mask;
    std::unordered_map<std::string, torch::Tensor> norm_feat_cache;

    bool override_base_state;

    std::vector<torch::Tensor> edge_idxs_per_modulo;
    std::shared_ptr<CGRA> spatial_cgra;


    std::shared_ptr<std::unordered_map<EnumFunctionalities, int>> functionality_to_idx = std::make_shared<std::unordered_map<EnumFunctionalities, int>>(
        std::unordered_map<EnumFunctionalities, int>{
            {EnumFunctionalities::ADD_AND_SUB, 0},
            {EnumFunctionalities::MULT_AND_DIV, 1},
            {EnumFunctionalities::LOGIC, 2},
            {EnumFunctionalities::SHIFT, 3},
            {EnumFunctionalities::COMPARISON, 4},
            {EnumFunctionalities::MEMORY_ACCESS, 5}
        });
    
    
    //dfg node features 
    static constexpr short dfg_in_degree_idx = 0;
    static constexpr short dfg_out_degree_idx = 1;
    static constexpr short dfg_opcode_idx = 2;
    static constexpr short dfg_id_node_idx = 3;
    static constexpr short dfg_sched_order_idx = 4;
    static constexpr short dfg_sched_time_slice_idx = 5;
    static constexpr short dfg_assigned_pe_idx = 6;
    static constexpr short dfg_num_node_features = 7;

    //dfg edge features
    static constexpr short dfg_delay_idx = 0;
    static constexpr short dfg_distance_idx = 1;
    static constexpr short dfg_id_edge_idx = 2;
    static constexpr short dfg_mapped_adg_edge_id_idx = 3;
    static constexpr short dfg_num_edge_features = 4;
    
    //adg node features
    static constexpr short adg_functionalities_initial_idx = 0;
    static constexpr short adg_in_degree_idx = 6;
    static constexpr short adg_out_degree_idx = 7;
    static constexpr short adg_id_idx = 8;
    static constexpr short adg_id_mapped_dfg_node_idx = 9;
    static constexpr short adg_num_node_features = 10;
    
    //adg edge features
    static constexpr short adg_id_edge_idx = 0;
    static constexpr short adg_mapped_dfg_edge_id_idx = 1;
    static constexpr short adg_num_edge_features = 2;

    std::vector<int> node_to_placement_order; 


public:

TransMapState() : BaseState(),
    dfg_node_features(torch::empty({})),
    dfg_edge_features(torch::empty({})),
    adg_node_features(torch::empty({})),
    adg_edge_features(torch::empty({})),
    mask(torch::empty({})),
    dfg_pad_mask(torch::empty({})),
    adg_pad_mask(torch::empty({})),
    override_base_state(false),
    edge_idxs_per_modulo(),
    spatial_cgra(nullptr),
    node_to_placement_order() {
}

TransMapState(const TransMapState& other) :
BaseState(other) {
    this->dfg_node_features = other.dfg_node_features.clone();
    this->dfg_edge_features = other.dfg_edge_features.clone();
    this->adg_node_features = other.adg_node_features.clone();
    this->adg_edge_features = other.adg_edge_features.clone();
    this->mask = other.mask.clone();
    this->dfg_pad_mask = other.dfg_pad_mask.clone();
    this->adg_pad_mask = other.adg_pad_mask.clone();
    this->override_base_state = other.override_base_state;
    this->edge_idxs_per_modulo = other.edge_idxs_per_modulo;
    this->spatial_cgra = other.spatial_cgra;
    this->node_to_placement_order = std::ref(other.node_to_placement_order);
}
TransMapState(TransMapState&& other) noexcept : BaseState(other){
    this->dfg_node_features = std::move(other.dfg_node_features);
    this->dfg_edge_features = std::move(other.dfg_edge_features);
    this->adg_node_features = std::move(other.adg_node_features);
    this->adg_edge_features = std::move(other.adg_edge_features);
    this->mask = std::move(other.mask);
    this->dfg_pad_mask = std::move(other.dfg_pad_mask);
    this->adg_pad_mask = std::move(other.adg_pad_mask);
    this->override_base_state = other.override_base_state;
    this->edge_idxs_per_modulo = std::move(other.edge_idxs_per_modulo);
    this->spatial_cgra = std::move(other.spatial_cgra);
    this->node_to_placement_order = std::move(other.node_to_placement_order);
}
TransMapState(std::shared_ptr<OrderedDFG> dfg, std::shared_ptr<CGRA> cgra,
    std::shared_ptr<zmq::socket_t> socket, const ServerConstants& server_k, 
    const TransMapConfig& transmap_config, StateConfig& state_config) 
: BaseState(dfg, cgra, state_config) 
{
    this->initialize();
    #ifdef DEBUG
    std::cout << "[TransMapState] Constructor called." << std::endl;
    #endif

    this->override_base_state = state_config.getOverrideBaseState();
    if(!this->dfg->has_scheduling() || this->dfg->is_mobility_sched()){
        throw std::runtime_error("TransMap doesn't support multi modulo processing yet.");
    } 

    this->II = this->cgra->get_II();
    
    #ifdef DEBUG
    std::cout << "[TransMapState] II = " << this->II << std::endl;
    #endif

    this->spatial_cgra = std::make_shared<CGRA>(
        1, 
        this->cgra->get_cgra_dimensions_(),
        this->cgra->get_pe_to_functionalities(),
        this->cgra->get_interconnection_styles(), 
        false
    );

    auto lpe_dim = transmap_config.getLpeDim();
    auto adg_edge_idx = create_edges_indices(spatial_cgra->get_edges_());
    auto dfg_edge_idx = create_edges_indices(this->dfg->get_edges());

    #ifdef DEBUG
    std::cout << "ADG Edge Indices: " << adg_edge_idx.size() << " edges." << std::endl;
    std::cout << "DFG Edge Indices: " << dfg_edge_idx.size() << " edges." << std::endl;
    #endif

    auto adg_edge_idx_tensor = matrix_to_tensor<int, int>(adg_edge_idx).to(torch::kInt64);
    auto dfg_edge_idx_tensor = matrix_to_tensor<int, int>(dfg_edge_idx).to(torch::kInt64);

    auto lpe_adg = get_laplacian_embeddings(socket, "gen_laplacian_embeddings", adg_edge_idx,
                                lpe_dim, cgra->number_of_spatial_PEs_(), false, server_k);
    auto lpe_dfg = get_laplacian_embeddings(socket, "gen_laplacian_embeddings", dfg_edge_idx,
                                lpe_dim, dfg->num_vertices(), false, server_k);

    #ifdef DEBUG
    std::cout << "LPE ADG sizes: " << lpe_adg.sizes() << std::endl;
    std::cout << "LPE DFG sizes: " << lpe_dfg.sizes() << std::endl;
    #endif


    #ifdef DEBUG
    std::cout << "[TransMapState] Initializing DFG features..." << std::endl;
    #endif

    initialize_dfg_features(lpe_dfg);

    #ifdef DEBUG
    std::cout << "[TransMapState] Initializing ADG features..." << std::endl;
    #endif

    initialize_adg_features(lpe_adg);

    this->mask = this->calc_masks();
    // this->update_node_to_be_mapped_features();

    dfg_pad_mask = torch::ones({this->calc_max_dfg_seq_len()}, torch::kBool);
    adg_pad_mask = torch::ones({this->calc_max_adg_seq_len()}, torch::kBool);

    #ifdef DEBUG
    std::cout << "[TransMapState] Constructor completed." << std::endl;
    #endif
}


void initialize_dfg_features(torch::Tensor lpe_dfg)
{
    #ifdef DEBUG
    std::cout << "[initialize_dfg_features] Starting..." << std::endl;
    #endif

    const auto& edges = this->dfg->get_edges();

    auto adg_edge_idx = create_edges_indices(this->cgra->get_edges_());
    auto dfg_edge_idx = create_edges_indices(this->dfg->get_edges());

    const auto& placement_order = this->dfg->get_placement_order();

    int num_dfg_rows = this->dfg->num_vertices();
    // auto& in_vertices = this->dfg->get_node_to_in_vertices();
    // auto& out_vertices = this->dfg->get_node_to_out_vertices();

    std::vector<std::vector<int>> node_features_vec(num_dfg_rows, std::vector<int>(7, 0));
    std::vector<std::vector<int>> edge_features_vec(edges.size(), std::vector<int>(4, 0));

    int fill_value = -1;
    int count = 0;
    this->node_to_placement_order = std::vector<int>(num_dfg_rows,-1);
    for (const int& node : placement_order) {
        if (node != -1) {
            node_to_placement_order[node] = count;
            node_features_vec[node] = this->create_dfg_node_feat_by_node(node, fill_value);
            count ++;
        }
    }



    this->dfg_node_features = matrix_to_tensor<int, float>(node_features_vec);

    this->dfg_node_features = torch::cat({this->dfg_node_features, lpe_dfg, lpe_dfg}, -1);

    auto edge_indices_tensor = matrix_to_tensor<int, int>(dfg_edge_idx);
    const auto& edge_to_id = this->dfg->get_edge_to_id();
    for(auto& edge: edges){
        int edge_id  = edge_to_id.at(edge);
        edge_features_vec[edge_id] = this->create_dfg_edge_feat_by_edge(edge, edge_id, fill_value);
    }
    this->dfg_edge_features = matrix_to_tensor<int, float>(edge_features_vec);

    this->dfg_edge_features = torch::cat({
        this->dfg_edge_features,
        lpe_dfg.index_select(0, edge_indices_tensor[0]),
        lpe_dfg.index_select(0, edge_indices_tensor[1])
    }, -1);

    #ifdef DEBUG
    std::cout << "[initialize_dfg_features] Completed." << std::endl;
    #endif
}

bool is_same_spatial_pe(int pe1, int pe2) {
    const auto& [cc1, r1, c1] = this->cgra->get_coord_by_PE_(pe1);
    const auto& [cc2, r2, c2] = this->cgra->get_coord_by_PE_(pe2);

    return r1 == r2 && c1 == c2;
}

std::vector<std::vector<std::pair<int,int>>> get_edges_per_modulo() {
    std::vector<std::vector<std::pair<int,int>>> edges_per_modulo(this->II, std::vector<std::pair<int,int>>());

    for (int i = 0; i < this->cgra->total_PEs(); ++i) {
        auto pe = i ;
        edges_per_modulo.reserve(this->spatial_cgra->get_num_edges());
        const auto& [cc1,r1,c1] =  this->cgra->get_coord_by_PE_(pe);
        // for (auto& father : this->cgra->get_in_vertices_by_id_(pe)) {
        //     auto edge = std::make_pair(father, pe);
        //     // if (!is_same_spatial_pe(edge.first, edge.second)) {
        //         edges_per_modulo[cc1].push_back(edge);
        //     // }
        // }

        for (const auto& child : this->cgra->get_out_vertices_by_id_(pe)) {
            auto edge = std::make_pair(pe, child);
            // if (!is_same_spatial_pe(edge.first, edge.second)) {
                edges_per_modulo[cc1].push_back(edge);
            // }
        }

        
    }
    // for (int i = 0; i < this->II; ++i) {
    //     assert(edges_per_modulo[i].size() ==  this->spatial_cgra->get_num_edges() + this->spatial_cgra->total_PEs());
    // }
#ifdef DEBUG
    std::cout << "[get_edges_per_modulo] Completed. Modules: " << edges_per_modulo.size() << std::endl;
#endif

    return edges_per_modulo;
}

std::vector<int> create_adg_node_feat_by_node(
    int vertice,
    int id_mapped_dfg_node
) {
    std::vector<int> adg_node_feat = std::vector<int>(10, -1);

    for (const auto& functionality : this->cgra->get_pe_to_functionalities().at(vertice)) {
        int idx = this->adg_functionalities_initial_idx + this->functionality_to_idx->at(functionality);
        if (idx < static_cast<int>(adg_node_feat.size()))
            adg_node_feat[idx] = 1;
    }

    auto relative_pe = this->calc_relative_PE(vertice);
    adg_node_feat[this->adg_in_degree_idx] = this->spatial_cgra->get_in_vertices_by_id_(relative_pe).size();
    adg_node_feat[this->adg_out_degree_idx] = this->spatial_cgra->get_out_vertices_by_id_(relative_pe).size();
    adg_node_feat[this->adg_id_idx] = vertice;
    adg_node_feat[this->adg_id_mapped_dfg_node_idx] = id_mapped_dfg_node;

#ifdef DEBUG
    std::cout << "[create_adg_node_feat_by_node] Features for vertice " << vertice << ":" << std::endl;
    for (size_t i = 0; i < adg_node_feat.size(); ++i) {
        std::cout << "  Feature[" << i << "] = " << adg_node_feat[i] << std::endl;
    }
             
            //   int a;
            //   std::cin >> a;
#endif
    return adg_node_feat;
}

std::vector<int> create_adg_edge_feat_by_edge(int edge_id, int id_mapped_dfg_edge) {
    std::vector<int> edge_features_vec = std::vector<int>(2, -1);

    edge_features_vec[this->adg_id_edge_idx] = edge_id;
    edge_features_vec[this->adg_mapped_dfg_edge_id_idx] = id_mapped_dfg_edge;

#ifdef DEBUG
    std::cout << "[create_adg_edge_feat_by_edge] Edge ID: " << edge_id << ", DFG Mapped ID: " << id_mapped_dfg_edge << std::endl;
#endif

    return edge_features_vec;
}

std::vector<int> create_dfg_node_feat_by_node(
    int node,
    int pe
) {

    const auto& in_vertices = this->dfg->get_in_vertices_by_node_id(node);
    const auto& out_vertices = this->dfg->get_out_vertices_by_node_id(node);
    const auto& planned_time_slice = this->dfg->is_asap_scheduling() ? this->dfg->get_asap() : this->dfg->get_alap();
    std::vector<int> dfg_node_feat = std::vector<int>(this->dfg_num_node_features,-1);
    int fill_value = -1;

    dfg_node_feat[this->dfg_in_degree_idx] = in_vertices.size();
    dfg_node_feat[this->dfg_out_degree_idx] = out_vertices.size();
    dfg_node_feat[this->dfg_opcode_idx] = get_opcode_by_operation(this->dfg->get_operation_by_node(node));
    dfg_node_feat[this->dfg_id_node_idx] = node;
    dfg_node_feat[this->dfg_sched_order_idx] = this->node_to_placement_order[node];
    dfg_node_feat[this->dfg_sched_time_slice_idx] = !planned_time_slice.empty() ? planned_time_slice[node] : fill_value;
    dfg_node_feat[this->dfg_assigned_pe_idx] = pe;

#ifdef DEBUG
    std::cout << "[DEBUG][create_dfg_node_feat_by_node]" << std::endl;
    std::cout << "  In-degree: " << dfg_node_feat[this->dfg_in_degree_idx] << std::endl;
    std::cout << "  Out-degree: " << dfg_node_feat[this->dfg_out_degree_idx] << std::endl;
    std::cout << "  Opcode: " << dfg_node_feat[this->dfg_opcode_idx] << std::endl;
    std::cout << " Node: " << node << std::endl;
    std::cout << "  Sched Order: " << this->node_to_placement_order[node] << "\n Time Slice: " << dfg_node_feat[this->dfg_sched_time_slice_idx] << std::endl;
    std::cout << " PE ID: " << fill_value << std::endl;
    // int a;
    // std::cin >> a;
#endif

    return dfg_node_feat;
}

std::vector<int> create_dfg_edge_feat_by_edge(const std::pair<int,int>& edge, const int& edge_id, int id_mapped_adg_edge){
    std::vector<int> dfg_edge_feat = std::vector<int>(this->dfg_num_edge_features,-1);

    dfg_edge_feat[this->dfg_delay_idx] = 1;

    auto& [src, dst] = edge;
    auto src_sched_mod = this->node_to_scheduled_time_slice.at(src);
    auto dst_sched_mod = this->node_to_scheduled_time_slice.at(dst);
    int distance; 
    distance = dst_sched_mod - src_sched_mod;


    dfg_edge_feat[this->dfg_distance_idx] = distance;
    dfg_edge_feat[this->dfg_id_edge_idx] = edge_id;
    dfg_edge_feat[this->dfg_mapped_adg_edge_id_idx] = id_mapped_adg_edge;

#ifdef DEBUG
    std::cout << "[DEBUG][create_dfg_edge_feat_by_edge] Edge ID: " << edge_id << std::endl;
    std::cout << "  Src: " << src << ", Dst: " << dst << std::endl;
    std::cout << "  Distance: " << distance << ", Delay: 1" << std::endl;
#endif

    return dfg_edge_feat;
}


void initialize_adg_features(torch::Tensor lpe_adg) {
    auto num_total_PEs = cgra->total_PEs();
    auto total_edges = cgra->get_num_edges();
    std::vector<std::vector<int>> node_features_vec, edge_features_vec;
    node_features_vec.resize(num_total_PEs, std::vector<int>(10, 0));
    edge_features_vec.resize(total_edges, std::vector<int>(2, 0));

    int fill_value = -1;

#ifdef DEBUG
    std::cout << "[DEBUG][initialize_adg_features] Initializing ADG node features..." << std::endl;
#endif

    for (const auto& vertice : cgra->get_vertices_()) {
        node_features_vec[vertice] = create_adg_node_feat_by_node(vertice, fill_value);
#ifdef DEBUG
        std::cout << "  Node " << vertice << " initialized." << std::endl;
#endif
    }

    std::vector<int> src_pe(total_edges, 0);
    std::vector<int> dst_pe(total_edges, 0);

#ifdef DEBUG
    std::cout << "[DEBUG][initialize_adg_features] Initializing ADG edge features..." << std::endl;
#endif

    const auto& edge_to_id = cgra->get_edge_to_id();
    for (const auto& edge : cgra->get_edges_()) {
        auto edge_id = edge_to_id.at(edge);
        edge_features_vec[edge_id] = create_adg_edge_feat_by_edge(edge_id, fill_value);
#ifdef DEBUG
        std::cout << "  Edge ID " << edge_id << " initialized." << std::endl;
#endif
    }
    
    const auto& edges_per_modulo = get_edges_per_modulo();

    for (auto i = 0; i < this->II; i++) {
        std::vector<int> edge_idx;
        for (auto& edge : edges_per_modulo.at(i)) {
            const auto& [src, dst] = edge;
            const auto& [cc1, r1, c1] = this->cgra->get_coord_by_PE_(src);
            const auto& [cc2, r2, c2] = this->cgra->get_coord_by_PE_(dst);
            auto src_spatial_pe = this->cgra->get_PE_by_coord(std::make_tuple(0, r1, c1));
            auto dst_spatial_pe = this->cgra->get_PE_by_coord(std::make_tuple(0, r2, c2));
            auto edge_id = edge_to_id.at({edge});
            src_pe[edge_id] = src_spatial_pe;
            dst_pe[edge_id] = dst_spatial_pe;
            edge_idx.emplace_back(edge_id);
        }

        this->edge_idxs_per_modulo.emplace_back(
            torch::tensor(edge_idx,torch::kInt32)
        );


    }

#ifdef DEBUG
    std::cout << "[DEBUG][initialize_adg_features] Normalizing and finalizing ADG feature tensors..." << std::endl;
#endif

    this->adg_node_features = matrix_to_tensor<int, float>(node_features_vec).resize_(
        {this->II, this->cgra->number_of_spatial_PEs_(), 10});

        
    auto lpe_per_modulo = lpe_adg.unsqueeze(0).repeat({this->II, 1, 1});
    this->adg_node_features = torch::cat({
        this->adg_node_features,
        lpe_per_modulo,
        lpe_per_modulo
    }, -1);
        
    auto edge_features_tens = matrix_to_tensor<int, float>(edge_features_vec);
    auto src_lpe = lpe_adg.index_select(0, torch::tensor(src_pe, torch::kInt64));
    auto dst_lpe = lpe_adg.index_select(0, torch::tensor(dst_pe, torch::kInt64));

    this->adg_edge_features = torch::cat({
        edge_features_tens,
        src_lpe,
        dst_lpe
    }, -1);


#ifdef DEBUG
    std::cout << "[DEBUG][initialize_adg_features] ADG features initialization complete." << std::endl;
    std::cout << "[DEBUG] Final adg_node_features size: " << this->adg_node_features.sizes() << std::endl;
    std::cout << "[DEBUG] Final adg_edge_features size: " << this->adg_edge_features.sizes() << std::endl;
    std::cout << "[DEBUG] Sample adg_node_features[0][0]: " << this->adg_node_features[0][0] << std::endl;
    std::cout << "[DEBUG] Sample adg_edge_features[0]: " << this->adg_edge_features[0] << std::endl;
#endif

}



void assign_PE_to_node(int PE, const int& node)
{
#ifdef DEBUG
    std::cout << "[assign_PE_to_node] Assigning PE " << PE << " to DFG node " << node << std::endl;
#endif
        
    dfg_node_features[node][this->dfg_assigned_pe_idx] = PE;

}

int calc_max_dfg_seq_len() const {
    return this->dfg->num_vertices() + this->dfg->get_num_edges();
}

int calc_max_adg_seq_len() const {
    return this->spatial_cgra->number_of_spatial_PEs_() + this->get_num_cgra_spatial_edges();
}

void assign_node_to_PE(int node, const int &PE)
{
#ifdef DEBUG
    std::cout << "[assign_node_to_PE] Assigning node " << node << " to PE " << PE << std::endl;
#endif
    int cur_modulo = PE / this->cgra->number_of_spatial_PEs_();
    auto relative_PE = this->calc_relative_PE(PE);
    this->adg_node_features[cur_modulo][relative_PE][this->adg_id_mapped_dfg_node_idx] = node;

    }
int calc_relative_PE(const int &PE)
{
    return PE % this->cgra->number_of_spatial_PEs_();
}
void place_node_in_PE(int node, int PE)
{
#ifdef DEBUG
    std::cout << "[place_node_in_PE] Placing node " << node << " in PE " << PE << std::endl;
#endif
    this->assign_node_to_PE(node, PE);
    this->assign_PE_to_node(PE, node);    
}

torch::Tensor calc_masks()
{
    auto num_spatial_pes = this->cgra->number_of_spatial_PEs_();

#ifdef DEBUG
    std::cout << "[calc_masks] Calculating mask with " << this->legal_actions.size() 
              << " legal actions and " << num_spatial_pes << " spatial PEs." << std::endl;
    assert(static_cast<int>(this->legal_actions.size()) <= num_spatial_pes);
#endif

    auto idx = torch::tensor(this->legal_actions, torch::kInt32) % num_spatial_pes;
    auto mask = torch::zeros({1, num_spatial_pes}, torch::kInt32);
    mask.index_put_({0, idx}, 1);
    return mask.reshape({1, num_spatial_pes});
}

void update_features_before_state_step(const int& action) override
{
#ifdef DEBUG
    std::cout << "[update_features_before_state_step] Updating features with action: " << action << std::endl;
#endif

    auto id_node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);
#ifdef DEBUG
    std::cout << "[update_features_before_state_step] Node to be mapped: " << id_node_to_be_mapped << std::endl;
#endif

    this->place_node_in_PE(id_node_to_be_mapped, action);

    const auto& in_vertices_cur_node = this->dfg->get_in_vertices_by_node_id(id_node_to_be_mapped);
    const auto& out_vertices_cur_node = this->dfg->get_out_vertices_by_node_id(id_node_to_be_mapped);

    const auto&  PEs_to_routing = this->get_PEs_to_routing_();
    const auto& dfg_edge_to_id = this->dfg->get_edge_to_id();
    const auto& adg_edge_to_id = this->cgra->get_edge_to_id();

    for (const auto& father : in_vertices_cur_node) {
        if (this->node_was_mapped(father)) {
            const auto& father_PE = this->get_assigned_PE_by_node_id(father);
            auto pe_routing_pair = std::make_pair(father_PE, action);
            const auto& cur_routing = PEs_to_routing.at(pe_routing_pair);
            int cur_routing_size = cur_routing.size();
            auto dfg_edge = std::make_pair(father, id_node_to_be_mapped);
            const auto& dfg_edge_id = dfg_edge_to_id.at(dfg_edge);

#ifdef DEBUG
            std::cout << "[update_features_before_state_step] Routing from father " << father
                      << " (PE " << father_PE << ") to node " << id_node_to_be_mapped 
                      << " (PE " << action << "), hops: " << cur_routing_size << std::endl;
#endif

            if (cur_routing_size > 2) {
                auto last_PE = father_PE;
                for (int i = 1; i < cur_routing_size - 1; i++) {
                    auto cur_PE = cur_routing[i];
                    this->assign_node_to_PE(-2, cur_PE);
                    auto cur_edge = std::make_pair(last_PE, cur_PE);
                    auto adg_edge_id = adg_edge_to_id.at(cur_edge);

                    if (i == 1) {
                        this->dfg_edge_features[dfg_edge_id][this->dfg_mapped_adg_edge_id_idx] = adg_edge_id;
                    }
                    
                    this->adg_edge_features[adg_edge_id][this->adg_mapped_dfg_edge_id_idx] = dfg_edge_id;
                    
                    last_PE = cur_PE;
                }
            }else if (cur_routing_size == 2) {
                auto adg_edge_id = adg_edge_to_id.at(pe_routing_pair);
                this->dfg_edge_features[dfg_edge_id][this->dfg_mapped_adg_edge_id_idx] = adg_edge_id;
                this->adg_edge_features[adg_edge_id][this->adg_mapped_dfg_edge_id_idx] = dfg_edge_id;
                    
            }
            
        }
    }

    for (const auto& child : out_vertices_cur_node) {
        if (this->node_was_mapped(child)) {
            auto child_PE = this->get_assigned_PE_by_node_id(child);
            auto pe_routing_pair = std::make_pair(action, child_PE);
            const auto& cur_routing = PEs_to_routing.at(pe_routing_pair);
            int cur_routing_size = cur_routing.size();
            auto dfg_edge = std::make_pair(id_node_to_be_mapped, child);
            auto dfg_edge_id = dfg_edge_to_id.at(dfg_edge);

#ifdef DEBUG
            std::cout << "[update_features_before_state_step] Routing to child " << child
                      << " (PE " << child_PE << ") from node " << id_node_to_be_mapped 
                      << " (PE " << action << "), hops: " << cur_routing_size << std::endl;
#endif

            if (cur_routing_size > 2) {
                auto last_PE = action;
                for (int i = 1; i < cur_routing_size - 1; i++) {
                    auto cur_PE = cur_routing[i];
                    this->assign_node_to_PE(-2, cur_PE);
                    auto adg_edge_id = adg_edge_to_id.at({last_PE, cur_PE});

                    if (i == 1) {
                        this->dfg_edge_features[dfg_edge_id][this->dfg_mapped_adg_edge_id_idx] = adg_edge_id;
                    }
                    
                    this->adg_edge_features[adg_edge_id][this->adg_mapped_dfg_edge_id_idx] = dfg_edge_id;

                    
                    last_PE = cur_PE;
                    
                }
            } else if (cur_routing_size == 2) {
                auto adg_edge_id = adg_edge_to_id.at(pe_routing_pair);
            
                this->dfg_edge_features[dfg_edge_id][this->dfg_mapped_adg_edge_id_idx] = adg_edge_id;
            
                this->adg_edge_features[adg_edge_id][this->adg_mapped_dfg_edge_id_idx] = dfg_edge_id;

            }
            
        }
    }
}

void update_features_after_state_step(const int& action) override
{
#ifdef DEBUG
    std::cout << "[update_features_after_state_step] Post-step update with action: " << action << std::endl;
#endif
    // this->update_node_to_be_mapped_features();
    this->mask = this->calc_masks();
}

// void update_node_to_be_mapped_features()
// {
//     auto id_node_to_be_mapped = this->dfg->get_node_in_placement_order_by_idx_(this->idx_id_node_to_be_mapped);
// #ifdef DEBUG
//     std::cout << "[update_node_to_be_mapped_features] Node to be mapped ID: " << id_node_to_be_mapped << std::endl;
// #endif
//     if (id_node_to_be_mapped != -1) {
//         this->node_to_be_mapped_features = dfg_node_features[id_node_to_be_mapped].unsqueeze_(0);
//     } else {
//         this->node_to_be_mapped_features = torch::tensor(
//             {0},
//             torch::dtype(torch::kFloat32).requires_grad(false));
// #ifdef DEBUG
//         std::cout << "[update_node_to_be_mapped_features] No node to be mapped. Assigned zero tensor." << std::endl;
// #endif
//     }
// }

int calc_mask_seq_len()
{
    int len = this->spatial_cgra->number_of_spatial_PEs_() +
              this->spatial_cgra->get_num_edges() +
              this->dfg->get_num_nodes() +
              this->dfg->get_num_edges();
#ifdef DEBUG
    std::cout << "[calc_mask_seq_len] Calculated mask sequence length: " << len << std::endl;
#endif
    return len;
}

void update_is_end_state()
{
#ifdef DEBUG
    std::cout << "[update_is_end_state] Checking if end state reached..." << std::endl;
#endif

    if (this->override_base_state) {
        bool all_nodes_were_mapped = this->all_nodes_were_mapped();
#ifdef DEBUG
        std::cout << "[update_is_end_state] all_nodes_were_mapped = " << all_nodes_were_mapped
                  << ", legal_actions.empty() = " << this->legal_actions.empty() << std::endl;
#endif
        if (all_nodes_were_mapped || this->legal_actions.empty()) {
            this->is_end_state = true;
#ifdef DEBUG
            std::cout << "[update_is_end_state] End state reached." << std::endl;
#endif
        }
    } else {
        BaseState::update_is_end_state();
#ifdef DEBUG
        std::cout << "[update_is_end_state] Called BaseState::update_is_end_state()." << std::endl;
#endif
    }
}

void print() const {
    std::cout << "TransMapState Debug Info:" << std::endl;
    
    std::cout << "dfg_node_features: " << dfg_node_features << std::endl;
    std::cout << "dfg_edge_features: " << dfg_edge_features << std::endl;
    
    std::cout << "adg_node_features: " << adg_node_features << std::endl;
    std::cout << "adg_edge_features: " << adg_edge_features << std::endl;
    
    // std::cout << "node_to_be_mapped_features: " << node_to_be_mapped_features << std::endl;
    std::cout << "mask: " << mask << std::endl;
    std::cout << "dfg_pad_mask: " << dfg_pad_mask << std::endl;
    std::cout << "adg_pad_mask: " << adg_pad_mask << std::endl;
    
    std::cout << "override_base_state: " << std::boolalpha << override_base_state << std::endl;
}

torch::Tensor get_dfg_pad_mask() const {
    return this->dfg_pad_mask;
}  

torch::Tensor get_adg_pad_mask() const  {
    return this->adg_pad_mask;
}

torch::Tensor get_mask() const  {
    return this->mask;
}

torch::Tensor get_node_to_be_mapped_features ()  {
    std::string dfg_key = "dfg_node_feat";
    // assert(
    //     this->norm_feat_cache.find(dfg_key) != this->norm_feat_cache.end() &&
    //     "Use get_dfg_node_features first"
    // );
    auto dfg_feat =  this->norm_feat_cache.at("dfg_node_feat");
    auto node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);
    return dfg_feat[node_to_be_mapped].unsqueeze(0);
}

torch::Tensor get_adg_node_features()  {
    std::string adg_node_feat_key = "adg_node_feat";
    if(this->norm_feat_cache.find(adg_node_feat_key) != this->norm_feat_cache.end()){
        return this->norm_feat_cache.at(adg_node_feat_key);
    }

    
    auto node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);
    auto cur_modulo = this->node_to_sched_mods.at(node_to_be_mapped).first;
    
    auto cur_features = this->adg_node_features[cur_modulo];
    
    // std::cout <<  "Current adg features: " << cur_features << std::endl;

    auto norm_features = normalize_features_by_column(cur_features.slice(1, 0, this->adg_num_node_features));

    auto all_features = cur_features.clone();

    all_features.slice(1, 0, this->adg_num_node_features) = norm_features;
    
    this->norm_feat_cache[adg_node_feat_key] = all_features;

    return all_features;
}

int get_num_cgra_spatial_edges() const{
    return this->spatial_cgra->get_num_edges() + (this->II > 1 ? this->spatial_cgra->total_PEs() : 0);
}


torch::Tensor get_adg_edge_features()  {
    std::string key = "adg_edge_feat";
    if (this->norm_feat_cache.find(key) != this->norm_feat_cache.end()) {
        return this->norm_feat_cache.at(key);
    }

    auto node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);
    auto cur_modulo = this->node_to_sched_mods.at(node_to_be_mapped).first;

    auto cur_features = this->adg_edge_features.index_select(0, edge_idxs_per_modulo[cur_modulo]);
    auto norm_features = normalize_features_by_column(cur_features.slice(1, 0, this->adg_num_edge_features));

    auto all_features = cur_features.clone();
    all_features.slice(1, 0, this->adg_num_edge_features) = norm_features;

    this->norm_feat_cache[key] = all_features;
    return all_features;
}

torch::Tensor get_dfg_node_features()  {
    std::string key = "dfg_node_feat";
    if (this->norm_feat_cache.find(key) != this->norm_feat_cache.end()) {
        return this->norm_feat_cache.at(key);
    }

    auto cur_features = this->dfg_node_features;
    auto norm_features = normalize_features_by_column(cur_features.slice(1, 0, this->dfg_num_node_features));

    auto all_features = cur_features.clone();
    all_features.slice(1, 0, this->dfg_num_node_features) = norm_features;

    this->norm_feat_cache[key] = all_features;
    return all_features;
}

torch::Tensor get_dfg_edge_features()  {
    std::string key = "dfg_edge_feat";
    if (this->norm_feat_cache.find(key) != this->norm_feat_cache.end()) {
        return this->norm_feat_cache.at(key);
    }

    auto cur_features = this->dfg_edge_features;
    auto norm_features = normalize_features_by_column(cur_features.slice(1, 0, this->dfg_num_edge_features));

    auto all_features = cur_features.clone();
    all_features.slice(1, 0, this->dfg_num_edge_features) = norm_features;

    this->norm_feat_cache[key] = all_features;
    return all_features;
}
void update_action_to_idx() override {
    this->action_to_idx.clear();
    this->idx_to_action.clear();

    #ifdef DEBUG
    std::cout << "(TransMap) Updating action_to_idx map..." << std::endl;
    #endif
    for(auto& pe: this->legal_actions){
        int idx = this->calc_relative_PE(pe);
        this->action_to_idx[pe] = idx;
        this->idx_to_action[idx] = pe;
        #ifdef DEBUG

        std::cout << "Action: " << pe << " -> Index: " << idx << std::endl;
        #endif
    }
    #ifdef DEBUG

    std::cout << "Update completed. Total actions: " << this->action_to_idx.size() << std::endl;
    #endif
}

bool model_is_general() const override {
    return false;
}


std::vector<c10::IValue> calc_initial_model_inputs() override {
    return std::vector<c10::IValue>();
};

std::vector<c10::IValue> get_model_inputs(c10::DeviceType device) 
{
    std::vector<c10::IValue> model_input;
    model_input.push_back(get_dfg_node_features().unsqueeze(0));
    model_input.push_back(get_dfg_edge_features().unsqueeze(0));
    model_input.push_back(get_adg_node_features().unsqueeze(0));
    model_input.push_back(get_adg_edge_features().unsqueeze(0));
    model_input.push_back(get_node_to_be_mapped_features());

    model_input.push_back(get_dfg_pad_mask().unsqueeze(0));
    model_input.push_back(get_adg_pad_mask().unsqueeze(0));
    // assert(get_adg_node_features().size(0) + get_adg_edge_features().size(0) ==  get_adg_pad_mask().size(0));
    
    model_input.push_back(get_mask());

    

    return model_input;
};

int get_dfg_node_features_size() const {
    return this->dfg_node_features.size(-1);
}

int get_dfg_edge_features_size() const {
    return this->dfg_edge_features.size(-1);
}


int get_adg_node_features_size() const {
    return this->adg_node_features.size(-1);
}

int get_adg_edge_features_size() const {
    return this->adg_edge_features.size(-1);
}


template<class Archive>
void serialize(Archive& ar) {
    ar(cereal::base_class<BaseState>(this));

    serialize_tensor(ar, dfg_node_features);
    serialize_tensor(ar, dfg_edge_features);
    serialize_tensor(ar, adg_node_features);
    serialize_tensor(ar, adg_edge_features);
    serialize_tensor(ar, mask);
    serialize_tensor(ar, dfg_pad_mask);
    serialize_tensor(ar, adg_pad_mask);

    serialize_unordered_map_string_tensor(ar, norm_feat_cache);
    serialize_vector_tensor(ar, edge_idxs_per_modulo);
    serialize_shared_ptr_unordered_map_enum_int(ar, functionality_to_idx);

    ar(node_to_placement_order);

    ar(spatial_cgra);
}

};

#endif