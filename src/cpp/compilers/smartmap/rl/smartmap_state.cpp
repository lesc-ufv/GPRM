#include "src/hpp/compilers/smartmap/rl/smartmap_state.hpp"

SmartMapState::SmartMapState(std::shared_ptr<OrderedDFG> dfg, std::shared_ptr<CGRA> cgra, StateConfig& state_config ): 
BaseState(dfg, cgra, state_config) {
    this->initialize();

}

SmartMapState::SmartMapState(const SmartMapState& other)
    : BaseState(other) ,
    sched_mod_time_slice_to_nodes(std::ref(other.sched_mod_time_slice_to_nodes))
    
{
}

void SmartMapState::update_features_before_state_step(const int& action)  {
    auto id_node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);
    this->place_node_in_PE(id_node_to_be_mapped, action); 
    this->update_num_nodes_mapped_in_same_modulo(action, id_node_to_be_mapped);
    this->assign_modulo_time_slice(action, id_node_to_be_mapped);

    add_routing_nodes(id_node_to_be_mapped, action);  

};

void SmartMapState::update_features_after_state_step(const int& action)  {
    this->update_node_to_be_mapped_features();
    this->update_actions_features();
    this->update_masks();
};


torch::Tensor SmartMapState::create_legal_actions_features
() const {
    auto legal_actions_tensor = torch::tensor(this->legal_actions, torch::kInt32);

    auto legal_actions_features = this->model_inputs[this->idx_cgra_features]
                                      .toTensor()
                                      .index_select(0, legal_actions_tensor);  
    return legal_actions_features.unsqueeze(0);
}


torch::Tensor SmartMapState::get_legal_actions_features_tensor() const {
    return this->model_inputs[this->idx_actions_features].toTensor();
}

std::vector<c10::IValue> SmartMapState::calc_initial_model_inputs() {
    // DFG Features
    std::vector<std::array<int,9>> dfg_features;
    dfg_features.resize(this->dfg->num_vertices(),std::array<int,9>());
    const auto& in_vertices_dfg = this->dfg->get_node_to_in_vertices();
    const auto& out_vertices_dfg = this->dfg->get_node_to_out_vertices();
    int count  = 0;
    int fill_value = -1;
    const auto& placement_order = this->dfg->get_placement_order();

    #ifdef DEBUG
    std::cout << "[DEBUG] Calculating DFG Features" << std::endl;
    #endif

    for (const int& node : placement_order) {
        if (node != -1){
            dfg_features[node][this->dfg_id_idx] = node;
            dfg_features[node][this->sched_order_idx] = count;
            dfg_features[node][this->num_nodes_mapped_in_same_modulo_idx] = fill_value;
            dfg_features[node][this->has_self_cycle] = in_vertices_dfg.at(node).find(node) != in_vertices_dfg.at(node).end();
            dfg_features[node][this->sched_mod_time_slice_idx] = fill_value;
            dfg_features[node][this->dfg_in_degree_idx] = in_vertices_dfg.at(node).size();
            dfg_features[node][this->dfg_out_degree_idx] = out_vertices_dfg.at(node).size();
            dfg_features[node][this->opcode_idx] = get_opcode_by_operation(this->dfg->get_operation_by_node(node));
            dfg_features[node][this->assigned_pe_idx] = fill_value;
            count++;

            #ifdef DEBUG
            std::cout << "[DEBUG] DFG Feature for node " << node << ": ";
            for (const auto& val : dfg_features[node]) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
            #endif
        }
    }

    // CGRA Features
    std::vector<std::array<int,7>> cgra_features;
    auto num_total_PEs = this->cgra->total_PEs();
    cgra_features.resize(num_total_PEs, std::array<int,7>());
    std::unordered_map<EnumFunctionalities,int> functionality_to_idx = {
        {EnumFunctionalities::LOGICAL, 0},
        {EnumFunctionalities::ARITHMETIC, 1},
        {EnumFunctionalities::MEMORY_ACCESS, 2}
    };
    const auto& pe_to_functionalities = this->cgra->get_pe_to_functionalities();

    #ifdef DEBUG
    std::cout << "[DEBUG] Calculating CGRA Features" << std::endl;
    #endif

    for (const auto& vertice : this->cgra->get_vertices_()){
        cgra_features[vertice][this->cgra_id_idx] = vertice;
        for (const auto& functionality: pe_to_functionalities.at(vertice)){
            cgra_features[vertice][this->functionalitie_initial_idx + functionality_to_idx.at(functionality)] = 1;
        }
        cgra_features[vertice][this->cgra_in_degree_idx] = this->cgra->get_in_vertices_by_id_(vertice).size();
        cgra_features[vertice][this->cgra_out_degree_idx] =  this->cgra->get_out_vertices_by_id_(vertice).size();
        cgra_features[vertice][this->id_mapped_dfg_node_idx] =  fill_value;

        #ifdef DEBUG
        std::cout << "[DEBUG] CGRA Feature for PE " << vertice << ": ";
        for (const auto& val : cgra_features[vertice]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        #endif
    }

    std::vector<std::vector<float>> cgra_features_vec, dfg_features_vec;
    for (const auto& arr : dfg_features) {
        dfg_features_vec.emplace_back(arr.begin(), arr.end());
    }

    for (const auto& arr : cgra_features) {
        cgra_features_vec.emplace_back(arr.begin(), arr.end());
    }

    const auto& dfg_edge_indices = this->dfg->get_edge_indices();
    const auto& cgra_edge_indices = this->cgra->get_edge_indices();

    torch::Tensor tensor_dfg_features = matrix_to_tensor<float,float>(dfg_features_vec);
    torch::Tensor tensor_dfg_edge_indices = matrix_to_tensor<int,int>(dfg_edge_indices);
    torch::Tensor tensor_cgra_features = matrix_to_tensor<float,float>(cgra_features_vec);
    torch::Tensor tensor_cgra_edge_indices = matrix_to_tensor<int,int>(cgra_edge_indices);

    #ifdef DEBUG
    std::cout << "[DEBUG] Tensor shapes: DFG Features: " << tensor_dfg_features.sizes() 
              << ", DFG Edge Indices: " << tensor_dfg_edge_indices.sizes() 
              << ", CGRA Features: " << tensor_cgra_features.sizes() 
              << ", CGRA Edge Indices: " << tensor_cgra_edge_indices.sizes() << std::endl;
    #endif

    auto id_node_to_be_mapped = this->dfg->get_node_in_placement_order_by_idx_(this->idx_id_node_to_be_mapped);
  
    auto legal_actions_tensor = torch::tensor(this->legal_actions, torch::kInt32);

    auto legal_actions_features = tensor_cgra_features.index_select(0, legal_actions_tensor).unsqueeze(0);  

    torch::Tensor node_to_be_mapped_features = tensor_dfg_features[id_node_to_be_mapped].unsqueeze_(0);

    torch::Tensor actions_pad_mask =  torch::ones({1,static_cast<int>(this->legal_actions.size())}, torch::kFloat32); 
    
    torch::Tensor dfg_pad_mask = torch::ones({1,this->dfg->get_num_nodes()}, torch::kFloat32); 
    torch::Tensor cgra_pad_mask = torch::ones({1,this->cgra->total_PEs()}, torch::kFloat32);

    std::vector<c10::IValue> model_inputs;
    model_inputs.reserve(9);

    model_inputs.emplace_back(tensor_dfg_features);      
    model_inputs.emplace_back(tensor_dfg_edge_indices);
    model_inputs.emplace_back(tensor_cgra_features);
    model_inputs.emplace_back(tensor_cgra_edge_indices);
    model_inputs.emplace_back(node_to_be_mapped_features);
    model_inputs.emplace_back(legal_actions_features);
    model_inputs.emplace_back(actions_pad_mask);
    model_inputs.emplace_back(dfg_pad_mask);
    model_inputs.emplace_back(cgra_pad_mask);

    #ifdef DEBUG
    std::cout << "[DEBUG] Model Inputs:" << std::endl;
    for (size_t i = 0; i < model_inputs.size(); ++i) {
        std::cout << "[DEBUG] Model input " << i << ": " << model_inputs[i].toTensor().sizes() << std::endl;
    }
    #endif

    return model_inputs;
}




void SmartMapState::update_node_to_be_mapped_features()
{
    auto id_node_to_be_mapped = this->dfg->get_node_in_placement_order_by_idx_(this->idx_id_node_to_be_mapped);
    if (id_node_to_be_mapped != -1){
        auto cur_node_features = this->model_inputs[this->idx_dfg_features].toTensor()[id_node_to_be_mapped].unsqueeze_(0);
        this->model_inputs[this->idx_node_to_be_mapped_features] = cur_node_features;
    }else{
         this->model_inputs[this->idx_node_to_be_mapped_features] = torch::tensor(
            {0},
            torch::dtype(torch::kFloat32).requires_grad(false));
    }

}


int SmartMapState::get_assigned_PE_by_node_id(const int& node_id) const {
    // return this->dfg->get_assigned_PE_by_node(node_id);
    return this->model_inputs[this->idx_dfg_features].toTensor()[node_id][this->assigned_pe_idx].item<int>();
}



void SmartMapState::assign_node_to_PE(int node, const int &PE)
{
    auto cgra_features_tensor = this->model_inputs[this->idx_cgra_features].toTensor();
    cgra_features_tensor[PE][this->id_mapped_dfg_node_idx] = node;
    this->model_inputs[this->idx_cgra_features] = cgra_features_tensor;
}

void SmartMapState::assign_PE_to_node(int PE,const int& node)
{
    this->node_to_pe[node] = PE;

    torch::Tensor tensor = this->model_inputs[this->idx_dfg_features].toTensor();
    tensor[node][this->assigned_pe_idx] = PE;
    this->model_inputs[this->idx_dfg_features]= tensor;
}

void SmartMapState::place_node_in_PE(int node, int PE){
    this->assign_node_to_PE(node,PE);
    this->assign_PE_to_node(PE, node);
    this->add_occupied_PE(PE);
}
void SmartMapState::update_masks(){
    this->model_inputs[this->idx_action_mask] = torch::ones({1,static_cast<int>(this->legal_actions.size())}) ;
}

void SmartMapState::update_actions_features(){
    if (static_cast<int>(this->legal_actions.size()) > 0){
        auto tensor_actions_features = this->create_legal_actions_features();
        this->model_inputs[this->idx_actions_features] = tensor_actions_features;
    }else{
        this->model_inputs[this->idx_actions_features] = torch::tensor({});

    }
}


void SmartMapState::assign_modulo_time_slice(const int& action, const int& node){
    const auto& coord = this->cgra->get_coord_by_PE_(action);
    auto modulo = std::get<0>(coord);
    torch::Tensor tensor = this->model_inputs[this->idx_dfg_features].toTensor();
    tensor[node][this->sched_mod_time_slice_idx] = modulo;
    this->model_inputs[this->idx_dfg_features] = tensor;
}


void SmartMapState::update_num_nodes_mapped_in_same_modulo(const int& action, const int& node){
    const auto& coord = this->cgra->get_coord_by_PE_(action);
    auto modulo = std::get<0>(coord);
    if(this->sched_mod_time_slice_to_nodes.find(modulo) == this->sched_mod_time_slice_to_nodes.end()){
        this->sched_mod_time_slice_to_nodes[modulo] = {node};
    }else{
        this->sched_mod_time_slice_to_nodes[modulo].insert(node);
    }
    int num_nodes_in_same_mod_time_slice =  this->sched_mod_time_slice_to_nodes.at(modulo).size();
    torch::Tensor tensor = this->model_inputs[this->idx_dfg_features].toTensor();
    for (auto& node:  this->sched_mod_time_slice_to_nodes.at(modulo)){
        tensor[node][this->num_nodes_mapped_in_same_modulo_idx] = num_nodes_in_same_mod_time_slice;
    }
    this->model_inputs[this->idx_dfg_features] = tensor;
}

void SmartMapState::add_routing(const std::pair<int, int> &tgt_routing_PEs, std::vector<int> &routing_path)
{
    this->cgra->add_routing(tgt_routing_PEs, routing_path);
}


const torch::Tensor SmartMapState::get_dfg_features() const{
    return this->model_inputs[this->idx_dfg_features].toTensor();
}

const torch::Tensor SmartMapState::get_dfg_edge_indices() const{
    return this->model_inputs[this->idx_dfg_edge_indices].toTensor();

}

const torch::Tensor SmartMapState::get_cgra_features() const{
    return this->model_inputs[this->idx_cgra_features].toTensor();

}

const torch::Tensor SmartMapState::get_cgra_edge_indices() const{
    return this->model_inputs[this->idx_cgra_edge_indices].toTensor();

}

const torch::Tensor SmartMapState::get_node_to_be_mapped_features() const {
    return this->model_inputs[this->idx_node_to_be_mapped_features].toTensor();

}

const torch::Tensor SmartMapState::get_action_pad_mask() const{
    return this->model_inputs[this->idx_action_mask].toTensor();
}

const torch::Tensor SmartMapState::get_dfg_pad_masks() const{
    return this->model_inputs[this->idx_dfg_pad_mask].toTensor();

}

const torch::Tensor SmartMapState::get_cgra_pad_masks() const{
    return this->model_inputs[this->idx_cgra_pad_mask].toTensor();
}

bool SmartMapState::is_bad_reward(const double& reward){
    return this->bad_reward == reward;
}

std::unordered_map<int, int> SmartMapState::gen_PE_to_node() const
{
    std::unordered_map<int, int> PE_to_node;
    auto cgra_features = this->model_inputs[this->idx_cgra_features].toTensor();
    for (int i =0; i < cgra_features.size(0); i++){
        PE_to_node[cgra_features[i][this->cgra_id_idx].item<int>()] = cgra_features[i][this->id_mapped_dfg_node_idx].item<int>();
    }
    return PE_to_node;
}

void SmartMapState::add_routing_nodes_by_neigh(const int& id_node_to_be_mapped,
                                const int& action, bool in_vertices){
    const auto& neigh_vertices = in_vertices ? BaseState::get_in_vertices_dfg_by_node_id(id_node_to_be_mapped) : 
                        BaseState::get_out_vertices_dfg_by_node_id(id_node_to_be_mapped);
    
    for (const auto& neigh : neigh_vertices) {
        if(this->node_was_mapped(neigh)){
            auto neigh_PE = BaseState::get_assigned_PE_by_node_id(neigh);
            auto pe_routing_pair = in_vertices ? std::make_pair(neigh_PE, action) : std::make_pair(action, neigh_PE);
            auto routing = this->PEs_to_routing.at(pe_routing_pair);
            for(int i = 1; i < static_cast<int>(routing.size()) - 1; i++){
                this->assign_node_to_PE(-2, neigh_PE);
        }
        }
    }
}

void SmartMapState::add_routing_nodes(const int& id_node_to_be_mapped,
                                const int& action){
    add_routing_nodes_by_neigh(id_node_to_be_mapped,action,true);
    add_routing_nodes_by_neigh(id_node_to_be_mapped,action,false);
}

bool SmartMapState::do_backtracking() const {
    return this->is_end_state && !this->mapping_is_valid;
}


void SmartMapState::print(){
    std::cout << "\nSmartMapState Information:\n";
    std::cout << "  Scheduled Mod Time Slice to Nodes:\n";
    for (const auto& pair : this->sched_mod_time_slice_to_nodes) {
        std::cout << "    Mod Time Slice " << pair.first << " -> Nodes: ";
        for (const int node : pair.second) {
            std::cout << node << " ";
        }
        std::cout << "\n";
    }
    std::stringstream ss;
    BaseState::print_base_state(ss);
    std::cout << ss.str();
}

void SmartMapState::update_action_to_idx() {
    this->action_to_idx.clear();  
    this->idx_to_action.clear();  

    for (int count = 0; count < static_cast<int>(this->legal_actions.size()); ++count) {
        auto action = this->legal_actions[count];
        this->action_to_idx[action] = count;
        this->idx_to_action[count] = action;

    }
}
