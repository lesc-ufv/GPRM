#include "src/hpp/compilers/mapzero/rl/mapzero_state.hpp"


MapZeroState::MapZeroState(std::shared_ptr<OrderedDFG> dfg, std::shared_ptr<CGRA> cgra ,StateConfig& state_config)
: 
BaseState(dfg, cgra, state_config), use_spatio_temporal_input(state_config.getUseSpatioTemporalInput()) {
    this->override_base_state = state_config.getOverrideBaseState();
    if(!this->dfg->has_scheduling() || this->dfg->is_mobility_sched()){
        throw std::runtime_error("MapZero doesn't support multi modulo processing yet.");
    } 

    this->II = this->cgra->get_II();
    
    //masks must be initialized first
    auto number_of_spatial_PEs = this->cgra->number_of_spatial_PEs_();
    this->cgra_features_per_module = torch::full({II,number_of_spatial_PEs,7},-1,torch::kFloat32);
    
    std::vector<std::vector<std::array<int,7>>> cgra_features;
    cgra_features.resize(II, std::vector<std::array<int,7>>());
    for ( int i = 0; i < static_cast<int>(II); i++){
        cgra_features[i].resize(number_of_spatial_PEs,std::array<int,7>()); 
    }
    
    std::unordered_map<EnumFunctionalities,int> functionality_to_idx = {
        {EnumFunctionalities::LOGICAL, 0},
        {EnumFunctionalities::ARITHMETIC, 1},
        {EnumFunctionalities::MEMORY_ACCESS, 2}
    };
    
    auto cgra_II1 = CGRA(1, this->cgra->get_cgra_dimensions_(),this->cgra->get_pe_to_functionalities(),
    this->cgra->get_interconnection_styles(), false);
    
    const auto& pe_to_functionalities = this->cgra->get_pe_to_functionalities();
    for (const auto& vertice : this->cgra->get_vertices_()){
        int i = vertice / number_of_spatial_PEs;
        int vertice_idx = vertice % number_of_spatial_PEs;  
        
        cgra_features[i][vertice_idx][this->cgra_id_idx] = vertice;
        
        cgra_features[i][vertice_idx][this->cgra_in_degree_idx] = this->use_spatio_temporal_input ? this->cgra->get_in_vertices_by_id_(vertice_idx).size() : cgra_II1.get_in_vertices_by_id_(vertice_idx).size();
        
        cgra_features[i][vertice_idx][this->cgra_out_degree_idx] =  this->use_spatio_temporal_input ? this->cgra->get_out_vertices_by_id_(vertice_idx).size() : cgra_II1.get_out_vertices_by_id_(vertice_idx).size();
        
        cgra_features[i][vertice_idx][this->id_mapped_dfg_node_idx] =  -1;
        
        for (const auto& functionality: pe_to_functionalities.at(vertice)){
            int functionality_idx = functionality_to_idx.at(functionality);
            #ifdef DEBUG
                std::cout << "[DEBUG] functionality: " << static_cast<int>(functionality) << ", functionality_idx: " << functionality_idx 
                << ", cgra_features[" << i << "][" << vertice_idx << "][" 
                << (this->functionalitie_initial_idx + functionality_idx) << "] = 1" << std::endl;
            #endif
            cgra_features[i][vertice_idx][this->functionalitie_initial_idx + functionality_idx] = 1;
        }
        #ifdef DEBUG

        std::cout << "[DEBUG] Vertex: " << vertice << ", i: " << i << ", vertice_idx: " << vertice_idx << std::endl;
        std::cout << "[DEBUG] cgra_id_idx[" << i << "][" << vertice_idx << "]: " << cgra_features[i][vertice_idx][this->cgra_id_idx] << std::endl;
        std::cout << "[DEBUG] cgra_in_degree_idx[" << i << "][" << vertice_idx << "]: " << cgra_features[i][vertice_idx][this->cgra_in_degree_idx] << std::endl;
        std::cout << "[DEBUG] cgra_out_degree_idx[" << i << "][" << vertice_idx << "]: " << cgra_features[i][vertice_idx][this->cgra_out_degree_idx] << std::endl;
        std::cout << "[DEBUG] id_mapped_dfg_node_idx[" << i << "][" << vertice_idx << "]: " << cgra_features[i][vertice_idx][this->id_mapped_dfg_node_idx] << std::endl;
        #endif
    }
    
    
    for (int i = 0; i < static_cast<int>(II); i++){
        auto cur_cgra_features = cgra_features[i];
        std::vector<float> flattened_cgra_features;
        for (const auto& feature_set : cur_cgra_features) {
            flattened_cgra_features.insert(flattened_cgra_features.end(), feature_set.begin(), feature_set.end());
        }
        
        torch::Tensor tensor_cgra_features = torch::from_blob(
            flattened_cgra_features.data(),
            {static_cast<long>(cur_cgra_features.size()), this->cgra_num_features},
            torch::kFloat32
        ).clone();
        
        this->cgra_features_per_module[i] = tensor_cgra_features;
        
    }
    
    // const auto& topo_order = this->dfg->get_topological_sort();
    #ifdef DEBUG
        std::cout<< "Features cgra: \n" << cgra_features_per_module << "\n";
    #endif
    this->initialize();
    
}

MapZeroState::MapZeroState(const MapZeroState& other) 
    :BaseState(other),
     cgra_features_per_module(other.cgra_features_per_module.clone()), override_base_state(other.override_base_state), 
     use_spatio_temporal_input(other.use_spatio_temporal_input)
    {
   
}


const int& MapZeroState::get_II() const{
    return this->II;

}


std::vector<c10::IValue> MapZeroState::calc_initial_model_inputs() {  
    std::unordered_map<int, int> num_nodes_to_be_mapped_same_mod_time_slice;
    std::vector<std::array<int,10>> dfg_features;
    std::unordered_set<int> modulos;
    #ifdef DEBUG
        std::cout << "Iterating over node_to_sched_mods..." << std::endl;
    #endif

    for (const auto& pair : this->node_to_sched_mods) {
        auto modulo = this->dfg->is_asap_scheduling() ? pair.second.first : pair.second.second;

    #ifdef DEBUG
        std::cout << "Node: " << pair.first << " | Modulo: " << modulo << std::endl;
    #endif

        if (modulos.find(modulo) == modulos.end()) {
            modulos.insert(modulo);
            num_nodes_to_be_mapped_same_mod_time_slice[modulo] = 1;

    #ifdef DEBUG
            std::cout << "New modulo inserted: " << modulo << std::endl;
    #endif

        } else {
            num_nodes_to_be_mapped_same_mod_time_slice[modulo]++;

    #ifdef DEBUG
            std::cout << "Incrementing count for modulo " << modulo 
                    << " | New Count: " << num_nodes_to_be_mapped_same_mod_time_slice[modulo] << std::endl;
    #endif
        }
    }


    const auto& placement_order = this->dfg->get_placement_order();
    const auto& dfg_in_vertices = this->dfg->get_node_to_in_vertices();
    const auto& dfg_out_vertices = this->dfg->get_node_to_out_vertices();
    
    #ifdef DEBUG
    std::cout << "[DEBUG] Placement Order: ";
    for (const auto& node : placement_order) std::cout << node << " ";
    std::cout << std::endl;
    #endif
    dfg_features.resize(this->dfg->num_vertices(), std::array<int,10>());
   
    #ifdef DEBUG
    std::cout << "Checking scheduling type..." << std::endl;
#endif

    const auto& planned_time_slice = this->dfg->is_asap_scheduling() ? this->dfg->get_asap() : this->dfg->get_alap();
    

    #ifdef DEBUG
        std::cout << "Planned Time Slice Size: " << planned_time_slice.size() << std::endl;
    #endif

    int count = 0;
    for (const int& node : placement_order) {
        if (node != -1){
    #ifdef DEBUG
            std::cout << "Processing node: " << node << std::endl;
    #endif
            dfg_features[node][this->dfg_id_idx] = node;
            dfg_features[node][this->sched_order_idx] = count;
            dfg_features[node][this->sched_time_slice_idx] = planned_time_slice.empty() ? -1 : planned_time_slice[node];

    #ifdef DEBUG
            std::cout << "Scheduled Time Slice: " << dfg_features[node][this->sched_time_slice_idx] << std::endl;
    #endif

            auto sched_mod = this->dfg->is_asap_scheduling() ? this->node_to_sched_mods.at(node).first
                                                            : this->node_to_sched_mods.at(node).second;

    #ifdef DEBUG
            std::cout << "Scheduling Mod: " << sched_mod << std::endl;
    #endif

            if(this->dfg->has_scheduling() && !this->dfg->is_mobility_sched()){
                dfg_features[node][this->sched_mod_time_slice_idx] = sched_mod;
            } else {
                dfg_features[node][this->sched_mod_time_slice_idx] = -1;
            }

            dfg_features[node][this->dfg_in_degree_idx] = dfg_in_vertices.at(node).size();
            dfg_features[node][this->dfg_out_degree_idx] = dfg_out_vertices.at(node).size();
            dfg_features[node][this->has_self_cycle] = dfg_in_vertices.at(node).find(node) != dfg_in_vertices.at(node).end();
            dfg_features[node][this->opcode_idx] = get_opcode_by_operation(this->dfg->get_operation_by_node(node));
            dfg_features[node][this->num_nodes_to_be_mapped_same_mod_idx] = num_nodes_to_be_mapped_same_mod_time_slice.empty() ? -1 
                                                    : num_nodes_to_be_mapped_same_mod_time_slice[sched_mod];
            dfg_features[node][this->assigned_pe_idx] = -1;            

    #ifdef DEBUG
            std::cout << "DFG In-Degree: " << dfg_features[node][this->dfg_in_degree_idx] << std::endl;
            std::cout << "DFG Out-Degree: " << dfg_features[node][this->dfg_out_degree_idx] << std::endl;
            std::cout << "Has Self Cycle: " << dfg_features[node][this->has_self_cycle] << std::endl;
            std::cout << "Opcode: " << dfg_features[node][this->opcode_idx] << std::endl;
            std::cout << "Nodes to be Mapped in Same Mod: " << dfg_features[node][this->num_nodes_to_be_mapped_same_mod_idx] << std::endl;
    #endif

            count++;
        }
    }

    
    const auto& dfg_edge_indices = this->dfg->get_edge_indices();
    auto cgra_II1 = CGRA(1, this->cgra->get_cgra_dimensions_(),this->cgra->get_pe_to_functionalities(),
    this->cgra->get_interconnection_styles(), false);
    const auto& cgra_edge_indices = cgra_II1.get_edge_indices();
    
    #ifdef DEBUG
    
    std::cout << "[DEBUG] DFG Edge Indices: ";

    print_matrix(dfg_edge_indices);
    
    std::cout << "[DEBUG] CGRA Edge Indices: ";
    print_matrix(cgra_edge_indices);
    std::cout << std::endl;
    #endif
    
    
    std::vector<float> flattened_dfg_features;
    for (const auto& feature_set : dfg_features) {
        flattened_dfg_features.insert(flattened_dfg_features.end(), feature_set.begin(), feature_set.end());
    }

    torch::Tensor tensor_dfg_features = torch::from_blob(
        flattened_dfg_features.data(),
        {static_cast<long>(dfg_features.size()), this->dfg_num_features},
        torch::kFloat32
    ).clone(); 

    std::vector<int> flattened_dfg_edge_indices;
    for (const auto& edge : dfg_edge_indices) {
        flattened_dfg_edge_indices.insert(flattened_dfg_edge_indices.end(), edge.begin(), edge.end());
    }

    torch::Tensor tensor_dfg_edge_indices = torch::from_blob(
        flattened_dfg_edge_indices.data(),
        {2, static_cast<long>(dfg_edge_indices[0].size())},
        torch::kInt32
    ).clone();

    std::vector<int> flattened_cgra_edge_indices;
    for (const auto& edge : cgra_edge_indices) {
        flattened_cgra_edge_indices.insert(flattened_cgra_edge_indices.end(), edge.begin(), edge.end());
    }

    torch::Tensor tensor_cgra_edge_indices = torch::from_blob(
        flattened_cgra_edge_indices.data(),
        {2, static_cast<long>(cgra_edge_indices[0].size())},
        torch::kInt32
    ).clone();

    auto id_node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);
    auto node_to_be_mapped_features = tensor_dfg_features[id_node_to_be_mapped].unsqueeze(0);
    
    auto tensor_cgra_features = this->use_spatio_temporal_input ? this->cgra_features_per_module.view({this->cgra->total_PEs(),7}) : this->cgra_features_per_module[0];
    torch::Tensor mask = this->calc_masks(id_node_to_be_mapped); 
    
    torch::Tensor dfg_pad_mask = torch::ones({1,this->dfg->get_num_nodes()}, torch::kFloat32); 
    torch::Tensor cgra_pad_mask = torch::ones({1,this->use_spatio_temporal_input ? \
        this->cgra->number_of_spatial_PEs_() * this->II : this->cgra->number_of_spatial_PEs_()}, torch::kFloat32);

    
    std::vector<c10::IValue> temp_model_inputs;
    temp_model_inputs.reserve(8);

    temp_model_inputs.emplace_back(tensor_dfg_features);      
    temp_model_inputs.emplace_back(tensor_dfg_edge_indices);
    temp_model_inputs.emplace_back(tensor_cgra_features);
    temp_model_inputs.emplace_back(tensor_cgra_edge_indices);
    temp_model_inputs.emplace_back(node_to_be_mapped_features);
    temp_model_inputs.emplace_back(mask);
    temp_model_inputs.emplace_back(dfg_pad_mask);
    temp_model_inputs.emplace_back(cgra_pad_mask);

    return temp_model_inputs;
}




void MapZeroState::assign_scheduled_time_slice_to_node(const int scheduled_time_slice, const int &node)
{
    this->node_to_scheduled_time_slice[node] = scheduled_time_slice;
    auto tensor = this->model_inputs[this->idx_dfg_features].toTensor();
    tensor[node][this->sched_time_slice_idx] = scheduled_time_slice;
    this->model_inputs[this->idx_dfg_features] = tensor;
}



void MapZeroState::update_node_to_be_mapped_features()
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




void MapZeroState::update_action_to_idx() {
    this->action_to_idx.clear();
    this->idx_to_action.clear();

    #ifdef DEBUG
    std::cout << "(MapZero) Updating action_to_idx map..." << std::endl;
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


int MapZeroState::calc_relative_PE(const int &PE)
{
    return PE % this->cgra->number_of_spatial_PEs_();
}

void MapZeroState::assign_node_to_PE(int node, const int &PE)
{
    // this->cgra->assign_node_to_PE(node, PE);
    int cur_modulo = PE / this->cgra->number_of_spatial_PEs_();
    auto relative_PE = this->calc_relative_PE(PE);
    #ifdef DEBUG
        auto value = this->cgra_features_per_module[cur_modulo][relative_PE][this->id_mapped_dfg_node_idx].item<int>();
        assert(value == -1);
    #endif
    this->cgra_features_per_module[cur_modulo][relative_PE][this->id_mapped_dfg_node_idx] = node;

}

void MapZeroState::assign_PE_to_node(int PE,const int& node)
{
    torch::Tensor tensor = this->model_inputs[this->idx_dfg_features].toTensor();
    #ifdef DEBUG
        assert(tensor[node][this->assigned_pe_idx].item<int>() == -1);
    #endif
    tensor[node][this->assigned_pe_idx] = PE;
    this->model_inputs[this->idx_dfg_features]= tensor;
}

torch::Tensor MapZeroState::calc_masks(int id_node_to_be_mapped){
    auto num_spatial_pes = this->cgra->number_of_spatial_PEs_();
  
    #ifdef DEBUG
        assert(static_cast<int>(this->legal_actions.size()) <= num_spatial_pes);
    #endif
    auto idx = torch::tensor(this->legal_actions, torch::kInt32) % num_spatial_pes;

    auto mask = torch::zeros({1, num_spatial_pes}, torch::kInt32);

    mask.index_put_({0, idx}, 1);
    return mask.reshape({1, num_spatial_pes});
}

void MapZeroState::place_node_in_PE(int node, int PE){
    this->assign_node_to_PE(node,PE);
    this->assign_PE_to_node(PE, node);    
}

void MapZeroState::update_masks_in_model_input(){
    int node = this->dfg->get_node_in_placement_order_by_idx_(this->idx_id_node_to_be_mapped); 
    if (node != -1){
        auto mask = this->calc_masks(node);
        this->model_inputs[this->idx_PE_mask] = mask;
    }
}


void MapZeroState::assign_modulo_time_slice(int modulo_time_slice, const int& node){
    torch::Tensor tensor = this->model_inputs[this->idx_dfg_features].toTensor();
    tensor[node][this->sched_mod_time_slice_idx] = modulo_time_slice;
    this->model_inputs[this->idx_dfg_features] = tensor;
}


int MapZeroState::get_scheduled_time_slice_by_node_id(const int& node_id)
{
    return this->model_inputs[this->idx_dfg_features].toTensor()[node_id][this->sched_time_slice_idx].item<int>();
}

void MapZeroState::update_is_end_state(){
    if(this->override_base_state){

        bool all_nodes_were_mapped = this->all_nodes_were_mapped();
        if (all_nodes_were_mapped || this->legal_actions.empty()){
            this->is_end_state = true;
    }
    }else{
        BaseState::update_is_end_state();
    }
}




const torch::Tensor MapZeroState::get_dfg_features() const{
    return this->model_inputs[this->idx_dfg_features].toTensor();
}

const torch::Tensor MapZeroState::get_dfg_edge_indices() const{
    return this->model_inputs[this->idx_dfg_edge_indices].toTensor();

}

const torch::Tensor MapZeroState::get_cgra_features() const{
    return this->model_inputs[this->idx_cgra_features].toTensor();

}

const torch::Tensor MapZeroState::get_cgra_edge_indices() const{
    return this->model_inputs[this->idx_cgra_edge_indices].toTensor();

}

const torch::Tensor MapZeroState::get_node_to_be_mapped_features() const {
    return this->model_inputs[this->idx_node_to_be_mapped_features].toTensor();

}

const torch::Tensor MapZeroState::get_mask() const{
    return this->model_inputs[this->idx_PE_mask].toTensor();
}

const torch::Tensor MapZeroState::get_dfg_pad_masks() const{
    return this->model_inputs[this->idx_dfg_pad_mask].toTensor();

}

const torch::Tensor MapZeroState::get_cgra_pad_masks() const{
    return this->model_inputs[this->idx_cgra_pad_mask].toTensor();
}


int MapZeroState::get_cgra_input_size(){
    if(!this->use_spatio_temporal_input){
        return this->cgra->number_of_spatial_PEs_();
    }else{
        return this->cgra->number_of_spatial_PEs_() * this->II;
    }
}


void MapZeroState::update_features_before_state_step(const int& action)  {
    auto id_node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);
    this->place_node_in_PE(id_node_to_be_mapped, action);
    const auto& in_vertices_cur_node = this->dfg->get_in_vertices_by_node_id(id_node_to_be_mapped);
    const auto& out_vertices_cur_node = this->dfg->get_out_vertices_by_node_id(id_node_to_be_mapped);

    const auto& PEs_to_routing = this->get_PEs_to_routing_();
    for (const auto& father : in_vertices_cur_node) {
        if(this->node_was_mapped(father)){
            auto father_PE = this->get_assigned_PE_by_node_id(father);
            std::vector<int> routing_path;

            auto pe_routing_pair = std::make_pair(father_PE, action);

            auto cur_routing = PEs_to_routing.at(pe_routing_pair);
            int cur_routing_size = cur_routing.size();

            if (cur_routing_size > 2) {
                for (int i = 1; i < cur_routing_size - 1; i++) {
                    auto cur_PE = cur_routing[i];
                    this->assign_node_to_PE(-2, cur_PE);
                }
            }

        }
    }

    for (const auto& child : out_vertices_cur_node) {
        if(this->node_was_mapped(child)){
            auto child_PE = this->get_assigned_PE_by_node_id(child);
            std::vector<int> routing_path;

            auto pe_routing_pair = std::make_pair(action,child_PE);

            auto cur_routing = PEs_to_routing.at(pe_routing_pair);
            int cur_routing_size = cur_routing.size();

            if (cur_routing_size > 2) {
                for (int i = 1; i < cur_routing_size - 1; i++) {
                    auto cur_PE = cur_routing[i];
                    this->assign_node_to_PE(-2, cur_PE);
                }
            }
        }
    }

};

void MapZeroState::update_features_after_state_step(const int& action)  {
    int node_to_be_mapped = this->dfg->get_node_in_placement_order_by_idx_(this->idx_id_node_to_be_mapped); 
    this->update_node_to_be_mapped_features();

    if (node_to_be_mapped != -1 && static_cast<int>(this->legal_actions.size()) > 0){
        auto first_pe = this->legal_actions[0];
        auto pe_mod = first_pe / this->cgra->number_of_spatial_PEs_();
        auto next_cgra_features = this->cgra_features_per_module[pe_mod];
        this->model_inputs[this->idx_cgra_features] = this->use_spatio_temporal_input ? this->cgra_features_per_module.view({this->cgra->total_PEs(),7}) :  next_cgra_features;
    
    }
    this->update_masks_in_model_input();
};


bool MapZeroState::do_backtracking() const {
    if(this->override_base_state) return this->contains_invalid_routing || (this->is_end_state && !this->mapping_is_valid);
    return BaseState::do_backtracking();
}

void MapZeroState::print() const
{
        std::cout << "MapZeroState:" << std::endl;
        
        if (cgra_features_per_module.defined()) {
            std::cout << "CGRA Features Per Module: " << cgra_features_per_module.sizes() << std::endl;
            std::cout << cgra_features_per_module << std::endl;
        } else {
            std::cout << "CGRA Features Per Module: Undefined" << std::endl;
        }
        std::cout << "override_base_state: " << override_base_state << std::endl;
        std::cout << "use_spatio_temporal_input: " << use_spatio_temporal_input << std::endl;

        std::stringstream ss;
        BaseState::print_base_state(ss);
        std::cout << ss.str();}
