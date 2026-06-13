#include "src/hpp/rl/base_state.hpp"
// #define DEBUG
const int& BaseState::get_id_node_to_be_mapped_by_idx(const int& idx_id_node_to_be_mapped) const
{
    return this->dfg->get_node_in_placement_order_by_idx_(idx_id_node_to_be_mapped);
    
}

BaseState::BaseState(std::shared_ptr<OrderedDFG> dfg, std::shared_ptr<CGRA> cgra, StateConfig& state_config)
: is_end_state(false), mapping_is_valid(false), invalid_mapping_reason(""), idx_id_node_to_be_mapped(0), 
contains_invalid_routing(false), contains_invalid_scheduling(false), num_mapped_edges(0), dfg(dfg), cgra(cgra)
{
    #ifdef DEBUG
        std::cout << "[DEBUG] Initializing BaseState..." << std::endl;
    #endif
    this->node_was_placed_tensor = torch::empty({});
    this->remain_in_edges = torch::empty({});
    this->remain_out_edges = torch::empty({});

    this->II = this->cgra->get_II();
    this->free_connections = this->cgra->get_free_connections();
    this->use_routing_mask = state_config.getUseCongestionEdgeMask();
    this->use_symmetry_mask = state_config.getUseSymmetricMask();
    this->use_used_pe_mask = state_config.getUseUsedPeMask();
    this->use_rt_scheduler = state_config.getUseRTScheduler();
    this->use_sched_mask = state_config.getUseSchedMask();
    this->greedy = this->dfg->is_greedy(); 
    this->enum_end_state = state_config.getEndState();
    this->use_sync_routing = state_config.getUseSyncRouting();

    #ifdef DEBUG
        std::cout << "[DEBUG] Initial II: " << this->II << std::endl;
        std::cout << "[DEBUG] Total PEs: " << this->cgra->total_PEs() << std::endl;
    #endif

    if(this->dfg->has_scheduling()){
        for(int i = 0; i < this->dfg->num_vertices(); i++){
            int asap,alap;
            if(this->dfg->is_asap_scheduling()){
                asap = this->dfg->get_asap_value_by_node(i);
                int mod = asap % this->II;
                node_to_sched_mods[i] = std::make_pair(mod,mod);
                this->node_to_scheduled_time_slice[i] = asap;
            }else if(this->dfg->is_alap_scheduling()){
                alap = this->dfg->get_alap_value_by_node(i);
                this->node_to_scheduled_time_slice[i] = alap;
                int mod = alap % this->II;
                node_to_sched_mods[i] = std::make_pair(mod,mod);
            }else if(this->dfg->is_mobility_sched()){
                asap = this->dfg->get_asap_value_by_node(i);
                alap = this->dfg->get_alap_value_by_node(i);
                int asap_mod, alap_mod;
                asap_mod = asap % this->II;
                alap_mod = alap % this->II;
                node_to_sched_mods[i] = std::make_pair(asap_mod,alap_mod);
            }else if (this->dfg->is_tight_sched()){
                auto tight_value = this->dfg->get_tight_value_by_node(i);
                this->node_to_scheduled_time_slice[i] = tight_value;
                int mod = tight_value % this->II;
                node_to_sched_mods[i] = std::make_pair(mod,mod);
            }
        }
    }
    
    this->node_was_placed_tensor = torch::zeros({this->cgra->total_PEs()},torch::kInt32);
    if(this->use_routing_mask){
        this->remain_in_edges = torch::empty(this->cgra->total_PEs(), torch::kInt32);
        this->remain_out_edges = torch::empty(this->cgra->total_PEs(), torch::kInt32);

        for(int i = 0; i<this->cgra->total_PEs();i++ ){
            this->remain_in_edges[i] = this->cgra->get_in_vertices_by_id_(i).size();
            this->remain_out_edges[i] = this->cgra->get_out_vertices_by_id_(i).size();
        }
    }
    // keep order
    this->calc_initial_legal_actions();
    #ifdef DEBUG
    std::cout << "[DEBUG] BaseState initialization complete." << std::endl;
    #endif



    // if(this->use_rt_scheduler){
    //     this->rt_scheduler = RTScheduler();
    // }
}


BaseState::BaseState()
: rt_scheduler(),                         
  II(0),
  is_end_state(false),
  mapping_is_valid(false),
  invalid_mapping_reason(""),
  idx_id_node_to_be_mapped(0),
  contains_invalid_routing(false),
  contains_invalid_scheduling(false),
  num_mapped_edges(0),
  dfg(nullptr),
  cgra(nullptr),
  greedy(false),
  model_inputs(),
  enum_end_state(EnumEndState::DEFAULT), 
  node_to_scheduled_time_slice(),
  node_to_pe(),
  pe_to_node(),
  PEs_to_routing(),
  free_connections(),
  node_was_placed_tensor(torch::empty({})),
  remain_in_edges(torch::empty({})),
  remain_out_edges(torch::empty({})),
  legal_actions(),
  action_to_idx(),
  idx_to_action(),
  node_to_sched_mods(),
  use_used_pe_mask(false),
  use_routing_mask(false),
  use_symmetry_mask(false),
  use_rt_scheduler(false),
  use_sync_routing(false),
    use_sched_mask(true)
{}
void BaseState::initialize() {
    this->update_action_to_idx();
    this->model_inputs = this->calc_initial_model_inputs();
};


BaseState::BaseState(const BaseState& other)

: rt_scheduler(other.rt_scheduler),
     II(other.II),
        is_end_state(other.is_end_state),
      mapping_is_valid(other.mapping_is_valid),
      invalid_mapping_reason(other.invalid_mapping_reason),
      idx_id_node_to_be_mapped(other.idx_id_node_to_be_mapped),
      contains_invalid_routing(other.contains_invalid_routing),
      contains_invalid_scheduling(other.contains_invalid_scheduling),
      num_mapped_edges(other.num_mapped_edges),
      dfg(other.dfg),
      cgra(other.cgra),
      greedy(other.greedy),
      node_to_scheduled_time_slice(other.node_to_scheduled_time_slice),
      node_to_pe(other.node_to_pe),
      pe_to_node(other.pe_to_node),
      PEs_to_routing(other.PEs_to_routing),
      free_connections(other.free_connections),
      node_was_placed_tensor(other.node_was_placed_tensor.clone()),
      legal_actions(other.legal_actions),
      action_to_idx(other.action_to_idx),
      idx_to_action(other.idx_to_action),
      node_to_sched_mods(std::ref(other.node_to_sched_mods)), 
      use_used_pe_mask(other.use_used_pe_mask), 
      use_routing_mask(other.use_routing_mask),
      use_symmetry_mask(other.use_symmetry_mask),
      use_rt_scheduler(other.use_rt_scheduler),
      use_sync_routing(other.use_sync_routing),
      use_sched_mask(other.use_sched_mask)
    {
  
    if(other.remain_in_edges.defined()){
        remain_in_edges = other.remain_in_edges.clone();
    }

    if(other.remain_out_edges.defined()){
        remain_out_edges = other.remain_out_edges.clone();
    }
    

    this->model_inputs.reserve(other.model_inputs.size());

    for (size_t i = 0; i < other.model_inputs.size(); ++i) {
        this->model_inputs.emplace_back(other.model_inputs[i].toTensor().clone());
    }
}

BaseState::BaseState(BaseState&& other) noexcept{
    this->rt_scheduler = std::move(other.rt_scheduler);
    this->II = other.II;
    this->is_end_state = other.is_end_state;
    this->mapping_is_valid = other.mapping_is_valid;
    this->invalid_mapping_reason = std::move(other.invalid_mapping_reason);
    this->idx_id_node_to_be_mapped = other.idx_id_node_to_be_mapped;
    this->contains_invalid_routing = other.contains_invalid_routing;
    this->contains_invalid_scheduling = other.contains_invalid_scheduling;
    this->num_mapped_edges = other.num_mapped_edges;
    this->dfg = std::move(other.dfg);
    this->cgra = std::move(other.cgra);
    this->greedy = other.greedy;
    this->node_to_scheduled_time_slice = std::move(other.node_to_scheduled_time_slice);
    this->node_to_pe = std::move(other.node_to_pe);
    this->pe_to_node = std::move(other.pe_to_node);
    this->PEs_to_routing = std::move(other.PEs_to_routing);
    this->free_connections = std::move(other.free_connections);
    this->node_was_placed_tensor = std::move(other.node_was_placed_tensor);
    this->legal_actions = std::move(other.legal_actions);
    this->action_to_idx = std::move(other.action_to_idx);
    this->idx_to_action = std::move(other.idx_to_action);
    this->node_to_sched_mods = std::ref(other.node_to_sched_mods); 
    this->use_used_pe_mask = other.use_used_pe_mask; 
    this->use_routing_mask = other.use_routing_mask;
    this->use_symmetry_mask = other.use_symmetry_mask;
    this->use_rt_scheduler = other.use_rt_scheduler;
    this->use_sync_routing = other.use_sync_routing;
    this->use_sched_mask = other.use_sched_mask;

    if (other.remain_in_edges.defined()) {
        remain_in_edges = std::move(other.remain_in_edges);
    }

    if (other.remain_out_edges.defined()) {
        remain_out_edges = std::move(other.remain_out_edges);
    }
    this->model_inputs = std::move(other.model_inputs);
}


void BaseState::route(bool route_in_vertices, int action){
    auto id_node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);
    auto arch_dims = this->get_cgra_dimensions_();
    const auto& pe_to_coord = this->get_PE_to_coord_();
    const auto& PEs_to_routing = this->get_PEs_to_routing_();
    const auto& free_connections = this->get_free_connections_();
    const auto& out_vertices_cgra = this->get_out_vertices_cgra_();
    const auto& neigh_vertices = route_in_vertices ? this->get_in_vertices_dfg_by_node_id(id_node_to_be_mapped) : 
                        this->get_out_vertices_dfg_by_node_id(id_node_to_be_mapped);
    int cost = -1;

    for (const auto& neigh : neigh_vertices) {
            if (this->node_was_mapped(neigh))
            {
                auto occupied_PEs = this->get_occupied_PEs_();
                auto neigh_PE = this->get_assigned_PE_by_node_id(neigh);
                std::vector<int> routing_path;
                std::tie(routing_path, cost) = std::move(Router::route(route_in_vertices? neigh_PE: action
                , arch_dims, route_in_vertices ? action : neigh_PE, pe_to_coord, occupied_PEs, PEs_to_routing, free_connections, out_vertices_cgra));

                auto pe_routing_pair = route_in_vertices ? std::make_pair(neigh_PE, action) : std::make_pair(action, neigh_PE);
                this->route_by_routing_path(pe_routing_pair, routing_path);

                // PEs_to_routing = this->get_PEs_to_routing_();
                const auto& cur_PEs_to_routing = PEs_to_routing.at(pe_routing_pair);
                int cur_PEs_to_routing_size = cur_PEs_to_routing.size();
                if (cur_PEs_to_routing_size > 0) this->num_mapped_edges += 1;
                if (cur_PEs_to_routing_size > 2) {
                    for (int i = 1; i < cur_PEs_to_routing_size - 1; i++) {
                        auto cur_PE = cur_PEs_to_routing[i];
                        this->add_occupied_PE(cur_PE);
                        #ifdef DEBUG
                            assert(this->pe_to_node.find(cur_PE) == this->pe_to_node.end());
                        #endif
                        this->pe_to_node[cur_PE] = -2;
                    }
                }else if (cur_PEs_to_routing_size == 0){
                    this->contains_invalid_routing = true;
                }

                if(this->use_routing_mask){
                    for(int k = 0; k < static_cast<int>(cur_PEs_to_routing.size()) - 1;k++){
                        auto in_node = cur_PEs_to_routing[k];
                        auto out_node = cur_PEs_to_routing[k+1];
                        this->remain_in_edges[out_node] -= 1;
                        this->remain_out_edges[in_node] -= 1;
                        if(this->II == 1){
                            this->remain_in_edges[in_node] -= 1;
                            this->remain_out_edges[out_node] -= 1;
                        }
                        #ifdef DEBUG
                            assert(this->remain_in_edges[out_node].item<int>() >= 0);
                            assert(this->remain_out_edges[in_node].item<int>() >= 0);
                        #endif
                    }
                }
                
            }
        }
}

void BaseState::reroute(int in_pe, int out_pe, int distance){
    const auto& PEs_to_routing = this->get_PEs_to_routing_();
    const auto& out_vertices_cgra = this->get_out_vertices_cgra_();
    const auto& pe_routing_pair = std::make_pair(in_pe, out_pe);
    const auto& old_PEs_to_routing = PEs_to_routing.at(pe_routing_pair);

#ifdef DEBUG
    std::cerr << "[DEBUG] Rerouting (" << in_pe << " -> " << out_pe 
              << ") with distance " << distance << "\n";
    std::cerr << "[DEBUG] Old path size: " << old_PEs_to_routing.size() << "\n";
#endif

    auto occupied_PEs = this->get_occupied_PEs_();
    for(int i = 1; i < static_cast<int>(old_PEs_to_routing.size()) - 1; i++){
        auto cur_PE = old_PEs_to_routing[i];
        occupied_PEs.erase(cur_PE);
    }

    for (int i = 0; i < static_cast<int>(old_PEs_to_routing.size()) - 1; ++i) {
        this->free_connections[old_PEs_to_routing[i]].insert(old_PEs_to_routing[i+1]);
    }

    std::vector<int> routing_path = Router::find_path_exact_distance(
        in_pe, out_pe, distance, occupied_PEs, free_connections, out_vertices_cgra
    );

#ifdef DEBUG
    std::cerr << "[DEBUG] New routing path size: " << routing_path.size() << "\n";
#endif

    if(routing_path.empty()){
#ifdef DEBUG
        std::cerr << "[DEBUG] No valid routing path found!\n";
#endif
        for (int i = 0; i < static_cast<int>(routing_path.size()) - 1; ++i) {
            free_connections[routing_path[i]].erase(routing_path[i+1]);
        }
        return;
    }

    //undo last routing
    int old_PEs_to_routing_size = old_PEs_to_routing.size();
    for (int i = 1; i < old_PEs_to_routing_size - 1; i++) {
        auto cur_PE = old_PEs_to_routing[i];
        this->remove_occupied_PE(cur_PE);
        this->pe_to_node[cur_PE] = -1;
    }
   
    if(this->use_routing_mask){
        for(int k = 0; k < static_cast<int>(old_PEs_to_routing.size()) - 1;k++){
            auto in_node = old_PEs_to_routing[k];
            auto out_node = old_PEs_to_routing[k+1];
            this->remain_in_edges[out_node] += 1;
            this->remain_out_edges[in_node] += 1;
            if(this->II == 1){
                this->remain_in_edges[in_node] += 1;
                this->remain_out_edges[out_node] += 1;
            }
        }
    }

    int cur_routing_path_size = routing_path.size();
    for (int i = 1; i < cur_routing_path_size - 1; i++) {
        auto cur_PE = routing_path[i];
        this->add_occupied_PE(cur_PE);
        this->pe_to_node[cur_PE] = -2;
    }
   
    if(this->use_routing_mask){
        for(int k = 0; k < static_cast<int>(routing_path.size()) - 1;k++){
            auto in_node = routing_path[k];
            auto out_node = routing_path[k+1];
            this->remain_in_edges[out_node] -= 1;
            this->remain_out_edges[in_node] -= 1;
            if(this->II == 1){
                this->remain_in_edges[in_node] -= 1;
                this->remain_out_edges[out_node] -= 1;
            }
        }
    }

#ifdef DEBUG
    std::cerr << "[DEBUG] Successfully rerouted (" << in_pe << " -> " << out_pe << ")\n";
#endif

    this->route_by_routing_path(pe_routing_pair, routing_path);
}


void BaseState::add_occupied_PE(int pe) {
    // #ifdef DEBUG
    // std::cout << "[DEBUG] Adding occupied PE: " << pe << std::endl;
    // std::cout << "[DEBUG] Number of occupied PEs before: " << this->node_was_mapped.size() << std::endl;
    // std::cout << "[DEBUG] Number of not occupied PEs before: " << this->not_occupied_PEs.size() << std::endl;
    // #endif

    this->node_was_placed_tensor[pe] = 1;

    // #ifdef DEBUG
    // std::cout << "[DEBUG] Number of occupied PEs after: " << this->node_was_mapped.size() << std::endl;
    // std::cout << "[DEBUG] Number of not occupied PEs after: " << this->not_occupied_PEs.size() << std::endl;
    // #endif
}


void BaseState::place_and_route(const int& action){
    this->add_occupied_PE(action);
    auto id_node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);
    this->node_to_pe[id_node_to_be_mapped] = action;
    this->pe_to_node[action] = id_node_to_be_mapped;
    this->route(true, action);
    this->route(false, action);
}

void BaseState::analyze_mapping(){
    bool mapping_is_valid = true;
    std::string reason;

#ifdef DEBUG
    std::cout << "[DEBUG] Starting analyze_mapping..." << std::endl;
#endif

    if (!this->all_nodes_were_mapped()) {
#ifdef DEBUG
        std::cout << "[DEBUG] Invalid placement detected." << std::endl;
#endif
        reason = get_invalid_mapping_reason(EnumInvalidMappingReason::INVALID_PLACEMENT);
        mapping_is_valid = false;
    } else if (!Router::routing_is_valid(PEs_to_routing)) {
#ifdef DEBUG
        std::cout << "[DEBUG] Invalid routing detected." << std::endl;
#endif
        reason = get_invalid_mapping_reason(EnumInvalidMappingReason::INVALID_ROUTING);
        mapping_is_valid = false;
    }

    if (mapping_is_valid) {
#ifdef DEBUG
        std::cout << "[DEBUG] Placement and routing are valid. Proceeding to scheduling validation..." << std::endl;
#endif

        // const auto& dfg_vertices = this->dfg->get_vertices();
        // const auto& node_to_scheduled_time_slice = this->get_node_to_scheduled_time_slice_();
        const auto& dfg_node_to_in_vertices = this->get_dfg_node_to_in_vertices_();
        const auto& node_to_pe = this->get_node_to_pe_();
        const auto& root_nodes = this->get_root_nodes();
        const auto& dfg_node_to_out_vertices = this->get_dfg_node_to_out_vertices_();

  
#ifdef DEBUG
        std::cout << "[DEBUG] Scheduling not found, performing scheduling..." << std::endl;
#endif
        if(this->use_rt_scheduler){
            this->rt_scheduler.update_schedule_time_with_min_time();
            this->node_to_scheduled_time_slice = this->rt_scheduler.get_node_to_scheduled_time_slice_();
            mapping_is_valid = !this->contains_invalid_scheduling;
        } 
        // else if (!this->node_to_scheduled_time_slice.empty()){
        //     mapping_is_valid =  Scheduler::scheduling_is_valid(this->dfg->get_vertices(),
        //     this->node_to_scheduled_time_slice, this->dfg->get_node_to_in_vertices(),
        //     this->node_to_pe,this->PEs_to_routing);
        // } 
        else {
        
            std::tie(node_to_scheduled_time_slice, mapping_is_valid) =std::move(Scheduler::schedule(root_nodes, node_to_pe, 
                                                                                            dfg_node_to_in_vertices, 
                                                                                            dfg_node_to_out_vertices, 
                                                                                            this->PEs_to_routing));
            this->set_node_to_scheduled_time_slice(node_to_scheduled_time_slice);
        }


        if (!mapping_is_valid) {
#ifdef DEBUG
            std::cout << "[DEBUG] Invalid scheduling detected." << std::endl;
#endif
            reason = get_invalid_mapping_reason(EnumInvalidMappingReason::INVALID_SCHEDULING);
        } else {
#ifdef DEBUG
            std::cout << "[DEBUG] Mapping is valid!" << std::endl;
#endif
            reason = get_invalid_mapping_reason(EnumInvalidMappingReason::NONE);
        }

        this->mapping_is_valid = mapping_is_valid;
    }
    this->invalid_mapping_reason = reason;

#ifdef DEBUG
    std::cout << "[DEBUG] analyze_mapping finished. Result: " << (mapping_is_valid ? "Valid" : "Invalid") << " mapping." << std::endl;
#endif
}

const std::pair<int, int> &BaseState::get_cgra_dimensions_() const
{
    return this->cgra->get_cgra_dimensions_();
}
const std::vector<int>& BaseState::get_vertices() const{
    return this->dfg->get_vertices();
}
const std::unordered_map<std::pair<int, int>, std::vector<int>> &BaseState::get_PEs_to_routing_() const
{
    return this->PEs_to_routing;
}

const std::unordered_map<int,std::tuple<int, int, int>>& BaseState::get_PE_to_coord_() const {   
    return this->cgra->get_PE_to_coord_();
}

const std::unordered_map<std::tuple<int, int, int>, int>& BaseState::get_coord_to_PE_() const {   
    return this->cgra->get_coord_to_PE_();
}

const std::unordered_map<int,std::unordered_set<int>>& BaseState::get_free_connections_() const{
    return this->free_connections;
}



const std::unordered_map<int, std::unordered_set<int>> &BaseState::get_out_vertices_cgra_() const
{
    return this->cgra->get_out_vertices_();
}


const std::unordered_set<int>& BaseState::get_in_vertices_dfg_by_node_id(const int &node) const
{
    return this->dfg->get_in_vertices_by_node_id(node);
}

const std::unordered_set<int>& BaseState::get_out_vertices_dfg_by_node_id(const int &node) const
{
    return this->dfg->get_out_vertices_by_node_id(node);
}

bool BaseState::all_nodes_were_mapped() const
{
    return static_cast<int>(this->node_to_pe.size()) == this->dfg->get_num_nodes();
}

const std::unordered_map<int, int> &BaseState::get_PE_to_node() const
{
    return this->pe_to_node;
}

bool BaseState::node_was_mapped(const int &node) const
{
    return this->node_to_pe.find(node) != this->node_to_pe.end();
}

const std::string BaseState::get_real_node(const int& node_id) const {
    return this->dfg->get_name_by_node(node_id);
}

const std::unordered_map<int, int>& BaseState::get_node_to_scheduled_time_slice_() const
{   if(this->use_rt_scheduler){
        return this->rt_scheduler.get_node_to_scheduled_time_slice_();
    }
    return this->node_to_scheduled_time_slice;
}

const std::unordered_map<int, std::unordered_set<int>>& BaseState::get_dfg_node_to_in_vertices_() const
{
    return this->dfg->get_node_to_in_vertices();
}

const std::unordered_map<int, std::unordered_set<int>>& BaseState::get_dfg_node_to_out_vertices_() const
{
    return this->dfg->get_node_to_out_vertices();
}
const std::unordered_map<int, int>& BaseState::get_node_to_pe_() const
{
    return this->node_to_pe;
}
void BaseState::set_routing_is_invalid(){
    if(!this->contains_invalid_routing)
    this->contains_invalid_routing = true;
}

bool BaseState::get_routing_is_invalid() const {
    return this->contains_invalid_routing;
}

const std::vector<int>& BaseState::get_root_nodes() const {
    return this->dfg->get_root_nodes();
}

void BaseState::set_node_to_scheduled_time_slice(const std::unordered_map<int,int>& new_node_to_scheduled_time_slice)
{
    this->node_to_scheduled_time_slice = std::move(new_node_to_scheduled_time_slice);
}

const std::vector<std::pair<int,int>>& BaseState::get_dfg_edges() const {
    return this->dfg->get_edges();
}


void BaseState::route_by_routing_path(const std::pair<int, int>& tgt_routing_PEs, const std::vector<int>& routing_path)
{
    this->PEs_to_routing[tgt_routing_PEs] = routing_path;

    for (int i = 0; i < static_cast<int>(routing_path.size()) - 1; ++i) {
        this->free_connections[routing_path[i]].erase(routing_path[i+1]);
    }
}




std::vector<int> BaseState::get_legal_actions_with_traversal(const int& idx_id_node_to_be_mapped){
    std::vector<int> cur_legal_actions = {};
auto node_to_be_mapped = this->dfg->get_node_in_placement_order_by_idx_(idx_id_node_to_be_mapped);

#ifdef DEBUG
    std::cout << "Node to be mapped: " << node_to_be_mapped << std::endl;
#endif

if (node_to_be_mapped != -1) {
    const auto& traversal_edge = this->dfg->get_traversal_edge_by_node(node_to_be_mapped);
    auto prev_mapped_node = std::get<0>(traversal_edge);
    auto dir = std::get<2>(traversal_edge);
    auto mapped_pe = this->node_to_pe.at(prev_mapped_node);
    
#ifdef DEBUG
    std::cout << "Previous mapped node: " << prev_mapped_node << std::endl;
    std::cout << "Direction: " << dir << std::endl;
    std::cout << "Mapped PE: " << mapped_pe << std::endl;
#endif

    auto neighs = dir ? this->cgra->get_in_vertices_().at(mapped_pe) 
                      : this->cgra->get_out_vertices_().at(mapped_pe);

    auto num_in_neigh = this->get_in_vertices_dfg_by_node_id(node_to_be_mapped).size();
    auto num_out_neigh = this->get_out_vertices_dfg_by_node_id(node_to_be_mapped).size();

#ifdef DEBUG
    std::cout << "Number of input neighbors: " << num_in_neigh << std::endl;
    std::cout << "Number of output neighbors: " << num_out_neigh << std::endl;
#endif

    for (auto& neigh_node : neighs) {
#ifdef DEBUG
        std::cout << "Checking neighbor node: " << neigh_node << std::endl;
#endif
        if (this->satisfies_mask_condition(neigh_node, num_in_neigh, num_out_neigh)) {
            cur_legal_actions.push_back(neigh_node);

#ifdef DEBUG
            std::cout << "Neighbor node " << neigh_node << " added to legal actions." << std::endl;
#endif
        }
    }
}

#ifdef DEBUG
    std::cout << "Total legal actions found: " << cur_legal_actions.size() << std::endl;
#endif

return cur_legal_actions;

}

int BaseState::get_action_idx_by_action(const int& action) const{
        return this->action_to_idx.at(action);
}

int BaseState::get_action_by_action_idx(const int& idx){
    return this->idx_to_action.at(idx);
}


bool BaseState::satisfies_mask_condition(const int& PE ,const int& num_in_neigh, const int& num_out_neigh) const {
    
    bool aux_condition;
    if(this->use_routing_mask){
        if(this->II == 1) {
            aux_condition =  (this->remain_in_edges[PE].item<int>() >= num_in_neigh) && (this->remain_out_edges[PE].item<int>() - num_in_neigh >= num_out_neigh);
        } else{ 
            aux_condition =  (this->remain_in_edges[PE].item<int>() >= num_in_neigh) && (this->remain_out_edges[PE].item<int>() >= num_out_neigh); 
    }
    }else{
        aux_condition = false;
    }
    bool condition1 = (this->use_used_pe_mask && (this->node_was_placed_tensor[PE].item<int>() == 0)) || !this->use_used_pe_mask;
    bool condition2 = !this->use_routing_mask || (this->use_routing_mask && aux_condition);

    return condition1 && condition2;
}

std::vector<int> BaseState::get_legal_actions_with_annotations(const int idx_id_node_to_be_mapped) {
    std::vector<int> cur_legal_actions;
    auto node_to_be_mapped = this->dfg->get_node_in_placement_order_by_idx_(idx_id_node_to_be_mapped);
    std::unordered_map<int, int> node_to_count;

    if (node_to_be_mapped != -1) {
        const auto& traversal_edge = this->dfg->get_traversal_edge_by_node(node_to_be_mapped); 
        auto prev_mapped_node = std::get<0>(traversal_edge);
        auto dir = std::get<2>(traversal_edge);
        auto mapped_pe = this->node_to_pe.at(prev_mapped_node);
        const auto& neighs = dir ? this->cgra->get_in_vertices_().at(mapped_pe) : this->cgra->get_out_vertices_().at(mapped_pe);
        int num_in_neigh = this->get_in_vertices_dfg_by_node_id(node_to_be_mapped).size();
        int num_out_neigh = this->get_out_vertices_dfg_by_node_id(node_to_be_mapped).size();

        #ifdef DEBUG
        std::cout << "[DEBUG] Node to be mapped: " << node_to_be_mapped << std::endl;
        std::cout << "[DEBUG] Previous mapped node: " << prev_mapped_node << std::endl;
        std::cout << "[DEBUG] Direction: " << (dir ? "IN" : "OUT") << std::endl;
        std::cout << "[DEBUG] Neighbors: ";
        for (auto n : neighs) std::cout << n << " ";
        std::cout << std::endl;
        #endif

        for (auto& neigh_node : neighs) {
            if (this->satisfies_mask_condition(neigh_node, num_in_neigh, num_out_neigh)) {
                node_to_count[neigh_node] = 0;
            }
        }

        auto edge = std::make_pair(std::get<0>(traversal_edge), std::get<1>(traversal_edge));
        const auto& annotation_vector = this->dfg->get_annotation_by_edge(edge);
        const auto& node_to_in_dist_to_nodes = this->cgra->get_node_to_in_dist_to_nodes();
        const auto& node_to_out_dist_to_nodes = this->cgra->get_node_to_out_dist_to_nodes();
        int annot_to_ignore = 0;

        for(auto& [conv_node, annotation_list]: annotation_vector){
            auto conv_pe = this->node_to_pe.at(conv_node);
            int max_dist_conv_pe = 0;
            
            for(const auto& [dist,_]: node_to_in_dist_to_nodes.at(conv_pe)){
                if(dist > max_dist_conv_pe) max_dist_conv_pe = dist;
            }
            bool skip = false;
            int temp_dist = 0;
            for(auto& [direction, dist]: annotation_list){
                temp_dist += dist;
                if(temp_dist > max_dist_conv_pe){
                    skip = true;
                    break;
                }
            }
            if(skip){
                annot_to_ignore += 1;
                continue;
            }
            std::unordered_set<int> pes_to_analyze = {conv_pe};
            for(auto& [direction, dist]: annotation_list){
                // 1 == IN; 0 == OUT
                std::unordered_set<int> pes_in_dist;
                for(auto& pe: pes_to_analyze){
                    if(direction == 1){
                        pes_in_dist.insert(node_to_in_dist_to_nodes.at(pe).at(dist).begin(), 
                        node_to_in_dist_to_nodes.at(pe).at(dist).end());
                        #ifdef DEBUG
                        std::cout << "[DEBUG] Analyzing IN direction for PE: " << pe << " at distance: " << dist << std::endl;
                        std::cout << "[DEBUG] Found PEs: ";
                        for (auto p : pes_in_dist) std::cout << p << " ";
                        std::cout << std::endl;
                        #endif
                    }else{
                        #ifdef DEBUG
                        std::cout << "[DEBUG] Analyzing OUT direction for PE: " << pe << " at distance: " << dist << std::endl;
                        std::cout << "[DEBUG] node_to_out_dist.at(pe): \n";
                        for (const auto& [d, nodes] : node_to_out_dist_to_nodes.at(pe)) {
                            std::cout << "  Distance " << d << ": ";
                            for (const auto& n : nodes) {
                                std::cout << n << " ";
                            }
                            std::cout << std::endl;
                        }
                        #endif

                        pes_in_dist.insert(node_to_out_dist_to_nodes.at(pe).at(dist).begin(), 
                                                node_to_out_dist_to_nodes.at(pe).at(dist).end());
                        
                        #ifdef DEBUG
                        std::cout << "[DEBUG] Analyzing OUT direction for PE: " << pe << " at distance: " << dist << std::endl;
                        std::cout << "[DEBUG] Found PEs: ";
                        for (auto p : pes_in_dist) std::cout << p << " ";
                        std::cout << std::endl;
                        #endif
                                            }
                }
                pes_to_analyze = std::move(pes_in_dist);
            }
            for(auto& pe: pes_to_analyze) {
                #ifdef DEBUG
                    std::cout << "[DEBUG] Analyzing PE: " << pe << " for node: " << conv_node << std::endl;
                #endif
                if (node_to_count.find(pe) != node_to_count.end()) {
                    node_to_count[pe] += 1;

                    #ifdef DEBUG
                    std::cout << "[DEBUG] PE: " << pe << ", Count: " << node_to_count[pe] << std::endl;
                    #endif
                }
            }

        }
 


        for(auto& pair:node_to_count){
            if (pair.second == static_cast<int>(annotation_vector.size()) - annot_to_ignore) {
                cur_legal_actions.emplace_back(pair.first);

                #ifdef DEBUG
                std::cout << "[DEBUG] Added PE: " << pair.first << " to legal actions." << std::endl;
                #endif
            }
        
    }

    }

    #ifdef DEBUG
    std::cout << "[DEBUG] Final Legal Actions: ";
    for (auto action : cur_legal_actions) std::cout << action << " ";
    std::cout << std::endl;
    #endif

    return cur_legal_actions;
}


std::vector<int> BaseState::get_greedy_legal_actions(const int idx_id_node_to_be_mapped) {
    std::vector<int> cur_legal_actions = {};
    auto node_to_be_mapped = this->dfg->get_node_in_placement_order_by_idx_(idx_id_node_to_be_mapped);
    std::unordered_map<int, int> node_to_count;

#ifdef DEBUG
    std::cout << "[DEBUG] Node to be mapped: " << node_to_be_mapped << std::endl;
#endif

    if (node_to_be_mapped != -1) {
        int edges_to_map = 0;

        for (const auto& father : this->get_in_vertices_dfg_by_node_id(node_to_be_mapped)) {
            if (this->node_was_mapped(father)) {
                auto mapped_pe = this->node_to_pe.at(father);
                edges_to_map += 1;

#ifdef DEBUG
                std::cout << "[DEBUG] Father node mapped to PE: " << mapped_pe << std::endl;
#endif

                for (const auto& pe : this->cgra->get_out_vertices_by_node_id(mapped_pe)) {
                    node_to_count[pe] += 1;
                }
            }
        }

        if (!this->dfg->is_order_topological()) {
            for (const auto& child : this->get_out_vertices_dfg_by_node_id(node_to_be_mapped)) {
                if (this->node_was_mapped(child)) {
                    auto mapped_pe = this->node_to_pe.at(child);
                    edges_to_map += 1;

#ifdef DEBUG
                    std::cout << "[DEBUG] Child node mapped to PE: " << mapped_pe << std::endl;
#endif

                    for (const auto& pe : this->cgra->get_in_vertices_by_node_id(mapped_pe)) {
                        node_to_count[pe] += 1;
                    }
                }
            }
        }

        int num_in_neigh = this->get_in_vertices_dfg_by_node_id(node_to_be_mapped).size();
        int num_out_neigh = this->get_out_vertices_dfg_by_node_id(node_to_be_mapped).size();

#ifdef DEBUG
        std::cout << "[DEBUG] Num In Neigh: " << num_in_neigh << ", Num Out Neigh: " << num_out_neigh << std::endl;
        std::cout << "[DEBUG] Edges to map: " << edges_to_map << std::endl;
#endif

        for (auto& pair : node_to_count) {
#ifdef DEBUG
            std::cout << "[DEBUG] Checking PE: " << pair.first << " with edge count: " << pair.second << std::endl;
#endif
            if (pair.second == edges_to_map) {
#ifdef DEBUG
                std::cout << "[DEBUG] PE " << pair.first << " passed edge count check" << std::endl;
#endif

                if (this->satisfies_mask_condition(pair.first,num_in_neigh,num_out_neigh)) {
                    cur_legal_actions.push_back(pair.first);

#ifdef DEBUG
                    std::cout << "[DEBUG] PE " << pair.first << " added to legal actions." << std::endl;
#endif
                }
            }
        }

#ifdef DEBUG
        std::cout << "[DEBUG] Final legal actions: ";
        for (const auto& action : cur_legal_actions) {
            std::cout << action << " ";
        }
        std::cout << std::endl;
#endif
    }

    return cur_legal_actions;
}



void BaseState::update_idx_id_node_to_be_mapped()
{
    this->idx_id_node_to_be_mapped += 1;
}

const bool &BaseState::get_mapping_is_valid_() const
{
    return this->mapping_is_valid;
}

void BaseState::update_state(){
    #ifdef DEBUG
    std::cout << "[DEBUG] Updating state..." << std::endl;
    if  (this->idx_id_node_to_be_mapped > 0)
    std::cout << "[DEBUG] Previous mapped node: " << this->get_prev_mapped_node() << std::endl;
    #endif

    this->update_idx_id_node_to_be_mapped();

    #ifdef DEBUG
    std::cout << "[DEBUG] Current node to be mapped index: " << this->idx_id_node_to_be_mapped << std::endl;
    #endif

    bool last_node_was_placed = this->node_to_pe.find(this->get_prev_mapped_node()) != this->node_to_pe.end();

    #ifdef DEBUG
    std::cout << "[DEBUG] Last node was placed: " << std::boolalpha << last_node_was_placed << std::endl;
    std::cout << "[DEBUG] Total DFG vertices: " << this->dfg->num_vertices() << std::endl;
    #endif

    if (last_node_was_placed && (static_cast<int>(this->idx_id_node_to_be_mapped) != this->dfg->num_vertices())) {
        #ifdef DEBUG
        std::cout << "[DEBUG] Updating legal actions and action-to-index mapping." << std::endl;
        #endif

        this->update_legal_actions();
        this->update_action_to_idx();
       
    } else {
        #ifdef DEBUG
        std::cout << "[DEBUG] Clearing legal actions and action-to-index mapping." << std::endl;
        #endif

        this->legal_actions.clear();
        this->action_to_idx.clear();
        this->idx_to_action.clear();
    }

    this->update_is_end_state();

    #ifdef DEBUG
    std::cout << "[DEBUG] Is end state: " << std::boolalpha << this->is_end_state << std::endl;
    #endif

    if (this->is_end_state) {
        #ifdef DEBUG
        std::cout << "[DEBUG] Analyzing final mapping..." << std::endl;
        #endif
        this->analyze_mapping();
    }
}


std::vector<int> BaseState::get_actions_with_sched(int node_to_be_mapped) {
    auto num_spatial_pes = this->cgra->number_of_spatial_PEs_();
    std::vector<int> cur_legal_actions;
    auto modulos = this->node_to_sched_mods.at(node_to_be_mapped);
    int initial_PE = num_spatial_pes * modulos.first;
    int final_PE = num_spatial_pes * (modulos.second + 1);
    auto total_pes = this->cgra->total_PEs();
    #ifdef DEBUG
    std::cout << "[DEBUG] Node to be mapped: " << node_to_be_mapped << std::endl;
    std::cout << "[DEBUG] Initial PE: " << initial_PE << ", Final PE: " << final_PE << std::endl;
    std::cout << "[DEBUG] Num Spatial PEs: " << num_spatial_pes << std::endl;
    #endif

    if (final_PE > initial_PE) {
        cur_legal_actions.resize(final_PE - initial_PE);
        std::iota(cur_legal_actions.begin(), cur_legal_actions.end(), initial_PE);
    } 
    else if (final_PE < initial_PE) {
        cur_legal_actions.resize(total_pes - initial_PE + final_PE);
        std::iota(cur_legal_actions.begin(), cur_legal_actions.begin() + (total_pes - initial_PE), initial_PE);
        std::iota(cur_legal_actions.begin() + (total_pes - initial_PE), cur_legal_actions.end(), 0);
    } 
    else { 
        cur_legal_actions.resize(total_pes);
        std::iota(cur_legal_actions.begin(), cur_legal_actions.end(), 0);
    }

    #ifdef DEBUG
    std::cout << "[DEBUG] Legal Actions: ";
    for (const auto& action : cur_legal_actions) {
        std::cout << action << " ";
    }
    std::cout << std::endl;
    #endif

    return cur_legal_actions;
}
torch::Tensor BaseState::mask_congested_edges(torch::Tensor mask, int node_to_be_mapped) {
    int num_in_neigh = this->get_in_vertices_dfg_by_node_id(node_to_be_mapped).size();
    int num_out_neigh = this->get_out_vertices_dfg_by_node_id(node_to_be_mapped).size();

#ifdef DEBUG
    std::cout << "[DEBUG] Masking congested edges for node: " << node_to_be_mapped << std::endl;
    std::cout << "[DEBUG] Num in-neighbors: " << num_in_neigh << ", Num out-neighbors: " << num_out_neigh << std::endl;
    std::cout << "[DEBUG] Mask " << mask << std::endl;

#endif
    torch::Tensor mask1,mask2;
    if(this->II == 1) {
        mask1 = this->remain_in_edges >= num_in_neigh;
        mask2 = this->remain_out_edges - num_in_neigh >= num_out_neigh;
    }else{ 
        mask1 = this->remain_in_edges >= num_in_neigh;
        mask2 = this->remain_out_edges >= num_out_neigh;
    }

#ifdef DEBUG
    std::cout << "[DEBUG] Mask1 (remain_in_edges >= num_in_neigh): " << mask1 << std::endl;
    std::cout << "[DEBUG] Mask2 (remain_out_edges - num_in_neigh >= num_out_neigh): " << mask2 << std::endl;
#endif

    auto final_mask = torch::logical_and(torch::logical_and(mask1, mask2), mask);

#ifdef DEBUG
    std::cout << "[DEBUG] Final mask after applying congestion constraints: " << final_mask << std::endl;
#endif

    return final_mask;
}



void BaseState::calc_initial_legal_actions() {
    auto node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);

#ifdef DEBUG
    std::cout << "[DEBUG] Calculating initial legal actions for node: " << node_to_be_mapped << std::endl;
#endif

    this->legal_actions.clear();
    int initial_PE, final_PE;

    if (this->dfg->has_scheduling()) {
        auto num_spatial_pes = this->cgra->number_of_spatial_PEs_();
        auto modulos = this->node_to_sched_mods.at(node_to_be_mapped);
        initial_PE = num_spatial_pes * modulos.first;
        final_PE = num_spatial_pes * (modulos.second + 1);

#ifdef DEBUG
        std::cout << "[DEBUG] DFG has scheduling enabled." << std::endl;
        std::cout << "[DEBUG] num_spatial_pes: " << num_spatial_pes << std::endl;
        std::cout << "[DEBUG] Scheduling modulos: [" << modulos.first << ", " << modulos.second << "]" << std::endl;
        std::cout << "[DEBUG] initial_PE: " << initial_PE << ", final_PE: " << final_PE << std::endl;
#endif

    } else {
        initial_PE = 0;
        final_PE = this->cgra->total_PEs();

#ifdef DEBUG
        std::cout << "[DEBUG] DFG has no scheduling. Using full range of PEs." << std::endl;
        std::cout << "[DEBUG] initial_PE: " << initial_PE << ", final_PE: " << final_PE << std::endl;
#endif
    }

    this->legal_actions = get_masked_actions(initial_PE, final_PE, this->use_symmetry_mask);

#ifdef DEBUG
    std::cout << "[DEBUG] Calculated " << this->legal_actions.size() << " legal actions." << std::endl;
#endif
}

std::vector<int> BaseState::get_masked_actions(int initial_PE, int final_PE, bool use_sym_mask) {
    auto num_total_PEs = this->cgra->total_PEs();

    #ifdef DEBUG
    std::cout << "[DEBUG] Initial PE: " << initial_PE 
              << ", Final PE: " << final_PE 
              << ", Num total PEs: " << num_total_PEs << std::endl;
    #endif

    torch::Tensor indices;
    auto node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);

    torch::Tensor mask = torch::ones(this->cgra->total_PEs());

    if(this->use_used_pe_mask){
        auto pe_mask = this->node_was_placed_tensor == 0;
        mask = torch::logical_and(mask,pe_mask);
    }

    if(use_sym_mask){
        auto cgra_dims = this->cgra->get_cgra_dimensions_();
        float n = cgra_dims.first;
        float m = cgra_dims.second;

        // torch::Tensor indices = torch::arange(n * m, torch::kInt32).view({n, m});
        auto sym_mask = torch::zeros({static_cast<int>(n), static_cast<int>(m)}, torch::kBool).reshape({static_cast<int>(n * m)});
        for (int i = 0; i < n / 2.; ++i) {
            for (int j = 0; j < m / 2.; ++j) {
                if (j >= i) {
                    int pe_id = i * m + j;
                    sym_mask[pe_id] = true;

        #ifdef DEBUG
                    std::cout << "[DEBUG] Activating PE (" << i << ", " << j << ") with pe_id = " << pe_id << std::endl;
        #endif
                }
            }
        }

        sym_mask = sym_mask.reshape(-1).repeat({this->II});

        mask = torch::logical_and(mask, sym_mask);
    }

    if (this->use_routing_mask) {
        mask = this->mask_congested_edges(mask,node_to_be_mapped);

        #ifdef DEBUG
        std::cout << "[DEBUG] Applied congestion mask: " << mask << std::endl;
        #endif
    }


    if (final_PE > initial_PE) {
        auto sliced_pes = mask.slice(0, initial_PE, final_PE);
        indices = torch::nonzero(sliced_pes).to(torch::kInt32).squeeze(-1) + initial_PE;

        #ifdef DEBUG
        std::cout << "[DEBUG] Sliced mask (Initial PE to Final PE): " << sliced_pes << std::endl;
        #endif
    } else if (final_PE < initial_PE) { 
        auto sliced_part1 = mask.slice(0, initial_PE, num_total_PEs);
        auto sliced_part2 = mask.slice(0, 0, final_PE);

        auto indices_part1 = torch::nonzero(sliced_part1).to(torch::kInt32).squeeze(-1) + initial_PE;
        auto indices_part2 = torch::nonzero(sliced_part2).to(torch::kInt32).squeeze(-1);

        indices = torch::cat({indices_part2, indices_part1}, 0);

        #ifdef DEBUG
        std::cout << "[DEBUG] Wrapped slicing: Part1 (" << initial_PE << " to " << num_total_PEs << ") -> " 
                  << sliced_part1 << ", Part2 (0 to " << final_PE << ") -> " << sliced_part2 << std::endl;
        #endif
    } else {
        indices = torch::nonzero(mask).to(torch::kInt32).squeeze(-1);
    }

    #ifdef DEBUG
    std::cout << "[DEBUG] Nonzero indices (available PEs): " << indices << std::endl;
    #endif

    std::vector<int> real_actions(indices.data_ptr<int>(), indices.data_ptr<int>() + indices.numel());

    #ifdef DEBUG
    std::cout << "[DEBUG] Final Actions: ";
    for (const auto &action : real_actions) {
        std::cout << action << " ";
    }
    std::cout << std::endl;
    #endif

    return real_actions;
}



std::vector<int> BaseState::get_non_greedy_legal_actions(int node_to_be_mapped){
    std::vector<int> cur_legal_actions;
    if(this->dfg->has_scheduling()){
        auto num_spatial_pes = this->cgra->number_of_spatial_PEs_();
        auto modulos = this->node_to_sched_mods.at(node_to_be_mapped);
        int initial_PE = num_spatial_pes * modulos.first;
        int final_PE = num_spatial_pes * (modulos.second + 1);
        
        cur_legal_actions = this->get_masked_actions(initial_PE,final_PE, false);
        // if (this->use_used_pe_mask || this->use_routing_mask){
        //     cur_legal_actions =  this->get_masked_actions(initial_PE, final_PE);
        // }else{
        //     cur_legal_actions = this->get_actions_with_sched(node_to_be_mapped);
        // }

    }else{
        if(this->use_used_pe_mask || this->use_routing_mask){
            cur_legal_actions =  this->get_masked_actions(0, this->cgra->total_PEs(),false);
        } else{
            return this->legal_actions;
        }
    }

    #ifdef DEBUG
    std::cout << "[DEBUG] Using non-greedy mode. Legal actions for node " << node_to_be_mapped << ": ";
    for (const auto& action : cur_legal_actions) {
        std::cout << action << " ";
    }
    std::cout << std::endl;
    #endif

    return cur_legal_actions;
}


void BaseState::update_legal_actions() {
    #ifdef DEBUG
    std::cout << "[DEBUG] Updating legal actions..." << std::endl;
    std::cout << "Greedy mode: " << (this->greedy ? "ON" : "off") << std::endl;
    #endif
 
    auto node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);

    if (!this->greedy) {
        this->legal_actions = this->get_non_greedy_legal_actions(node_to_be_mapped);
    } else {
        if(!this->some_neigh_was_mapped(node_to_be_mapped)){
            this->legal_actions = this->get_non_greedy_legal_actions(node_to_be_mapped);
        }else{
            if (this->dfg->is_greedy_type()) {
                this->legal_actions = this->get_greedy_legal_actions(this->idx_id_node_to_be_mapped);
                #ifdef DEBUG
                std::cout << "[DEBUG] Using greedy legal actions. Legal actions: ";
                #endif
            }else{
                if (this->dfg->use_annotations()) {
                    this->legal_actions = this->get_legal_actions_with_annotations(this->idx_id_node_to_be_mapped);
                    
                    #ifdef DEBUG
                    std::cout << "[DEBUG] Using traversal with annotations. Legal actions: ";
                    #endif
                } else {
                    this->legal_actions = this->get_legal_actions_with_traversal(this->idx_id_node_to_be_mapped);
                    
                    #ifdef DEBUG
                    std::cout << "[DEBUG] Using traversal without annotations. Legal actions: ";
                    #endif  
                }
            }
            
            #ifdef DEBUG
            for (const auto& action : this->legal_actions) {
                std::cout << action << " ";
            }
            std::cout << std::endl;
            #endif
        }
    }

    #ifdef DEBUG
    std::cout << "[DEBUG] Legal actions updated successfully." << std::endl;
    #endif
}

bool BaseState::some_neigh_was_mapped(int id) {
    #ifdef DEBUG
        std::cout << "[DEBUG] Checking if any neighbor of node " << id << " was mapped." << std::endl;
    #endif
    
        for (const auto& neigh : this->dfg->get_in_vertices_by_node_id(id)) {
    #ifdef DEBUG
            std::cout << "[DEBUG] Checking incoming neighbor: " << neigh << std::endl;
    #endif
            if (this->node_to_pe.find(neigh) != this->node_to_pe.end()) {
    #ifdef DEBUG
                std::cout << "[DEBUG] Incoming neighbor " << neigh << " is mapped!" << std::endl;
    #endif
                return true;
            }
        }
    
        for (const auto& neigh : this->dfg->get_out_vertices_by_node_id(id)) {
    #ifdef DEBUG
            std::cout << "[DEBUG] Checking outgoing neighbor: " << neigh << std::endl;
    #endif
            if (this->node_to_pe.find(neigh) != this->node_to_pe.end()) {
    #ifdef DEBUG
                std::cout << "[DEBUG] Outgoing neighbor " << neigh << " is mapped!" << std::endl;
    #endif
                return true;
            }
        }
    
    #ifdef DEBUG
        std::cout << "[DEBUG] No neighbors of node " << id << " were mapped." << std::endl;
    #endif
        return false;
    }
    

void BaseState::update_is_end_state() {
    bool all_nodes_were_mapped = this->all_nodes_were_mapped();
    bool last_node_was_placed = this->node_to_pe.find(this->get_prev_mapped_node()) != this->node_to_pe.end();
    bool all_PEs_used = static_cast<int>(this->pe_to_node.size()) == this->cgra->total_PEs();
    
    #ifdef DEBUG
        std::cout << "[DEBUG] Checking end state..." << std::endl;
        std::cout << "[DEBUG] All nodes mapped: " << all_nodes_were_mapped << std::endl;
        std::cout << "[DEBUG] Last node was placed: " << last_node_was_placed << std::endl;
        std::cout << "[DEBUG] Legal actions empty: " << this->legal_actions.empty() << std::endl;
        std::cout << "[DEBUG] Contains invalid routing: " << this->contains_invalid_routing << std::endl;
        std::cout << "[DEBUG] Contains invalid scheduling: " << this->contains_invalid_routing << std::endl;

    #endif

    bool contains_invalid_rout, contains_inv_sched;

    if(this->enum_end_state == EnumEndState::MAPZERO){
        contains_invalid_rout = false;
        contains_inv_sched = false;
    }else if(this->enum_end_state == EnumEndState::SMARTMAP){
        contains_inv_sched = false;
        contains_invalid_rout = this->contains_invalid_routing;
    }else{
        contains_inv_sched =  this->contains_invalid_scheduling;
        contains_invalid_rout = this->contains_invalid_routing;
    }


    if (!last_node_was_placed || this->legal_actions.empty() || all_PEs_used || all_nodes_were_mapped || contains_invalid_rout  ||
    contains_inv_sched
        ) {
        this->is_end_state = true;

        #ifdef DEBUG
            std::cout << "[DEBUG] Marking as end state!" << std::endl;
        #endif
    }
}


int BaseState::get_assigned_PE_by_node_id(const int& node_id) const {
    return this->node_to_pe.at(node_id);
}


const std::unordered_set<int> BaseState::get_occupied_PEs_() const {
    torch::Tensor indices = torch::nonzero(this->node_was_placed_tensor == 1).to(torch::kInt32).squeeze(-1);
    return  std::unordered_set<int>(indices.data_ptr<int>(), indices.data_ptr<int>() + indices.numel());
}

const std::unordered_set<int> BaseState::get_not_occupied_PEs_() const {
    torch::Tensor indices = torch::nonzero(this->node_was_placed_tensor == 0).to(torch::kInt32).squeeze(-1);
    return  std::unordered_set<int>(indices.data_ptr<int>(), indices.data_ptr<int>() + indices.numel());
}
const std::vector<int> BaseState::get_not_occupied_PEs_vector() const {
    torch::Tensor indices = torch::nonzero(this->node_was_placed_tensor == 0).to(torch::kInt32).squeeze(-1);
    return  std::vector<int>(indices.data_ptr<int>(), indices.data_ptr<int>() + indices.numel());
}

bool BaseState::pe_is_occupied(const int& PE) const {
        return this->node_was_placed_tensor[PE].item<int>() == 1;
}

const int& BaseState::get_prev_mapped_node() const {

    return this->dfg->get_node_in_placement_order_by_idx_(this->idx_id_node_to_be_mapped - 1);
}


std::vector<c10::IValue> BaseState::get_model_inputs(c10::DeviceType device) const
{
     if (device == torch::kCUDA){
        std::vector<c10::IValue> inputs_on_device;
        for (auto& input : this->model_inputs) {
            if (input.isTensor()) {
                inputs_on_device.push_back(input.toTensor().to( torch::kCUDA));
            } else {
                inputs_on_device.push_back(input);
            }
        }
        return inputs_on_device;
        }
    return this->model_inputs;   
};



void BaseState::step(const int& action){
    if(this->pe_to_node.find(action) == this->pe_to_node.end()){
        #ifdef DEBUG
        std::cout << "\n[DEBUG] Step called with action: " << action << std::endl;
        #endif
        
        #ifdef DEBUG
        std::cout << "[DEBUG] Calling place_and_route with action: " << action << std::endl;
        #endif
        this->place_and_route(action);
        if(this->use_rt_scheduler){
        //     int node,
        // std::unordered_map<int, std::unordered_set<int>>& node_to_in,
        // std::unordered_map<int, std::unordered_set<int>>& node_to_out,
        // std::unordered_map<std::pair<int, int>, std::vector<int>>& PEs_to_routing,
        // std::unordered_map<int, int>& node_to_pe
        this->contains_invalid_scheduling = !this->rt_scheduler.schedule(this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped),
        this->dfg->get_node_to_in_vertices(),
        this->dfg->get_node_to_out_vertices(),
        this->PEs_to_routing,
        this->node_to_pe);
        
        if(this->use_sync_routing){
            auto node_to_be_mapped = this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped);
            if(this->contains_invalid_scheduling && this->node_was_mapped(node_to_be_mapped)){
        
                std::vector<int> conv_nodes;
                const auto& in_vertices = this->dfg->get_node_to_in_vertices();
                if(static_cast<int>(in_vertices.at(node_to_be_mapped).size()) > 1){
                    conv_nodes =  {node_to_be_mapped};
                }
        
                for(auto& node: in_vertices.at(node_to_be_mapped)){
                    if(static_cast<int>(in_vertices.at(node).size()) > 1){
                        conv_nodes.push_back(node);
                    }
                }
                
                const auto& out_vertices = this->dfg->get_node_to_out_vertices();
                for(auto& node: out_vertices.at(node_to_be_mapped)){
                    if(static_cast<int>(out_vertices.at(node).size()) > 1){
                        conv_nodes.push_back(node);
                    }
                }
        
                const auto& topo_order = this->dfg->get_topological_sort();
        
        #ifdef DEBUG
                std::cout << "[DEBUG] Checking sync routing for node " << node_to_be_mapped 
                          << " | conv_nodes = { ";
                for(auto n : conv_nodes) std::cout << n << " ";
                std::cout << "}" << std::endl;
        #endif
        
                this->rt_scheduler.update_schedule_time_with_max_delay(topo_order, in_vertices, this->PEs_to_routing, this->node_to_pe);
                auto& cur_node_to_sched_mod = this->rt_scheduler.get_node_to_scheduled_time_slice_();
                bool all_conv_satisfied = true;
        
                for(auto node: conv_nodes){
                    for(auto in_node: in_vertices.at(node)){
                        if(!this->node_was_mapped(in_node)){
                            continue;
                        }
                        auto in_pe = this->node_to_pe.at(in_node);
                        auto out_pe = this->node_to_pe.at(node);
                        auto delay = static_cast<int>(this->PEs_to_routing.at({in_pe, out_pe}).size()) - 1;
                        auto time_diff = cur_node_to_sched_mod.at(node) - delay - cur_node_to_sched_mod.at(in_node);
        
        #ifdef DEBUG
                        std::cout << "[DEBUG] Checking edge " << in_node << " (" << in_pe << ") -> " 
                                  << node << " (" << out_pe << ") | delay = " << delay 
                                  << " | sched_diff = " << time_diff << std::endl;
        #endif
        
                        if (time_diff != 0 ){
        #ifdef DEBUG
                            std::cout << "[DEBUG]   -> Time mismatch, rerouting...\n";
        #endif
                            this->reroute(in_pe, out_pe, cur_node_to_sched_mod.at(node) - cur_node_to_sched_mod.at(in_node));
                        }
        
                        auto new_delay = static_cast<int>(this->PEs_to_routing.at({in_pe, out_pe}).size()) - 1;
                        time_diff = cur_node_to_sched_mod.at(node) - new_delay - cur_node_to_sched_mod.at(in_node);
        
        #ifdef DEBUG
                        std::cout << "[DEBUG]   -> After reroute: new_delay = " << new_delay 
                                  << " | new sched_diff = " << time_diff << std::endl;
        #endif
        
                        if (time_diff != 0 ){
                            all_conv_satisfied = false;
        #ifdef DEBUG
                            std::cout << "[DEBUG] Invalid routing after scheduling for node: " << node << std::endl;
        #endif
                            break;
                        }
                    }
                    if(!all_conv_satisfied){
                        break;
                    }
                }
                this->contains_invalid_scheduling = !all_conv_satisfied;
            }
        }
        
        }
        #ifdef DEBUG
        std::cout << "[DEBUG] Calling update_features_before_state_step with action: " << action << std::endl;
        #endif
        this->update_features_before_state_step(action);
        
        #ifdef DEBUG
        std::cout << "[DEBUG] Calling update_state with action: " << action << std::endl;
        #endif
        this->update_state();
        
        #ifdef DEBUG
        std::cout << "[DEBUG] Calling update_features_after_state_step with action: " << action << std::endl;
        #endif
        this->update_features_after_state_step(action);
        
        #ifdef DEBUG
        std::cout << "[DEBUG] Step completed with action: " << action << "\n\n";
        #endif
    }else{
        #ifdef DEBUG
        std::cout << "[DEBUG] Calling update_state with action after placement in a used PE " << action << std::endl;
        #endif
        this->update_state();
    }
}


int BaseState::get_num_dfg_nodes() const{
    return this->dfg->get_num_nodes();
}

int BaseState::get_num_mapped_nodes() const{
    return static_cast<int>(this->node_to_pe.size());
}

int BaseState::get_num_mapped_edges() const{
    
    return this->num_mapped_edges;
}


const std::vector<int> &BaseState::get_legal_actions_() const
{
    return this->legal_actions;
}
int BaseState::get_num_dfg_edges() const
{
    return this->dfg->get_num_edges();
}

int BaseState::get_num_cgra_nodes() const{
    return this->cgra->total_PEs();
}

int BaseState::get_num_cgra_edges() const{
    return this->cgra->get_num_edges();
}




const std::vector<int>& BaseState:: get_legal_actions() const{
    return this->legal_actions;
}


int BaseState::get_legal_actions_size() const{
    return this->legal_actions.size();
}

std::pair<int,int> BaseState::get_PE_pos_by_PE(const int& action) const {
    return this->cgra->get_pe_pos(action);
}

const bool& BaseState::get_is_end_state() const {
    return this->is_end_state;
}


const std::string& BaseState::get_dfg_name() const{
    return this->dfg->get_dfg_name();
}



const std::string& BaseState::get_invalid_mapping_reason_() const{
    return this->invalid_mapping_reason;
}

const std::string BaseState::get_dfg_node_name_by_id(int id){
    return this->dfg->get_name_by_node(id);
}

int BaseState::get_dfg_size() const {
    return this->dfg->get_num_nodes();
}

int BaseState::get_number_of_spatial_PEs() const {
    return this->cgra->number_of_spatial_PEs_();
}


void BaseState::draw_mapping_md(std::stringstream &ss, bool final_mapping,
    const std::vector<float>* action_probs) const {

    auto II = this->II;
    const auto& dims = this->get_cgra_dimensions_();
    std::unordered_map<std::pair<int, int>, std::vector<int>> PEs_to_routing = this->get_PEs_to_routing_();
    const auto& node_to_scheduled_time_slice = this->get_node_to_scheduled_time_slice_();
    const auto& pe_to_node = this->get_PE_to_node();
    std::string invalid_mapping_reason = this->get_invalid_mapping_reason_();

    std::unordered_set<int> legal_actions_set(this->legal_actions.begin(), this->legal_actions.end());

    auto get_color_from_probability = [&](double p) {
        p = std::max(0.0, std::min(1.0, p));
    
        int r, g, b;
    
        if (p < 0.33) {
            double t = p / 0.33;
            r = static_cast<int>(0   + t * (0   - 0));   // 0 → 0
            g = static_cast<int>(0   + t * (255 - 0));   // 0 → 255
            b = static_cast<int>(255 + t * (0   - 255)); // 255 → 0
        } 
        else if (p < 0.66) {
            double t = (p - 0.33) / 0.33;
            r = static_cast<int>(0   + t * (255 - 0));   // 0 → 255
            g = 255;
            b = 0;
        } 
        else {
            double t = (p - 0.66) / 0.34;
            r = 255;
            g = static_cast<int>(255 + t * (0 - 255));   // 255 → 0
            b = 0;
        }
    
        std::stringstream ss;
        ss << "color:rgb(" << r << "," << g << "," << b << "); font-weight:bold;";
        return ss.str();
    };
    std::vector<std::vector<std::vector<int>>> placement(
        II, std::vector<std::vector<int>>(dims.first, std::vector<int>(dims.second, -1))
    );

    for (const auto& [pe, node] : pe_to_node) {
        int cycle = pe / (dims.first * dims.second);
        int spatial_pe = pe % (dims.first * dims.second);
        int row = spatial_pe / dims.second;
        int col = spatial_pe % dims.second;
        placement[cycle][row][col] = node;
    }

    std::unordered_map<int,int> routing_sched_time;
    if (!node_to_scheduled_time_slice.empty()){
        for (const auto& [pe_pos, routing_list] : PEs_to_routing) {
            if(routing_list.empty()) continue;

            int initial_sched = node_to_scheduled_time_slice.at(pe_to_node.at(routing_list[0]));
            for(int i = 1; i < (int)routing_list.size() - 1; ++i){
                int pe = routing_list[i];
                routing_sched_time[pe] = initial_sched + i;
            }
        }
    }

    for (int cycle = 0; cycle < II; ++cycle) {
        ss << "#### Placement for cycle " << cycle << "\n\n";
        ss << "<pre style='line-height:1.4em;'>\n";

        for (int row = 0; row < dims.first; ++row) {
            for (int col = 0; col < dims.second; ++col) {

                int node = placement[cycle][row][col];
                int pe_id = col + row * dims.second + dims.first * dims.second * cycle;

                std::string label = "-";
                std::string time = "-";
                std::string color = "color:#444444;";

                if (node != -1) {
                    auto rout_pe_sched = routing_sched_time.count(pe_id) ?
                        std::to_string(routing_sched_time.at(pe_id)) : "-";

                    int scheduled_time =
                        node_to_scheduled_time_slice.count(node) ?
                        node_to_scheduled_time_slice.at(node) :
                        std::numeric_limits<int>::max();

                    label = !this->node_is_routing(node) ? this->dfg->get_name_by_node(node) : "R";
                    time = !this->node_is_routing(node)
                            ? (scheduled_time != std::numeric_limits<int>::max()
                                ? std::to_string(scheduled_time) : "-")
                            : rout_pe_sched;

                    if (final_mapping) {
                        if (this->node_is_routing(node))
                            color = "color:#8C6A03; font-weight:bold;";
                        else if (this->dfg->is_root(node))
                            color = "color:#4A1A7F; font-weight:bold;";
                        else if (this->dfg->is_leaf(node))
                            color = "color:#A3541C; font-weight:bold;";
                        else
                            color = "color:#0A4C8A; font-weight:bold;";
                    }
                }

                std::string tooltip = "";
                if (!final_mapping && action_probs != nullptr) {
                    if (legal_actions_set.count(pe_id) == 0) {
                        tooltip = "illegal action";
                        color = "color:#8B0000; font-weight:bold;";
                    } else {
                        int idx = this->get_action_idx_by_action(pe_id);
                        if (idx >= 0 && idx < (int)action_probs->size()) {

                            double p = (*action_probs)[idx];
                            color = get_color_from_probability(p);

                            std::stringstream tt;
                            tt << "aidx=" << idx << ", p=" 
                               << std::fixed << std::setprecision(3) << p;

                            tooltip = tt.str();
                        }
                    }
                }

                ss << "<span style='" << color << "' title='" << tooltip << "'>"
                   << "[" << std::setw(3) << pe_id << " | "
                   << std::setw(8) << label << " | "
                   << std::setw(3) << time << "]"
                   << "</span> ";
            }

            ss << "\n";
        }

        ss << "</pre>\n\n";
    }

    /* ------------------ ROUTING ------------------ */
    ss << "#### Routing Information\n\n";
    for (const auto& [pe_pos, routing_list] : PEs_to_routing) {
        ss << "- From PE " << pe_pos.first << " to PE " << pe_pos.second << " → Routing: ";
        if (routing_list.empty()) ss << "`-`";
        else {
            ss << "`";
            for (auto r : routing_list) ss << r << " ";
            ss << "`";
        }
        ss << "\n";
    }

    ss << "\n---\n";
}




void BaseState::print_base_state(std::stringstream& ss) const {
    ss << "BaseState Information:\n";
    ss << "II: " << II << "\n";
    ss << "DFG name: " << this->dfg->get_dfg_name() << "\n";
    ss << "Is End State: " << std::boolalpha << is_end_state << "\n";
    ss << "Mapping Is Valid: " << std::boolalpha << mapping_is_valid << "\n";
    ss << "Scheduling Is Valid: " << std::boolalpha << !contains_invalid_scheduling << "\n";
    ss << "Using RTScheduler: " << std::boolalpha << this->use_rt_scheduler << "\n";
    ss << "Using Sync Routing: " << std::boolalpha << this->use_sync_routing << "\n";
    ss << "Using Scheduling mask" << std::boolalpha << this->use_sched_mask << "\n";
    ss << "Invalid Mapping Reason: " << invalid_mapping_reason << "\n";
    ss << "Idx ID Node to Be Mapped: " << idx_id_node_to_be_mapped << "\n";
    ss << "Contains Invalid Routing: " << std::boolalpha << contains_invalid_routing << "\n";
    ss << "Greedy: " << std::boolalpha << greedy << "\n";
    ss << "Num Mapped Edges: " << num_mapped_edges << "\n";
    ss << "Use PE Mask: " << std::boolalpha << use_used_pe_mask << "\n";
    ss << "Use Congestion Mask: " << std::boolalpha << use_routing_mask << "\n";
    ss << "Use Symmetric Mask: " << std::boolalpha << use_symmetry_mask << "\n";


    ss << "Legal Actions: ";
    for (const auto& action : legal_actions) ss << action << " ";
    ss << "\n";

    ss << "Action to Index: ";
    for (const auto& pair : action_to_idx)
        ss << "[" << pair.first << " -> " << pair.second << "] ";
    ss << "\n";

    ss << "Idx to action: ";
    for (const auto& pair : idx_to_action)
        ss << "[" << pair.first << " -> " << pair.second << "] ";
    ss << "\n";

    ss << "Node to Scheduled Time Slice: ";
    for (const auto& pair : get_node_to_scheduled_time_slice_())
        ss << "[" << pair.first << " -> " << pair.second << "] ";
    ss << "\n";

    ss << "Node to PE: ";
    for (const auto& pair : node_to_pe)
        ss << "[" << pair.first << " -> " << pair.second << "] ";
    ss << "\n";

    ss << "PE to Node: ";
    for (const auto& pair : pe_to_node)
        ss << "[" << pair.first << " -> " << pair.second << "] ";
    ss << "\n";

    ss << "PEs to Routing:\n";
    for (const auto& pair : PEs_to_routing) {
        ss << "[" << pair.first.first << "," << pair.first.second << "] -> { ";
        for (const auto& v : pair.second) ss << v << " ";
        ss << "} ";
    }
    ss << "\n";

    ss << "Free Connections:\n";
    for (const auto& pair : free_connections) {
        ss << pair.first << " -> { ";
        for (const auto& v : pair.second) ss << v << " ";
        ss << "} ";
    }
    ss << "\n";

    ss << "Node to Sched Mods:\n";
    for (const auto& pair : node_to_sched_mods)
        ss << "[" << pair.first << " -> (" << pair.second.first << ", " << pair.second.second << ")] ";
    ss << "\n";

    if (node_was_placed_tensor.defined()) {
        ss << "Node Was Placed Tensor: " << node_was_placed_tensor.sizes() << "\n";
        ss << node_was_placed_tensor << "\n";
    } else {
        ss << "Node Was Placed Tensor: Undefined\n";
    }

    ss << "Model Inputs Size: " << model_inputs.size() << "\n";

}

bool BaseState::do_backtracking() const {
    return this->is_end_state && !this->mapping_is_valid;
}
void BaseState::update_features_before_state_step(const int& action) {
    throw std::runtime_error("Not implemented");
};

void BaseState::update_features_after_state_step(const int& action) {
    throw std::runtime_error("Not implemented");
};


std::vector<c10::IValue> BaseState::calc_initial_model_inputs(){
    throw std::runtime_error("Not implemented");

}


void BaseState::update_action_to_idx(){
    throw std::runtime_error("Not implemented");
}


bool BaseState::model_is_general() const {
    throw std::runtime_error("Not implemented");
}
