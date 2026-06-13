#ifndef BASE_STATE_HPP
#define BASE_STATE_HPP

#include <iostream>
#include <execution>
#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>
#include "src/hpp/utils/cgra/router.hpp"
#include "src/hpp/utils/util_torch.hpp"
#include "src/hpp/data_structures/dfgs/ordered_dfg.hpp"
#include "src/hpp/utils/cgra/scheduler.hpp"
#include "src/hpp/data_structures/cgras/cgra.hpp"
#include "src/hpp/configs/state_config.hpp"
#include "src/hpp/utils/cgra/rt_scheduler.hpp"
#include "src/hpp/enums/enum_end_state.hpp"
#include "src/hpp/utils/util_serialize.hpp"

class BaseState{ 
protected:
    int id;
    RTScheduler rt_scheduler;
    int II;
    bool is_end_state;
    bool mapping_is_valid;
    std::string invalid_mapping_reason;
    unsigned int idx_id_node_to_be_mapped;
    bool contains_invalid_routing, contains_invalid_scheduling;
    int num_mapped_edges;
    std::shared_ptr<OrderedDFG> dfg;
    std::shared_ptr<CGRA> cgra;
    bool greedy;
    std::vector<c10::IValue> model_inputs;
    EnumEndState enum_end_state;

    std::unordered_map<int,int> node_to_scheduled_time_slice;
    std::unordered_map<int, int> node_to_pe;
    std::unordered_map<int, int> pe_to_node;

    std::unordered_map<std::pair<int, int>, std::vector<int>> PEs_to_routing;
    std::unordered_map<int,std::unordered_set<int>> free_connections;

    torch::Tensor node_was_placed_tensor;
    torch::Tensor remain_in_edges;
    torch::Tensor remain_out_edges;

    std::vector<int> legal_actions;
    std::unordered_map<int,int> action_to_idx;
    std::unordered_map<int,int> idx_to_action;
    std::unordered_map<int,std::pair<int,int>> node_to_sched_mods;
    bool use_used_pe_mask, use_routing_mask, use_symmetry_mask, use_rt_scheduler, use_sync_routing, use_sched_mask;
    // bool skip_state = false;
    int total_routing_nodes = -1;
    int state_key = -1;
public:

    std::string get_key(){
        auto dims = this->cgra->get_cgra_dimensions_();
        auto connection_bit = this->cgra->get_connection_bitset().to_ulong();
        return std::format("{}_{}_{}_{}_{}_{}",id, this->II, dims.first,dims.second, connection_bit, this->dfg->get_dfg_name());
    }

    std::string get_key_with_placement_idx(){
        auto dims = this->cgra->get_cgra_dimensions_();
        auto connection_bit = this->cgra->get_connection_bitset().to_ulong();
        return std::format("{}_{}_{}_{}_{}_{}", this->II, dims.first,dims.second, connection_bit, this->idx_id_node_to_be_mapped,this->dfg->get_dfg_name());
    }

    void set_id(int id){
        this->id = id;
    };

    int get_id() const{
        return this->id;
    };

    std::shared_ptr<CGRA> get_cgra() const{
        return this->cgra;
    };

    BaseState(std::shared_ptr<OrderedDFG> dfg, std::shared_ptr<CGRA> cgra, StateConfig& state_config);
    BaseState(BaseState&& other) noexcept;
    BaseState(const BaseState& other);
    BaseState();
    // void set_skip_state(bool skip){
    //     this->skip_state = skip;
    // }
    // bool get_skip_state() const{
    //     return this->skip_state;
    // }
    void reroute(int in_pe, int out_pe, int distance);
    const int& get_id_node_to_be_mapped_by_idx(const int& idx_id_node_to_be_mapped) const;
    void route(bool route_in_vertices, int action);
    void place_and_route(const int& action);
    void analyze_mapping();
    int get_num_mapped_nodes() const;
    int get_num_mapped_edges() const;
    void initialize();
    void remove_occupied_PE(int pe) {
        this->node_was_placed_tensor[pe] = 0;
    }
    int get_II() const {
        return this->II;
    };
    int get_total_rec_paths() const {
        return this->dfg->get_total_rec_paths();
    }
    double get_complexity() {
        auto cur_conn = this->cgra->get_num_spatial_edges();
        auto cur_pes = this->get_number_of_spatial_PEs();
        double dfg_val = this->dfg->get_dfg_value();
        double mii_ii_ratio = this->get_MII_II_ratio();
        double denom = std::pow(static_cast<double>(cur_conn + cur_pes), 1.0 / 3.0) * mii_ii_ratio;
        double result = dfg_val / denom;
    
    #ifdef DEBUG
        std::cout << "[DEBUG] get_complexity()" << std::endl;
        std::cout << "  cur_conn: " << cur_conn << std::endl;
        std::cout << "  cur_pes: " << cur_pes << std::endl;
        std::cout << "  dfg_val: " << dfg_val << std::endl;
        std::cout << "  MII/II ratio: " << mii_ii_ratio << std::endl;
        std::cout << "  denominator: " << denom << std::endl;
        std::cout << "  result: " << result << std::endl;
    #endif
    
        return result;
    }
    
    bool is_action_available(int action){
        return this->action_to_idx.find(action) != this->action_to_idx.end();
    }

    double get_complexity_v2(double lambda_map_complexity) {
        auto occp = this->get_min_pe_occupancy();
        double occp_w = occp < lambda_map_complexity ? (1.01) : 1;
        auto cur_conn = this->cgra->get_num_spatial_edges();
        double dfg_val = this->dfg->get_dfg_value();
        double mii_ii_ratio = this->get_MII_II_ratio();
        double denom = std::pow(static_cast<double>(cur_conn), 1.0 / 3.0) * mii_ii_ratio;
        double result = dfg_val * occp_w / denom;
        
    #ifdef DEBUG
        std::cout << "[DEBUG] get_complexity_v2()" << std::endl;
        std::cout << "  occp: " << occp << std::endl;
        std::cout << "  lambda_map_complexity: " << lambda_map_complexity << std::endl;
        std::cout << "  occp_w: " << occp_w << std::endl;
        std::cout << "  cur_conn: " << cur_conn << std::endl;
        std::cout << "  dfg_val: " << dfg_val << std::endl;
        std::cout << "  MII/II ratio: " << mii_ii_ratio << std::endl;
        std::cout << "  denominator: " << denom << std::endl;
        std::cout << "  result: " << result << std::endl;
    #endif
    
        return result;
    }
    
    const std::unordered_set<EnumInterconnectionStyles>& get_interconnection_styles() const {
        return this->cgra->get_interconnection_styles();
    };

    int get_num_routing_nodes() {
        if(this->total_routing_nodes != -1){
            return this->total_routing_nodes;
        }
        this->total_routing_nodes = 0;
        for(auto& [key, value] : this->PEs_to_routing){
            if(value.size() > 0){
                total_routing_nodes += value.size() - 2;
            }
        }
        return this->total_routing_nodes;
    }
    int get_num_dfg_nodes() const;

    const std::vector<int>& get_legal_actions_() const;

    double get_MII_II_ratio() const {
        double mii = static_cast<double>(this->get_MII());
        double ii = static_cast<double>(this->II);
        double ratio = mii / ii;
    
    #ifdef DEBUG
        std::cout << "[DEBUG] MII: " << mii << std::endl;
        std::cout << "[DEBUG] II: " << ii << std::endl;
        std::cout << "[DEBUG] MII/II Ratio: " << ratio << std::endl;
    #endif
    
        return ratio;
    }

    int calc_num_rout_nodes() const {
        int rout_nodes_sum = 0;
        
        for(auto& [pair,list]: this->PEs_to_routing){
            rout_nodes_sum += list.size() - 2;
        }

        return rout_nodes_sum;
    }

    const std::pair<int, int>& get_cgra_dimensions_() const;
    const std::unordered_map<std::pair<int, int>, std::vector<int>>& get_PEs_to_routing_() const;
    const std::unordered_map<int,std::tuple<int, int, int>>& get_PE_to_coord_() const;

    const std::unordered_map<std::tuple<int, int, int>, int>& get_coord_to_PE_() const ;
    const std::unordered_map<int,std::unordered_set<int>>& get_free_connections_() const;

    const std::unordered_map<int, std::unordered_set<int>>& get_out_vertices_cgra_() const;
    const std::unordered_set<int>& get_in_vertices_dfg_by_node_id(const int &node) const;
    const std::unordered_set<int>& get_out_vertices_dfg_by_node_id(const int &node) const;
    const bool& get_mapping_is_valid_() const;
    
    bool all_nodes_were_mapped() const ;

    const std::unordered_map<int, int>& get_PE_to_node() const;


    bool node_was_mapped(const int &node) const;
    const std::vector<int>& get_vertices() const;

    const std::string get_real_node(const int& node_id) const;
    const std::unordered_map<int, int>& get_node_to_scheduled_time_slice_() const;
    double get_dfg_complexity() const {
        return this->dfg->get_dfg_value();
    }
    const std::unordered_map<int, std::unordered_set<int>>& get_dfg_node_to_in_vertices_() const;
    const std::unordered_map<int, std::unordered_set<int>>& get_dfg_node_to_out_vertices_() const;
    const std::unordered_map<int, int> &get_node_to_pe_() const ;
    const std::string& get_invalid_mapping_reason_() const;

    std::string get_str_dims() const {
        return std::to_string(this->cgra->get_cgra_dimensions_().first) + "x" + std::to_string(this->cgra->get_cgra_dimensions_().second);
    };

    void set_routing_is_invalid();

    bool get_routing_is_invalid() const;

    const std::vector<int>& get_root_nodes() const;
    
    void set_node_to_scheduled_time_slice(const std::unordered_map<int,int>& new_node_to_scheduled_time_slice);

    const std::vector<std::pair<int,int>>& get_dfg_edges() const;

    void route_by_routing_path(const std::pair<int, int>& tgt_routing_PEs, const std::vector<int>& routing_path);

    virtual const std::unordered_set<int> get_occupied_PEs_() const ;

    virtual int get_assigned_PE_by_node_id(const int& node_id) const;

    virtual std::vector<c10::IValue> get_model_inputs(c10::DeviceType device) const ;


    virtual void update_features_before_state_step(const int& action);
    
    virtual void update_features_after_state_step(const int& action);

    void step(const int& action);

    virtual void update_state();

    virtual void calc_initial_legal_actions();

    virtual void update_legal_actions();

    virtual void update_is_end_state();

    virtual bool pe_is_occupied(const int& PE) const;


    int get_num_dfg_edges() const;
    virtual void add_occupied_PE(int pe);

    int get_num_cgra_nodes() const;
    int get_num_cgra_edges() const;
    int get_num_spatial_edges () const {
        return this->cgra->get_num_spatial_edges();
    }

    int get_total_reconvergent_levels() const {
        return 0;
    }
    
    virtual void update_action_to_idx();
    int get_action_idx_by_action(const int& action) const ;
    int get_legal_actions_size() const;
    std::vector<int> get_legal_actions_with_annotations(const int idx_id_node_to_be_mapped);
    std::vector<int> get_legal_actions_with_traversal(const int& idx_id_node_to_be_mapped);
    std::vector<int> get_greedy_legal_actions(const int idx_id_node_to_be_mapped);
    void update_idx_id_node_to_be_mapped();
    virtual std::vector<c10::IValue> calc_initial_model_inputs();
    const std::vector<int>&  get_legal_actions() const;
    const std::unordered_set<int> get_not_occupied_PEs_() const;
    const int& get_prev_mapped_node() const;
    const std::vector<int> get_not_occupied_PEs_vector() const ;
    std::pair<int,int> get_PE_pos_by_PE(const int& action) const;

    const bool& get_is_end_state() const;
    const std::string& get_dfg_name() const;
    const std::string get_dfg_node_name_by_id(int id);
    virtual bool do_backtracking() const;
    virtual bool model_is_general() const;

    std::shared_ptr<OrderedDFG> get_dfg() const {
        return this->dfg;
    }

    size_t get_placement_hash() const {
        size_t h = 0;
        for (auto& [k, v] : this->node_to_pe) {
            h ^= std::hash<int>{}(k) ^ (std::hash<int>{}(v) << 1);
        }
        return h;
    }

    int get_dfg_size() const;
    int get_number_of_spatial_PEs() const ;
    // void draw_mapping(std::stringstream& ss) const ;
    void print_base_state(std::stringstream& ss) const;
    std::vector<int> get_masked_actions(int initial_PE, int final_PE, bool use_sym_mask);
    bool some_neigh_was_mapped(int id);
    std::vector<int> get_non_greedy_legal_actions(int node_to_be_mapped);
    std::vector<int> get_actions_with_sched(int node_to_be_mapped);
    bool satisfies_mask_condition(const int& PE ,const int& num_in_neigh, const int& num_out_neigh) const;
    torch::Tensor mask_congested_edges(torch::Tensor mask, int node_to_be_mapped);
    int get_action_by_action_idx(const int& idx);

    int get_num_convergent_nodes() const {
        return this->dfg->get_num_convergent_nodes();
    };

    int get_num_convergent_edges() const {
        return this->dfg->get_num_convergent_edges();
    };
    
    double get_min_pe_occupancy() const {
        return static_cast<double>(this->get_num_dfg_nodes())/static_cast<double>(this->cgra->total_PEs());
    }

    int get_num_valid_convergent_edges() const {
        auto&  cur_node_to_sched_time_slice = this->get_node_to_scheduled_time_slice_();
        int total_valid_convergent_edges = 0;
        if (cur_node_to_sched_time_slice.empty()) return 0;
        const auto& node_to_in_vertices = this->dfg->get_node_to_in_vertices();
        for (const auto& node : this->dfg->get_vertices()) {
            if(cur_node_to_sched_time_slice.find(node) == cur_node_to_sched_time_slice.end() ||
                node_to_pe.find(node) == node_to_pe.end()){
                    continue;
                }
            int cur_time_slice = cur_node_to_sched_time_slice.at(node);
            int child_pe = node_to_pe.at(node);
            auto neighs = node_to_in_vertices.at(node);
    
            #ifdef DEBUG
                std::cout << "[DEBUG] Processing node: " << node << std::endl;
                std::cout << "    Scheduled time slice: " << cur_time_slice << std::endl;
                std::cout << "    Assigned PE: " << child_pe << std::endl;
                std::cout << "    Number of parent nodes: " << neighs.size() << std::endl;
            #endif
    
            if (neighs.size() > 1) {
                for (const auto& father : neighs) {
                    if(cur_node_to_sched_time_slice.find(father) == cur_node_to_sched_time_slice.end() ||
                        node_to_pe.find(father) == node_to_pe.end()) continue;
                    int father_pe = node_to_pe.at(father);
                    int father_sched_time = cur_node_to_sched_time_slice.at(father);
                    int routing_delay = PEs_to_routing.at(std::make_pair(father_pe, child_pe)).size() - 1;
                    #ifdef DEBUG
                        std::cout << "    Parent node: " << father << std::endl;
                        std::cout << "        Parent PE: " << father_pe << std::endl;
                        std::cout << "        Parent scheduled time slice: " << father_sched_time << std::endl;
                        std::cout << "        Routing delay: " << routing_delay << std::endl;
                        std::cout << " Pair routing: " << father_pe << " " << child_pe << std::endl;

                        std::cout << "        Expected child execution time: " 
                                    << (father_sched_time + routing_delay) << std::endl;
                    #endif
    
                    if ((father_sched_time + routing_delay) == cur_time_slice) {
                        total_valid_convergent_edges += 1;
                        
                        #ifdef DEBUG
                            std::cout << "        [DEBUG] Valid scheduling detected! Incrementing final reward." << std::endl;
                        #endif
                    }
                }
            }
        }
        #ifdef DEBUG
                            std::cout << "        [DEBUG] Valid convergent edges: " << total_valid_convergent_edges << std::endl;
                        #endif
        return total_valid_convergent_edges;
    };

    int get_num_valid_convergent_nodes() const {
        auto& cur_node_to_sched_time_slice = this->get_node_to_scheduled_time_slice_();
        int total_valid_convergent_nodes = 0;
        if (cur_node_to_sched_time_slice.empty()) return 0;
        const auto& node_to_in_vertices = this->dfg->get_node_to_in_vertices();

        for (const auto& node : this->dfg->get_vertices()) {
            if(cur_node_to_sched_time_slice.find(node) == cur_node_to_sched_time_slice.end() ||
            node_to_pe.find(node) == node_to_pe.end()){
                continue;
            }
            int cur_time_slice = cur_node_to_sched_time_slice.at(node);
            int child_pe = node_to_pe.at(node);
            auto neighs = node_to_in_vertices.at(node);
    
            #ifdef DEBUG
                std::cout << "[DEBUG] Processing node: " << node << std::endl;
                std::cout << "    Scheduled time slice: " << cur_time_slice << std::endl;
                std::cout << "    Assigned PE: " << child_pe << std::endl;
                std::cout << "    Number of parent nodes: " << neighs.size() << std::endl;
            #endif
    
            if (neighs.size() > 1) {
                int count = 0; 
                for (const auto& father : neighs) {
                    if(cur_node_to_sched_time_slice.find(father) == cur_node_to_sched_time_slice.end() ||
                        node_to_pe.find(father) == node_to_pe.end()) continue;
                    int father_pe = node_to_pe.at(father);
                    int father_sched_time = cur_node_to_sched_time_slice.at(father);
                    int routing_delay = PEs_to_routing.at(std::make_pair(father_pe, child_pe)).size() - 1;
                    #ifdef DEBUG
                        std::cout << "    Parent node: " << father << std::endl;
                        std::cout << "        Parent PE: " << father_pe << std::endl;
                        std::cout << "        Parent scheduled time slice: " << father_sched_time << std::endl;
                        std::cout << "        Routing delay: " << routing_delay << std::endl;
                        std::cout << " Pair routing: " << father_pe << " " << child_pe << std::endl;

                        std::cout << "        Expected child execution time: " 
                                    << (father_sched_time + routing_delay) << std::endl;
                    #endif
    
                    if ((father_sched_time + routing_delay) == cur_time_slice) {
                        count += 1;
                    }
                }
                if(count == static_cast<int>(neighs.size())){
                    total_valid_convergent_nodes += 1;
                }
            }
        }
        return total_valid_convergent_nodes;
    };

    bool node_is_routing(int node) const {
        return node == -2 || this->node_to_pe.find(node) == this->node_to_pe.end();
    }

    void draw_mapping_md(std::stringstream &ss, bool final_mapping,
        const std::vector<float>* action_probs = nullptr) const;
    
    void print_cgra_info_md(std::stringstream& ss) const {
      this->cgra->write_cgra_md(ss);
    }
    
    void print_dfg_info_md(std::stringstream& ss) const {
        this->dfg->write_dfg_md(ss);
      }

    std::string get_arch_name() const{
        return this->cgra->get_arch_name();
    }

    int get_MII() const {
        auto& dims = this->get_cgra_dimensions_();
        int num_nodes = static_cast<int>(this->dfg->num_vertices());
        int area = dims.first * dims.second;
        int mii = static_cast<int>(std::ceil(static_cast<double>(num_nodes) / static_cast<double>(area)));
    
    #ifdef DEBUG
        std::cout << "[DEBUG] CGRA dims: " << dims.first << "x" << dims.second << " = " << area << std::endl;
        std::cout << "[DEBUG] DFG nodes: " << num_nodes << std::endl;
        std::cout << "[DEBUG] Computed MII: " << mii << std::endl;
    #endif
    
        return mii;
    }

    std::string get_node_to_be_mapped_name() const {
        return this->dfg->get_name_by_node(this->get_id_node_to_be_mapped_by_idx(this->idx_id_node_to_be_mapped));
    }

    const std::vector<int>& get_placement_order(){
        return this->dfg->get_placement_order();
    }


    void write_final_mapping_information_md(std::stringstream& ss){
        if (!invalid_mapping_reason.empty()) {
            if (this->mapping_is_valid) {
                ss << "**✅ Mapping is valid!**\n\n";
            } else {
                ss << "**❌ Invalid mapping reason:** `" << invalid_mapping_reason << "`\n\n";
            }
        }
    
        
        ss << "### Placement Information\n\n";
        ss << "Ordering approach: " << this->dfg->get_placement_approach_str() << "\n\n";
        ss << "Mask type: " << this->dfg->get_greedy_type_str() << "\n\n";
        ss << "**Placement order:**\n\n";
        this->write_placement_md(ss);
    
        ss << "### Scheduling Information\n\n";
        ss << "Using RTScheduler: " << std::boolalpha << this->use_rt_scheduler << "\n\n";
        ss << "Using SyncRouting: " << std::boolalpha << this->use_sync_routing << "\n\n";
        ss << "Using Scheduling mask: " << std::boolalpha << this->use_sched_mask << "\n\n";
        
        ss << "Pre-scheduling: " << std::boolalpha << this->dfg->get_scheduling_str() << "\n\n";
    
        ss << "### State Information\n\n";
        ss << "Using PE mask: " << std::boolalpha << this->use_used_pe_mask << "\n\n";
        ss << "Using routing mask: " << std::boolalpha << this->use_routing_mask << "\n\n";
        ss << "Using symmetry mask: " << std::boolalpha << this->use_symmetry_mask << "\n\n";
    
    }
    void write_placement_md(std::stringstream& ss) const {

    
        std::vector<std::string> placement_sequence;
        for (int i = 0; i < this->dfg->num_vertices(); ++i) {
            int node = this->get_id_node_to_be_mapped_by_idx(i);
            if (node != -1) {
                placement_sequence.push_back(this->dfg->get_name_by_node(node));
            }
        }
    
        ss << "`";
        for (size_t i = 0; i < placement_sequence.size(); ++i) {
            ss << placement_sequence[i];
            if (i < placement_sequence.size() - 1) {
                ss << " -> ";
            }
        }
        ss << "`\n\n";
    }
    
    template <class Archive>
    void serialize(Archive& ar) {
        ar(rt_scheduler,
           II,
           is_end_state,
           mapping_is_valid,
           invalid_mapping_reason,
           idx_id_node_to_be_mapped,
           contains_invalid_routing,
           contains_invalid_scheduling,
           num_mapped_edges,
           dfg,
           cgra,
           greedy
        );

        ar(model_inputs);
    
        int enum_val = static_cast<int>(enum_end_state);
        ar(enum_val);

        serialize_int_enum(ar, enum_end_state);
    
        ar(node_to_scheduled_time_slice,
           node_to_pe,
           pe_to_node,
           PEs_to_routing,
           free_connections,
           legal_actions,
           action_to_idx,
           idx_to_action,
           node_to_sched_mods,
           use_used_pe_mask,
           use_routing_mask,
           use_symmetry_mask,
           use_rt_scheduler,
           use_sync_routing,
           use_sched_mask,
           id
        );
    
        serialize_tensor(ar, node_was_placed_tensor);
        serialize_tensor(ar, remain_in_edges);
        serialize_tensor(ar, remain_out_edges);
    }
};
#endif