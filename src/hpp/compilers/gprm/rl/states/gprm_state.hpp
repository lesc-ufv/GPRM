#ifndef GPRM_BASE_STATE_HPP
#define GPRM_BASE_STATE_HPP

#include <tuple>
#include <vector>
#include <string>
#include "src/hpp/utils/cgra/router.hpp"
#include <execution>
#include "src/hpp/utils/util_torch.hpp"
#include "src/hpp/rl/base_state.hpp"
#include "src/hpp/utils/util_torch.hpp"
#include "src/hpp/compilers/gprm/configs/gprm_config.hpp"
#include "src/hpp/utils/util_server.hpp"
#include <zmq.hpp>
#include <torch/torch.h>
#include "src/hpp/configs/state_config.hpp"
#include "src/hpp/compilers/gprm/rl/states/gprm_state.hpp"

class GPRMState: public BaseState { 
private:
    int placed_id;
    int not_placed_id;
    int to_place_id, to_not_place_id;
    std::unordered_map<std::pair<int,int>,int> mapping_node_to_id;

    //Static
    torch::Tensor lpe_dfg;
    torch::Tensor dfg_placement_order_ids;
    torch::Tensor dfg_node_scheduling_ids;
    torch::Tensor dfg_edge_indices;
    torch::Tensor dfg_operation_ids;
    
    torch::Tensor lpe_adg;
    torch::Tensor adg_pos;
    torch::Tensor adg_edge_indices;
    torch::Tensor adg_operation_ids;  
    //Dynamic
    
    // torch::Tensor dfg_routing_type_ids;
    // torch::Tensor dfg_placement_type_ids;
    // torch::Tensor adg_placement_type_ids;
    // torch::Tensor adg_routing_type_ids;
    
    torch::Tensor mapping_edge_indices;
    
    torch::Tensor placements;
    torch::Tensor routings;
    
    
    torch::Tensor action_batch_idx;
    
    torch::Tensor batch_idx;
    torch::Tensor seq_idx;
    torch::Tensor contextual_action_seq_idx;
    torch::Tensor action_seq_idx;
    
    torch::Tensor state_seq_idx;

    int max_seq_len, max_legal_actions;

    torch::Tensor max_range_values;
    torch::Tensor max_zeros_values;
public:
    int calc_max_possible_seq_len(){
        return 2*(this->cgra->total_PEs() +2*this->cgra->get_num_edges()) + 
        this->dfg->num_vertices() + 
        this->dfg->get_num_edges(); 
    }
    GPRMState()
    : BaseState(), 
      placed_id(0),
      not_placed_id(0),
      to_place_id(0),
      to_not_place_id(0),
      mapping_node_to_id(),
      lpe_dfg(torch::Tensor()),
      dfg_placement_order_ids(torch::Tensor()),
      dfg_node_scheduling_ids(torch::Tensor()),
      dfg_edge_indices(torch::Tensor()),
      dfg_operation_ids(torch::Tensor()),
      lpe_adg(torch::Tensor()),
      adg_pos(torch::Tensor()),
      adg_edge_indices(torch::Tensor()),
      adg_operation_ids(torch::Tensor()),
      mapping_edge_indices(torch::Tensor()),
      placements(torch::Tensor()),
      routings(torch::Tensor()),
      action_batch_idx(torch::Tensor()),
      batch_idx(torch::Tensor()),
      seq_idx(torch::Tensor()),
      contextual_action_seq_idx(torch::Tensor()),
      action_seq_idx(torch::Tensor()),
      state_seq_idx(torch::Tensor()),
      max_seq_len(0),
      max_legal_actions(0),
      max_range_values(torch::Tensor()),
      max_zeros_values(torch::Tensor())
{}

    GPRMState(std::shared_ptr<OrderedDFG> dfg, std::shared_ptr<CGRA> cgra,
    std::shared_ptr< zmq::socket_t> socket, const ServerConstants& server_k, const GPRMConfig& gprm_config,StateConfig& state_config);
    GPRMState(const GPRMState& other);
    GPRMState(GPRMState&& other) noexcept;
    // void update_dfg_routing_type_by_id(const int& node_to_be_mapped, const int& id);

    // void update_adg_routing_type_by_id(const int& node_to_be_mapped, const int& id);

    void update_idx_and_mask();

    int calc_max_seq_len() const ;
    void update_features_before_state_step(const int& action) override;

    void update_features_after_state_step(const int& action) override;

    void add_routing_nodes_placement_by_neigh(std::vector<std::vector<int>>& placements, const int& id_node_to_be_mapped,
                                    const int& action, bool in_vertices);

    void add_routing_nodes_placement(std::vector<std::vector<int>>& placements, const int& id_node_to_be_mapped,
                                    const int& action);

    void add_routing_by_neigh(std::vector<std::vector<int>>& routings, const int& id_node_to_be_mapped,
                                    const int& action, bool in_vertices);

    void add_routing(std::vector<std::vector<int>>& routings, const int& id_node_to_be_mapped,
                                    const int& action);

    std::vector<c10::IValue> get_model_inputs(c10::DeviceType device) const override;

    void print() const;

    const torch::Tensor& getLpeDfg() const ;
    const torch::Tensor& getDfgNodeSchedulingIds() const ;
    const torch::Tensor& getDfgEdgeIndices() const ;
    const torch::Tensor& getDfgOperationIds() const ;
    const torch::Tensor& getDfgPlacementOrderIds() const {
        return this->dfg_placement_order_ids;
    } ;

    const torch::Tensor& getLpeAdg() const ;
    const torch::Tensor& getAdgEdgeIndices() const ;
    const torch::Tensor& getAdgOperationIds() const ;
    
    // const torch::Tensor& getDfgRoutingTypeIds() const ;
    // const torch::Tensor& getDfgPlacementTypeIds() const ;
    // const torch::Tensor& getAdgPlacementTypeIds() const ;
    // const torch::Tensor& getAdgRoutingTypeIds() const ;
    
    const torch::Tensor& getMappingEdgeIndices() const ;
    const torch::Tensor& getPlacements() const ;
    const torch::Tensor& getRoutings() const ;
    const torch::Tensor& getActionBatchIdx() const ;
    
    const torch::Tensor& getBatchIdx() const ;
    const torch::Tensor& getSeqIdx() const ;
    const torch::Tensor& getContextualActionSeqIdx() const ;
    const torch::Tensor& getActionSeqIdx() const ;
    
    const torch::Tensor& getStateSeqIdx() const ;

    int get_num_mapping_nodes() const;
    int get_num_mapping_edges() const;

    int get_max_seq_len() const;

    bool do_backtracking() const override;
    void update_action_to_idx() override;
    std::vector<c10::IValue> calc_initial_model_inputs() override {
        return std::vector<c10::IValue>();
    };


    const torch::Tensor& getADGPos() const ;
    bool model_is_general() const override {return true;};

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::base_class<BaseState>(this));

        ar(placed_id,
        not_placed_id,
        to_place_id,
        to_not_place_id);

        serialize_mapping_node_to_id(ar, mapping_node_to_id);

        // Serialize tensors
        serialize_tensor(ar, lpe_dfg);
        serialize_tensor(ar, dfg_placement_order_ids);
        serialize_tensor(ar, dfg_node_scheduling_ids);
        serialize_tensor(ar, dfg_edge_indices);
        serialize_tensor(ar, dfg_operation_ids);

        serialize_tensor(ar, lpe_adg);
        serialize_tensor(ar, adg_pos);
        serialize_tensor(ar, adg_edge_indices);
        serialize_tensor(ar, adg_operation_ids);

        serialize_tensor(ar, mapping_edge_indices);

        serialize_tensor(ar, placements);
        serialize_tensor(ar, routings);

        serialize_tensor(ar, action_batch_idx);

        serialize_tensor(ar, batch_idx);
        serialize_tensor(ar, seq_idx);
        serialize_tensor(ar, contextual_action_seq_idx);
        serialize_tensor(ar, action_seq_idx);
        serialize_tensor(ar, state_seq_idx);

        ar(max_seq_len,
        max_legal_actions);

        serialize_tensor(ar, max_range_values);
        serialize_tensor(ar, max_zeros_values);
    }

};
#endif