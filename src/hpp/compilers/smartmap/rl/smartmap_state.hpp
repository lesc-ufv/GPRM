#ifndef SMARTMAP_STATE_HPP
#define SMARTMAP_STATE_HPP

#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <string>
#include "src/hpp/utils/cgra/router.hpp"
#include <execution>
#include "src/hpp/utils/util_torch.hpp"
#include <functional>
#include "src/hpp/rl/base_state.hpp"
#include "src/hpp/configs/state_config.hpp"
#include "src/hpp/utils/util_serialize.hpp"
#include <cereal/types/base_class.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>

class SmartMapState: public BaseState{ 
private:
    std::unordered_map<int,std::unordered_set<int>> sched_mod_time_slice_to_nodes;
public:
    static constexpr int idx_dfg_features = 0;
    static constexpr int idx_dfg_edge_indices = 1;
    static constexpr int idx_cgra_features = 2;
    static constexpr int idx_cgra_edge_indices = 3;
    static constexpr int idx_node_to_be_mapped_features= 4;
    static constexpr int idx_actions_features = 5;
    static constexpr int idx_action_mask = 6;
    static constexpr int idx_dfg_pad_mask = 7;
    static constexpr int idx_cgra_pad_mask = 8;


    static constexpr double bad_reward = -100.0;

    //dfg idx
    static constexpr short dfg_id_idx = 0;
    static constexpr short sched_order_idx = 1;
    static constexpr short num_nodes_mapped_in_same_modulo_idx = 2;
    static constexpr short has_self_cycle = 3;
    static constexpr short sched_mod_time_slice_idx = 4;
    static constexpr short dfg_in_degree_idx = 5;
    static constexpr short dfg_out_degree_idx = 6;
    static constexpr short opcode_idx = 7;
    static constexpr short assigned_pe_idx = 8;
    static constexpr short num_dfg_features = 9;

    //cgra idx
    static constexpr short cgra_id_idx = 0;
    static constexpr short functionalitie_initial_idx = 1;
    static constexpr short cgra_in_degree_idx = 4;
    static constexpr short cgra_out_degree_idx = 5;
    static constexpr short id_mapped_dfg_node_idx = 6;
    
    static constexpr short num_cgra_features = 7;

    SmartMapState(std::shared_ptr<OrderedDFG> dfg, std::shared_ptr<CGRA> cgra , StateConfig& state_config);
    SmartMapState(const SmartMapState& other);
    SmartMapState(SmartMapState&& other) noexcept : BaseState(other){
        this->sched_mod_time_slice_to_nodes = std::ref(other.sched_mod_time_slice_to_nodes);
    }
    
    void update_features_before_state_step(const int& action) override ;
    void update_features_after_state_step(const int& action) override;
    SmartMapState(): BaseState(),
    sched_mod_time_slice_to_nodes() {
    }
    torch::Tensor create_legal_actions_features() const;

    torch::Tensor get_legal_actions_features_tensor() const;
    std::vector<c10::IValue> calc_initial_model_inputs() ;
    void update_node_to_be_mapped_features();

    int get_assigned_PE_by_node_id(const int& node_id) const ;
    void assign_node_to_PE(int node, const int &PE);

    void assign_PE_to_node(int PE,const int& node);
    void place_node_in_PE(int node, int PE);
    void update_masks();
    void update_actions_features();
    void assign_modulo_time_slice(const int& action, const int& node);
    void update_num_nodes_mapped_in_same_modulo(const int& action, const int& node);
    void add_routing(const std::pair<int, int> &tgt_routing_PEs, std::vector<int> &routing_path);
    const torch::Tensor get_dfg_features() const;
    const torch::Tensor get_dfg_edge_indices() const;

    const torch::Tensor get_cgra_features() const;
    const torch::Tensor get_cgra_edge_indices() const;
    const torch::Tensor get_node_to_be_mapped_features() const ;
    const torch::Tensor get_action_pad_mask() const;
    const torch::Tensor get_dfg_pad_masks() const;

    const torch::Tensor get_cgra_pad_masks() const;
    bool is_bad_reward(const double& reward);
    std::unordered_map<int, int> gen_PE_to_node() const;

    void add_routing_nodes_by_neigh(const int& id_node_to_be_mapped,
                                    const int& action, bool in_vertices);

    void add_routing_nodes(const int& id_node_to_be_mapped,
                                    const int& action);
    bool do_backtracking() const override ;
    void print();
    void update_action_to_idx();
    bool model_is_general() const override {return true;};

    template<class Archive>
    void serialize(Archive& ar) {
        ar(cereal::base_class<BaseState>(this));
        serialize_unordered_map_int_unorderedset_int(ar, sched_mod_time_slice_to_nodes);
    }

};

#endif