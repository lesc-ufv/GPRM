#ifndef MAPZERO_STATE_HPP
#define MAPZERO_STATE_HPP

#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <string>
#include "src/hpp/utils/cgra/router.hpp"
#include <execution>
#include "src/hpp/rl/base_state.hpp"
#include "src/hpp/configs/state_config.hpp"
#include "src/hpp/utils/util_serialize.hpp"
#include <cereal/types/base_class.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/utility.hpp> 
#include <memory>

class MapZeroState: public BaseState { 
private:      
    torch::Tensor cgra_features_per_module;
    bool override_base_state;
    //model inputs
    static constexpr int idx_dfg_features = 0;
    static constexpr int idx_dfg_edge_indices = 1;
    static constexpr int idx_cgra_features = 2;
    static constexpr int idx_cgra_edge_indices = 3;
    static constexpr int idx_node_to_be_mapped_features= 4;
    static constexpr int idx_PE_mask = 5;
    static constexpr int idx_dfg_pad_mask = 6;
    static constexpr int idx_cgra_pad_mask = 7;

    //dfg
    static constexpr short dfg_id_idx = 0;
    static constexpr short sched_order_idx = 1;
    static constexpr short sched_time_slice_idx = 2;
    static constexpr short sched_mod_time_slice_idx = 3;
    static constexpr short dfg_in_degree_idx = 4;
    static constexpr short dfg_out_degree_idx = 5;
    static constexpr short opcode_idx = 6;
    static constexpr short has_self_cycle = 7;
    static constexpr short num_nodes_to_be_mapped_same_mod_idx = 8;
    static constexpr short assigned_pe_idx = 9;
    static constexpr short dfg_num_features = 10;

    //cgra
    static constexpr short cgra_id_idx = 0;
    static constexpr short cgra_in_degree_idx = 1;
    static constexpr short cgra_out_degree_idx = 2;
    static constexpr short functionalitie_initial_idx = 3;

    static constexpr short id_mapped_dfg_node_idx = 6;
    static constexpr short cgra_num_features = 7;
    bool use_spatio_temporal_input;
    static constexpr double bad_reward = -100.0;
 
public:

    MapZeroState(std::shared_ptr<OrderedDFG> dfg, std::shared_ptr<CGRA> cgra , StateConfig& state_config);
    MapZeroState(const MapZeroState& other);
    MapZeroState(MapZeroState&& other) noexcept : BaseState(std::move(other)) {
        this->cgra_features_per_module = std::move(other.cgra_features_per_module);
        this->override_base_state = other.override_base_state;
        this->use_spatio_temporal_input = other.use_spatio_temporal_input;
    }


    const int& get_II() const;

    MapZeroState(): BaseState(),
    cgra_features_per_module(torch::empty({})),
    override_base_state(false) {
    }
    int get_cgra_input_size();

    std::vector<c10::IValue> calc_initial_model_inputs();

    void assign_scheduled_time_slice_to_node(const int scheduled_time_slice, const int &node);
    void update_node_to_be_mapped_features();

    void assign_node_to_PE(int node, const int &PE);

    void assign_PE_to_node(int PE,const int& node);

    void place_node_in_PE(int node, int PE);
    void update_masks_in_model_input();

    void assign_modulo_time_slice(int modulo_time_slice, const int& node);

    int get_scheduled_time_slice_by_node_id(const int& node_id);
    void update_is_end_state();

    const torch::Tensor get_dfg_features() const;
    const torch::Tensor get_dfg_edge_indices() const;
    const torch::Tensor get_cgra_features() const;

    const torch::Tensor get_cgra_edge_indices() const;
    const torch::Tensor get_node_to_be_mapped_features() const ;

    const torch::Tensor get_mask() const;
    const torch::Tensor get_dfg_pad_masks() const;

    const torch::Tensor get_cgra_pad_masks() const;

    void update_features_before_state_step(const int& action) override;
    void update_features_after_state_step(const int& action) override;

    int calc_relative_PE(const int &PE);
    
    torch::Tensor calc_masks(int id_node_to_be_mapped);
    
    void update_action_to_idx() override;

    bool do_backtracking() const override;
    bool model_is_general() const override {return false;};

    
    void print() const;

    template <class Archive>
    void serialize(Archive& ar) {
    ar(cereal::base_class<BaseState>(this));
    ar(use_spatio_temporal_input);
    serialize_tensor(ar, cgra_features_per_module);
    ar(override_base_state);
    }

    

};
#endif