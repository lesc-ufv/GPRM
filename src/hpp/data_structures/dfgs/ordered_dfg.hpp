#ifndef ORDERED_DFG_HPP
#define ORDERED_DFG_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <array>
#include <unordered_set>
#include "src/hpp/data_structures/dfgs/dfg.hpp"
#include "src/hpp/enums/enum_operation.hpp"
#include "src/hpp/utils/util_print.hpp"
#include <zmq.hpp>
#include "src/hpp/utils/util_server.hpp"
#include <torch/torch.h>
#include "src/hpp/enums/enum_placement_approach.hpp"
#include "src/hpp/enums/enum_greedy_type.hpp"
#include "src/hpp/utils/util_placement.hpp"
#include "src/hpp/enums/enum_placement_approach.hpp"
#include <random>
#include "src/hpp/enums/enum_scheduling.hpp"
#include "src/hpp/configs/dfg_config.hpp"
#include "src/hpp/rng_singleton.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp> 
#include <cereal/types/base_class.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/base_class.hpp>
#include "src/hpp/utils/util_serialize.hpp"
#include <cereal/types/memory.hpp>

class OrderedDFG: public DFG {
private:
    std::vector<int> placement_order;
    std::vector<std::tuple<int, int, int>> traversal_edges;
    std::unordered_map<int,std::tuple<int,int,int>> node_to_traversal_edge;
    std::unordered_map<std::pair<int,int>, std::vector<Annotation>> edge_to_annotations;
    std::vector<int> asap;
    std::vector<int> alap;
    std::vector<int> tight;
    EnumPlacementApproach placement_approach;
    EnumGreedyType greedy_type;
    EnumScheduling scheduling;
    static constexpr std::vector<Annotation> empty_vector = {};
public:
    OrderedDFG()
        : DFG(),
        placement_order(),
        traversal_edges(),
        node_to_traversal_edge(),
        edge_to_annotations(),
        asap(),
        alap(),
        placement_approach(EnumPlacementApproach::DEFAULT),
        greedy_type(EnumGreedyType::GREEDY),
        scheduling(EnumScheduling::TIGHT)
    {}

    int get_tight_value_by_node(int node  ) const{
        return this->tight[node];
    };
    
    const std::vector<int>& get_tight() const{
        return this->tight;
    };
    
    bool is_tight_sched() const {
        return this->scheduling == EnumScheduling::TIGHT;
    };

    const std::unordered_map<std::pair<int,int>, std::vector<Annotation>>& get_edge_to_annotations() const {
        return edge_to_annotations;
    }

    bool are_structurally_isomorphic(const OrderedDFG& dfg) {
        return boost::isomorphism(this->graph, dfg.get_graph());
    }

    bool placement_ordering_equals(const std::shared_ptr<OrderedDFG>& dfg) {
        return this->placement_order == dfg->get_placement_order();
    }

    bool are_structurally_isomorphic(const std::shared_ptr<OrderedDFG>& dfg) {
        return boost::isomorphism(this->graph, dfg->get_graph());
    }


    void build_with_config(DFGConfig& dfg_config){
        auto generator = RNG::get();
    

        placement_approach = dfg_config.getPlacementApproach();
        greedy_type = dfg_config.getGreedyType();
        scheduling = dfg_config.getScheduling();
        
        if (scheduling == EnumScheduling::ASAP || scheduling == EnumScheduling::MOBILITY || scheduling ==  EnumScheduling::TIGHT) {
            asap = calc_scheduling(this->topo_order, this->in_vertices, this->out_vertices, this->root_nodes, true);
        }
    
        if (scheduling == EnumScheduling::ALAP || scheduling == EnumScheduling::MOBILITY || scheduling == EnumScheduling::TIGHT) {
            alap = calc_scheduling(this->topo_order, this->in_vertices, this->out_vertices, this->root_nodes, false);
        }

        if(scheduling == EnumScheduling::TIGHT){
            tight = this->balanced_schedule();
        }
    
        switch (placement_approach) {
            case EnumPlacementApproach::RANDOM:
                this->placement_order = this->vertices;
                std::shuffle(this->placement_order.begin(), this->placement_order.end(), generator);
                break;
    
            case EnumPlacementApproach::TOPOLOGICAL_ORDER:
                this->placement_order = this->get_topological_sort();
                this->topo_order = placement_order;
                break;
    
            default:
                bool use_stack = (placement_approach == EnumPlacementApproach::ZIGZAG) || (placement_approach == EnumPlacementApproach::DFS);
                bool start_from_outputs = dfg_config.getStartTraversalFromOutput();
                bool get_annotations = (EnumGreedyType::YOTT == greedy_type);
                int max_depth = dfg_config.getMaxDepth();
                
                std::tie(this->traversal_edges, this->edge_to_annotations) = std::move(do_traversal(
                    this->vertices, this->edges, this->in_vertices, this->out_vertices,
                    max_depth, use_stack, start_from_outputs, get_annotations, false, placement_approach == EnumPlacementApproach::ZIGZAG
                ));
                
    
                this->placement_order = gen_placement_order(this->traversal_edges);
                this->node_to_traversal_edge = gen_node_to_traversal_edge(this->traversal_edges);
    
        }
    
        this->placement_order.emplace_back(-1);
    
        #ifdef DEBUG
        std::cout << "[DEBUG] Final Placement Order: ";
        for (auto node : this->placement_order) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
        #endif
        std::cout << std::flush;
        assert(this->placement_order.size() == this->vertices.size() + 1);
    }
    void update_placement_order(const std::vector<int>& placement_order, int max_idx, const boost::property_map<DFGType, vertex_name_t>::type& node_to_name){
        std::unordered_map<std::string, int> name_to_node;
        for(auto node: this->vertices){
            auto name = boost::get(this->node_to_name, node);
            name_to_node[name] = node;
        }

        this->placement_order.clear();
        int idx = 0;
        for(auto& node: placement_order){
            if(idx > max_idx) break;
            auto name = node_to_name[node];
            this->placement_order.push_back(name_to_node.at(name));
            idx++;
        }

        assert(this->placement_order.size() == this->vertices.size());

        this->placement_order.push_back(-1);
    }

    

    void build_from_string(std::string& dot_string, std::string dfg_name, DFGConfig& dfg_config){
        this->build_dfg_from_string(dot_string, dfg_name, dfg_config);
        this->build_with_config(dfg_config);
    };

    EnumPlacementApproach get_placement_approach() const { return placement_approach; }
    std::string get_placement_approach_str() const {
        return enumPlacementToString(placement_approach);
    }

    std::string get_scheduling_str() const {
        return enumSchedulingToString(scheduling);
    }

    std::string get_greedy_type_str() const {
        return enumToString(greedy_type);

    }

    OrderedDFG(const std::string& dot_filename, DFGConfig& dfg_config);
    OrderedDFG(const OrderedDFG& other);
    const std::vector<int>& get_placement_order() const;
    int get_num_edges() const;
    int get_num_nodes() const;
    const std::vector<std::vector<int>>& get_edge_indices() const;
    const int& get_node_in_placement_order_by_idx_(const int& idx) const;
    const std::tuple<int, int, int>& get_traversal_edge_by_node(const int& node);
    bool is_order_topological() const;
    bool is_order_traversal() const;
    bool contains_annotations() const;
    const std::vector<Annotation>& get_annotation_by_edge(const std::pair<int,int>& edge) const;

    const std::string& get_dfg_name() const;
    int get_edge_to_annotations_size() const;
    int get_asap_value_by_node(int node  ) const;
    
    int get_alap_value_by_node(int node  ) const;

    const std::vector<int>& get_asap() const;
    const std::vector<int>& get_alap() const;
    bool is_asap_scheduling() const;
    bool is_alap_scheduling() const;
    EnumScheduling get_scheduling()const {
        return this->scheduling;
    }
    
    bool is_mobility_sched() const;
    bool is_greedy() const;
    bool is_greedy_type() const;
    bool use_annotations() const ;
    void print() const {
        std::cout << "OrderedDFG State Dump (" << get_dfg_name() << ")\n";
        std::cout << "================================\n";
        std::cout << "- Placement Order: ";
        for (const auto& node : placement_order) {
            std::cout << node << " ";
        }
        std::cout << "\n- Traversal Edges: " << traversal_edges.size() << "\n";
        std::cout << "- Node to Traversal Edge: " << node_to_traversal_edge.size() << "\n";
        std::cout << "- Edge to Annotations: " << edge_to_annotations.size() << "\n";
        std::cout << "- ASAP: ";
        for (const auto& val : asap) {
            std::cout << val << " ";
        }
        std::cout << "\n- ALAP: ";
        for (const auto& val : alap) {
            std::cout << val << " ";
        }
        std::cout << "\n- Placement Approach: " << get_placement_approach_str() 
                  << "\n- Greedy Type: " << get_greedy_type_str()
                  << "\n- Scheduling: " << get_scheduling_str() 
                  << "\n";
    }
    
    bool has_scheduling() const;


    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::base_class<DFG>(this));
        ar(placement_order);
        ar(traversal_edges);
        ar(node_to_traversal_edge);
        ar(edge_to_annotations);
        ar(asap);
        ar(alap);
        ar(tight);
        serialize_int_enum(ar, placement_approach);
        serialize_int_enum(ar, greedy_type);
        serialize_int_enum(ar, scheduling);
    }
    

};
#endif