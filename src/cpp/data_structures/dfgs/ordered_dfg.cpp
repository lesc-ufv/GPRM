#include "src/hpp/data_structures/dfgs/ordered_dfg.hpp"

OrderedDFG::OrderedDFG(const std::string &dot_filename, DFGConfig& dfg_config) : DFG(dot_filename, dfg_config)
{
   this->build_with_config(dfg_config);
}



OrderedDFG::OrderedDFG(const OrderedDFG& other)
    : DFG(other),
      placement_order(other.placement_order),
      traversal_edges(other.traversal_edges),
      node_to_traversal_edge(other.node_to_traversal_edge),
      edge_to_annotations(other.edge_to_annotations),
      asap(other.asap),
      alap(other.alap),
      tight(other.tight)
    { }


const std::vector<int>& OrderedDFG::get_placement_order() const{
    return this->placement_order;
}

const int& OrderedDFG::get_node_in_placement_order_by_idx_(const int& idx) const{
    return this->placement_order[idx];
}


int OrderedDFG::get_num_edges() const{
    return this->get_edges().size();
}

int OrderedDFG::get_num_nodes() const{
    return this->vertices.size();
}

const std::vector<std::vector<int>>& OrderedDFG::get_edge_indices() const{
    return this->edge_indices;
}



const std::tuple<int, int, int>&  OrderedDFG::get_traversal_edge_by_node(const int& node){
    return this->node_to_traversal_edge.at(node);
}


bool OrderedDFG::is_order_topological() const{
    return this->placement_approach == EnumPlacementApproach::TOPOLOGICAL_ORDER;
}
bool OrderedDFG::is_order_traversal() const {
    return (this->placement_approach == EnumPlacementApproach::ZIGZAG) ||
    (this->placement_approach == EnumPlacementApproach::DFS) ||
    (this->placement_approach == EnumPlacementApproach::BFS);
}


bool OrderedDFG::contains_annotations() const{
    return !this->edge_to_annotations.empty();
}

const std::vector<Annotation>& OrderedDFG::get_annotation_by_edge(const std::pair<int,int>& edge) const{
    if (this->edge_to_annotations.find(edge) == this->edge_to_annotations.end()) return this->empty_vector;
    else return this->edge_to_annotations.at(edge);
}

const std::string& OrderedDFG::get_dfg_name() const{
    return this->dfg_name;
}

int OrderedDFG::get_edge_to_annotations_size() const{
    return this->edge_to_annotations.size();
};

bool OrderedDFG::is_asap_scheduling() const{
    return this->scheduling == EnumScheduling::ASAP;
};

bool OrderedDFG::is_alap_scheduling() const{
    return this->scheduling == EnumScheduling::ALAP;
};

bool OrderedDFG::is_mobility_sched() const{
    return this->scheduling == EnumScheduling::MOBILITY;
};

bool OrderedDFG::has_scheduling() const{
    return this->scheduling != EnumScheduling::NONE;
};

bool OrderedDFG::is_greedy() const {
    return this->greedy_type != EnumGreedyType::NONE;
};

bool OrderedDFG::use_annotations() const{
    return  this->greedy_type == EnumGreedyType::YOTT;
}

bool OrderedDFG::is_greedy_type() const{
    return this->greedy_type == EnumGreedyType::GREEDY;
};


int OrderedDFG::get_asap_value_by_node(int node  ) const{
    return this->asap[node];
};

int OrderedDFG::get_alap_value_by_node(int node  ) const{
    return this->alap[node];
};

const std::vector<int>& OrderedDFG::get_asap() const{
    return this->asap;
};

const std::vector<int>& OrderedDFG::get_alap() const{
    return this->alap;
};

