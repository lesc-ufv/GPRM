#ifndef RT_SCHEDULER_HPP
#define RT_SCHEDULER_HPP

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
// #define DEBUG

class RTScheduler {
private:
    std::unordered_map<int, int> node_to_set;
    std::unordered_map<int, std::unordered_set<int>> set_to_nodes;

    std::unordered_map<int, int> node_to_scheduled_time_slice;
    int min_scheduled_time_slice;
    int num_sets;

public:
    RTScheduler() {
        min_scheduled_time_slice = std::numeric_limits<int>::max();
        num_sets = 0;
    }

    template<class Archive>
    void serialize(Archive & ar) {
        ar(
            node_to_set,
            set_to_nodes,
            node_to_scheduled_time_slice,
            min_scheduled_time_slice,
            num_sets
        );
    }
    
    RTScheduler(const RTScheduler&) = default;
    RTScheduler(RTScheduler&&) noexcept = default;
    RTScheduler& operator=(const RTScheduler&) = default;
    RTScheduler& operator=(RTScheduler&&) noexcept = default;

    const std::unordered_map<int, int>& get_node_to_scheduled_time_slice_() const {
        return this->node_to_scheduled_time_slice;
    }

    int merge_sets(int cur_set, int neigh_set) {

        int cur_set_size = this->set_to_nodes.at(cur_set).size();
        int neigh_set_size = this->set_to_nodes.at(neigh_set).size();
        int new_set ;

#ifdef DEBUG
        std::cout << "  cur_set size: " << cur_set_size << ", neigh_set size: " << neigh_set_size << "\n";
#endif

        if (cur_set_size > neigh_set_size) {
#ifdef DEBUG
        std::cout << "[merge_sets] Merging set " << neigh_set << " into " << cur_set << "\n";
#endif

            for (auto& node : this->set_to_nodes.at(neigh_set)) {
                this->node_to_set[node] = cur_set;
                this->set_to_nodes.at(cur_set).insert(node);
            }
            this->set_to_nodes.erase(neigh_set);
            new_set = cur_set;
        } else {
            #ifdef DEBUG
            std::cout << "[merge_sets] Merging set " << cur_set << " into " << neigh_set << "\n";
    #endif

            for (auto& node : this->set_to_nodes.at(cur_set)) {
                this->node_to_set[node] = neigh_set;
                this->set_to_nodes.at(neigh_set).insert(node);
            }
            this->set_to_nodes.erase(cur_set);
            new_set = neigh_set;
        }

        return new_set;
    }

    void update_node_to_scheduled_time_slice_by_set(int neigh_set, int sched_diff) {
#ifdef DEBUG
        std::cout << "[update_node_to_scheduled_time_slice_by_set] Updating set " << neigh_set << " with diff " << sched_diff << "\n";
        std::cout << "  Before update:\n";
#endif
        for (auto& node : this->set_to_nodes.at(neigh_set)) {
#ifdef DEBUG
            std::cout << "    Node " << node << " old time: " << this->node_to_scheduled_time_slice[node];
#endif
            this->node_to_scheduled_time_slice[node] += sched_diff;
            this->min_scheduled_time_slice = std::min(this->min_scheduled_time_slice, this->node_to_scheduled_time_slice[node]);
#ifdef DEBUG
            std::cout << ", new time: " << this->node_to_scheduled_time_slice[node] << "\n";
#endif
        }
    }

    bool schedule_by_neigh(int node_to_schedule, int& cur_scheduled_time_slice, int& cur_set,
        const std::unordered_set<int>& neighs,
        const std::unordered_map<int, int>& node_to_pe,
        bool input,
        const std::unordered_map<std::pair<int, int>, std::vector<int>>& PEs_to_routing) {

        auto pe = node_to_pe.at(node_to_schedule);
        bool scheduling_is_valid = true;

#ifdef DEBUG
        std::cout << "[schedule_by_neigh] Checking neighbors for node " << node_to_schedule << " | PE: " << pe << " | input: " << input << "\n";
#endif

        for (auto& neigh : neighs) {
            
        // #ifdef DEBUG
        // std::cout << " \nNeigh : " << neigh << "\n";
        // #endif
            if (node_to_pe.find(neigh) != node_to_pe.end()) {
                auto neigh_pe = node_to_pe.at(neigh);
                auto rout_pair = input ? std::make_pair(neigh_pe, pe) : std::make_pair(pe, neigh_pe);
                auto routing_path_size = PEs_to_routing.at(rout_pair).size() - 1;
                auto neigh_scheduled_time_slice = node_to_scheduled_time_slice.at(neigh);
                int neigh_set = this->node_to_set.at(neigh);

#ifdef DEBUG
                std::cout << " Neigh sched time: " << neigh_scheduled_time_slice << " ↔ Neighbor " << neigh << ", path size = " << routing_path_size << "\n";
#endif

                if (cur_scheduled_time_slice != std::numeric_limits<int>::min()) {
                    int sched_diff = (cur_scheduled_time_slice + (input ? -routing_path_size : routing_path_size)) - neigh_scheduled_time_slice;
#ifdef DEBUG
                    std::cout << "    sched_diff = " << sched_diff << ", cur_set = " << cur_set << ", neigh_set = " << neigh_set << "\n";
#endif
                    if (neigh_set != cur_set) {
                        if(sched_diff != 0){
                            this->update_node_to_scheduled_time_slice_by_set(neigh_set, sched_diff);
                        }
                        cur_set = this->merge_sets(cur_set, neigh_set);
                    } else {
                        scheduling_is_valid = sched_diff == 0;
                        if(!scheduling_is_valid && !input){
                            node_to_scheduled_time_slice[neigh] = cur_scheduled_time_slice + routing_path_size;
                        }
                        #ifdef DEBUG
                        std::cout << "[DEBUG] neigh_scheduled_time_slice: " << neigh_scheduled_time_slice 
                                  << ", routing_path_size: " << routing_path_size 
                                  << ", cur_scheduled_time_slice: " << cur_scheduled_time_slice 
                                  << ", sched_diff: " << sched_diff 
                                  << ", scheduling_is_valid: " << scheduling_is_valid 
                                  << std::endl;
                        #endif
                    }
                    
                } else {
                    int to_schedule_time_slice = input ? neigh_scheduled_time_slice + routing_path_size : neigh_scheduled_time_slice - routing_path_size;
                    cur_set = neigh_set;
                    cur_scheduled_time_slice = to_schedule_time_slice;
#ifdef DEBUG
                    std::cout << "    Initial scheduled_time set to " << cur_scheduled_time_slice << "\n";
                    std::cout << "    Initial set set to " << cur_set << "\n";

#endif
                }
            }
        }
        #ifdef DEBUG
        std::cout << "Scheduling is valid: " << scheduling_is_valid << "\n";
        #endif
        return scheduling_is_valid;
    }

    void update_schedule_time_with_min_time() {
#ifdef DEBUG
        std::cout << "[update_schedule_time_with_min_time] Adjusting all slices by min = " << this->min_scheduled_time_slice << "\n";
#endif
        for (auto& [node, time] : this->node_to_scheduled_time_slice) {
#ifdef DEBUG
            std::cout << "  Node " << node << " time before: " << time;
#endif
            this->node_to_scheduled_time_slice[node] -= this->min_scheduled_time_slice;
#ifdef DEBUG
            std::cout << ", after: " << this->node_to_scheduled_time_slice[node] << "\n";
#endif
        }
        this->min_scheduled_time_slice = 0;
    }

    void update_schedule_time_with_max_delay( const std::vector<int>& topo_order, 
        const std::unordered_map<int, std::unordered_set<int>>& node_to_in,
        const std::unordered_map<std::pair<int, int>, std::vector<int>>& PEs_to_routing,
        const std::unordered_map<int, int>& node_to_pe) {
        for(auto& node: topo_order){
            if(node_to_pe.find(node) == node_to_pe.end()){
                continue;
            }
            int max_sched_time = -1;
            auto pe = node_to_pe.at(node);
            for(auto& in_node: node_to_in.at(node)){
                if(node_to_pe.find(in_node) == node_to_pe.end()){
                    continue;
                }
                auto in_pe = node_to_pe.at(in_node);
                auto delay = static_cast<int>(PEs_to_routing.at({in_pe, pe}).size()) - 1;
                auto in_node_sched_time = this->node_to_scheduled_time_slice.at(in_node);
                max_sched_time = std::max(max_sched_time, in_node_sched_time + delay);
            }
            this->node_to_scheduled_time_slice[node] = max_sched_time;
        }
    }

    bool schedule(int node,
        const std::unordered_map<int, std::unordered_set<int>>& node_to_in,
        const std::unordered_map<int, std::unordered_set<int>>& node_to_out,
        const std::unordered_map<std::pair<int, int>, std::vector<int>>& PEs_to_routing,
        const std::unordered_map<int, int>& node_to_pe) {


#ifdef DEBUG
        std::cout << "\n[schedule] Scheduling node " << node << "\n";
#endif
        int scheduled_time = std::numeric_limits<int>::min();
        int cur_set = std::numeric_limits<int>::min();
        bool scheduling_is_valid = true;

        scheduling_is_valid &= this->schedule_by_neigh(node, scheduled_time, cur_set, node_to_in.at(node), node_to_pe, true, PEs_to_routing);
        scheduling_is_valid &= this->schedule_by_neigh(node, scheduled_time, cur_set, node_to_out.at(node), node_to_pe, false, PEs_to_routing);

        if (scheduled_time == std::numeric_limits<int>::min()) {
            this->node_to_scheduled_time_slice[node] = 0;
            this->node_to_set[node] = this->num_sets;
            set_to_nodes[this->num_sets].insert(node);
            this->num_sets++;
            this->min_scheduled_time_slice = std::min(this->min_scheduled_time_slice, 0);

#ifdef DEBUG
            std::cout << "  Node scheduled as root, time slice 0, set = " << this->num_sets - 1 << "\n";
#endif
        } else {
            this->node_to_scheduled_time_slice[node] = scheduled_time;
            this->node_to_set[node] = cur_set;
            set_to_nodes[cur_set].insert(node);
            this->min_scheduled_time_slice = std::min(this->min_scheduled_time_slice, scheduled_time);

#ifdef DEBUG
            std::cout << "  Node scheduled at time slice " << scheduled_time << ", set = " << cur_set << "\n";
#endif
        }

#ifdef DEBUG
        std::cout << "[schedule] Final assignment for node " << node << ":\n";
        std::cout << "    Time slice: " << this->node_to_scheduled_time_slice[node] 
                  << ", Set ID: " << this->node_to_set[node] << "\n";
#endif
        return scheduling_is_valid;
    }

    int get_time_slice(int node) const {
        return node_to_scheduled_time_slice.at(node);
    }
};

#endif  // RT_SCHEDULER_HPP
