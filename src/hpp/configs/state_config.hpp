#ifndef STATE_CONFIG_HPP
#define STATE_CONFIG_HPP

#include <sstream>
#include "src/hpp/enums/enum_end_state.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

class StateConfig {
private:
    bool use_used_pe_mask;
    bool use_congestion_edge_mask;
    bool use_symmetric_mask;
    bool override_base_state;
    bool use_rt_scheduler;
    EnumEndState end_state;
    bool use_spatio_temporal_input;
    bool use_sync_routing;
    bool use_sched_mask = true;


public:
    StateConfig()
        : use_used_pe_mask(false),
          use_congestion_edge_mask(false),
          use_symmetric_mask(false),
          override_base_state(false),
          use_rt_scheduler(false),
          end_state(EnumEndState::DEFAULT),
          use_spatio_temporal_input(false),
          use_sync_routing(false) {}
        
    void set_config_from_json(const json& js) {
        use_used_pe_mask = js.value("use_pe_mask", use_used_pe_mask);
        use_congestion_edge_mask = js.value("use_edge_mask", use_congestion_edge_mask);
        use_symmetric_mask = js.value("use_sym_mask", use_symmetric_mask);
        override_base_state = js.value("override_base_state", override_base_state);
        use_rt_scheduler = js.value("use_rt_scheduler", use_rt_scheduler);
        end_state = static_cast<EnumEndState>(js.value("end_state_id", static_cast<int>(end_state)));
        use_spatio_temporal_input = js.value("use_spatio_temporal_input", use_spatio_temporal_input);
        use_sync_routing = js.value("use_sync_routing", use_sync_routing);
      }
    
    bool getUseSchedMask() const { return use_sched_mask; }
    void setUseSchedMask(bool value) { use_sched_mask = value; }

    bool getUseSyncRouting() const { return use_sync_routing;}
    void setUseSyncRouting(bool value) { use_sync_routing = value;}
    
    EnumEndState getEndState() const { return end_state; }
    void setEndState(EnumEndState value) { end_state = value; }

    bool getUseUsedPeMask() const { return use_used_pe_mask; }
    void setUseUsedPeMask(bool value) { use_used_pe_mask = value; }

    bool getUseRTScheduler() const { return use_rt_scheduler; }
    void setUseRTScheduler(bool value) { use_rt_scheduler = value; }

    bool getUseSpatioTemporalInput() const { return use_spatio_temporal_input; }
    void setUseSpatioTemporalInput(bool value) { use_spatio_temporal_input = value; }

    bool getUseCongestionEdgeMask() const { return use_congestion_edge_mask; }
    void setUseCongestionEdgeMask(bool value) { use_congestion_edge_mask = value; }

    bool getUseSymmetricMask() const { return use_symmetric_mask; }
    void setUseSymmetricMask(bool value) { use_symmetric_mask = value; }

    bool getOverrideBaseState() const { return override_base_state; }
    void setOverrideBaseState(bool value) { override_base_state = value; }

    void print_ss(std::stringstream& ss) const {
        ss << "\nState Configuration:\n"
           << "\tUse Used PE Mask: " << (use_used_pe_mask ? "true" : "false") << "\n"
           << "\tUse Congestion Edge Mask: " << (use_congestion_edge_mask ? "true" : "false") << "\n"
           << "\tUse Symmetric Mask: " << (use_symmetric_mask ? "true" : "false") << "\n"
              << "\tUse RT Scheduler: " << (use_rt_scheduler ? "true" : "false") << "\n"

              << "\tEnd State: " << enumEndStateToString(end_state) << "\n"
           << "\tOverride Base State: " << (override_base_state ? "true" : "false") << "\n"
           << "Use spatio temporal input: " << (use_spatio_temporal_input ? "true" : "false");
    }
};

#endif
