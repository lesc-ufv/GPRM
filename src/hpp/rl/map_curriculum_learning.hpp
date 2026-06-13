#ifndef MAP_CURRICULUM_LEARNING_HPP
#define MAP_CURRICULUM_LEARNING_HPP
#include <vector>
#include <unordered_map>
#include <tuple>
#include <queue>
#include <set>
#include <cassert>
#include <memory>
#include "src/hpp/configs/global_config.hpp"
#include "src/hpp/enums/enum_curriculum.hpp"
#include <utility>

typedef std::tuple<int,int,int,int,int,int,int> OrderedValues;
typedef std::tuple<int,int,int,int,int,int,int,int> OrderedValues2;
typedef std::tuple<double,int,int,int,int,double,int> OrderedValues3;

template<typename StateT, typename ModelConfig>
class MapCurriculumLearning {
    private:
        std::vector<std::shared_ptr<StateT>> curriculum_states; 
        int current_index = 0;
        double lambda = 1; 
        EnumCurriculum curriculum_learning;
        bool use_sequence_curriculum;
        bool add_all = false;
    public:
        MapCurriculumLearning() : current_index(0) {}
        MapCurriculumLearning(int current_index) : current_index(current_index) {}
        MapCurriculumLearning(const std::vector<std::shared_ptr<StateT>> train_states,
        GlobalConfig<ModelConfig>& global_config){
            this->build_curriculum_states(train_states, global_config);
        }
        int get_total_curriculum_steps() const {
            if (this->use_sequence_curriculum){
                return static_cast<int>(curriculum_states.size());
            }else{
                if(this->curriculum_learning == EnumCurriculum::DFG_SIZE){
                    int total_steps = 0;
                    int i = 0;
                    while (i < static_cast<int>(curriculum_states.size())) {
                        auto current_dfg_size = curriculum_states[i]->get_dfg_size();
                        while (i < static_cast<int>(curriculum_states.size()) && curriculum_states[i]->get_dfg_size() == current_dfg_size) {
                            i++;
                        }
                        total_steps++;
                    }
                    return total_steps - 1;
                }else{
                    throw std::runtime_error("Grouped Curriculum not implemented for current EnumCurriculum with id " + std::to_string(static_cast<int>(this->curriculum_learning)));
                }
            }
        }
        std::vector<std::shared_ptr<StateT>> fill_states_until_current_index(std::vector<std::shared_ptr<StateT>>& states) {
            int i = 0;
            std::vector<std::shared_ptr<StateT>> added_states;
            while (i < this->current_index) {
                // states.push_back(curriculum_states[i]);
                added_states.push_back(curriculum_states[i]);
                i++;
            }
            return added_states;
        }

        const int& get_current_index() const {
            return this->current_index;
        }
        std::vector<std::shared_ptr<StateT>> get_curriculum_states() const {
            return this->curriculum_states;
        }

        void build_curriculum_states(const std::vector<std::shared_ptr<StateT>> train_states,
            GlobalConfig<ModelConfig>& global_config){
            auto curriculum_type = global_config.getTrainConfig().getEnumCurriculum();
            this->use_sequence_curriculum = global_config.getTrainConfig().getUseSequenceCurriculum();
            this->curriculum_learning = curriculum_type;
            this->add_all = global_config.getTrainConfig().getAddAllCurriculumStates();
            if(curriculum_type == EnumCurriculum::DFG_SIZE){
                this->curriculum_states = create_curriculum_by_dfg_size(train_states);
            }else if (curriculum_type == EnumCurriculum::HUMAN){
                this->curriculum_states = create_gprm_curriculum_generic(
                    train_states,
                    [this](std::shared_ptr<StateT> s) { return this->get_tuple_v1(s); },
                    {1, 2}
                );
            }else if (curriculum_type == EnumCurriculum::HUMANV2){
                this->curriculum_states = create_gprm_curriculum_generic(
                    train_states,
                    [this](std::shared_ptr<StateT> s) { return this->get_tuple_v2(s); },
                    {1, 2, 7}
                );
            }else if (curriculum_type == EnumCurriculum::HUMANV2_OCCUPANCY){
                this->lambda = global_config.getTrainConfig().getLambdaHumanComplexity();
                this->curriculum_states = create_gprm_curriculum_generic(
                    train_states,
                    [this](std::shared_ptr<StateT> s) { return this->get_tuple_v3(s); },
                    {0, 5}
                );
            }else if (curriculum_type == EnumCurriculum::MAP_COMPLEXITY){
                this->curriculum_states = create_gprm_curriculum_generic(
                    train_states,
                    [this](std::shared_ptr<StateT> s) { return this->get_tuple_v4(s); },
                    {}
                );
            }else if (curriculum_type == EnumCurriculum::MAP_COMPLEXITYV2){
                this->lambda = global_config.getTrainConfig().getLambdaComplexity();
                this->curriculum_states = create_gprm_curriculum_generic(
                    train_states,
                    [this](std::shared_ptr<StateT> s) { return this->get_tuple_v5(s); },
                    {}
                );
            }else if (curriculum_type == EnumCurriculum::DFG_COMPLEXITY){
                this->curriculum_states = create_gprm_curriculum_generic(
                    train_states,
                    [this](std::shared_ptr<StateT> s) { return this->get_tuple_v6(s); },
                    {}
                );
            }else{
                throw std::runtime_error("Invalid curriculum type");
            }
        }
        
        std::vector<std::shared_ptr<StateT>> create_curriculum_by_dfg_size(const std::vector<std::shared_ptr<StateT>>& train_states) {
            std::vector<std::shared_ptr<StateT>> sorted_states = train_states;
        
            std::sort(sorted_states.begin(), sorted_states.end(), 
                [](const std::shared_ptr<StateT>& a, const std::shared_ptr<StateT>& b) {
                    return a->get_dfg_size() < b->get_dfg_size();
                });
            return sorted_states;
        }

        template <typename Tuple, std::size_t... Is>
        bool tuple_compare(const Tuple& t1, const Tuple& t2, const std::vector<size_t>& descending_indices, std::index_sequence<Is...>) {
            bool result = false;

            auto compare = [&](auto I) -> std::optional<bool> {
                constexpr std::size_t i = decltype(I)::value;
                const auto& v1 = std::get<i>(t1);
                const auto& v2 = std::get<i>(t2);

        #ifdef DEBUG
                std::cout << "Comparing index " << i << ": v1 = " << v1 << ", v2 = " << v2 << std::endl;
        #endif

                if (v1 == v2) return std::nullopt;

                bool is_desc = std::find(descending_indices.begin(), descending_indices.end(), i) != descending_indices.end();

        #ifdef DEBUG
                std::cout << "Direction: " << (is_desc ? "descending" : "ascending") << std::endl;
                std::cout << "Result: " << (is_desc ? (v1 > v2) : (v1 < v2)) << std::endl << std::flush;
        #endif

                return is_desc ? (v1 > v2) : (v1 < v2);
            };

            bool decided = false;
            (
                ([&]() {
                    if (!decided) {
                        auto cmp = compare(std::integral_constant<std::size_t, Is>{});
                        if (cmp.has_value()) {
                            result = *cmp;
                            decided = true;
        #ifdef DEBUG
                            std::cout << "Decision made at index " << Is << ": " << std::boolalpha << result << std::endl;
        #endif
                        }
                    }
                }()), ...
            );

            return result;
        }

        
        template<typename TupleGetter>
        auto create_gprm_curriculum_generic(
            const std::vector<std::shared_ptr<StateT>>& train_states,
            TupleGetter get_tuple,
            const std::vector<size_t>& descending_indices
        ) -> std::vector<std::shared_ptr<StateT>> {

            std::vector<std::shared_ptr<StateT>> sorted_states = train_states;
            std::sort(sorted_states.begin(), sorted_states.end(),
                [this, &get_tuple, &descending_indices](const auto& a, const auto& b) {
                    const auto& t1 = get_tuple(a);
                    using TupleType = std::remove_const_t<std::remove_reference_t<decltype(t1)>>;
                    const auto& t2 = get_tuple(b);
                    return tuple_compare(t1, t2, descending_indices,
                        std::make_index_sequence<std::tuple_size<TupleType>::value>{}
                    );
                });
        return sorted_states;    
    }                

        OrderedValues get_tuple_v1(std::shared_ptr<StateT> state){
            return std::make_tuple(
                state->get_II(),
                state->get_number_of_spatial_PEs(), // Desc
                state->get_num_spatial_edges(), // Desc
                state->get_num_dfg_nodes(),
                state->get_num_dfg_edges(),
                state->get_num_convergent_edges(),
                state->get_total_rec_paths()
            );
        }
        
        OrderedValues2 get_tuple_v2(std::shared_ptr<StateT> state){
            return std::make_tuple(
                state->get_MII(),
                state->get_number_of_spatial_PEs(), //Desc
                state->get_num_spatial_edges(), //Desc
                state->get_num_dfg_nodes(),
                state->get_num_dfg_edges(),
                state->get_num_convergent_edges(),
                state->get_total_rec_paths(),
                state->get_II() //Desc
            );
        }

        OrderedValues3 get_tuple_v3(std::shared_ptr<StateT> state){
            auto occupancy = state->get_min_pe_occupancy();
            return std::make_tuple(
                state->get_MII_II_ratio(),
                state->get_num_dfg_nodes(),
                state->get_num_dfg_edges(),
                state->get_num_convergent_edges(),
                state->get_total_rec_paths(),
                occupancy  > this->lambda ? -occupancy : occupancy,
                state->get_num_spatial_edges()
            );
        }

    std::tuple<double> get_tuple_v4(std::shared_ptr<StateT> state){
        return std::make_tuple(
            state->get_complexity()
        );
    }

    std::tuple<double> get_tuple_v5(std::shared_ptr<StateT> state){
        return std::make_tuple(
            state->get_complexity_v2(this->lambda)
        );
    }
    
    std::tuple<double> get_tuple_v6(std::shared_ptr<StateT> state){
        return std::make_tuple(
            state->get_dfg_complexity()
        );
    }
    std::vector<std::shared_ptr<StateT>> add_next_states(std::vector<std::shared_ptr<StateT>>& cur_train_states, bool upd_idx = false){
        if (current_index >= static_cast<int>(curriculum_states.size())) {
            return {}; // No more states to add
        }
        std::vector<std::shared_ptr<StateT>> states_to_add;
        if (add_all){
            while (current_index < static_cast<int>(curriculum_states.size())) {
                std::cout << "[INFO] Adding state with index " << current_index << " to the train states.\n";
                auto next_state = curriculum_states[current_index];
                cur_train_states.push_back(next_state);
                states_to_add.push_back(next_state);
                if (!upd_idx) current_index++;
            }
            return states_to_add;
        }
        
        if (this->use_sequence_curriculum){
            std::cout << "[INFO] Adding state with index " << current_index << " to the train states.\n";
            auto next_state = curriculum_states[current_index];
            cur_train_states.push_back(next_state);
            if (!upd_idx) current_index++;
            #ifdef DEBUG
                std::cout << "Current index: " << current_index << std::endl;
            #endif
            states_to_add.push_back(next_state);
        }else if(this->curriculum_learning == EnumCurriculum::DFG_SIZE){
            auto next_state = curriculum_states[current_index];
            auto cur_dfg_size = next_state->get_dfg_size();
            while(cur_dfg_size == next_state->get_dfg_size()){
                cur_train_states.push_back(next_state);
                states_to_add.push_back(next_state);
                if (!upd_idx) {
                    current_index++;
                    if(current_index >= static_cast<int>(curriculum_states.size())){
                        break;
                    }
                }
                next_state = curriculum_states[current_index];
            }
        }else{
            throw std::runtime_error("Grouped Curriculum not implemented for current EnumCurriculum with id " + std::to_string(static_cast<int>(this->curriculum_learning)));
        }
        return states_to_add;
    }

    template<class Archive>
    void serialize(Archive& ar) {
        ar(current_index);
        ar(use_sequence_curriculum);
        ar(lambda);
        ar(add_all);
    }
};
#endif  