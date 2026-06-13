#include "src/hpp/utils/cgra/util_cgra.hpp"

template <typename CGRAT, typename... Args>
std::shared_ptr<CGRAT> initialize_homogeneous_cgra(const std::unordered_set<EnumInterconnectionStyles>& interconnection_styles, const int& II,
                                                    const std::pair<int,int>& arch_dimensions, 
                                                    const std::unordered_set<EnumFunctionalities>& functionalities,const Args&... args ){
    auto pe_to_functionalities = init_homogeneous_pe_to_functionalities(arch_dimensions.first * arch_dimensions.second * II, functionalities); 
    std::shared_ptr<CGRAT> cgra;
    cgra = std::make_shared<CGRAT>(II, arch_dimensions, pe_to_functionalities, interconnection_styles, args...);
    return cgra;
}


template <typename CGRAT, typename... Args>
std::vector<std::shared_ptr<CGRAT>> initialize_homogeneous_cgras(const std::vector<std::unordered_set<EnumInterconnectionStyles>>& interconnection_styles_list, 
                                                    const int& II, const std::pair<int,int>& arch_dimensions,
                                            const std::vector<std::unordered_set<EnumFunctionalities>>& functionalities_list, const Args&... args ){
    std::vector<std::shared_ptr<CGRAT>> cgras;
    for (auto& functionalities: functionalities_list){
        for(auto interconnection_styles: interconnection_styles_list){
            cgras.emplace_back(
                initialize_homogeneous_cgra<CGRAT>(interconnection_styles, II, arch_dimensions, functionalities, args...)
                );
        }
    }
    return cgras;
}