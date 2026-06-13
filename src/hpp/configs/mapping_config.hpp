#ifndef MAPPING_CONFIG_HPP
#define MAPPING_CONFIG_HPP

#include <iostream>
#include <sstream>

class MappingConfig {
private:
    bool use_backtracking;
    bool get_first_valid_solution;

public:
    MappingConfig() :
        use_backtracking(false),
        get_first_valid_solution(false) {}

    bool getUseBacktracking() const {
        return use_backtracking;
    }

    bool getFirstValidSolution() const {
        return get_first_valid_solution;
    }

    void setUseBacktracking(bool value) {
        use_backtracking = value;
    }

    void setFirstValidSolution(bool value) {
        get_first_valid_solution = value;
    }

    void print_ss(std::stringstream& ss) const {
        ss << "Mapping Configuration:\n"
           << "  Use Backtracking: " << (use_backtracking ? "true" : "false") << "\n"
           << "  Get First Valid Solution: " << (get_first_valid_solution ? "true" : "false") << "\n";
    }
};

#endif
