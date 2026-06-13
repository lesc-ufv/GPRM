#ifndef MCTS_SELECTION_HPP
#define MCTS_SELECTION_HPP

#include "src/hpp/utils/min_max_stats.hpp"
#include <iostream>
#include <cmath>
#include <vector>


double uct_score(const double& mean_value, const double& constant, const int& father_visit_count, const int& child_visit_count);
double uct_score(const double& mean_value, const double& constant, const double& prior, const int& father_visit_count, const int& child_visit_count);
double smartmap_ucb_score(const int& father_visit_count,const int& child_visit_count, const double& c1, const double& c2, const double& child_prior, 
const double& discount, const double& reward, const double& mean_value, MinMaxStats& min_max_stats);

#endif