#include "src/hpp/utils/mcts.hpp"

double uct_score(const double& mean_value, const double& constant, const int& father_visit_count, const int& child_visit_count)
{   
    if (child_visit_count == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return mean_value + 2*constant*sqrt((2*log(father_visit_count != 0 ? father_visit_count : 1 ))/(child_visit_count != 0 ? child_visit_count : 1));
}

double uct_score(const double& mean_value, const double& constant, const double& prior, const int& father_visit_count, const int& child_visit_count)
{   
    if (child_visit_count == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return mean_value + prior*2*constant*sqrt((2*log(father_visit_count != 0 ? father_visit_count : 1 ))/( child_visit_count));
}

// double uct_score(const double& mean_value, const double& constant, const int& father_visit_count, const int& child_visit_count)
// {   
//     if (child_visit_count == 0) {
//         return std::numeric_limits<double>::infinity();
//     }
//         return mean_value + constant * sqrt((2 * log(father_visit_count)) / child_visit_count);
// }

double smartmap_ucb_score(
    const int& father_visit_count,
    const int& child_visit_count,
    const double& c1,
    const double& c2,
    const double& child_prior, 
    const double& discount,
    const double& reward,
    const double& mean_value,
    MinMaxStats& min_max_stats)
{
    double pb_c = log((father_visit_count + c2 + 1) / c2) + c1;
    pb_c *= sqrt(father_visit_count) / (child_visit_count + 1);

#ifdef DEBUG
    std::cout << "[DEBUG] pb_c: " << pb_c << std::endl;
#endif

    double prior_score = pb_c * child_prior;

#ifdef DEBUG
    std::cout << "[DEBUG] prior_score: " << prior_score << std::endl;
#endif

    double value_score = 0.0;
    if (child_visit_count > 0) {
        value_score = min_max_stats.normalize(reward + discount * mean_value);

#ifdef DEBUG
        std::cout << "[DEBUG] reward: " << reward << std::endl;
        std::cout << "[DEBUG] discount: " << discount << std::endl;
        std::cout << "[DEBUG] mean_value: " << mean_value << std::endl;
        std::cout << "[DEBUG] value_score: " << value_score << std::endl;
#endif
    }

#ifdef DEBUG
    std::cout << "[DEBUG] total_score (prior + value): " 
              << prior_score + value_score << std::endl;
#endif

    return prior_score + value_score;
}


