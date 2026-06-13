#include "src/hpp/utils/min_max_stats.hpp"

MinMaxStats::MinMaxStats() {
    maximum = -std::numeric_limits<double>::infinity(); 
    minimum = std::numeric_limits<double>::infinity();
}


void MinMaxStats::update(double value) {
    if (value > this->maximum) {
        this->maximum = value; 
    }
    if (value < this->minimum) {
        this->minimum = value;
    }
}

double MinMaxStats::normalize(double value) {
    if (this->maximum > this->minimum) {
        return (value - this->minimum) / (this->maximum - this->minimum);
    }
    return value; 
}
