#ifndef MIN_MAX_STATS_HPP
#define MIN_MAX_STATS_HPP

#include <limits>
#include <cassert>
class MinMaxStats {
private:
    double maximum;
    double minimum; 
public:
    MinMaxStats();

    void update(double value);

    double normalize(double value);

    double normalize2(double value) {
        double ret;
        if (this->maximum > this->minimum) {
            ret = (value - this->minimum) / (this->maximum - this->minimum);
        }else{
            ret = 1.0; 
        }
        // assert(1 - ret >= -1e-9);
        return ret;
    }

    template<typename Archive>
    void serialize(Archive& ar) {
        ar(maximum, minimum);
    }

    double get_min() { return minimum; }
    double get_max() { return maximum; }
};


#endif