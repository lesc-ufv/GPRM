#ifndef BUFFER_CONFIG_HPP
#define BUFFER_CONFIG_HPP

#include <iostream>
#include <vector>
#include <sstream>
#include <utility>
#include <iomanip>

class BufferConfig {
private:
    int buffer_size;
    bool reanalyze;
    bool PER;
    double PER_alpha;
    double alpha_reduce;
    bool use_ISW;
    double ISW_beta;
    bool augment_data;
    bool norm_isw_with_max;
    std::vector<int> degrees;
    std::vector<std::pair<int, int>> shifts, flips;
    bool norm_advantages, norm_returns;
    bool norm_rewards_smartmap_pt_buffer, clear_buffer_after_training;

public:
    BufferConfig()
        : buffer_size(0), reanalyze(false), PER(false), PER_alpha(0.0),
          alpha_reduce(0.0), use_ISW(false), ISW_beta(0.0), augment_data(false),
          norm_isw_with_max(false), norm_advantages(false), norm_returns(false),
          norm_rewards_smartmap_pt_buffer(false), clear_buffer_after_training(false) {}
    int getBufferSize() const { return buffer_size; }
    bool getUseReanalyze() const { return reanalyze; }
    bool getPER() const { return PER; }
    double getPERAlpha() const { return PER_alpha; }
    double getAlphaReduce() const { return alpha_reduce; }
    double getISWBeta() const { return ISW_beta; }
    bool getUseISW() const { return use_ISW; }
    bool getNormISWWithMax() const { return norm_isw_with_max; }
    bool getNormAdvantages() const { return norm_advantages; }
    bool getNormReturns() const { return norm_returns; }
    bool getNormRewardsPTbuffer() const { return norm_rewards_smartmap_pt_buffer; }
    bool getAugmentData() const { return augment_data; }
    bool getClearBufferAfterTraining() const {return clear_buffer_after_training;}
    void setClearBufferAfterTraining(bool value)  {clear_buffer_after_training = value;}

    const std::vector<int>& getDegrees() const { return degrees; }
    const std::vector<std::pair<int, int>>& getShifts() const { return shifts; }
    const std::vector<std::pair<int, int>>& getFlips() const { return flips; }

    void setBufferSize(int value) { 
        assert(value != 0);
        buffer_size = value; }
    void setUseReanalyze(bool value) { reanalyze = value; }
    void setPER(bool value) { PER = value; }
    void setPERAlpha(double value) { PER_alpha = value; }
    void setAlphaReduce(double value) { alpha_reduce = value; }
    void setISWBeta(double value) { ISW_beta = value; }
    void setUseISW(bool value) { use_ISW = value; }
    void setNormISWWithMax(bool value) { norm_isw_with_max = value; }
    void setNormAdvantages(bool value) { norm_advantages = value; }
    void setNormReturns(bool value) { norm_returns = value; }

    void setNormRewardsPTbuffer(bool value) { norm_rewards_smartmap_pt_buffer = value; }
    void setAugmentData(bool value) { augment_data = value; }
    void setDegrees(const std::vector<int>& values) { degrees = values; }
    void setShifts(const std::vector<std::pair<int, int>>& values) { shifts = values; }
    void setFlips(const std::vector<std::pair<int, int>>& values) { flips = values; }

    void print_ss(std::stringstream& ss) const {
        ss << "\nBuffer Configuration:\n"
           << "\tBuffer Size: " << buffer_size << "\n"
           << "\tReanalyze: " << (reanalyze ? "true" : "false") << "\n"
           << "\tPER: " << (PER ? "true" : "false") << "\n"
           << "\tPER Alpha: " << PER_alpha << "\n"
           << "\tAlpha Reduce: " << alpha_reduce << "\n"
           << "\tUse ISW: " << (use_ISW ? "true" : "false") << "\n"
           << "\tISW Beta: " << ISW_beta << "\n"
           << "\tNormalize ISW with Max: " << (norm_isw_with_max ? "true" : "false") << "\n"
           << "\tNormalize Advantages: " << (norm_advantages ? "true" : "false") << "\n"
           << "\tNormalize Returns: " << (norm_returns ? "true" : "false") << "\n"
           << "\tNormalize Rewards PT Buffer: " << (norm_rewards_smartmap_pt_buffer ? "true" : "false") << "\n"
           << "\tClear buffer after training: " << (clear_buffer_after_training ? "true" : "false") << "\n"
           << "\tAugment Data: " << (augment_data ? "true" : "false") << "\n"
           << "\tDegrees: ";
        for (int d : degrees) ss << d << " ";
        ss << "\n  Shifts: ";
        for (const auto& s : shifts) ss << "(" << s.first << "," << s.second << ") ";
        ss << "\n  Flips: ";
        for (const auto& f : flips) ss << "(" << f.first << "," << f.second << ") ";
        ss << "\n";
    }

    void print() const {
        std::stringstream ss;
        print_ss(ss);
        std::cout << ss.str() << std::endl;
    }
};

#endif
