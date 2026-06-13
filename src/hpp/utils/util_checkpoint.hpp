#ifndef UTIL_CHECKPOINT_HPP
#define UTIL_CHECKPOINT_HPP

#include <iostream>
#include <vector>


inline void set_model_config_from_json(GlobalConfigVariant& variant, const json& js) {
    std::visit([&js](auto& config){
        using ConfigType = std::decay_t<decltype(config)>;
        config.set_config_from_json(js);
    }, variant);
}

#endif