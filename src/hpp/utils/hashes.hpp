#ifndef HASHES_HPP
#define HASHES_HPP

#include <functional>
#include <utility>
#include <tuple>
#include "src/hpp/utils/hashes.hpp"

namespace std {
    template <typename T1, typename T2>
    struct hash<std::pair<T1, T2>> {
        size_t operator()(const std::pair<T1, T2>& p) const {
            auto hash1 = std::hash<T1>{}(p.first);
            auto hash2 = std::hash<T2>{}(p.second);
            return hash1 ^ (hash2 << 1);  
        }
    };

    template <typename... Args>
    struct hash<std::tuple<Args...>> {
        size_t operator()(const std::tuple<Args...>& t) const {
            return std::apply([](const Args&... args) {
                size_t seed = 0;
                (..., (seed ^= std::hash<Args>{}(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2)));
                return seed;
            }, t); 
        }
    };

    template <>
    struct hash<std::tuple<int, int, int>> {
        size_t operator()(const std::tuple<int, int, int>& t) const {
            size_t hash1 = std::hash<int>{}(std::get<0>(t));
            size_t hash2 = std::hash<int>{}(std::get<1>(t));
            size_t hash3 = std::hash<int>{}(std::get<2>(t));
            return hash1 ^ (hash2 << 1) ^ (hash3 << 2);
        }
    };

    template <>
    struct hash<std::tuple<int, int, int, int>> {
        size_t operator()(const std::tuple<int, int, int, int>& t) const {
            size_t h1 = std::hash<int>{}(std::get<0>(t));
            size_t h2 = std::hash<int>{}(std::get<1>(t));
            size_t h3 = std::hash<int>{}(std::get<2>(t));
            size_t h4 = std::hash<int>{}(std::get<3>(t));
            
            size_t combined = h1;
            combined ^= h2 + 0x9e3779b9 + (combined << 6) + (combined >> 2);
            combined ^= h3 + 0x9e3779b9 + (combined << 6) + (combined >> 2);
            combined ^= h4 + 0x9e3779b9 + (combined << 6) + (combined >> 2);
            
            return combined;
        }
    };
    
    template <>
    struct hash<std::tuple<int, int, int, int, int,int >> {
        std::size_t operator()(const std::tuple<int,int,int,int,int,int>& t) const noexcept {
            auto h1 = std::hash<int>{}(std::get<0>(t));
            auto h2 = std::hash<int>{}(std::get<1>(t));
            auto h3 = std::hash<int>{}(std::get<2>(t));
            auto h4 = std::hash<int>{}(std::get<3>(t));
            auto h5 = std::hash<int>{}(std::get<4>(t));
            auto h6 = std::hash<int>{}(std::get<5>(t));
    
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5);
        }
    };
    
}

#endif  
