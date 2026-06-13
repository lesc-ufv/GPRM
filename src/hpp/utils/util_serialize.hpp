#ifndef UTIL_SERIALIZE_HPP
#define UTIL_SERIALIZE_HPP

#include <cereal/cereal.hpp>
#include <torch/torch.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <string>


// === ENUM SERIALIZATION ===
template<class Archive, typename Enum>
void serialize_int_enum(Archive& ar, Enum& e) {
    ar(reinterpret_cast<int&>(e));
}

template<class Archive, class Enum>
void serialize_set_of_enum(Archive& ar, std::unordered_set<Enum>& s) {
    std::vector<int> tmp;
    if constexpr (Archive::is_saving::value) {
        for (auto& e : s)
            tmp.push_back(static_cast<int>(e));
        ar(tmp);
    } else {
        ar(tmp);
        s.clear();
        for (auto& v : tmp)
            s.insert(static_cast<Enum>(v));
    }
}

template<class Archive>
void serialize_map_of_set_of_enum(
    Archive& ar,
    std::unordered_map<int, std::unordered_set<EnumFunctionalities>>& map) 
{
    if constexpr (Archive::is_saving::value) {
        std::unordered_map<int, std::vector<int>> tmp;
        for (auto& [k, vset] : map) {
            std::vector<int> vec;
            for (auto& e : vset)
                vec.push_back(static_cast<int>(e));
            tmp[k] = std::move(vec);
        }
        ar(tmp);
    } else {
        std::unordered_map<int, std::vector<int>> tmp;
        ar(tmp);
        map.clear();
        for (auto& [k, vec] : tmp) {
            std::unordered_set<EnumFunctionalities> vset;
            for (auto& val : vec)
                vset.insert(static_cast<EnumFunctionalities>(val));
            map[k] = std::move(vset);
        }
    }
}

template <class Archive>
void serialize_generator_state(Archive& ar, at::Generator& gen) {
    if constexpr (Archive::is_saving::value) {
        torch::Tensor state = gen.get_state();
        if (!state.defined()) {
            int64_t n0 = 0;
            ar(n0);
            return;
        }
        state = state.to(torch::kCPU).to(torch::kUInt8).contiguous();
        int64_t n = state.numel() * state.element_size(); 
        ar(n);
        ar(cereal::binary_data(state.data_ptr(), n));
        #ifdef DEBUG

        std::cout << "[SAVE] gen state: dtype=" << state.dtype()
                  << ", device=" << state.device()
                  << ", numel=" << state.numel()
                  << ", bytes=" << n << std::endl;
        #endif
    } else {
        gen = torch::make_generator<torch::CPUGeneratorImpl>();  
        int64_t n;
        ar(n);

        if (n == 0) {
            return;
        }

        torch::Tensor state = torch::empty({n}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)).contiguous();
        ar(cereal::binary_data(state.data_ptr(), n));
        #ifdef DEBUG
        std::cout << "[LOAD] loaded gen state: dtype=" << state.dtype()
                  << ", device=" << state.device()
                  << ", numel=" << state.numel()
                  << ", bytes=" << (state.numel() * state.element_size()) << std::endl;
        
        assert(state.defined());
        assert(state.device().is_cpu());
        assert(state.dtype() == torch::kUInt8);
        assert(state.numel() > 0);
        #endif

        gen.set_state(state);
    }
}

template <class Archive>
void serialize_tensor(Archive& ar, torch::Tensor& tensor) {
    if constexpr (Archive::is_saving::value) {
        std::ostringstream oss;
        torch::save(tensor, oss);
        std::string bytes = oss.str();
        if (bytes.empty()) {
            throw std::runtime_error("torch::save produced an empty buffer");
        }
        ar(bytes);
    } else {
        std::string bytes;
         try {
            ar(bytes);
        } catch (const cereal::Exception& e) {
            throw std::runtime_error(std::string("Failed to deserialize tensor byte payload from archive: ") + e.what());
        }

        try {
            std::istringstream iss(bytes);
            torch::Tensor tmp;
            torch::load(tmp, iss);
            tensor = tmp;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to reconstruct torch::Tensor from serialized payload: ") + e.what());
        }
    }
}

constexpr int IVALUE_TAG_INT = 0;
constexpr int IVALUE_TAG_DOUBLE = 1;
constexpr int IVALUE_TAG_STRING = 2;
constexpr int IVALUE_TAG_TENSOR = 3;



template<class Archive>
void serialize_ivalue(Archive& ar, c10::IValue& v) {
    if constexpr (Archive::is_saving::value) {
        int type_tag = -1;
        if (v.isInt()) {
            type_tag = IVALUE_TAG_INT;
            int64_t i = v.toInt();
            ar(type_tag, i);
        } else if (v.isDouble()) {
            type_tag = IVALUE_TAG_DOUBLE;
            double d = v.toDouble();
            ar(type_tag, d);
        } else if (v.isString()) {
            type_tag = IVALUE_TAG_STRING;
            std::string s = v.toStringRef();
            ar(type_tag, s);
        } else if (v.isTensor()) {
            type_tag = IVALUE_TAG_TENSOR;
            torch::Tensor t = v.toTensor();
            ar(type_tag);
            serialize_tensor(ar, t);
        } else {
            throw std::runtime_error("Unsupported IValue type for serialization.");
        }
    } else {
        int type_tag;
        ar(type_tag);
        if (type_tag == IVALUE_TAG_INT) {
            int64_t i;
            ar(i);
            v = c10::IValue(i);
        } else if (type_tag == IVALUE_TAG_DOUBLE) {
            double d;
            ar(d);
            v = c10::IValue(d);
        } else if (type_tag == IVALUE_TAG_STRING) {
            std::string s;
            ar(s);
            v = c10::IValue(s);
        } else if (type_tag == IVALUE_TAG_TENSOR) {
            torch::Tensor t;
            serialize_tensor(ar, t);
            v = c10::IValue(t);
        } else {
            throw std::runtime_error("Unknown IValue type tag during deserialization.");
        }
    }
}

template<class Archive>
void serialize_unordered_map_string_tensor(
    Archive& ar,
    std::unordered_map<std::string, torch::Tensor>& map)
{
    if constexpr (Archive::is_saving::value) {
        size_t sz = map.size();
        ar(sz);
        for (const auto& kv : map) {
            ar(kv.first);
            serialize_tensor(ar, const_cast<torch::Tensor&>(kv.second));
        }
    } else {
        size_t sz;
        ar(sz);
        map.clear();
        for (size_t i = 0; i < sz; ++i) {
            std::string key;
            torch::Tensor value;
            ar(key);
            serialize_tensor(ar, value);
            map[key] = value;
        }
    }
}

template<class Archive>
void serialize_shared_ptr_unordered_map_enum_int(
    Archive& ar,
    std::shared_ptr<std::unordered_map<EnumFunctionalities, int>>& ptr)
{
    if constexpr (Archive::is_saving::value) {
        bool exists = ptr != nullptr;
        ar(exists);
        if (exists) {
            size_t sz = ptr->size();
            ar(sz);
            for (const auto& kv : *ptr) {
                int enum_val = static_cast<int>(kv.first);
                ar(enum_val, kv.second);
            }
        }
    } else {
        bool exists;
        ar(exists);
        if (exists) {
            size_t sz;
            ar(sz);
            auto map_ptr = std::make_shared<std::unordered_map<EnumFunctionalities, int>>();
            for (size_t i = 0; i < sz; ++i) {
                int enum_val, value;
                ar(enum_val, value);
                (*map_ptr)[static_cast<EnumFunctionalities>(enum_val)] = value;
            }
            ptr = map_ptr;
        } else {
            ptr = nullptr;
        }
    }
}

template<class Archive>
void serialize_vector_tensor(
    Archive& ar,
    std::vector<torch::Tensor>& vec)
{
    if constexpr (Archive::is_saving::value) {
        size_t sz = vec.size();
        ar(sz);
        for (auto& tensor : vec) {
            serialize_tensor(ar, tensor);
        }
    } else {
        size_t sz;
        ar(sz);
        vec.resize(sz);
        for (size_t i = 0; i < sz; ++i) {
            serialize_tensor(ar, vec[i]);
        }
    }
}

template <class Archive, typename T>
void serialize_shared_ptr(Archive& ar, std::shared_ptr<T>& ptr) {
    if constexpr (Archive::is_saving::value) {
        bool has = (ptr != nullptr);
        ar(has);
        if (has) {
            ar(*ptr); 
        }
    } else {
        bool has;
        ar(has);
        if (has) {
            ptr = std::make_shared<T>();
            ar(*ptr);
        } else {
            ptr = nullptr;
        }
    }
}

template <class Archive, typename T>
void serialize_vector_shared_ptr(Archive& ar, std::vector<std::shared_ptr<T>>& vec) {
    if constexpr (Archive::is_saving::value) {
        size_t sz = vec.size();
        ar(sz);
        for (auto& e : vec) {
            serialize_shared_ptr(ar, e);
        }
    } else {
        size_t sz;
        ar(sz);
        vec.resize(sz);
        for (size_t i = 0; i < sz; ++i) {
            serialize_shared_ptr(ar, vec[i]);
        }
    }
}

template <class Archive, typename T>
void serialize_vector_vector_shared_ptr(Archive& ar, std::vector<std::vector<std::shared_ptr<T>>>& vv) {
    if constexpr (Archive::is_saving::value) {
        size_t outer = vv.size();
        ar(outer);
        for (auto& inner : vv) {
            size_t inner_sz = inner.size();
            ar(inner_sz);
            for (auto& e : inner) serialize_shared_ptr(ar, e);
        }
    } else {
        size_t outer;
        ar(outer);
        vv.resize(outer);
        for (size_t i = 0; i < outer; ++i) {
            size_t inner_sz;
            ar(inner_sz);
            vv[i].resize(inner_sz);
            for (size_t j = 0; j < inner_sz; ++j) serialize_shared_ptr(ar, vv[i][j]);
        }
    }
}

template<class Archive>
void serialize_unordered_map_int_unorderedset_int(
    Archive& ar,
    std::unordered_map<int, std::unordered_set<int>>& map)
{
    if constexpr (Archive::is_saving::value) {
        size_t sz = map.size();
        ar(sz);
        for (const auto& kv : map) {
            ar(kv.first);
            size_t set_sz = kv.second.size();
            ar(set_sz);
            for (const auto& val : kv.second) {
                ar(val);
            }
        }
    } else {
        size_t sz;
        ar(sz);
        map.clear();
        for (size_t i = 0; i < sz; ++i) {
            int key;
            ar(key);
            size_t set_sz;
            ar(set_sz);
            std::unordered_set<int> uset;
            for (size_t j = 0; j < set_sz; ++j) {
                int val;
                ar(val);
                uset.insert(val);
            }
            map[key] = std::move(uset);
        }
    }
}

namespace cereal {

template<class Archive>
void serialize(Archive& ar, c10::IValue& v) {
    serialize_ivalue(ar, v);
}


} 

template<class Archive>
void serialize_mapping_node_to_id(Archive& ar, std::unordered_map<std::pair<int,int>,int>& map) {
    if constexpr (std::is_same<Archive, cereal::BinaryOutputArchive>::value) {
        size_t sz = map.size();
        ar(sz);
        for (auto& kv : map) {
            ar(kv.first.first, kv.first.second, kv.second);
        }
    } else {
        size_t sz;
        ar(sz);
        map.clear();
        for (size_t i = 0; i < sz; ++i) {
            int a, b, v;
            ar(a, b, v);
            map[{a, b}] = v;
        }
    }
    
}
template<class Archive>
void serialize_gen(Archive& ar, std::mt19937& gen) {
    if constexpr (Archive::is_saving::value) {
        std::ostringstream oss;
        oss << gen;
        std::string state = oss.str();
        ar(state);
    } else {
        std::string state;
        ar(state);
        std::istringstream iss(state);
        iss >> gen;
    }
}
template<class Archive>
void serialize_ivalue(Archive& ar, std::vector<c10::IValue>& vec) {
    size_t n = vec.size();
    ar(n);
    if constexpr (Archive::is_loading::value) {
        vec.resize(n);
    }
    for (auto& v : vec) {
        serialize_ivalue(ar, v);
    }
}

template<class Archive>
void serialize_ivalue(Archive& ar, std::vector<std::vector<c10::IValue>>& batches) {
    size_t n = batches.size();
    ar(n);
    if constexpr (Archive::is_loading::value) {
        batches.resize(n);
    }
    for (auto& batch : batches) {
        serialize_ivalue(ar, batch);
    }
}


#endif
