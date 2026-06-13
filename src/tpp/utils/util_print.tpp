#include <vector>

template <typename List>
void print_vector(const List& vec) {
    std::cout << "[";
    long unsigned int i = 0;
    for (const auto& value: vec) {
        std::cout << value;
        if (i < vec.size() - 1)
            std::cout << ", "; 
        i += 1;
    }
    std::cout << "]" << std::endl;
}

template <typename Matrix>
void print_matrix(const Matrix& matrix) {
    for (const auto& row : matrix) {
        std::cout << "[ ";
        for (const auto& elem : row) {
            std::cout << elem << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

template <typename T>
void print_matrix_of_tuples(const T& matrix) {
    for (const auto& row : matrix) {
        print_vector_of_tuples(row);
    }
}

template <typename MapType>
void print_element_key_element_value(const MapType& map) {
    for (const auto& pair : map) {
        std::cout << "Key " << pair.first << " Value: " <<pair.second << std::endl; 
    }
}


template <typename MapType>
void print_element_key_list_value(const MapType& map) {
    for (const auto& pair : map) {
        std::cout << "Key " << pair.first << ": ";
        const auto& values = pair.second; 
        print_vector(values);
    }
}

template <typename MapType>
void print_element_key_tuple_value(const MapType& map) {
    for (const auto& pair : map) {
        std::cout << "Key " << pair.first << ": ";
        print_tuple(pair.second); 
        std::cout << std::endl;
    }
}

template <typename MapType>
void print_tuple_key_element_value(const MapType& map) {
    for (const auto& pair : map) {
         
        std::cout << "Key ";
        print_tuple(pair.first);
        std::cout << ": ";
        std::cout << pair.second;
        std::cout << std::endl;
    }
}

template <typename MapType>
void print_tuple_key_list_value(const MapType& map) {
    for (const auto& pair : map) {
        std::cout << "Key " << print_tuple(pair.second); pair.first << ": ";
        print_tuple(pair.second);
        std::cout << std::endl;
    }
}

template <typename Tuple, std::size_t... I>
void print_tuple_impl(const Tuple& t, std::index_sequence<I...>) {
    std::cout << "(";
    ((std::cout << (I == 0 ? "" : ", ") << std::get<I>(t)), ...);
    std::cout << ")";
}

template <typename... Args>
void print_tuple(const std::tuple<Args...>& t) {
    print_tuple_impl(t, std::index_sequence_for<Args...>{});
}

template <typename List>
void print_tuple_vector(const List& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        print_tuple(vec[i]);  
        if (i < vec.size() - 1) {
            std::cout << ", ";  
        }
    }
    std::cout << "]" << std::endl;  
}

template <typename T1, typename T2>
void print_pair(const std::pair<T1, T2>& p) {
    std::cout << "(" << p.first << ", " << p.second << ")";
}

template <typename List>
void print_pair_vector(const List& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        print_pair(vec[i]);  
        if (i < vec.size() - 1) {
            std::cout << ", ";  
        }
    }
    std::cout << "]" << std::endl;  
}

template <typename T1, typename T2, typename T3>
std::ostream& operator<<(std::ostream& os, const std::tuple<T1, T2, T3>& t) {
    os << "(" << std::get<0>(t) << ", " << std::get<1>(t) << ", " << std::get<2>(t) << ")";
    return os;
}
