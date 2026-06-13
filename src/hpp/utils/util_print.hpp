#ifndef UTIL_PRINT_H
#define UTIL_PRINT_H
#include <vector>
#include <utility>
#include <iostream>
#include <map>
#include <torch/torch.h>
#include <sstream>

void print_line(int quantity);

void printTensor(const torch::Tensor& tensor, const std::string& label);

template <typename List>
void print_vector(const List& vec);

template <typename Matrix>
void print_matrix(const Matrix& matrix);

template <typename T>
void print_matrix_of_tuples(const T& matrix);

template <typename MapType>
void print_element_key_list_value(const MapType& map);

template <typename MapType>
void print_element_key_tuple_value(const MapType& map);


template <typename MapType>
void print_tuple_key_list_value(const MapType& map);

template <typename Tuple, std::size_t... I>
void print_tuple_impl(const Tuple& t, std::index_sequence<I...>);

template <typename... Args>
void print_tuple(const std::tuple<Args...>& t);

template <typename List>
void print_tuple_vector(const List& vec);

template <typename MapType>
void print_tuple_key_element_value(const MapType& map);

template <typename T1, typename T2>
void print_pair(const std::pair<T1, T2>& p);

template <typename List>
void print_pair_vector(const List& vec);

template <typename MapType>
void print_element_key_element_value(const MapType& map);

template <typename T1, typename T2, typename T3>
std::ostream& operator<<(std::ostream& os, const std::tuple<T1, T2, T3>& t);


#include "src/tpp/utils/util_print.tpp"

#endif