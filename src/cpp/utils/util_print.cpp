#include "src/hpp/utils/util_print.hpp"

void printTensor(const torch::Tensor& tensor, const std::string& label) {
    std::cout << label << ":\n" << tensor << '\n';
}

void print_line(int quantity){
    std::cout << "\n";
    for (int i = 0; i< quantity; i++) std::cout << "-";
    std::cout << "\n";
}



