#ifndef ENUM_OPERATION_H
#define ENUM_OPERATION_H
#include <iostream>

enum class EnumOperation{
    add = 0,
    mult = 1,
    constant = 2,
    load = 3,
    output = 4,
    store = 5,
    sub = 6,
    div = 7,
    shift = 8,
    compare = 9

};
inline int get_opcode_by_operation(const std::string& operation){
    if (operation == "add") {
        return static_cast<int>(EnumOperation::add);
    } else if (operation == "mult") {
        return  static_cast<int>(EnumOperation::mult);
    } else if (operation == "constant") {
        return  static_cast<int>(EnumOperation::constant);
    } else if (operation == "load") {
        return  static_cast<int>(EnumOperation::load);
    } else if (operation == "output") {
        return  static_cast<int>(EnumOperation::output);
    } else if (operation == "store") {
        return  static_cast<int>(EnumOperation::store);
    } else if (operation == "sub") {
        return  static_cast<int>(EnumOperation::sub);
    } else if (operation == "div") {
        return  static_cast<int>(EnumOperation::div);
    } else if (operation == "shift") {
        return  static_cast<int>(EnumOperation::shift);
    } else if (operation == "compare") {
        return  static_cast<int>(EnumOperation::compare);
    } else {
        throw std::invalid_argument("Invalid operation: " + operation);
        return -1;
    }
}
#endif