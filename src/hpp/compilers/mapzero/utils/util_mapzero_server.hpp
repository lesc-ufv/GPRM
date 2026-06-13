#ifndef UTIL_MAPZERO_SERVER_HPP
#define UTIL_MAPZERO_SERVER_HPP

#include "src/hpp/compilers/mapzero/configs/mapzero_config.hpp"
#include "src/hpp/utils/constants/server_constants.hpp"
#include <unordered_map>
#include <string>
#include <zmq.hpp>
#include <msgpack.hpp>
#include "src/hpp/utils/util_server.hpp"
#include "src/hpp/configs/global_config.hpp"

void initialize_mapzero_models(std::shared_ptr<zmq::socket_t> socket, ServerConstants& server_k, GlobalConfig<MapZeroConfig>& global_config);

#endif