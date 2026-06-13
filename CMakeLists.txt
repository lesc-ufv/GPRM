cmake_minimum_required(VERSION 3.10)
project(GPRM)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_USE_PCH ON)
set(CMAKE_UNITY_BUILD ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(Python3_EXECUTABLE "{PYTHON_INTERPERTER_PATH}")
set(CMAKE_PREFIX_PATH "/usr/local/lib/libtorch")

link_directories("/usr/lib/x86_64-linux-gnu")
include_directories("/usr/include", "${CMAKE_SOURCE_DIR}/src/externals")

set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})  
cmake_policy(SET CMP0074 NEW)

find_package(Boost REQUIRED COMPONENTS graph filesystem)
find_package(Torch REQUIRED)
find_package(Catch2 REQUIRED)
find_package(TBB REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(CUDA REQUIRED)
find_package(msgpack REQUIRED)
find_package(cppzmq REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Msgpack_INCLUDE_DIRS})

file(GLOB_RECURSE H_SRCS ${CMAKE_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE CXX_SRCS ${CMAKE_SOURCE_DIR}/src/cpp/*.cpp)
file(GLOB_RECURSE TPP_SRCS ${CMAKE_SOURCE_DIR}/src/tpp/*.tpp)
file(GLOB_RECURSE CXX_TESTS ${CMAKE_SOURCE_DIR}/tests/*.cpp)
file(GLOB_RECURSE CXX_EXP ${CMAKE_SOURCE_DIR}/experiments/*.cpp)

add_executable(run_gprm_mrl_exp experiments/mapping_representation_learning/run_gprm_mrl_exp.cpp ${H_SRCS} ${CXX_SRCS} ${TPP_SRCS} )
target_link_libraries(run_gprm_mrl_exp 
PRIVATE
Catch2::Catch2WithMain
    PRIVATE
    Boost::graph
    Boost::filesystem 
    TBB::tbb
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    zmq 
    cppzmq
    msgpackc
    )
set_target_properties(run_gprm_mrl_exp PROPERTIES LINKER_LANGUAGE CXX)

add_executable(run_gprm_ablation experiments/mapping_representation_learning/run_gprm_ablation.cpp ${H_SRCS} ${CXX_SRCS} ${TPP_SRCS} )
target_link_libraries(run_gprm_ablation 
PRIVATE
Catch2::Catch2WithMain
    PRIVATE
    Boost::graph
    Boost::filesystem 
    TBB::tbb
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    zmq 
    cppzmq
    msgpackc
    )
set_target_properties(run_gprm_ablation PROPERTIES LINKER_LANGUAGE CXX)


add_executable(run_smartmap_mrl_exp experiments/mapping_representation_learning/run_smartmap_mrl_exp.cpp ${H_SRCS} ${CXX_SRCS} ${TPP_SRCS} )
target_link_libraries(run_smartmap_mrl_exp 
PRIVATE
Catch2::Catch2WithMain
    PRIVATE
    Boost::graph
    Boost::filesystem 
    TBB::tbb
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    zmq 
    cppzmq
    msgpackc
    )
set_target_properties(run_smartmap_mrl_exp PROPERTIES LINKER_LANGUAGE CXX)


add_executable(run_mapzero_mrl_exp experiments/mapping_representation_learning/run_mapzero_mrl_exp.cpp ${H_SRCS} ${CXX_SRCS} ${TPP_SRCS} )
target_link_libraries(run_mapzero_mrl_exp 
PRIVATE
Catch2::Catch2WithMain
    PRIVATE
    Boost::graph
    Boost::filesystem 
    TBB::tbb
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    zmq 
    cppzmq
    msgpackc
    )
set_target_properties(run_mapzero_mrl_exp PROPERTIES LINKER_LANGUAGE CXX)

add_executable(run_transmap_mrl_exp experiments/mapping_representation_learning/run_transmap_mrl_exp.cpp ${H_SRCS} ${CXX_SRCS} ${TPP_SRCS} )
target_link_libraries(run_transmap_mrl_exp 
PRIVATE
Catch2::Catch2WithMain
    PRIVATE
    Boost::graph
    Boost::filesystem 
    TBB::tbb
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    zmq 
    cppzmq
    msgpackc
    )
set_target_properties(run_transmap_mrl_exp PROPERTIES LINKER_LANGUAGE CXX)
