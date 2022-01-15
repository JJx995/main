cmake_minimum_required(VERSION 3.21)
project(tch1)

set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
set(Torch_DIR /usr/local/Libtorch/share/cmake/Torch)
set(CMAKE_CXX_STANDARD 23)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(tch1 main.cpp)
target_link_libraries(tch1 "${TORCH_LIBRARIES}")
set_property(TARGET tch1 PROPERTY CXX_STANDARD 23)
