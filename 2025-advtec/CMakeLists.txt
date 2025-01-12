cmake_minimum_required(VERSION 3.15)
project(dac)
set(CMAKE_CXX_STANDARD 14)  #用于指定从C++标准，与这句话后，将在编译器中添加 -std=c++1４
set(CMAKE_CXX_STANDARD_REQUIRED ON) #强制开启c++标准版本检查

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -pthread  -std=c++14")
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)  #设置cmake编译后生成的文件的位置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)  #设置运行时产生的结果的存放位置

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED ENV{NEUWARE_HOME})
    set(ENV{NEUWARE_HOME} /usr/local/neuware)
endif()

find_package(Boost REQUIRED)
find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED)

# cnrt
include_directories("$ENV{NEUWARE_HOME}/include")   #头文件
link_directories("$ENV{NEUWARE_HOME}/lib64")        #runtime lib

message(STATUS "mlu-env $ENV{NEUWARE_HOME}")

#opencv 
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "opencv include dirs---${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv libs---${OpenCV_LIBS}")

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src/")
set(LIB_DIR "${PROJECT_SOURCE_DIR}/lib")

# 将子目录包含进来
include_directories(${SRC_DIR})
message(STATUS "src-dir---${SRC_DIR}")
include_directories(src/mlu100)
include_directories(src/mlu)
include_directories(src/readimage)
include_directories(${CMAKE_SOURCE_DIR})

message(STATUS "project source dir: ${PROJECT_SOURCE_DIR}")

add_subdirectory(lib)
add_subdirectory(test)

add_executable(hello main.cpp ${SRC_DIR}/bean/Task.hpp ${SRC_DIR}/readimage/ImageReader.cpp  ${SRC_DIR}/mlu100/Launch.hpp ${SRC_DIR}/mlu100/Launch.cpp) 
target_link_libraries(hello cnrt ${OpenCV_LIBS}) 

add_executable(hello2 main2.cpp)
target_link_libraries(hello2 cnrt ${OpenCV_LIBS})

add_executable(hello3 main3.cpp)
target_link_libraries(hello3 cnrt ${OpenCV_LIBS})

add_executable(hello4 main4.cpp)
target_link_libraries(hello4 cnrt ${OpenCV_LIBS})

add_executable(generate generate.cpp)
target_link_libraries(generate)

# add_executable(test ${SRC_DIR}/readimage/LabelReader.cpp)
# target_link_libraries(test)