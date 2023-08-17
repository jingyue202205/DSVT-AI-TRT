cmake_minimum_required(VERSION 2.6)

project(dsvt-ai-trt)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)


find_package(CUDA REQUIRED)
# set(CUDA_NVCC_FLAGS -G;-g)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugins/include)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt

include_directories(/home/xxx/data/softwares/trt/TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-10.2.cudnn8.2/TensorRT-8.2.1.8/include)
link_directories(/home/xxx/data/softwares/trt/TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-10.2.cudnn8.2/TensorRT-8.2.1.8/lib)


#cnpy
# include_directories(/home/xxx/code/cnpy-master)
# link_directories(/home/xxx/code/cnpy-master/build)
# add_library(cnpy SHARED IMPORTED)
# set_target_properties(cnpy 
#                                     PROPERTIES IMPORTED_LOCATION
#                                     /home/xxx/code/cnpy-master/build/libcnpy.so)                                       

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")                     


cuda_add_library(voxelGeneratorlayer SHARED ${PROJECT_SOURCE_DIR}/plugins/src/points2Features.cu)
target_link_libraries(voxelGeneratorlayer nvinfer cudart)

cuda_add_library(torchScatterMaxop SHARED ${PROJECT_SOURCE_DIR}/plugins/src/torchScatterMax.cu)
target_link_libraries(torchScatterMaxop nvinfer cudart)

cuda_add_library(windowPartitionlayer SHARED ${PROJECT_SOURCE_DIR}/plugins/src/windowPartition.cu)
target_link_libraries(windowPartitionlayer nvinfer cudart)

cuda_add_library(getSetOp SHARED ${PROJECT_SOURCE_DIR}/plugins/src/getSet.cu)
target_link_libraries(getSetOp nvinfer cudart)

cuda_add_library(getValueByIndexOp SHARED ${PROJECT_SOURCE_DIR}/plugins/src/getValueByIndex.cu)
target_link_libraries(getValueByIndexOp nvinfer cudart)

cuda_add_library(mapSetFeature2VoxelOp SHARED ${PROJECT_SOURCE_DIR}/plugins/src/mapSetFeature2voxel.cu)
target_link_libraries(mapSetFeature2VoxelOp nvinfer cudart)

cuda_add_library(layerNormlayer SHARED ${PROJECT_SOURCE_DIR}/plugins/src/layerNorm.cu)
target_link_libraries(layerNormlayer nvinfer cudart)

cuda_add_library(gelulayer SHARED ${PROJECT_SOURCE_DIR}/plugins/src/gelu.cu)
target_link_libraries(gelulayer nvinfer cudart)

cuda_add_library(map2bevlayer SHARED ${PROJECT_SOURCE_DIR}/plugins/src/map2bev.cu)
target_link_libraries(map2bevlayer nvinfer cudart)

cuda_add_library(filterBoxByScorelayer SHARED ${PROJECT_SOURCE_DIR}/plugins/src/filterBoxByScore.cu)
target_link_libraries(filterBoxByScorelayer nvinfer cudart)


# find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs visualization_msgs pcl_conversions QUIET)
# find_package(PCL REQUIRED COMPONENTS filters QUIET)
find_package(Boost REQUIRED COMPONENTS filesystem QUIET)

# set(INC_DIRS
#     ./
#     ${catkin_INCLUDE_DIRS}
#     ${PCL_INCLUDE_DIRS})


set(LINK_LIBS
    ${Boost_LIBRARIES})



add_executable(dsvt-ai-trt ${PROJECT_SOURCE_DIR}/src/dsvt-ai-trt.cpp)
target_link_libraries(dsvt-ai-trt nvinfer)
target_link_libraries(dsvt-ai-trt cudart)
target_link_libraries(dsvt-ai-trt voxelGeneratorlayer)
target_link_libraries(dsvt-ai-trt torchScatterMaxop)
target_link_libraries(dsvt-ai-trt windowPartitionlayer)
target_link_libraries(dsvt-ai-trt getSetOp)
target_link_libraries(dsvt-ai-trt getValueByIndexOp)
target_link_libraries(dsvt-ai-trt mapSetFeature2VoxelOp)
target_link_libraries(dsvt-ai-trt layerNormlayer)
target_link_libraries(dsvt-ai-trt gelulayer)
target_link_libraries(dsvt-ai-trt map2bevlayer)
target_link_libraries(dsvt-ai-trt filterBoxByScorelayer)



target_link_libraries(dsvt-ai-trt PRIVATE ${LINK_LIBS})

add_definitions(-O2 -pthread)

