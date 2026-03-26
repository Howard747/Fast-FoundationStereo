cmake_minimum_required(VERSION 3.8)
project(fast_foundation_stereo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -fopenmp")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
set(dependencies
    ament_cmake
    Eigen3
    rclcpp
    rclcpp_components
    rclcpp_lifecycle
    message_filters
    sensor_msgs
    std_msgs
    std_srvs
    tf2
    tf2_msgs
    tf2_ros
    tf2_sensor_msgs
    tf2_geometry_msgs
    nav_msgs
    visualization_msgs
    cv_bridge
    pcl_conversions
    depth_image_proc
)

foreach(dep IN LISTS dependencies)
    find_package(${dep} REQUIRED)
endforeach()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(PCL REQUIRED)



include_directories(${PCL_INCLUDE_DIRS})



# library
cuda_add_library(stereo_estimator SHARED
  src/estimator/fast_foundation_stereo_estimator.cpp
  src/estimator/gwc_volume_kernel.cu
)
ament_target_dependencies(stereo_estimator 
  rclcpp 
  cv_bridge
  CUDA OpenCV
  message_filters
  sensor_msgs
  std_msgs
)

# Add include directories
target_include_directories(stereo_estimator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Add library directories
target_link_libraries(stereo_estimator nvinfer nvonnxparser)



cuda_add_executable(dnn_stereo_depth_node src/dnn_stereo_depth_node.cpp)
ament_target_dependencies(dnn_stereo_depth_node 
  rclcpp 
  cv_bridge
  CUDA OpenCV
  message_filters
  sensor_msgs
  std_msgs
  std_srvs
)
target_link_libraries(dnn_stereo_depth_node stereo_estimator)


add_executable(disparity2pc_node src/disparity_to_pc_node.cpp)
ament_target_dependencies(disparity2pc_node 
  rclcpp 
  cv_bridge
  CUDA OpenCV
  message_filters
  sensor_msgs
  std_msgs
  pcl_conversions
  depth_image_proc
  Eigen3 PCL
)
target_link_libraries(disparity2pc_node)



# executable
cuda_add_executable(fast_foundation_stereo_test src/test/fast_foundation_stereo_test.cpp)
ament_target_dependencies(fast_foundation_stereo_test 
rclcpp 
cv_bridge
CUDA OpenCV
)
target_link_libraries(fast_foundation_stereo_test stereo_estimator)



install(TARGETS
  fast_foundation_stereo_test
  stereo_estimator
  dnn_stereo_depth_node
  disparity2pc_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch rviz
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
