cmake_minimum_required(VERSION 3.8)
project(ur3_rg2_moveit_config)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(geometric_shapes REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install configuration, launch, rviz, and srdf directories
install(
  DIRECTORY config launch rviz srdf
  DESTINATION share/${PROJECT_NAME}
)

# Test executable for verifying collision-free state and collision matrix
add_executable(test_collision_matrix test/test_collision_matrix.cpp)
ament_target_dependencies(test_collision_matrix
  rclcpp
  moveit_core
  moveit_ros_planning
  geometric_shapes
)

install(
  TARGETS test_collision_matrix
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python action client node
install(
  PROGRAMS scripts/pose_target_action_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
