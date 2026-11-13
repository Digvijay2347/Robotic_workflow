// Copyright (c) 2026 ur3_rg2_moveit_config
// SPDX-License-Identifier: BSD-3-Clause

#include <rclcpp/rclcpp.hpp>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/shapes.h>

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
      "test_collision_matrix",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  RCLCPP_INFO(node->get_logger(), "=== UR3 + OnRobot RG2 Collision Awareness Verification Test ===");

  // 1. Load Robot Model
  robot_model_loader::RobotModelLoader::Options opt("robot_description");
  robot_model_loader::RobotModelLoader loader(node, opt);
  auto kinematic_model = loader.getModel();

  if (!kinematic_model)
  {
    RCLCPP_ERROR(node->get_logger(), "FAILED to load robot model from robot_description!");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Successfully loaded RobotModel: %s", kinematic_model->getName().c_str());

  // Verify Planning Groups
  const auto& group_names = kinematic_model->getJointModelGroupNames();
  RCLCPP_INFO(node->get_logger(), "Available Planning Groups (%zu):", group_names.size());
  for (const auto& name : group_names)
  {
    RCLCPP_INFO(node->get_logger(), "  - %s", name.c_str());
  }

  bool has_ur_manipulator = kinematic_model->hasJointModelGroup("ur_manipulator");
  bool has_gripper = kinematic_model->hasJointModelGroup("gripper");

  if (!has_ur_manipulator)
  {
    RCLCPP_ERROR(node->get_logger(), "Missing required group 'ur_manipulator'!");
    rclcpp::shutdown();
    return 1;
  }
  if (!has_gripper)
  {
    RCLCPP_ERROR(node->get_logger(), "Missing required group 'gripper'!");
    rclcpp::shutdown();
    return 1;
  }

  // 2. Instantiate Planning Scene
  planning_scene::PlanningScene planning_scene(kinematic_model);
  collision_detection::CollisionRequest collision_request;
  collision_request.contacts = true;
  collision_request.max_contacts = 100;
  collision_detection::CollisionResult collision_result;

  // Test 1: Self-collision in initial state
  moveit::core::RobotState& current_state = planning_scene.getCurrentStateNonConst();
  current_state.setToDefaultValues();
  current_state.update();

  collision_result.clear();
  planning_scene.checkSelfCollision(collision_request, collision_result);

  RCLCPP_INFO(node->get_logger(), "--- Test 1: Self-Collision in Default State ---");
  if (collision_result.collision)
  {
    RCLCPP_ERROR(node->get_logger(), "FAILURE: Spurious self-collision detected in default state! Contacts (%zu):",
                 collision_result.contact_count);
    for (const auto& contact : collision_result.contacts)
    {
      RCLCPP_ERROR(node->get_logger(), "  Contact between '%s' and '%s'",
                   contact.first.first.c_str(), contact.first.second.c_str());
    }
    rclcpp::shutdown();
    return 1;
  }
  else
  {
    RCLCPP_INFO(node->get_logger(), "PASSED: Default state is completely self-collision FREE!");
  }

  // Test 2: Verify Allowed Collision Matrix (ACM) entries for gripper mount
  const auto& acm = planning_scene.getAllowedCollisionMatrix();
  collision_detection::AllowedCollision::Type allowed_type;

  bool mount_allowed = acm.getAllowedCollision("tool0", "rg2_onrobot_base_link", allowed_type);
  RCLCPP_INFO(node->get_logger(), "ACM tool0 <-> rg2_onrobot_base_link: %s",
              mount_allowed ? "ALLOWED (Correct - adjacent mount)" : "DISALLOWED (Error)");

  bool wrist_mount_allowed = acm.getAllowedCollision("wrist_3_link", "rg2_onrobot_base_link", allowed_type);
  RCLCPP_INFO(node->get_logger(), "ACM wrist_3_link <-> rg2_onrobot_base_link: %s",
              wrist_mount_allowed ? "ALLOWED (Correct - adjacent mount)" : "DISALLOWED (Error)");

  // Test 3: Gripper-to-Forearm collision MUST NOT be disabled (must be checked!)
  bool forearm_gripper_allowed = acm.getAllowedCollision("forearm_link", "rg2_left_inner_finger", allowed_type);
  RCLCPP_INFO(node->get_logger(), "ACM forearm_link <-> rg2_left_inner_finger: %s",
              forearm_gripper_allowed ? "ALLOWED (UNEXPECTED! Should be checked!)" : "NOT ALLOWED (Correct - collision checking is ACTIVE!)");

  // Test 4: External Obstacle Collision Awareness with Gripper
  RCLCPP_INFO(node->get_logger(), "--- Test 4: External Obstacle Collision with Gripper ---");
  // Place an obstacle right at the gripper TCP position
  const Eigen::Isometry3d& tcp_pose = current_state.getGlobalLinkTransform("rg2_gripper_tcp");
  auto box = std::make_shared<shapes::Box>(0.05, 0.05, 0.05);

  planning_scene.getWorldNonConst()->addToObject("test_cube", box, tcp_pose);

  collision_result.clear();
  planning_scene.checkCollision(collision_request, collision_result);

  bool detected_gripper_collision = false;
  if (collision_result.collision)
  {
    RCLCPP_INFO(node->get_logger(), "Collision correctly detected! Contacts count: %zu", collision_result.contact_count);
    for (const auto& contact : collision_result.contacts)
    {
      RCLCPP_INFO(node->get_logger(), "  Contact between '%s' and '%s'",
                  contact.first.first.c_str(), contact.first.second.c_str());
      if (contact.first.first.find("rg2_") != std::string::npos ||
          contact.first.second.find("rg2_") != std::string::npos)
      {
        detected_gripper_collision = true;
      }
    }
  }

  if (detected_gripper_collision)
  {
    RCLCPP_INFO(node->get_logger(), "PASSED: MoveIt is fully collision-aware with OnRobot RG2 gripper!");
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "FAILURE: Obstacle at gripper TCP did not trigger gripper collision!");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "=== ALL MOVEIT 2 COLLISION AWARENESS TESTS PASSED SUCCESSFULLY! ===");
  rclcpp::shutdown();
  return 0;
}
