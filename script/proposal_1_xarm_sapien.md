# Proposal 1: SAPIEN Engine + xArm6 Lite Robot

This document details the implementation plan for replacing the dual-arm Aloha robot with a single-arm xArm6 Lite within the original SAPIEN-based GPC framework.

---

## 1. Technical Architecture

### Simulator & Engine
- **Sim Engine**: SAPIEN (Vulkan/Ray-tracing).
- **Path Planner**: **mplib** (Open Motion Planning Library backend).
- **Observations**: GPC Observation Protocol (Unprojected Point Clouds + Camera RGB).

### Robot Configuration (xArm6 Lite)
- **URDF**: `sim-robot/urdf/xarm6_robotiq_85.urdf`.
- **Action Space**: **7-DOF** (6 positional joints + 1 continuous gripper joint).
- **Joint Mapping**:
    - Arm: `link1` to `link6`.
    - Gripper: `robotiq_85` mimic joints (controlled as 1 scalar value).

---

## 2. Implementation Steps

### Phase A: Environment Integration
1.  **Robot Loading**: Update `Base_task.load_robot` to support the xArm URDF. Implement a polymorphic switch to handle `robot_type="xarm6"`.
2.  **Planner Reconfiguration**: Re-initialize `mplib.planner` with the xArm6 URDF and corresponding SRDF for collision checking.
3.  **Action Dispatch**: Update `Base_task.step()` to route 7D actions to the motor controllers.

### Phase B: Policy & Configuration
1.  **7-DOF Policy Configs**: Create `task_config/xarm_7dof.yaml`. Define `action_dim: 7` and update `shape_meta`.
2.  **Composition Logic**: Ensure `apply_composed_policy` handles the reduction from 14D (dual) to 7D (single) configurations based on the loaded robot.

---

## 3. Pending Clarifications

To proceed with implementation, we need to address the following:

1.  **Mesh Path Resolution**: SAPIEN requires valid mesh paths. We need to verify if the meshes in `sim-robot/meshes/` are correctly referenced in the URDF for SAPIEN's renderer.
2.  **Gripper Command Semantics**: For the Robotiq 85, should the policy command absolute width (0-0.85m) or a normalized 0-1 value?
3.  **Prototype Task**: Which task should be the first to be re-calibrated for xArm6? (e.g., `blocks_stack_hard` or a simpler pick-and-place).
4.  **Policy Re-training**: Do we have existing 7-DOF single-arm datasets, or will we generate new ones using an xArm6-specific script?
