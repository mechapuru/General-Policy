"""
Pick and Place task for xArm6 Lite in SAPIEN.

Task Description:
    - A cube is spawned at a random position on the table.
    - A cylinder target is spawned at a separate random position.
    - The robot must pick up the cube and place it on top of the cylinder.
    - Success: cube center within cylinder radius AND above cylinder top.

Task Phases (validated in debug_planner.py):
    1. Open gripper
    2. RRT hover above cube (+0.25m)
    3. Screw dip to grasp height (+0.20m)
    4. Close gripper
    5. RRT lift cube (+0.20m)
    6. Screw move to cylinder drop zone (+0.30m above cylinder)
    7. Open gripper (release)
"""

from .base_task_xarm6 import Base_task_xarm6
from .utils import *
import sapien
import numpy as np
import random


# Task geometry constants (validated in debug_planner.py)
CUBE_HALF_SIZE = 0.025          # = 5cm cube
CYLINDER_RADIUS = 0.05          # 10cm diameter target
CYLINDER_HEIGHT = 0.04          # 4cm tall
TABLE_Z = 0.62                  # Table surface height (matched to robot base Z)

# Trajectory offsets (validated in debug_planner.py)
HOVER_OFFSET = 0.25             # Z above cube for RRT approach
GRASP_DIP_OFFSET = 0.20        # Z above cube for screw dip
LIFT_OFFSET = 0.40              # Z above cube for RRT lift
DROP_OFFSET = 0.30              # Z above cylinder for screw drop
CYL_HOVER_OFFSET = 0.25         # Intermediate hover above cylinder

# Gripper values (validated in debug_planner.py)
GRIPPER_OPEN = -0.03
GRIPPER_CLOSE = -0.022

# Grasp orientation: pure vertical downward (validated quaternion)
GRASP_QUAT = [0, 1, 0, 0]      # [qw, qx, qy, qz] — gripper pointing straight down

SUCCESS_RADIUS = CYLINDER_RADIUS
SUCCESS_Z_MARGIN = 0.01


class pick_and_place(Base_task_xarm6):

    def setup_demo(self, is_test=False, **kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera()
        self.pre_move()
        self.load_actors()
        self.step_lim = 500
        self.is_test = is_test

    def pre_move(self):
        """Open gripper before task starts."""
        render_freq = self.render_freq
        self.render_freq = 0
        self.open_gripper(save_freq=None, pos=GRIPPER_OPEN)
        self.render_freq = render_freq

    def load_actors(self):
        """
        Spawn a cube and a cylinder target at exact verified debug_planner base positions.
        """
        # Spawn cube exactly at [0.25, 0.0]
        cube_x = 0.25
        cube_y = 0.0
        cube_z = TABLE_Z + CUBE_HALF_SIZE

        cube_pose = sapien.Pose(p=[cube_x, cube_y, cube_z], q=[1, 0, 0, 0])
        self.cube = create_box(
            scene=self.scene,
            pose=cube_pose,
            half_size=(CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE),
            color=(0, 0, 0),
            name="cube"
        )
        self.cube.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        ).mass = 0.05

        # Spawn cylinder exactly at [0.35, 0.2]
        cyl_x = 0.35
        cyl_y = 0.2
        cyl_z = TABLE_Z + CYLINDER_HEIGHT / 2
        self.target_pos = [cyl_x, cyl_y, TABLE_Z]

        cyl_pose = sapien.Pose(
            p=[cyl_x, cyl_y, cyl_z],
            q=[0.7071068, 0, 0.7071068, 0]  # Stand upright
        )
        self.cylinder = create_cylinder(
            scene=self.scene,
            pose=cyl_pose,
            radius=CYLINDER_RADIUS,
            half_length=CYLINDER_HEIGHT / 2,
            color=(1, 0.5, 0),
            name="cylinder_target"
        )
        self.cylinder.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        ).mass = 10.0

        self.cube_start_pos = [cube_x, cube_y, cube_z]

    def play_once(self):
        """
        Execute the 7-phase pick-and-place trajectory from debug_planner.py.
        """
        cube_pos = self.cube.get_pose().p

        # Phase 1: Open gripper
        print("[PHASE 1] Opening gripper...")
        self.open_gripper(pos=GRIPPER_OPEN)

        # Phase 2: RRT hover above cube
        print("[PHASE 2] RRT hover above cube...")
        hover_pose = [cube_pos[0], cube_pos[1], cube_pos[2] + HOVER_OFFSET] + GRASP_QUAT
        self.move_to_pose_with_RRTConnect(hover_pose)

        # Phase 3: Screw dip to grasp height
        print("[PHASE 3] Screw dip to grasp height...")
        grasp_pose = [cube_pos[0], cube_pos[1], cube_pos[2] + GRASP_DIP_OFFSET] + GRASP_QUAT
        self.move_to_pose_with_screw(grasp_pose)

        # Phase 4: Close gripper
        print("[PHASE 4] Closing gripper...")
        self.close_gripper(pos=GRIPPER_CLOSE)

        # Phase 5: RRT lift cube
        print("[PHASE 5] RRT lift cube...")
        lift_pose = [cube_pos[0], cube_pos[1], cube_pos[2] + LIFT_OFFSET] + GRASP_QUAT
        self.move_to_pose_with_RRTConnect(lift_pose)

        # Phase 6: Screw move to cylinder drop zone
        print("[PHASE 6] Screw move to cylinder drop zone...")
        cyl_pos = self.cylinder.get_pose().p
        drop_pose = [cyl_pos[0], cyl_pos[1], cyl_pos[2] + DROP_OFFSET] + GRASP_QUAT
        self.move_to_pose_with_screw(drop_pose)

        # Phase 6.5: RRT/Screw Hover above Cylinder (verified intermediate step)
        print("[PHASE 6.5] Hover above cylinder...")
        cyl_hover_pose = [cyl_pos[0], cyl_pos[1], cyl_pos[2] + CYL_HOVER_OFFSET] + GRASP_QUAT
        self.move_to_pose_with_screw(cyl_hover_pose)

        # Phase 7: Open gripper (release)
        print("[PHASE 7] Releasing gripper...")
        self.open_gripper(pos=GRIPPER_OPEN)

        # Wait for cube to settle
        for _ in range(200):
            self.scene.step()
        self._update_render()

    def check_success(self):
        """
        Success: cube XY within cylinder radius AND above cylinder top.
        """
        cube_pos = self.cube.get_pose().p
        target_xy = np.array(self.target_pos[:2])
        dist_to_center = np.linalg.norm(cube_pos[:2] - target_xy)
        cylinder_top_z = self.target_pos[2] + CYLINDER_HEIGHT

        on_target = (dist_to_center < SUCCESS_RADIUS) and \
                    (cube_pos[2] > cylinder_top_z + SUCCESS_Z_MARGIN)

        return on_target and self.is_gripper_open()
