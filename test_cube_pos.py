import sys
sys.path.append('./')
import sapien.core as sapien
import numpy as np
from envs.pick_and_place import pick_and_place

class FixedPickAndPlace(pick_and_place):
    def load_actors(self):
        cube_x = 0.25
        cube_y = 0.0
        cube_z = 0.62 + 0.025
        cube_pose = sapien.Pose(p=[cube_x, cube_y, cube_z], q=[1, 0, 0, 0])
        from envs.utils import create_box, create_cylinder
        self.cube = create_box(self.scene, cube_pose, (0.025, 0.025, 0.025), (0,0,0), "cube")
        self.cube.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.05

        cyl_x = 0.35
        cyl_y = 0.2
        cyl_z = 0.62 + 0.02
        self.target_pos = [cyl_x, cyl_y, 0.62]
        cyl_pose = sapien.Pose(p=[cyl_x, cyl_y, cyl_z], q=[0.7071068, 0, 0.7071068, 0])
        self.cylinder = create_cylinder(self.scene, cyl_pose, 0.05, 0.02, (1,0.5,0), "cylinder")
        self.cylinder.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 10.0
        self.cube_start_pos = [cube_x, cube_y, cube_z]

if __name__ == "__main__":
    env = FixedPickAndPlace()
    env.setup_demo(is_test=True, render_freq=0, dual_arm=False, head_camera_fovy=37, head_camera_w=320, head_camera_h=240,
                   front_camera_fovy=37, front_camera_w=320, front_camera_h=240, wrist_camera_fovy=37, wrist_camera_w=320, wrist_camera_h=240,
                   pcd_crop=False, is_save=False)
    env.play_once()
    
    cube_pos = env.cube.get_pose().p
    target_xy = np.array(env.target_pos[:2])
    dist_to_center = np.linalg.norm(cube_pos[:2] - target_xy)
    cylinder_top_z = env.target_pos[2] + 0.04
    
    print(f"Final Cube Pos: {cube_pos}")
    print(f"Target Pos: {env.target_pos}")
    print(f"Distance to target center XY: {dist_to_center}")
    print(f"Cube Z: {cube_pos[2]}, Cylinder Top Z: {cylinder_top_z}")
    print(f"Is Gripper Open: {env.is_gripper_open()}")
    print("FIXED RUN FAILED" if not (env.plan_success and env.check_success()) else "FIXED RUN SUCCESS")
