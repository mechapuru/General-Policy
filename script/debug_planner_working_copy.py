"""
Visual debug script for pick-and-place trajectories.
Shows the environment in SAPIEN viewer to diagnose execution failures.
Usage: python3 script/debug_planner.py
"""
import sys
import numpy as np
import sapien
import sapien.core
import mplib

# Import environment utilities
sys.path.append('./')
from envs.utils import create_box, create_cylinder, create_table, create_visual_box

def main():
    print("=" * 60)
    print("Initializing environment and SAPIEN viewer...")
    
    engine = sapien.Engine()
    
    # [Match base_task.py] Configure renderer limits
    from sapien.render import set_global_config
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    
    # [Match base_task.py] Ray-tracing rendering parameters
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")
    
    # [Modify] Shrink the invisible forcefield collision padding (20mm default -> 2mm)
    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    
    scene.set_timestep(1/250)
    ground_mat = sapien.render.RenderMaterial()
    ground_mat.base_color = [0.75, 0.75, 0.72, 1]  # Light concrete gray
    ground_mat.roughness = 0.8
    scene.add_ground(altitude=0, render_material=ground_mat)
    # [Match base_task.py] default physical material uses exactly 0.5 friction
    scene.default_physical_material = scene.create_physical_material(0.5, 0.5, 0.0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    
    # 1. Create table and objects
    table = create_table(scene, sapien.Pose([0.3, 0, 0.62]), length=0.8, width=0.8, height=0.62, color=(0.55, 0.35, 0.17))
    
    # Walls for depth camera reference (visual-only, no collision)
    wall_color = (0.7, 0.75, 0.8)  # Soft blue-gray
    create_visual_box(scene, sapien.Pose([-1.2, 0, 1.0]),   half_size=(0.01, 2.0, 1.0), color=wall_color, name="back_wall")
    create_visual_box(scene, sapien.Pose([ 1.5, 0, 1.0]),   half_size=(0.01, 2.0, 1.0), color=wall_color, name="front_wall")
    create_visual_box(scene, sapien.Pose([0.15, -1.5, 1.0]), half_size=(1.5, 0.01, 1.0), color=wall_color, name="left_wall")
    create_visual_box(scene, sapien.Pose([0.15,  1.5, 1.0]), half_size=(1.5, 0.01, 1.0), color=wall_color, name="right_wall")
    
    CUBE_HALF_SIZE = 0.025
    CYLINDER_RADIUS = 0.05
    CYLINDER_HEIGHT = 0.04
    TABLE_Z = 0.625
    
    cube_x, cube_y, cube_z = 0.25, 0.0, TABLE_Z + CUBE_HALF_SIZE
    cube = create_box(scene, sapien.Pose([cube_x, cube_y, cube_z]), (CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE), color=(0,0,0), name="cube")
    cube.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.05
    
    cyl_x, cyl_y, cyl_z = 0.35, 0.2, TABLE_Z + CYLINDER_HEIGHT / 2
    # SAPIEN cylinders are X-aligned by default. 
    # Use q=[0.7071068, 0, 0.7071068, 0] to stand it upright along Z.
    cyl_pose_upright = sapien.Pose([cyl_x, cyl_y, cyl_z], [0.7071068, 0, 0.7071068, 0])
    cylinder = create_cylinder(scene, cyl_pose_upright, CYLINDER_RADIUS, CYLINDER_HEIGHT / 2, color=(1,0.5,0), name="cylinder")
    cylinder.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 10.0

    # 2. Load robot
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = "sim-robot/lite-6-updated-urdf/lite_6_new.urdf"
    srdf_path = "sim-robot/lite6.srdf"
    
    robot = loader.load(urdf_path)
    robot.set_root_pose(sapien.Pose([0, 0, 0.62], [1, 0, 0, 0]))
    
    # Paint the robot arm with an industrial xArm6 color scheme
    arm_colors = {
        'link_base': [0.85, 0.85, 0.85, 1],  # Light silver
        'link1':     [0.95, 0.95, 0.95, 1],  # White
        'link2':     [0.85, 0.85, 0.85, 1],  # Light silver
        'link3':     [0.95, 0.95, 0.95, 1],  # White
        'link4':     [0.85, 0.85, 0.85, 1],  # Light silver
        'link5':     [0.95, 0.95, 0.95, 1],  # White
        'link6':     [0.2,  0.2,  0.2,  1],  # Dark charcoal (wrist)
        'base_link': [0.5,  0.5,  0.5,  1],  # Medium gray (gripper base)
        'left_finger':  [0.2, 0.4, 0.8, 1],  # Blue pad
        'right_finger': [0.2, 0.7, 0.3, 1],  # Green pad
    }
    for link in robot.get_links():
        lname = link.get_name()
        if lname in arm_colors:
            try:
                render_body = link.entity.find_component_by_type(sapien.render.RenderBodyComponent)
                if render_body:
                    mat = sapien.render.RenderMaterial()
                    mat.base_color = arm_colors[lname]
                    mat.metallic = 0.3
                    mat.roughness = 0.4
                    for shape in render_body.render_shapes:
                        shape.set_material(mat)
            except Exception as e:
                print(f"[WARN] Could not color {lname}: {e}")
    
    # SAPIEN 3 explicitly migrated contact_offset from the global Scene to the granular CollisionShapes!
    # By shrinking this property natively on the URDF links, we eliminate the 20mm repulsion forcefield.
    for link in robot.get_links():
        for shape in link.get_collision_shapes():
            shape.contact_offset = 0.002 
            shape.rest_offset = 0.000
    
    active_joints = robot.get_active_joints()
    all_joint_names = [j.get_name() for j in active_joints]
    print("names of all the joints are", all_joint_names)
    arm_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    arm_ids = [all_joint_names.index(n) for n in arm_names]
    gripper_id = all_joint_names.index('left_finger_joint')
    right_gripper_id = all_joint_names.index('right_finger_joint')
    
    # Create planner
    planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group="link_eef")
    planner.set_base_pose([0, 0, 0.62, 1, 0, 0, 0])
    
    # Option 1: Instantly start at a safe pose rather than all zeros (which collides with the table)
    initial_qpos = np.zeros(len(active_joints))
    joint3_idx = arm_ids[2] # joint3 is the 3rd arm joint
    joint5_idx = arm_ids[4] # joint5 is wrist pitch
    # joint6_idx = arm_ids[5] # joint6 is wrist roll
    initial_qpos[joint3_idx] = np.deg2rad(60) # Safe offset to avoid table 
    initial_qpos[joint5_idx] = np.deg2rad(60) # Try counter-pitching exactly 60
    # initial_qpos[joint6_idx] = np.deg2rad(-30) # User requested -30 for joint6
    initial_qpos[gripper_id] = -0.03# Open at spawn
    initial_qpos[right_gripper_id] = -0.03   
    robot.set_qpos(initial_qpos)
    
    active_joints = robot.get_active_joints()
    for i, j in enumerate(active_joints):
        if j.name == 'right_finger_joint':
            j.set_drive_property(stiffness=0, damping=0) # Let the URDF mimic gear drive it passively!
        else:
            j.set_drive_property(stiffness=1000, damping=200)
        j.set_drive_target(initial_qpos[i])

    # Let physics settle the robot securely into its set_drive_target equilibrium
    for _ in range(100):
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        scene.step()
        
    print("\n--- Phase 0: Geometry Audit ---")
    def print_pose(name, pose):
        print(f"[AUDIT] {name:10} | Pos: {[round(x, 4) for x in pose.p]} | Quat (wxyz): {[round(x, 4) for x in pose.q]}")
        
    print_pose("Robot Base", robot.get_root_pose())
    print_pose("Table Base", table.get_pose())
    print_pose("Cube", cube.get_pose())
    print_pose("Cylinder", cylinder.get_pose())
    
    # Audit EEF
    link_eef = next(l for l in robot.get_links() if l.name == "link6" or l.name == "link_eef")
    print_pose("EEF(link)", link_eef.get_entity_pose())
    
    # Audit Fingers
    for link in robot.get_links():
        if "finger" in link.name:
            print_pose(link.name, link.get_entity_pose())
            
    print("-" * 31)
    
    # 4. Viewer
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(0.8, -0.5, 1.2)
    viewer.set_camera_rpy(0, -0.6, 2.0)
    
    def step_viewer():
        scene.step()
        scene.update_render()
        viewer.render()
        
    def follow_screw_path(target_pose):
        full_qpos = np.array(robot.get_qpos(), dtype=np.float64)
        print("full qpos is ", full_qpos)
        print(f"[DEBUG] Starting follow_screw_path to target: {[round(x, 4) for x in target_pose]}")
        
        try:
            col = planner.check_for_self_collision(planner.robot, full_qpos)
            print(f"[DEBUG] Self-collision at start qpos? {col}")
        except Exception as e:
            print(f"[DEBUG] Self-collision check failed: {e}")

        result = planner.plan_screw(target_pose=target_pose, qpos=full_qpos, time_step=1/250, verbose=True)
        if result['status'] != 'Success':
            print(f"FAILED to plan screw! Status: {result['status']}")
            return False
            
        print(f"Executing trajectory... Steps: {result['position'].shape[0]}")
        for i in range(result['position'].shape[0]):
            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            robot.set_qf(qf)
            for j, arm_id in enumerate(arm_ids):
                active_joints[arm_id].set_drive_target(result["position"][i][j])
                active_joints[arm_id].set_drive_velocity_target(result["velocity"][i][j])
            step_viewer()
        return True

    def follow_path_rrt(target_pose):
        full_qpos = np.array(robot.get_qpos(), dtype=np.float64)
        print(f"[DEBUG] Starting follow_path_rrt to target: {[round(x, 4) for x in target_pose]}")
        result = planner.plan_qpos_to_pose(target_pose, full_qpos, time_step=1/250, planner_name="RRTConnect")
        if result['status'] != 'Success':
            print(f"FAILED to plan RRT! Status: {result['status']}")
            return False
            
        print(f"Executing RRT trajectory... Steps: {result['position'].shape[0]}")
        for i in range(result['position'].shape[0]):
            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            robot.set_qf(qf)
            for j, arm_id in enumerate(arm_ids):
                active_joints[arm_id].set_drive_target(result["position"][i][j])
                active_joints[arm_id].set_drive_velocity_target(result["velocity"][i][j])
            step_viewer()
        return True
        
    def move_gripper(pos):
        print(f"[DEBUG] Linear kinematic teleport of gripper to {pos}...")
        current_finger = robot.get_qpos()[gripper_id]
        steps = 200
        step = (pos - current_finger) / steps
        val = current_finger
        
        # Capture exactly where the arm is right now and use it as the rigid anchor!
        frozen_arm_qpos = robot.get_qpos()
        
        for i in range(steps):
            val += step
            
            # FREEZE ARM IN ITS CURRENT POSITION so it doesn't drool OR snap to spawn!
            qpos = frozen_arm_qpos.copy() 
            qpos[gripper_id] = val
            qpos[right_gripper_id] = val
            robot.set_qpos(qpos)
            
            # Counteract physics gravity
            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            robot.set_qf(qf)
            step_viewer()
        
        # Lock the PD targets so the gripper doesn't snap back during arm movement
        active_joints[gripper_id].set_drive_target(pos)
        active_joints[right_gripper_id].set_drive_target(pos)
        print(f"       -> Gripper at {pos}")

    print("\n[DEBUG] Starting execution in 3 seconds. Watch the SAPIEN viewer!")
    for _ in range(300):
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        step_viewer()
        
    # print("\n--- Phase 1: Open and Close Gripper ---")
    # move_gripper(-0.01) # Close gripper
    # move_gripper(-0.04) # Open gripper
    
    
    print("\n--- Phase 2: Hover above Cube (RRT Connect) ---")
    cube_pose = cube.get_pose().p
    print(f"[DEBUG] Cube Pose: {[round(x, 4) for x in cube_pose]}")
    # PyBullet offset: Hovering is 0.90 Z, cube is 0.65 Z. Offset = +0.25m
    hover_pose = [cube_pose[0], cube_pose[1], cube_pose[2] + 0.25, 0, 1, 0, 0]
    print(f"[DEBUG] Target RRT Hover Pose: {[round(x, 4) for x in hover_pose]}")
    follow_path_rrt(hover_pose)
    
    print("\n--- Phase 3: Dip down to Grasp (Screw Motion) ---")
    # PyBullet offset: Grasping is 0.82 Z. Offset = +0.17m
    grasp_pose = [cube_pose[0], cube_pose[1], cube_pose[2] + 0.2, 0, 1, 0, 0]
    print(f"[DEBUG] Target Screw Dip Pose: {[round(x, 4) for x in grasp_pose]}")
    follow_screw_path(grasp_pose)

    print("\n--- Phase 4: Close Gripper ---")
    move_gripper(-0.022) # Firm grip limit identified by user

    print("\n--- Phase 5: Lift Cube (RRT Motion) ---")
    # Reduced from +0.35 to +0.20 — robot workspace ceiling is ~Z=1.0
    lift_pose = [cube_pose[0], cube_pose[1], cube_pose[2] + 0.4, 0, 1, 0, 0]
    print(f"[DEBUG] Target Lift Pose: {[round(x, 4) for x in lift_pose]}")
    follow_path_rrt(lift_pose)
    
    print("\n--- Phase 6: Move to Cylinder Drop Zone (Screw - direct line) ---")
    cyl_pose = cylinder.get_pose().p
    drop_pose = [cyl_pose[0], cyl_pose[1], cyl_pose[2] + 0.3, 0, 1, 0, 0]
    print(f"[DEBUG] Target Cylinder Drop Pose: {[round(x, 4) for x in drop_pose]}")
    follow_screw_path(drop_pose)

    print("\n--- Phase 6.5: Hover above Cylinder (RRT Connect) ---")
    hover_pose = [cyl_pose[0], cyl_pose[1], cyl_pose[2] + 0.25, 0, 1, 0, 0]
    print(f"[DEBUG] Target Cylinder Hover Pose: {[round(x, 4) for x in hover_pose]}")
    follow_screw_path(hover_pose)

    print("\n--- Phase 7: Open Gripper (Release) ---")
    move_gripper(-0.03)

    print("\n[DEBUG] Finished Full Pick and Place! Keeping viewer open to inspect. (Ctrl+C in terminal to exit)")
    while not viewer.closed:
        step_viewer()

if __name__ == "__main__":
    main()
