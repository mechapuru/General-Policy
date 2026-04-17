"""
Base_task_xarm6: Single-arm xArm6 Lite environment for SAPIEN.

Inherits from Base_task and overrides all Aloha-specific logic
(dual-arm joint IDs, hardcoded gripper indices, wrist camera links,
14D action slicing) with 7-DOF single-arm equivalents.

Method Audit Legend (vs Base_task):
    [INHERIT]  - Safe to use parent's implementation as-is.
    [OVERRIDE] - Contains Aloha-specific hardcoding; must be replaced.
    [NEW]      - New functionality specific to xArm6.
"""
import sapien.core as sapien
import numpy as np
import torch
import collections
import os
import json
from typing import Dict, Callable, List
from .base_task import Base_task, dict_apply
from .utils import *
import mplib
import open3d as o3d


class Base_task_xarm6(Base_task):

    # ================================================================
    # [OVERRIDE] Initialization
    # ================================================================
    def _init(self, **kwags):
        '''
        Initialization for xArm6 Lite (single arm).
        Forces dual_arm=False and sets xArm-specific joint mappings.
        '''
        kwags['dual_arm'] = False
        # Update default crop bbox for Lite6 table height (0.62)
        if 'bbox' not in kwags:
            kwags['bbox'] = [[-0.8, -0.8, 0.629], [0.8, 0.8, 2.0]]
        
        super()._init(**kwags)

        # xArm6 joint configuration
        self.arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.gripper_joint_name = 'left_finger_joint'  # Primary actuator (prismatic)
        self.right_gripper_joint_name = 'right_finger_joint'  # Mimic joint
        # Lite6 gripper: -0.04 = open, 0.0 = closed

        # Gripper value tracker (replaces left_gripper_val / right_gripper_val)
        self.gripper_val = 0.0

        # Arm joint indices into self.active_joints (populated in load_robot)
        self.arm_joint_ids = []
        self.gripper_joint_id = None
        self.right_gripper_joint_id = None

    # ================================================================
    # [OVERRIDE] Robot Loading - replaces Aloha URDF
    # ================================================================
    def load_robot(self, **kwargs):
        """
        Load xArm6 Lite + Robotiq 85 URDF.
        """
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True

        urdf_path = kwargs.get("urdf_path", "sim-robot/lite-6-updated-urdf/lite_6_new.urdf")
        self.robot = loader.load(urdf_path)

        # Set root pose — matches sim-robot/pick_and_place_xarm6_gripper.py
        # PyBullet reference: Lite6Robot([0, 0, 0.62], [0, 0, 0])
        self.robot.set_root_pose(
            sapien.Pose(
                kwargs.get("robot_origin_xyz", [0, 0, 0.62]),
                kwargs.get("robot_origin_quat", [1, 0, 0, 0]),
            )
        )

        # Configure joint drives
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            if joint.name == 'right_finger_joint':
                joint.set_drive_property(stiffness=0, damping=0)
            else:
                joint.set_drive_property(
                    stiffness=kwargs.get("joint_stiffness", 1000),
                    damping=kwargs.get("joint_damping", 200),
                )

        self.all_joints = self.robot.get_joints()
        self.all_links = self.robot.get_links()

        # [FIX] Map the end-effector link for tracking
        self.endpose = self.robot.find_link_by_name("link_eef")

        # Map arm joint IDs by name
        active_joint_names = [j.get_name() for j in self.active_joints]
        self.arm_joint_ids = [active_joint_names.index(n) for n in self.arm_joint_names]
        self.gripper_joint_id = active_joint_names.index(self.gripper_joint_name)
        self.right_gripper_joint_id = active_joint_names.index(self.right_gripper_joint_name)

        # Initial non-singular rest pose - Joint 3 and 5 at 60 degrees (1.047 rad)
        # Matches the high-elbow posture validated in debug_planner.py
        self.arm_rest_poses = [0.0, 0.0, 1.047, 0.0, 1.047, 0.0]
        self.gripper_val = -0.03  # Open

        # Set initial drive targets so gravity doesn't pull joints out of limits
        init_qpos = self.robot.get_qpos().copy()
        for i, arm_id in enumerate(self.arm_joint_ids):
            init_qpos[arm_id] = self.arm_rest_poses[i]
            self.active_joints[arm_id].set_drive_target(self.arm_rest_poses[i])
        init_qpos[self.gripper_joint_id] = self.gripper_val
        self.active_joints[self.gripper_joint_id].set_drive_target(self.gripper_val)
        self.robot.set_qpos(init_qpos)

        # Settle the robot at rest pose
        for _ in range(100):
            qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()

        # [FIX] Shrink collision contact_offset on all links to prevent ghost repulsion
        for link in self.robot.get_links():
            for shape in link.get_collision_shapes():
                shape.contact_offset = 0.002
                shape.rest_offset = 0.0

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
        for link in self.robot.get_links():
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
                except:
                    pass

    # ================================================================
    # [OVERRIDE] Scene Setup - adds colored ground
    # ================================================================
    def setup_scene(self, **kwargs):
        super().setup_scene(**kwargs)
        # Add colored ground (overrides the default ground from super().setup_scene)
        ground_mat = sapien.render.RenderMaterial()
        ground_mat.base_color = [0.75, 0.75, 0.72, 1]  # Light concrete gray
        ground_mat.roughness = 0.8
        self.scene.add_ground(altitude=0, render_material=ground_mat)

    # ================================================================
    # [OVERRIDE] Table and Walls - matched to debug_planner aesthetics
    # ================================================================
    def create_table_and_wall(self):
        # Create table with walnut wood color
        self.table = create_table(
            self.scene, 
            sapien.Pose([0.3, 0, 0.62]), 
            length=0.8, 
            width=0.8, 
            height=0.62, 
            color=(0.55, 0.35, 0.17),
            is_static=self.table_static
        )
        
        # Walls for depth camera reference (visual-only, no collision)
        wall_color = (0.7, 0.75, 0.8)  # Soft blue-gray
        create_visual_box(self.scene, sapien.Pose([-1.2, 0, 1.0]),   half_size=(0.01, 2.0, 1.0), color=wall_color, name="back_wall")
        create_visual_box(self.scene, sapien.Pose([ 1.5, 0, 1.0]),   half_size=(0.01, 2.0, 1.0), color=wall_color, name="front_wall")
        create_visual_box(self.scene, sapien.Pose([0.15, -1.5, 1.0]), half_size=(1.5, 0.01, 1.0), color=wall_color, name="left_wall")
        create_visual_box(self.scene, sapien.Pose([0.15,  1.5, 1.0]), half_size=(1.5, 0.01, 1.0), color=wall_color, name="right_wall")

    # ================================================================
    # [OVERRIDE] Camera Loading - single external camera
    # ================================================================
    def load_camera(self):
        '''
        Single external camera setup for xArm6.
        Camera position matches sim-robot/pick_and_place_xarm6_gripper.py:
            Eye:    [0.7463, 0.3093, 1.1774]
            Target: [-0.0288, -0.0952, 0.6919]
            Up:     [0, 0, 1]
        '''
        near, far = 0.1, 100

        # Third-person camera — from real-world calibration (base-to-camera transform)
        # Matches: tp_cam_eye in pick_and_place_xarm6_gripper.py
        tp_cam_eye = np.array([0.7463, 0.3093, 1.1774])
        tp_cam_target = np.array([-0.0288, -0.0952, 0.6919])
        tp_cam_up = np.array([0, 0, 1])

        # Compute camera extrinsic matrix from eye/target/up
        tp_forward = tp_cam_target - tp_cam_eye
        tp_forward = tp_forward / np.linalg.norm(tp_forward)
        tp_left = np.cross(tp_cam_up, tp_forward)
        tp_left = tp_left / np.linalg.norm(tp_left)
        tp_up = np.cross(tp_forward, tp_left)
        head_mat44 = np.eye(4)
        head_mat44[:3, :3] = np.stack([tp_forward, tp_left, tp_up], axis=1)
        head_mat44[:3, 3] = tp_cam_eye

        # Observer camera — side angle for video logging
        observer_cam_pos = np.array([0.8, -0.4, 1.3])
        observer_cam_target = np.array([0.2, 0.0, 0.65])
        obs_forward = observer_cam_target - observer_cam_pos
        obs_forward = obs_forward / np.linalg.norm(obs_forward)
        obs_left = np.cross(tp_cam_up, obs_forward)
        obs_left = obs_left / np.linalg.norm(obs_left)
        obs_up = np.cross(obs_forward, obs_left)
        observer_mat44 = np.eye(4)
        observer_mat44[:3, :3] = np.stack([obs_forward, obs_left, obs_up], axis=1)
        observer_mat44[:3, 3] = observer_cam_pos

        # Head camera (primary — third-person view from real calibration)
        self.head_camera = self.scene.add_camera(
            name="head_camera",
            width=self.head_camera_w,
            height=self.head_camera_h,
            fovy=np.deg2rad(60),  # Matches PyBullet FOV=60
            near=near, far=far,
        )

        # Front camera (same as head for single-camera setup; can be recalibrated)
        self.front_camera = self.scene.add_camera(
            name="front_camera",
            width=self.front_camera_w,
            height=self.front_camera_h,
            fovy=np.deg2rad(60),
            near=near, far=far,
        )

        self.observer_camera = self.scene.add_camera(
            name="observer_camera",
            width=320, height=240,
            fovy=np.deg2rad(60),
            near=near, far=far,
        )

        self.head_camera.entity.set_pose(sapien.Pose(head_mat44))
        self.observer_camera.entity.set_pose(sapien.Pose(observer_mat44))

        # Mount front_camera on the wrist link (link_eef) — tracks end-effector
        self.wrist_camera_link = self.robot.find_link_by_name("link_eef")
        self.front_camera.entity.set_pose(self.wrist_camera_link.get_pose())

        self.scene.step()
        self.scene.update_render()

    # ================================================================
    # [OVERRIDE] Motion Planner - single planner instead of left/right
    # ================================================================
    def setup_planner(self, **kwargs):
        """
        Initialize a single mplib planner for xArm6.
        """
        self.planner = mplib.Planner(
            urdf=kwargs.get("urdf_path", "sim-robot/lite-6-updated-urdf/lite_6_new.urdf"),
            srdf=kwargs.get("srdf_path", "sim-robot/lite6.srdf"),
            move_group=kwargs.get("move_group", "link_eef"),
        )
        robot_pose_in_world = [0, 0, 0.62, 1, 0, 0, 0]
        self.planner.set_base_pose(robot_pose_in_world)

    # ================================================================
    # [OVERRIDE] Render Update - tracks wrist camera
    # ================================================================
    def _update_render(self):
        """
        Update rendering. Tracks wrist camera to end-effector link.
        """
        self.front_camera.entity.set_pose(self.wrist_camera_link.get_pose())
        self.scene.update_render()
        self.scene.update_render()

    # ================================================================
    # [OVERRIDE] Follow Path - single arm
    # ================================================================
    def follow_path(self, result, save_freq=-1):
        """
        Follow a planned path using the xArm6 arm joints.
        Replaces left_follow_path / right_follow_path.
        """
        save_freq = self.save_freq if save_freq == -1 else save_freq
        n_step = result["position"].shape[0]

        if n_step > 2000:
            self.plan_success = False
            return

        if save_freq is not None:
            self._take_picture()

        for i in range(n_step):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            for j, arm_id in enumerate(self.arm_joint_ids):
                self.active_joints[arm_id].set_drive_target(result["position"][i][j])
                self.active_joints[arm_id].set_drive_velocity_target(result["velocity"][i][j])

            self.scene.step()
            if i % 5 == 0:
                self._update_render()
                if self.render_freq and i % self.render_freq == 0:
                    self.viewer.render()

            if save_freq is not None and i % save_freq == 0:
                self._take_picture()

        if save_freq is not None:
            self._take_picture()

    # Legacy redirects so task scripts can still call left/right
    def left_follow_path(self, result, save_freq=-1):
        self.follow_path(result, save_freq)

    def right_follow_path(self, result, save_freq=-1):
        self.follow_path(result, save_freq)

    def together_follow_path(self, left_result, right_result, save_freq=-1):
        # For a single arm, just execute one path
        self.follow_path(left_result, save_freq)

    # ================================================================
    # [OVERRIDE] Gripper Control - replaces Aloha active_joints[34:38]
    # ================================================================
    def set_gripper(self, pos=-0.03, save_freq=-1, **kwargs):
        """
        Set gripper to target position using kinematic teleport.
        Bypasses PD controller deadlocks caused by URDF mimic constraints.
        Lite6 prismatic gripper: -0.03 = open, -0.022 = closed.
        """
        save_freq = self.save_freq if save_freq == -1 else save_freq
        if save_freq is not None:
            self._take_picture()

        steps = 200
        current_finger = self.robot.get_qpos()[self.gripper_joint_id]
        step_size = (pos - current_finger) / steps
        val = current_finger

        # Freeze arm at current position to prevent drool during teleport
        frozen_qpos = self.robot.get_qpos()

        for i in range(steps):
            val += step_size
            qpos = frozen_qpos.copy()
            qpos[self.gripper_joint_id] = val
            self.robot.set_qpos(qpos)

            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)

            self.scene.step()
            if i % 5 == 0:
                self._update_render()
                if self.render_freq and i % self.render_freq == 0:
                    self.viewer.render()
            if save_freq is not None and i % save_freq == 0:
                self._take_picture()

        # Lock PD drive targets so gripper doesn't snap back during arm movement
        self.active_joints[self.gripper_joint_id].set_drive_target(pos)

        if save_freq is not None:
            self._take_picture()
        self.gripper_val = pos

    def open_gripper(self, save_freq=-1, pos=-0.04):
        save_freq = self.save_freq if save_freq == -1 else save_freq
        self.set_gripper(pos=pos, save_freq=save_freq)

    def close_gripper(self, save_freq=-1, pos=0.0):
        save_freq = self.save_freq if save_freq == -1 else save_freq
        self.set_gripper(pos=pos, save_freq=save_freq)

    # Legacy redirects
    def open_left_gripper(self, save_freq=-1, pos=-0.04):
        self.open_gripper(save_freq=save_freq, pos=pos)

    def close_left_gripper(self, save_freq=-1, pos=0.0):
        self.close_gripper(save_freq=save_freq, pos=pos)

    def open_right_gripper(self, save_freq=-1, pos=-0.04):
        self.open_gripper(save_freq=save_freq, pos=pos)

    def close_right_gripper(self, save_freq=-1, pos=0.0):
        self.close_gripper(save_freq=save_freq, pos=pos)

    def together_open_gripper(self, save_freq=-1, **kwargs):
        self.open_gripper(save_freq=save_freq)

    def together_close_gripper(self, save_freq=-1, **kwargs):
        self.close_gripper(save_freq=save_freq)

    # ================================================================
    # [OVERRIDE] Motion Planning - single planner
    # ================================================================
    def move_to_pose_with_screw(self, pose, use_point_cloud=False, use_attach=False, save_freq=-1):
        """
        Plan and execute screw motion to target pose.
        pose: [x, y, z, qw, qx, qy, qz] in world frame (transforms3d wxyz convention)
        """
        save_freq = self.save_freq if save_freq == -1 else save_freq
        # mplib expects full qpos (all joints including gripper) as numpy array
        full_qpos = np.array(self.robot.get_qpos(), dtype=np.float64)

        result = self.planner.plan_screw(
            target_pose=pose,
            qpos=full_qpos,
            time_step=1 / 250,
            use_point_cloud=use_point_cloud,
            use_attach=use_attach,
        )

        if result["status"] == "Success":
            self.follow_path(result, save_freq=save_freq)
            return 0
        else:
            print(f"\n arm planning failed! status={result['status']}")
            print(f"   target_pose={pose}")
            print(f"   current_qpos={np.round(full_qpos, 4)}")
            self.plan_success = False
            return -1

    # Legacy redirects
    def left_move_to_pose_with_screw(self, pose, **kwargs):
        return self.move_to_pose_with_screw(pose, **kwargs)

    def right_move_to_pose_with_screw(self, pose, **kwargs):
        return self.move_to_pose_with_screw(pose, **kwargs)

    def together_move_to_pose_with_screw(self, left_target_pose, right_target_pose=None, **kwargs):
        return self.move_to_pose_with_screw(left_target_pose, **kwargs)

    def move_to_pose_with_RRTConnect(self, pose, use_point_cloud=False, use_attach=False, freq=10):
        """
        Plan and follow a path using RRTConnect.
        """
        result = self.planner.plan_qpos_to_pose(
            pose,
            current_qpos=self.robot.get_qpos(),
            time_step=1 / 250,
            use_point_cloud=use_point_cloud,
            use_attach=use_attach,
            planner_name="RRTConnect",
        )
        if result["status"] != "Success":
            print(f"\n arm RRT planning failed! status={result['status']}")
            self.plan_success = False
            return -1
        self.follow_path(result, freq)
        return 0

    # ================================================================
    # [OVERRIDE] Gripper Status - replaces active_joints[34]/[36]
    # ================================================================
    def is_gripper_open(self):
        return self.active_joints[self.gripper_joint_id].get_drive_target()[0] <= -0.029

    def is_gripper_close(self):
        return self.active_joints[self.gripper_joint_id].get_drive_target()[0] > -0.01

    def is_left_gripper_open(self):
        return self.is_gripper_open()

    def is_right_gripper_open(self):
        return self.is_gripper_open()

    def is_left_gripper_open_half(self):
        return self.active_joints[self.gripper_joint_id].get_drive_target()[0] < -0.015

    def is_right_gripper_open_half(self):
        return self.is_left_gripper_open_half()

    def is_left_gripper_close(self):
        return self.is_gripper_close()

    def is_right_gripper_close(self):
        return self.is_gripper_close()

    # ================================================================
    # [OVERRIDE] Joint State Getters - replaces Aloha joint ID arrays
    # ================================================================
    def get_arm_jointState(self) -> list:
        """
        Returns 7D joint state: [j1..j6, gripper_val]
        """
        jointState_list = []
        for arm_id in self.arm_joint_ids:
            jointState_list.append(
                self.active_joints[arm_id].get_drive_target()[0].astype(float)
            )
        jointState_list.append(self.gripper_val)
        return jointState_list

    # Legacy redirects
    def get_left_arm_jointState(self) -> list:
        return self.get_arm_jointState()

    def get_right_arm_jointState(self) -> list:
        return self.get_arm_jointState()

    # ================================================================
    # [OVERRIDE] End-effector Pose Getters
    # ================================================================
    def get_endpose_pose(self):
        return self.endpose.global_pose

    def get_left_endpose_pose(self):
        return self.get_endpose_pose()

    def get_right_endpose_pose(self):
        return self.get_endpose_pose()

    # ================================================================
    # [OVERRIDE] get_obs - 7D joint_action instead of 14D
    # ================================================================
    def get_obs(self):
        """
        Get observations with 7D joint_action and single-camera obs.
        """
        self.scene.step()
        self._update_render()
        self._update_render()

        arm_endpose = self.endpose_transform(self.endpose, self.gripper_val)
        jointState = self.get_arm_jointState()
        jointState_array = np.array(jointState)

        self.head_camera.take_picture()
        self.front_camera.take_picture()

        head_pcd = self._get_camera_pcd(self.head_camera, point_num=0)
        front_pcd = self._get_camera_pcd(self.front_camera, point_num=0)
        head_rgba = self._get_camera_rgba(self.head_camera)
        front_rgba = self._get_camera_rgba(self.front_camera)
        head_depth = self._get_camera_depth(self.head_camera)
        front_depth = self._get_camera_depth(self.front_camera)

        # Merge PointCloud
        if self.data_type and self.data_type.get("conbine", False):
            conbine_pcd = np.vstack((head_pcd, front_pcd))
        else:
            conbine_pcd = head_pcd

        pcd_array, index = fps(conbine_pcd[:, :3], self.pcd_down_sample_num)

        obs = {
            "observation": {
                "head_camera": {},
                "front_camera": {},
            },
            "pointcloud": [],
            "joint_action": [],
            "endpose": []
        }

        # Camera intrinsics/extrinsics
        for cam_name, cam_obj in [("head_camera", self.head_camera), ("front_camera", self.front_camera)]:
            obs["observation"][cam_name] = {
                "intrinsic_cv": cam_obj.get_intrinsic_matrix(),
                "extrinsic_cv": cam_obj.get_extrinsic_matrix(),
                "cam2world_gl": cam_obj.get_model_matrix(),
            }

        obs["observation"]["head_camera"]["rgb"] = head_rgba[:, :, :3]
        obs["observation"]["front_camera"]["rgb"] = front_rgba[:, :, :3]
        obs["observation"]["head_camera"]["depth"] = head_depth
        obs["observation"]["front_camera"]["depth"] = front_depth

        obs["pointcloud"] = conbine_pcd[index.detach().cpu().numpy()[0]]
        obs["endpose"] = np.array([
            arm_endpose["x"], arm_endpose["y"], arm_endpose["z"],
            arm_endpose["roll"], arm_endpose["pitch"], arm_endpose["yaw"],
            arm_endpose["gripper"],
        ])
        obs["joint_action"] = jointState_array  # 7D

        return obs

    # ================================================================
    # [OVERRIDE] get_cam_obs - single-arm camera observations
    # ================================================================
    def get_cam_obs(self, observation: dict) -> dict:
        head_cam = np.moveaxis(observation['observation']['head_camera']['rgb'], -1, 0) / 255
        front_cam = np.moveaxis(observation['observation']['front_camera']['rgb'], -1, 0) / 255
        return dict(
            head_cam=head_cam,
            front_cam=front_cam,
        )

    # ================================================================
    # [OVERRIDE] _take_picture - single-arm data saving
    # ================================================================
    def _take_picture(self):
        if not self.is_save:
            return

        print('saving: episode = ', self.ep_num, ' index = ', self.PCD_INDEX, end='\r')
        self._update_render()
        self.head_camera.take_picture()
        self.front_camera.take_picture()
        self.observer_camera.take_picture()

        if self.PCD_INDEX == 0:
            self.file_path = {
                "observer_color": f"{self.save_dir}/episode{self.ep_num}/camera/color/observer/",
                "h_color": f"{self.save_dir}/episode{self.ep_num}/camera/color/head/",
                "h_depth": f"{self.save_dir}/episode{self.ep_num}/camera/depth/head/",
                "h_pcd":   f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/head/",
                "f_color": f"{self.save_dir}/episode{self.ep_num}/camera/color/front/",
                "f_depth": f"{self.save_dir}/episode{self.ep_num}/camera/depth/front/",
                "f_pcd":   f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/front/",
                "h_seg_mesh":  f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/head/mesh/",
                "f_seg_mesh":  f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/front/mesh/",
                "h_seg_actor": f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/head/actor/",
                "f_seg_actor": f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/front/actor/",
                "arm_ep":    f"{self.save_dir}/episode{self.ep_num}/arm/endPose/",
                "arm_joint": f"{self.save_dir}/episode{self.ep_num}/arm/jointState/",
                "pkl":       f"{self.save_dir}_pkl/episode{self.ep_num}/",
                "conbine_pcd": f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/conbine/",
            }
            for directory in self.file_path.values():
                if os.path.exists(directory):
                    for file in os.listdir(directory):
                        os.remove(os.path.join(directory, file))

        pkl_dic = {
            "observation": {
                "head_camera": {},
                "front_camera": {},
            },
            "pointcloud": [],
            "joint_action": [],
            "endpose": []
        }

        # Camera matrices
        for cam_name, cam_obj, prefix in [
            ("head_camera", self.head_camera, "h"),
            ("front_camera", self.front_camera, "f"),
        ]:
            pkl_dic["observation"][cam_name] = {
                "intrinsic_cv": cam_obj.get_intrinsic_matrix(),
                "extrinsic_cv": cam_obj.get_extrinsic_matrix(),
                "cam2world_gl": cam_obj.get_model_matrix(),
            }

        # RGB
        if self.data_type.get('rgb', False):
            head_rgba = self._get_camera_rgba(self.head_camera)
            front_rgba = self._get_camera_rgba(self.front_camera)

            if self.save_type.get('raw_data', True):
                if self.data_type.get('observer', False):
                    observer_rgba = self._get_camera_rgba(self.observer_camera)
                    save_img(self.file_path["observer_color"] + f"{self.PCD_INDEX}.png", observer_rgba)
                save_img(self.file_path["h_color"] + f"{self.PCD_INDEX}.png", head_rgba)
                save_img(self.file_path["f_color"] + f"{self.PCD_INDEX}.png", front_rgba)

            if self.save_type.get('pkl', True):
                pkl_dic["observation"]["head_camera"]["rgb"] = head_rgba[:, :, :3]
                pkl_dic["observation"]["front_camera"]["rgb"] = front_rgba[:, :, :3]

        # Segmentation
        if self.data_type.get('mesh_segmentation', False):
            head_seg = self._get_camera_segmentation(self.head_camera, level="mesh")
            front_seg = self._get_camera_segmentation(self.front_camera, level="mesh")
            if self.save_type.get('raw_data', True):
                save_img(self.file_path["h_seg_mesh"] + f"{self.PCD_INDEX}.png", head_seg)
                save_img(self.file_path["f_seg_mesh"] + f"{self.PCD_INDEX}.png", front_seg)
            if self.save_type.get('pkl', True):
                pkl_dic["observation"]["head_camera"]["mesh_segmentation"] = head_seg
                pkl_dic["observation"]["front_camera"]["mesh_segmentation"] = front_seg

        if self.data_type.get('actor_segmentation', False):
            head_seg = self._get_camera_segmentation(self.head_camera, level="actor")
            front_seg = self._get_camera_segmentation(self.front_camera, level="actor")
            if self.save_type.get('raw_data', True):
                save_img(self.file_path["h_seg_actor"] + f"{self.PCD_INDEX}.png", head_seg)
                save_img(self.file_path["f_seg_actor"] + f"{self.PCD_INDEX}.png", front_seg)
            if self.save_type.get('pkl', True):
                pkl_dic["observation"]["head_camera"]["actor_segmentation"] = head_seg
                pkl_dic["observation"]["front_camera"]["actor_segmentation"] = front_seg

        # Depth
        if self.data_type.get('depth', False):
            head_depth = self._get_camera_depth(self.head_camera)
            front_depth = self._get_camera_depth(self.front_camera)
            if self.save_type.get('raw_data', True):
                save_img(self.file_path["h_depth"] + f"{self.PCD_INDEX}.png", head_depth.astype(np.uint16))
                save_img(self.file_path["f_depth"] + f"{self.PCD_INDEX}.png", front_depth.astype(np.uint16))
            if self.save_type.get('pkl', True):
                pkl_dic["observation"]["head_camera"]["depth"] = head_depth
                pkl_dic["observation"]["front_camera"]["depth"] = front_depth

        # Endpose
        if self.data_type.get('endpose', False):
            endpose_data = self.endpose_transform(self.endpose, self.gripper_val)
            if self.save_type.get('raw_data', True):
                save_json(self.file_path["arm_ep"] + f"{self.PCD_INDEX}.json", endpose_data)
            if self.save_type.get('pkl', True):
                pkl_dic["endpose"] = np.array([
                    endpose_data["x"], endpose_data["y"], endpose_data["z"],
                    endpose_data["roll"], endpose_data["pitch"], endpose_data["yaw"],
                    endpose_data["gripper"],
                ])

        # Joint State
        if self.data_type.get('qpos', False):
            jointstate = {
                "effort": [0] * 7,
                "position": self.get_arm_jointState(),
                "velocity": [0] * 7,
            }
            if self.save_type.get('raw_data', True):
                save_json(self.file_path["arm_joint"] + f"{self.PCD_INDEX}.json", jointstate)
            if self.save_type.get('pkl', True):
                pkl_dic["joint_action"] = np.array(jointstate["position"])

        # PointCloud
        if self.data_type.get('pointcloud', False):
            head_pcd = self._get_camera_pcd(self.head_camera, point_num=0)
            front_pcd = self._get_camera_pcd(self.front_camera, point_num=0)

            if self.data_type.get("conbine", False):
                conbine_pcd = np.vstack((head_pcd, front_pcd))
            else:
                conbine_pcd = head_pcd

            pcd_array, index = conbine_pcd[:, :3], np.array(range(len(conbine_pcd)))
            if self.pcd_down_sample_num > 0:
                pcd_array, index = fps(conbine_pcd[:, :3], self.pcd_down_sample_num)
                index = index.detach().cpu().numpy()[0]

            if self.save_type.get('raw_data', True):
                ensure_dir(self.file_path["h_pcd"] + f"{self.PCD_INDEX}.pcd")
                o3d.io.write_point_cloud(self.file_path["h_pcd"] + f"{self.PCD_INDEX}.pcd",
                    self.arr2pcd(head_pcd[:, :3], head_pcd[:, 3:]))
                ensure_dir(self.file_path["f_pcd"] + f"{self.PCD_INDEX}.pcd")
                o3d.io.write_point_cloud(self.file_path["f_pcd"] + f"{self.PCD_INDEX}.pcd",
                    self.arr2pcd(front_pcd[:, :3], front_pcd[:, 3:]))
                if self.data_type.get("conbine", False):
                    ensure_dir(self.file_path["conbine_pcd"] + f"{self.PCD_INDEX}.pcd")
                    o3d.io.write_point_cloud(self.file_path["conbine_pcd"] + f"{self.PCD_INDEX}.pcd",
                        self.arr2pcd(pcd_array, conbine_pcd[index, 3:]))

            if self.save_type.get('pkl', True):
                pkl_dic["pointcloud"] = conbine_pcd[index]

        if self.save_type.get('pkl', True):
            save_pkl(self.file_path["pkl"] + f"{self.PCD_INDEX}.pkl", pkl_dic)

        self.PCD_INDEX += 1

    # ================================================================
    # [OVERRIDE] apply_dp - 7D single-arm DP (Vision Policy) execution
    # ================================================================
    def apply_dp(self, model, args):
        cnt = 0
        self.test_num += 1
        eval_video_log = args['eval_video_log']
        video_size = str(args['head_camera_w']) + 'x' + str(args['head_camera_h'])
        save_dir = 'dp/' + str(args['task_name']) + '_' + str(args['head_camera_type']) + '_' + str(args['expert_data_num']) + '_seed' + str(args['expert_seed'])

        if eval_video_log:
            import subprocess
            from pathlib import Path
            save_dir = Path('eval_video') / save_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            ffmpeg = subprocess.Popen([
                'ffmpeg', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', video_size, '-framerate', '10', '-i', '-',
                '-pix_fmt', 'yuv420p', '-vcodec', 'libx264', '-crf', '23',
                f'{save_dir}/{self.test_num}.mp4'
            ], stdin=subprocess.PIPE)

        success_flag = False
        self._update_render()
        if self.render_freq:
            self.viewer.render()
        self.actor_pose = True

        observation = self.get_obs()
        if eval_video_log:
            ffmpeg.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())

        while cnt < self.step_lim:
            observation = self.get_obs()
            obs = self.get_cam_obs(observation)
            obs['agent_pos'] = observation['joint_action']  # 7D
            model.update_obs(obs)
            actions = model.get_action()
            obs = model.get_last_obs()

            # 7D: actions[:, :6] = arm, actions[:, 6] = gripper
            arm_actions, gripper = actions[:, :6], actions[:, 6]
            current_qpos = obs['agent_pos'][:6]
            path = np.vstack((current_qpos, arm_actions))

            topp_flag = True
            try:
                times, pos, vel, acc, duration = self.planner.TOPP(path, 1 / 250, verbose=True)
                result = {'position': pos, 'velocity': vel}
                n_step = result["position"].shape[0]
                gripper = np.linspace(gripper[0], gripper[-1], n_step)
            except:
                topp_flag = False
                n_step = 1

            if n_step == 0:
                topp_flag = False
                n_step = 1

            cnt += actions.shape[0]
            obs_update_freq = max(1, n_step // actions.shape[0])

            now_id = 0 if topp_flag else int(1e9)
            i = 0

            while now_id < n_step:
                qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                self.robot.set_qf(qf)

                if topp_flag and now_id < n_step:
                    for j, arm_id in enumerate(self.arm_joint_ids):
                        self.active_joints[arm_id].set_drive_target(result["position"][now_id][j])
                        self.active_joints[arm_id].set_drive_velocity_target(result["velocity"][now_id][j])
                    if not self.fix_gripper:
                        self.active_joints[self.gripper_joint_id].set_drive_target(gripper[now_id])
                        self.active_joints[self.gripper_joint_id].set_drive_velocity_target(0.05)
                        self.gripper_val = gripper[now_id]
                    now_id += 1

                self.scene.step()
                self._update_render()

                if i != 0 and i % obs_update_freq == 0:
                    observation = self.get_obs()
                    obs = self.get_cam_obs(observation)
                    obs['agent_pos'] = observation['joint_action']
                    model.update_obs(obs)
                    self._take_picture()

                if i % 5 == 0:
                    self._update_render()
                    if self.render_freq and i % self.render_freq == 0:
                        self.viewer.render()

                i += 1
                if self.check_success():
                    success_flag = True
                    break
                if not self.actor_pose:
                    break

            self._update_render()
            if eval_video_log:
                ffmpeg.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())
            if self.render_freq:
                self.viewer.render()
            self._take_picture()
            print(f'step: {cnt} / {self.step_lim}', end='\r')

            if success_flag:
                print("\nsuccess!")
                self.suc += 1
                if eval_video_log: ffmpeg.stdin.close(); ffmpeg.wait(); del ffmpeg
                return
            if not self.actor_pose:
                break

        print("\nfail!")
        if eval_video_log: ffmpeg.stdin.close(); ffmpeg.wait(); del ffmpeg

    # ================================================================
    # [OVERRIDE] apply_dp3 - 7D single-arm DP3 (Point Cloud Policy) execution
    # ================================================================
    def apply_dp3(self, model, args):
        cnt = 0
        self.test_num += 1
        eval_video_log = args['eval_video_log']
        video_size = str(args['head_camera_w']) + 'x' + str(args['head_camera_h'])
        save_dir = 'dp3/' + str(args['task_name']) + '_' + str(args['head_camera_type']) + '_' + str(args['expert_data_num']) + '/seed' + str(args['expert_seed'])

        if eval_video_log:
            import subprocess
            from pathlib import Path
            save_dir = Path('eval_video') / save_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            ffmpeg = subprocess.Popen([
                'ffmpeg', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', video_size, '-framerate', '10', '-i', '-',
                '-pix_fmt', 'yuv420p', '-vcodec', 'libx264', '-crf', '23',
                f'{save_dir}/{self.test_num}.mp4'
            ], stdin=subprocess.PIPE)

        success_flag = False
        self._update_render()
        if self.render_freq:
            self.viewer.render()
        self.actor_pose = True

        observation = self.get_obs()
        if eval_video_log:
            ffmpeg.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())

        while cnt < self.step_lim:
            observation = self.get_obs()
            obs = dict()
            obs['point_cloud'] = observation['pointcloud']
            obs['agent_pos'] = observation['joint_action']
            assert obs['agent_pos'].shape[0] == 7, 'agent_pos shape error, expected 7D'

            actions = model.get_action(obs)

            # 7D: actions[:, :6] = arm, actions[:, 6] = gripper
            arm_actions, gripper = actions[:, :6], actions[:, 6]
            current_qpos = obs['agent_pos'][:6]
            path = np.vstack((current_qpos, arm_actions))

            topp_flag = True
            try:
                times, pos, vel, acc, duration = self.planner.TOPP(path, 1 / 250, verbose=True)
                result = {'position': pos, 'velocity': vel}
                n_step = result["position"].shape[0]
                gripper = np.linspace(gripper[0], gripper[-1], n_step)
            except:
                topp_flag = False
                n_step = 1

            if n_step == 0:
                topp_flag = False
                n_step = 1

            cnt += actions.shape[0]
            obs_update_freq = max(1, n_step // actions.shape[0])

            now_id = 0 if topp_flag else int(1e9)
            i = 0

            while now_id < n_step:
                qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                self.robot.set_qf(qf)

                if topp_flag and now_id < n_step:
                    for j, arm_id in enumerate(self.arm_joint_ids):
                        self.active_joints[arm_id].set_drive_target(result["position"][now_id][j])
                        self.active_joints[arm_id].set_drive_velocity_target(result["velocity"][now_id][j])
                    if not self.fix_gripper:
                        self.active_joints[self.gripper_joint_id].set_drive_target(gripper[now_id])
                        self.active_joints[self.gripper_joint_id].set_drive_velocity_target(0.05)
                        self.gripper_val = gripper[now_id]
                    now_id += 1

                self.scene.step()
                self._update_render()

                if i != 0 and i % obs_update_freq == 0:
                    observation = self.get_obs()
                    obs = dict()
                    obs['point_cloud'] = observation['pointcloud']
                    obs['agent_pos'] = observation['joint_action']
                    assert obs['agent_pos'].shape[0] == 7, 'agent_pos shape error'
                    model.update_obs(obs)
                    self._take_picture()

                if i % 5 == 0:
                    self._update_render()
                    if self.render_freq and i % self.render_freq == 0:
                        self.viewer.render()

                i += 1
                if self.check_success():
                    success_flag = True
                    break
                if not self.actor_pose:
                    break

            if eval_video_log:
                ffmpeg.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())
            self._update_render()
            if self.render_freq:
                self.viewer.render()
            self._take_picture()
            print(f'step: {cnt} / {self.step_lim}', end='\r')

            if success_flag:
                print("\nsuccess!")
                self.suc += 1
                if eval_video_log: ffmpeg.stdin.close(); ffmpeg.wait(); del ffmpeg
                return
            if not self.actor_pose:
                break

        print("\nfail!")
        if eval_video_log: ffmpeg.stdin.close(); ffmpeg.wait(); del ffmpeg

    # ================================================================
    # [OVERRIDE] apply_composed_policy - 7D MCDP composition
    # ================================================================
    def apply_composed_policy(self, model1, model2, args):
        cnt = 0
        self.test_num += 1
        eval_video_log = args['eval_video_log']
        video_size = str(args['head_camera_w']) + 'x' + str(args['head_camera_h'])
        save_dir = 'composed_dp/' + str(args['task_name']) + '_' + str(args['head_camera_type']) + '_' + str(args['expert_data_num']) + '_seed' + str(args['expert_seed']) + '_weight_' + str(args['dp_w']) + '_' + str(args['dp3_w'])

        if eval_video_log:
            dp_w = args['dp_w']
            dp3_w = args['dp3_w']
            import subprocess
            from pathlib import Path
            save_dir = Path('eval_video') / save_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            ffmpeg = subprocess.Popen([
                'ffmpeg', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', video_size, '-framerate', '10', '-i', '-',
                '-pix_fmt', 'yuv420p', '-vcodec', 'libx264', '-crf', '23',
                f'{save_dir}/{self.test_num}_{dp_w}_{dp3_w}.mp4'
            ], stdin=subprocess.PIPE)

        success_flag = False
        self._update_render()
        if self.render_freq:
            self.viewer.render()
        self.actor_pose = True

        observation = self.get_obs()
        if eval_video_log:
            ffmpeg.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())

        while cnt < self.step_lim:
            observation = self.get_obs()

            # Prepare DP (Vision)
            obs_dp2 = self.get_cam_obs(observation)
            obs_dp2['agent_pos'] = observation['joint_action']  # 7D
            model1.update_obs(obs_dp2)

            # Prepare DP3 (Point Cloud)
            obs_dp3 = dict()
            obs_dp3['point_cloud'] = observation['pointcloud']
            obs_dp3['agent_pos'] = observation['joint_action']
            assert obs_dp3['agent_pos'].shape[0] == 7, 'agent_pos shape error'

            # Core MCDP composition
            infer_data1 = model1.prepare_data()
            infer_data2 = model2.prepare_data(obs_dp3)
            actions = self.get_composed_action(infer_data1, infer_data2, args)

            # 7D: actions[:, :6] = arm, actions[:, 6] = gripper
            arm_actions, gripper = actions[:, :6], actions[:, 6]
            current_qpos = obs_dp3['agent_pos'][:6]
            path = np.vstack((current_qpos, arm_actions))

            topp_flag = True
            try:
                times, pos, vel, acc, duration = self.planner.TOPP(path, 1 / 250, verbose=True)
                result = {'position': pos, 'velocity': vel}
                n_step = result["position"].shape[0]
                gripper = np.linspace(gripper[0], gripper[-1], n_step)
            except:
                topp_flag = False
                n_step = 1

            if n_step == 0:
                topp_flag = False
                n_step = 1

            cnt += actions.shape[0]
            obs_update_freq = max(1, n_step // actions.shape[0])

            now_id = 0 if topp_flag else int(1e9)
            i = 0

            while now_id < n_step:
                qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                self.robot.set_qf(qf)

                if topp_flag and now_id < n_step:
                    for j, arm_id in enumerate(self.arm_joint_ids):
                        self.active_joints[arm_id].set_drive_target(result["position"][now_id][j])
                        self.active_joints[arm_id].set_drive_velocity_target(result["velocity"][now_id][j])
                    if not self.fix_gripper:
                        self.active_joints[self.gripper_joint_id].set_drive_target(gripper[now_id])
                        self.active_joints[self.gripper_joint_id].set_drive_velocity_target(0.05)
                        self.gripper_val = gripper[now_id]
                    now_id += 1

                self.scene.step()
                self._update_render()

                if i != 0 and i % obs_update_freq == 0:
                    observation = self.get_obs()
                    obs_dp2 = self.get_cam_obs(observation)
                    obs_dp2['agent_pos'] = observation['joint_action']
                    model1.update_obs(obs_dp2)

                    obs_dp3 = dict()
                    obs_dp3['point_cloud'] = observation['pointcloud']
                    obs_dp3['agent_pos'] = observation['joint_action']
                    self._take_picture()

                if i % 5 == 0:
                    self._update_render()
                    if self.render_freq and i % self.render_freq == 0:
                        self.viewer.render()

                i += 1
                if self.check_success():
                    success_flag = True
                    break
                if not self.actor_pose:
                    break

            if eval_video_log:
                ffmpeg.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())
            self._update_render()
            if self.render_freq:
                self.viewer.render()
            self._take_picture()
            print(f'step: {cnt} / {self.step_lim}', end='\r')

            if success_flag:
                print("\nsuccess!")
                self.suc += 1
                if eval_video_log: ffmpeg.stdin.close(); ffmpeg.wait(); del ffmpeg
                return
            if not self.actor_pose:
                break

        print("\nfail!")
        if eval_video_log: ffmpeg.stdin.close(); ffmpeg.wait(); del ffmpeg

    # ================================================================
    # [OVERRIDE] apply_policy_demo - 7D single-arm custom policy
    # ================================================================
    def apply_policy_demo(self, model):
        step_cnt = 0
        self.test_num += 1
        success_flag = False
        self._update_render()
        if self.render_freq:
            self.viewer.render()
        self.actor_pose = True

        while step_cnt < self.step_lim:
            obs = self.get_obs()
            actions = model.get_action(obs)  # (Horizon, 7)

            arm_actions, gripper = actions[:, :6], actions[:, 6]
            current_qpos = obs['joint_action'][:6]
            path = np.vstack((current_qpos, arm_actions))

            topp_flag = True
            try:
                times, pos, vel, acc, duration = self.planner.TOPP(path, 1 / 250, verbose=True)
                result = {'position': pos, 'velocity': vel}
                n_step = result["position"].shape[0]
                gripper = np.linspace(gripper[0], gripper[-1], n_step)
            except:
                topp_flag = False
                n_step = 1

            if n_step == 0:
                topp_flag = False
                n_step = 1

            step_cnt += actions.shape[0]
            obs_update_freq = max(1, n_step // actions.shape[0])

            now_id = 0 if topp_flag else int(1e9)
            i = 0

            while now_id < n_step:
                qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                self.robot.set_qf(qf)

                if topp_flag and now_id < n_step:
                    for j, arm_id in enumerate(self.arm_joint_ids):
                        self.active_joints[arm_id].set_drive_target(result["position"][now_id][j])
                        self.active_joints[arm_id].set_drive_velocity_target(result["velocity"][now_id][j])
                    if not self.fix_gripper:
                        self.active_joints[self.gripper_joint_id].set_drive_target(gripper[now_id])
                        self.active_joints[self.gripper_joint_id].set_drive_velocity_target(0.05)
                        self.gripper_val = gripper[now_id]
                    now_id += 1

                self.scene.step()
                self._update_render()

                if i % 5 == 0:
                    self._update_render()
                    if self.render_freq and i % self.render_freq == 0:
                        self.viewer.render()

                i += 1
                if self.check_success():
                    success_flag = True
                    break
                if not self.actor_pose:
                    break

            self._update_render()
            if self.render_freq:
                self.viewer.render()
            print(f'step: {step_cnt} / {self.step_lim}', end='\r')

            if success_flag:
                print("\nsuccess!")
                self.suc += 1
                return
            if not self.actor_pose:
                break

        print("\nfail!")

    # ================================================================
    # [INHERIT] The following methods are safe to use from Base_task:
    #   - __init__, setup_scene, create_table_and_wall
    #   - _get_camera_rgba, _get_camera_segmentation, _get_camera_depth
    #   - _get_camera_pcd, arr2pcd, get_camera_config
    #   - endpose_transform, dict_apply
    #   - get_composed_action (core MCDP noise blending)
    #   - get_grasp_pose_w_labeled_direction
    #   - get_grasp_pose_w_given_direction
    #   - get_target_pose_from_goal_point_and_direction
    #   - get_actor_goal_pose
    #   - play_once, check_success, pre_move (abstract/pass)
    # ================================================================
