# GPC Deep Dive: Technical Reading Guide

This guide provides a structured roadmap to understanding the **General Policy Composition (GPC)** framework. It covers the three pillars of the system: Data loading (Zarr), Training (DP3 vs DP-RGB), and Test-time Composition (MCDP).

## 1. Data Protocol (Zarr & Datasets)
GPC relies on a specific Zarr structure containing `state`, [action](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/env_runner/dp_runner.py#82-110), `point_cloud`, and `img` (or camera-specific keys like `head_camera`).

- **DP3 (Point Cloud) Dataset**
  - **File**: [robot_dataset.py](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/dataset/robot_dataset.py)
  - **Lines 26-27**: Shows the keys loaded from Zarr (`state`, [action](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/env_runner/dp_runner.py#82-110), `point_cloud`).
  - **Lines 60-80**: Shows the normalization logic for robot states and point clouds.

- **DP-RGB (Vision) Dataset**
  - **File**: [robot_image_dataset.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/dataset/robot_image_dataset.py)
  - **Lines 28-32**: Shows image key loading (`head_camera`, etc.).
  - **Lines 110-130**: Image normalization using `get_image_range_normalizer` (mapping [0, 255] to [0, 1]).

---

## 2. Training Workspaces
The two policies are trained in separate workspaces with different model architectures (PointNet vs ResNet/CNN).

- **DP3 Training (3D Diffusion)**
  - **File**: [train.py](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/train.py)
  - **Line 46**: [TrainDP3Workspace](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/train.py#46-473) class entry.
  - **Lines 196-200**: The core training step: `raw_loss, loss_dict = self.model.compute_loss(batch)`.

- **DP-RGB Training (Image Diffusion)**
  - **File**: [robotworkspace.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/workspace/robotworkspace.py)
  - **Line 33**: [RobotWorkspace](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/workspace/robotworkspace.py#33-279) class entry.
  - **Lines 120-150**: The training loop (standard Diffusion Policy implementation).

---

## 3. The Composition Layer (MCDP)
The "magic" happens during inference, where the noise predictions from both policies are blended at each denoising step.

- **The Main Evaluation Entry**
  - **File**: [eval_policy_composed_policy.py](file:///home/paddy/rrc/1cross/General-Policy/script/eval_policy_composed_policy.py)
  - **Lines 216-217**: Instantion of dual policies ([DP](file:///home/paddy/rrc/1cross/General-Policy/script/eval_policy_dp.py#47-61) and [DP3](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/dp3_policy.py#32-51)).
  - **Line 127**: Calling [apply_composed_policy](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py#1755-1982) on the environment task.

- **The Multi-Policy Inference Loop**
  - **File**: [base_task.py](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py)
  - **Line 1755**: [apply_composed_policy](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py#1755-1982) entry point.
  - **Lines 1815-1822**: Prepares denoising data from BOTH models and calls [get_composed_action](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py#1664-1753).

- **The Noise Blending Logic (THE CORE)**
  - **File**: [base_task.py](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py)
  - **Lines 1664-1752**: [get_composed_action](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py#1664-1753) implementation.
  - **Line 1697**: Initialization of the initial noise trajectory (shared between models).
  - **Lines 1720**: **Distribution-level composition**:
    ```python
    model_output = dp_w * model_output_dp + dp3_w * model_output_dp3
    ```
    This weighted average of the predicted noise residuals is what allows GPC to benefit from both vision and 3D geometric features.

---

## 4. Environment Runners (Inference API)
For the composition to work, the runners must support [prepare_data](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#113-145) which returns internal diffusion states (timesteps, noise, conditions) instead of just the final action.

- **DP-RGB Runner**: [dp_runner.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/env_runner/dp_runner.py) (Lines 112-139).
- **DP3 Policy Wrapper**: [dp3_policy.py](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/dp3_policy.py) contains the [prepare_data](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#113-145) wrapper for the 3D model.
