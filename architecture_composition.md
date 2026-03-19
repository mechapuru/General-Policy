# Architectural Document: General Policy Composition (GPC)

## 1. Overview
**General Policy Composition (GPC)** is a training-free framework designed to ensemble heterogeneous robot policies (e.g., RGB-based and Point Cloud-based) at inference time. Instead of training a single multi-modal model, GPC combines the "knowledge" of independent pre-trained experts by blending their predicted noise residuals during the diffusion denoising process.

### Core Philosophy
*   **Decoupled Training:** Experts are trained on their respective modalities independently.
*   **Distribution-Level Composition:** Composition happens in the action-score space (action gradient space).
*   **Zero-Shot Integration:** No additional training or fine-tuning is required to combine policies.

---

## 2. System Architecture

The pipeline consists of three distinct modules:
1.  **Expert A (DP - RGB):** A 2D Diffusion Policy using ResNet-18 for image encoding and a 1D CNN-based U-Net for action generation.
2.  **Expert B (DP3 - 3D):** A 3D Diffusion Policy using PointNet for point cloud encoding and a similar 1D CNN-based U-Net.
3.  **The GPC Orchestrator:** Located within the environment logic, it manually iterates through diffusion steps to combine the experts.

---

## 3. Data Preparation Phase

### A. Collection (RoboTwin Simulation)
*   **Execution:** `bash run_task.sh ${task_name}`
*   **Mechanism:** The simulation searches for valid random seeds. Once a successful trajectory is found, it is replayed to record multi-modal observations.
*   **Raw Output:** `.pkl` files containing:
    *   `rgb`: 320x180 images (usually from an L515 head camera).
    *   `depth`: Synchronized depth maps.
    *   `pointcloud`: 2500 sampled points in the robot base frame.
    *   `joint_action`: 14D vector (7D per arm) representing absolute joint positions.

### B. Format Conversion (Zarr)
The raw data is converted into Zarr format to allow for high-speed chunked access during training.
*   **RGB Path:** `script/pkl2zarr_dp.py` creates a Zarr with `head_camera` and `action` keys.
*   **3D Path:** `script/pkl2zarr_dp3.py` creates a Zarr with `point_cloud` and `action` keys.

---

## 4. Independent Training Phase

Each expert is trained to predict the noise $\epsilon$ added to an action trajectory.

### Training Constraints
For composition to work, both models **must** share an aligned action distribution space:
1.  **Consistent Scheduler:** Both must use the same noise scheduler (e.g., `DDPMScheduler`).
2.  **Consistent Prediction Type:** Both must predict either `epsilon` (noise) or `sample` (action).
3.  **Identical Horizon:** Both must have the same prediction horizon (e.g., 8 steps) and observation history (e.g., 2-3 steps).

### Model Structures
*   **DP (RGB):** `Images` $\rightarrow$ `ResNet-18` $\rightarrow$ `Global Cond` $\rightarrow$ `1D U-Net`.
*   **DP3 (3D):** `Point Cloud` $\rightarrow$ `PointNet` $\rightarrow$ `Global Cond` $\rightarrow$ `1D U-Net`.

---

## 5. Compositional Testing Phase (GPC Inference)

The composition logic is implemented in the `prepare_infer_data` method and the `apply_composed_policy` environment loop.

### A. The "Surgery" Hook (`prepare_infer_data`)
Standard policies only return a final action. The GPC version of the policies implements `prepare_infer_data()`, which returns:
*   The encoded **Observation Features**.
*   The **Noise Scheduler** instance.
*   The **U-Net Model** itself.
*   The **Action Normalizer**.

### B. The Distribution-Level Blend
During evaluation (`script/eval_policy_composed_policy.py`), the environment executes a manual denoising loop. The exact core mathematical composition resides in `envs/base_task.py` within the `get_composed_action` function [(lines 1707-1728)](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py#L1707-L1728):

1.  **Input:** Current RGB image ($O_{rgb}$) and Point Cloud ($O_{pcd}$).
2.  **Noise Prediction:**
    *   `model_output_dp`: $\epsilon_{rgb} = \text{Model}_{rgb}(\text{Trajectory}_t, \text{Step}_t, O_{rgb})$ [(line 1713)](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py#L1713)
    *   `model_output_dp3`: $\epsilon_{pcd} = \text{Model}_{pcd}(\text{Trajectory}_t, \text{Step}_t, O_{pcd})$ [(line 1716)](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py#L1716)
3.  **Linear Combination [(line 1720)](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py#L1720):**
    `model_output = dp_w * model_output_dp + dp3_w * model_output_dp3`
    *(Where $w_{rgb} + w_{pcd} = 1.0$)*
4.  **Denoising Step via DDPMScheduler [(line 1724)](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py#L1724):**
    `trajectory = dp3_scheduler.step(model_output, t, trajectory).prev_sample`
5.  **Repeat:** For $N$ inference steps.

### C. Execution
*   **Entry Point:** `bash eval_composed.sh ...`
*   **Parameters:** `dp_w` (RGB weight) and `dp3_w` (3D weight).
*   **Logic Location:** `envs/base_task.py` contains the `apply_composed_policy` method that orchestrates the per-step blending described above.

---

## 6. Summary of Component Locations

| Logic Component | File Path |
| :--- | :--- |
| **RGB Policy** | `policy/Diffusion-Policy/diffusion_policy/policy/diffusion_unet_image_policy.py` |
| **3D Policy** | `policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py` |
| **Composition Loop** | `envs/base_task.py` (Method: `apply_composed_policy` calling [`get_composed_action`](file:///home/paddy/rrc/1cross/General-Policy/envs/base_task.py#L1664)) |
| **Inference Orchestrator** | `script/eval_policy_composed_policy.py` |
| **Data Preprocessing** | `script/pkl2zarr_dp.py` & `script/pkl2zarr_dp3.py` |
