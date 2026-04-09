# GPC-PyBullet Migration Roadmap

> [!IMPORTANT]
> **CURRENT POSITION: Phase 2.1 (DP-RGB Training Infrastructure)**
> *   All Phase 1 (Core Foundations) tasks are verified and completed.
> *   We are now establishing the ResNet-based training pipeline for PyBullet image data.

## Phase 1: Infrastructure Foundations ✅ (COMPLETED)
1.  **Architecture Audit**: Mapping `Sim-DP3` vs Original GPC ([walkthrough.md](file:///home/paddy/.gemini/antigravity/brain/d125cdce-dc18-4ed5-81e9-e5085b1fbddc/walkthrough.md)).
2.  **Missing Utils**: Ported [replace_submodules](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/common/pytorch_util.py#43-76) to [pytorch_util.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/common/pytorch_util.py) (Line 43) to fix ResNet initialization errors.
3.  **Action Space Logic**: Confirmed [create_zarr.py](file:///home/paddy/rrc/1cross/General-Policy/sim-robot/create_zarr.py) implements the 7D action constraint (6D pos + 1D delta gripper).

## Phase 2: DP-RGB Training Infrastructure 🔄 (ACTIVE)
*The goal is to enable training of the vision-based baseline on PyBullet images.*

1.  **Step 2.1: Dataset Logic**: Implement `RRCImageDataset` ([rrc_image_dataset.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/rrc_image_dataset.py)).
    - *Dependency*: Must load `state`, [action](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#88-112), and `img` zarr keys.
2.  **Step 2.2: Task Configuration**: Author [task/rrc_sim_dp_rgb.yaml](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/rrc_sim_dp_rgb.yaml).
    - *Constraint*: Observation space must be specified as `[3, 224, 224]` to match the original ResNet-18 pooling.
3.  **Step 2.3: Training Workspace**: Implement [train_dp_rgb.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/train_dp_rgb.py) mirroring [RobotWorkspace](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/workspace/robotworkspace.py#33-279).
    - *Logic*: Needs to handle multi-image history processing as performed in `robotworkspace.py:33`.

## Phase 3: PyBullet Runner Alignment ⏳ (UPCOMING)
*The runner must expose the GPC Inference API to allow multi-model noise blending.*

1.  **Step 3.1: History Management**: Add [reset_obs()](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#70-72) and [update_obs()](file:///home/paddy/rrc/1cross/General-Policy/script/eval_policy_composed_policy.py#56-58) to [XArm6PyBulletRunner](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/pybullet_runner.py#62-787).
    - *Reference*: Mirror `dp_runner.py:112` fordeque-based observation buffering.
2.  **Step 3.2: Denoising Interface**: Implement [prepare_data()](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#113-145) in the runner.
    - *Logic*: This function returns the clean observation dictionary, device-transferred and ready for the policy's internal denoising loop.

## Phase 4: Composition Engine (MCDP) ⏳ (UPCOMING)
*The core GPC "Distribution-Level Composition" logic.*

1.  **Step 4.1: Policy Wrapper**: Create `policy/composed_policy.py`.
2.  **Step 4.2: Noise Blending**: Implement the dual-model prediction loop based on the original `base_task.py:1720`.
    - **Mathematics**: $\epsilon_{composed} = w_{dp3} \epsilon_{dp3} + w_{rgb} \epsilon_{rgb}$.
3.  **Step 4.3: Scheduler Sync**: Ensure the wrapper drives a shared `noise_scheduler` (DDIM/DDPM) for both residuals.

## Phase 5: Verification & Unified Eval ⏳ (UPCOMING)
1.  **Unified Script**: Create `eval_composed.py` supporting `--mode [dp3 | rgb | compose]`.
2.  **Metrics**: Log success rates and collision counts to `wandb`.
