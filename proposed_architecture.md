# Proposed Architecture for PyBullet General Policy Composition

## 1. Goal Description
The objective is to migrate the testing, training, and General Policy Composition (GPC) framework to the custom PyBullet (`sim-robot`) environment. 

### Terminology Disambiguation
There are currently two distinct repositories in your workspace:
1.  **RoboTwin DP3 Repo (`policy/3D-Diffusion-Policy/`)**: The original codebase tied to SAPIEN.
2.  **PyBullet DP3 Repo (`policy/DP3-sim/3D-Diffusion-Policy/`)**: The custom fork configured specifically for PyBullet environments.

**All proposed training and inference modifications will occur exclusively within the PyBullet DP3 Repo (`policy/DP3-sim/3D-Diffusion-Policy/`).**  

## 2. User Review Required
> [!IMPORTANT]
> - Please review the justifications provided for a unified evaluation script (`eval_pipeline.py`).
> - Please review the explicit handling of the Gripper dimension in the Zarr generation.
> - **[NEW]** Please review Section 5: Implementation Roadmap. Do the file-level tasks and verification gates align with the deployment strategy?

---

## 3. Proposed Changes

### A. Data Preprocessing: Absolute vs Delta Actions
*   **Approach:** We will modify `create_zarr.py` to support explicit action space parameterization via command-line arguments (e.g., `--action-mode absolute` or `--action-mode delta`).
    *   **Handling the Gripper (7th Dimension):** The PyBullet environment runner (`pybullet_runner.py`) uses internal threshold checks (`> 0.1` vs `< -0.05`) to discretize continuous gripper outputs. Therefore, regardless of whether the 6 *arm joints* are set to absolute or delta mode, the **7th gripper dimension MUST remain a delta value** ($Action\_Gripper_t = \text{Target\_Gripper}_{t+1} - \text{Current\_Gripper}_t$). This preserves PyBullet's current mechanical assumption without risking backwards incompatibility.

### B. Baseline Architecture Verification (DP3 vs DP-RGB)
*   **Verification:** The core denoising framework is identical across DP3, DP-RGB, and the original RoboTwin DP3 setup. 
    *   All three architectures mathematically rely on the exact same **1D CNN U-Net** structure.
    *   The only architectural divergence is the observation encoder: DP3 utilizes a **PointNet** backbone, whereas DP-RGB will utilize a **ResNet** backbone. This 1:1 U-Net parity is what guarantees the GPC noise predictions ($w_{rgb} \epsilon_{rgb} + w_{pcd} \epsilon_{pcd}$) map to identical latent distributions safely.

### C. The `ComposedPolicyWrapper` (Core Decoupled Logic)
*   **Context:** In the original RoboTwin repository, scheduler synchronization was handled implicitly/dangerously. SAPIEN's `base_task.py` just arbitrarily picked DP3's `noise_scheduler.timesteps` to govern the loop and silently ignored DP-RGB's scheduler, assuming their YAML configs were perfectly identical.
*   **Approach:** Build a standalone Policy Wrapper inside the PyBullet DP3 Repo (`policy.composed_policy.py`).
    *   **Safeguard:** We add an explicit `assert type(dp_rgb.noise_scheduler) == type(dp3.noise_scheduler)` during instantiation. This prevents silent but mathematically disastrous composition failures if a developer tunes one model's steps but forgets the other.
    *   **Implementation:** It splits `obs_dict`, queries both backbones, blends the noise linearly, and steps the shared internal scheduler.

### D. Single Unified Execution Script
*   **Context:** Rather than maintaining `eval_checkpoint.py` (DP3 only) and adding an `eval_composed_checkpoint.py`.
*   **Approach:** We will create a single unified `eval_pipeline.py --mode [dp3 | dp_rgb | compose]`.
    *   **Justification:** A single entry script ensures the `pybullet_runner` validation setup (e.g., seeding, log metrics, video capture, initialization wrappers) occurs in one place. If a bug is fixed in the PyBullet testing loop, that fix immediately benefits all three inference modes, drastically reducing code duplication and maintenance debt.

---

## 5. Implementation Roadmap

This section outlines the systematically phased approach to authoring the codebase. We will not proceed to subsequent phases until the Verification Step for the current phase has been fully passed.

### Phase 1: Action Space Toggles for Zarr Generation
*   **Target File:** `sim-robot/create_zarr.py`
*   **Modifications:** Use `argparse` to accept `--action-mode` (`absolute` or `delta`). Wrap line 72 (`actions[:, 6] = actions[:, 6] - states[:, 6]`) within logic ensuring the 7th dimension remains delta under all configurations. If `--action-mode delta` is declared, compute deltas across the remaining 6 positional joints dynamically. Format output paths to designate `dataset_{mode}.zarr`.
*   **Verification Step 1:** 
    *   Execute `python create_zarr.py --action-mode absolute` 
    *   Execute `python create_zarr.py --action-mode delta` 
    *   Draft a rapid Python script to load both output Zarr arrays and assert mathematically that the absolute vs delta offsets are correctly bounded and formatted within the 7D constraint.
*   **Git Checkpoint:**
    ```bash
    git add sim-robot/create_zarr.py
    git commit -m "feat(data): add absolute and delta action space toggles to zarr generation"
    ```

### Phase 2: DP-RGB Environment & Training
*   **Target File:** `policy/DP3-sim/3D-Diffusion-Policy/diffusion_policy_3d/config/dp_rgb.yaml` (New File)
*   **Modifications:** Duplicate the base `dp3.yaml` configuration. Update the observation keys from `point_cloud` to `image`. Replace the visual encoder backbone inside `shape_meta` to leverage the standard visually pooled ResNet instead of the PointNet.
*   **Verification Step 2:**
    *   Execute `bash policy/DP3-sim/scripts/train_policy.sh dp_rgb <task_name> 0` for 20 epochs.
    *   Verify locally through `wandb` terminal logging that the loss metrics are calculating and converging successfully on the PyBullet Zarr dataset without dimension-mismatch traceback errors.
*   **Git Checkpoint:**
    ```bash
    git add policy/DP3-sim/3D-Diffusion-Policy/diffusion_policy_3d/config/dp_rgb.yaml
    git commit -m "feat(config): introduce DP-RGB training config with ResNet encoder"
    ```

### Phase 3: Mathematical Composition Engine
*   **Target File:** `policy/DP3-sim/3D-Diffusion-Policy/diffusion_policy_3d/policy/composed_policy.py` (New File)
*   **Modifications:** Construct the `ComposedPolicyWrapper` subclassing `BaseImagePolicy`. Implement local `__init__` capturing and enforcing parameter/scheduler parity. Develop the multi-modal `predict_action` function combining internal noise residuals using hyperparameter weights ($w_{rgb}, w_{pcd}$).
*   **Verification Step 3:** 
    *   Write a self-contained unit test (`test_composed_policy.py`) initializing the wrapper with dummy models. 
    *   Pass a mocked PyBullet observation dictionary and mathematically assert that `predict_action` returns the accurate unnormalized continuous tensor output shape (e.g., `[batch, n_action_steps, action_dim]`).
*   **Git Checkpoint:**
    ```bash
    git add policy/DP3-sim/3D-Diffusion-Policy/diffusion_policy_3d/policy/composed_policy.py
    git add policy/DP3-sim/3D-Diffusion-Policy/diffusion_policy_3d/policy/__init__.py
    git commit -m "feat(policy): implement ComposedPolicyWrapper for test-time DP3+RGB composition"
    ```

### Phase 4: Unified Evaluation Framework
*   **Target File:** `policy/DP3-sim/eval_pipeline.py` (New File)
*   **Modifications:** Author a unified CLI integrating `argparse` mapping `--mode` (e.g. `dp3`, `dp-rgb`, `compose`). Instantiate `pybullet_runner` universally and programmatically load the `ComposedPolicyWrapper` exclusively for the `compose` flag. Route the wrapper back out to `env_runner.run()`.
*   **Verification Step 4:** 
    *   Execute all three sub-modes (`--mode dp3`, `--mode dp_rgb`, `--mode compose`) iteratively over a brief 10-episode PyBullet test block.
    *   Ensure PyBullet correctly spins up the simulator UI, drives the robot via discretized actions, logs MAE differences to the text log, and saves the final mp4 visual telemetry accurately across every format.
*   **Git Checkpoint:**
    ```bash
    git add policy/DP3-sim/eval_pipeline.py
    git commit -m "feat(eval): unify pybullet inference pipeline supporting dp3, dp_rgb, and compose modes"
    ```
