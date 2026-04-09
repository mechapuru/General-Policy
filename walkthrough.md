# Sim-DP3 Compatibility Audit
**Date:** 2026-03-30
**Scope:** Full comparison of General Policy training/testing pipeline vs. Sim-DP3

---

## Summary

The **General Policy** codebase (SAPIEN/RoboTwin) and **Sim-DP3** (PyBullet) share the same `diffusion_policy_3d` module namespace, but differ in their dataset, env_runner, policy, and vision model implementations. Specifically:

- **DP3 training** ([train.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/train.py)) is largely identical between the two repos — Sim-DP3 is a direct port. ✅
- **Common utils** (pytorch_util, checkpoint_util, replay_buffer, sampler, model utils) are replicated cleanly. ✅ (with one prior bug fix)
- **DP-RGB support** requires [rgb_multi_image_obs_encoder.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/rgb_multi_image_obs_encoder.py), [rgb_crop_randomizer.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/rgb_crop_randomizer.py), and [rgb_model_getter.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/rgb_model_getter.py) — Sim-DP3 **has all three**. ✅
- **Dataset**: Sim-DP3 has a custom [RRCDataset](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/rrc_dataset.py#12-92) but it **only loads DP3 keys** (`state`, [action](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/dp3_policy.py#39-42), `point_cloud`). DP-RGB requires an `img` key. ❌
- **Policy**: Sim-DP3 has [dp_rgb_unet_policy.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp_rgb_unet_policy.py) but **[train.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/train.py) only instantiates [DP3](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/dp3_policy.py#32-51)**. Training DP-RGB requires a separate workspace or an updated [train.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/train.py). ❌
- **[composed_policy.py](file:///home/paddy/rrc/1cross/General-Policy/script/eval_policy_composed_policy.py)**: Not yet implemented anywhere in Sim-DP3. ❌

---

## Detailed Module-by-Module Audit

### 1. `common/` — Utility Functions

| File | Function | In General Policy | In Sim-DP3 | Notes |
|---|---|---|---|---|
| [pytorch_util.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/common/pytorch_util.py) | [dict_apply](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/common/pytorch_util.py#6-17) | ✅ | ✅ | Used everywhere |
| [pytorch_util.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/common/pytorch_util.py) | [optimizer_to](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/common/pytorch_util.py#43-49) | ✅ | ✅ | Used in [train.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/train.py) |
| [pytorch_util.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/common/pytorch_util.py) | [replace_submodules](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/common/pytorch_util.py#43-76) | ❌ (missing) | ✅ | **Was just added (bug fix)**. Original only had it in the `Diffusion-Policy` repo |
| [pytorch_util.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/common/pytorch_util.py) | [pad_remaining_dims](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/common/pytorch_util.py#18-21), [dict_apply_split](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/common/pytorch_util.py#22-32), [dict_apply_reduce](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/common/pytorch_util.py#33-41) | ✅ | ✅ | Present, not actively used in training loop |
| [checkpoint_util.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/common/checkpoint_util.py) | [TopKCheckpointManager](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/common/checkpoint_util.py#4-60) | ✅ | ✅ | Identical |
| [replay_buffer.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/common/replay_buffer.py) | `ReplayBuffer` | ✅ | ✅ | Identical (20KB, same file) |
| [sampler.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/common/sampler.py) | `SequenceSampler`, `get_val_mask`, `downsample_mask` | ✅ | ✅ | Identical |
| [logger_util.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/common/logger_util.py) | `LargestKRecorder` | ✅ | ✅ | Minor size diff (1374 vs 1098 bytes), likely a subset |
| [model_util.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/common/model_util.py) | — | ✅ | ✅ | Identical |

---

### 2. [dataset/](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/rrc_dataset.py#48-59) — Data Loading

| File | Class | In General Policy | In Sim-DP3 | Notes |
|---|---|---|---|---|
| [robot_dataset.py](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/dataset/robot_dataset.py) | [RobotDataset](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/dataset/robot_dataset.py#13-91) (DP3) | ✅ | ✅ as [rrc_dataset.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/rrc_dataset.py) → [RRCDataset](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/rrc_dataset.py#12-92) | Loads `state`, [action](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/dp3_policy.py#39-42), `point_cloud` — Compatible for DP3 training |
| *(for DP-RGB)* | *[RRCDataset](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/rrc_dataset.py#12-92) with `img` key* | N/A | ❌ **MISSING** | `RRCDataset.__getitem__` does NOT load or return `img`. The [dp_rgb.yaml](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/dp_rgb.yaml) config will instantiate this, but there's no `img` in the Zarr or dataset. **Training will fail** when DP-RGB tries to access `obs["image"]` |

> [!CAUTION]
> [rrc_dataset.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/rrc_dataset.py) must be updated for DP-RGB: (1) the Zarr must have an `img` key (already in [create_zarr.py](file:///home/paddy/rrc/1cross/General-Policy/sim-robot/create_zarr.py)), (2) [_sample_to_data](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/rrc_dataset.py#74-86) must be updated to load `img` and return it as `obs["image"]`.

---

### 3. `env_runner/` — Evaluation Runner

| File | Class | In General Policy | In Sim-DP3 | Notes |
|---|---|---|---|---|
| [robot_runner.py](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py) | [RobotRunner](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#15-149) | ✅ | ❌ | SAPIEN-specific, incompatible with PyBullet |
| [pybullet_runner.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/pybullet_runner.py) | [XArm6PyBulletRunner](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/pybullet_runner.py#62-787) | ❌ | ✅ | Custom PyBullet runner — feature-complete |
| — | [get_action(policy, obs)](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/dp3_policy.py#39-42) | via [RobotRunner](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#15-149) | ✅ | `pybullet_runner` calls `policy.predict_action()` directly |
| — | [prepare_data](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#113-145) / [prepare_infer_data](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp_rgb_unet_policy.py#188-249) | ✅ in `robot_runner` | ❌ | `pybullet_runner` does NOT have [prepare_data()](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#113-145). This is needed for the composed policy inference that splits the denoising step. |

> [!IMPORTANT]
> The composed policy workflow (GPC) calls `dp3.env_runner.prepare_data()` and `dp.runner.prepare_data()` to get intermediate denoising tensors, then blends them. `pybullet_runner` currently only has a [run()](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#147-149) method. For Phase 3/4, [prepare_data()](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#113-145) must be added to `pybullet_runner`.

---

### 4. [policy/](file:///home/paddy/rrc/1cross/General-Policy/script/eval_policy_composed_policy.py#30-50) — Policy Models

| File | Class | In General Policy | In Sim-DP3 | Notes |
|---|---|---|---|---|
| [dp3.py](file:///home/paddy/rrc/1cross/General-Policy/script/pkl2zarr_dp3.py) | [DP3](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/dp3_policy.py#32-51) | ✅ | ✅ | Core DP3 policy, present and functional |
| [base_policy.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/base_policy.py) | `BasePolicy` | ✅ | ✅ | Identical |
| [dp_rgb_unet_policy.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp_rgb_unet_policy.py) | [DiffusionUnetImagePolicy](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp_rgb_unet_policy.py#15-330) | ❌ | ✅ | **New file for DP-RGB** — all inference methods present |
| [dp_rgb_base_policy.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp_rgb_base_policy.py) | `BaseImagePolicy` | ❌ | ✅ | Base class for DP-RGB |
| [composed_policy.py](file:///home/paddy/rrc/1cross/General-Policy/script/eval_policy_composed_policy.py) | `ComposedPolicyWrapper` | ❌ | ❌ | **Phase 3 — NOT YET IMPLEMENTED** |

---

### 5. `model/vision/` — Observation Encoders

| File | Class/Function | In General Policy | In Sim-DP3 | Notes |
|---|---|---|---|---|
| [pointnet_extractor.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py) | `PointNetEncoderXYZ` etc. | ✅ | ✅ | Same sizes — identical |
| [rgb_multi_image_obs_encoder.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/rgb_multi_image_obs_encoder.py) | [MultiImageObsEncoder](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/rgb_multi_image_obs_encoder.py#11-296) | ❌ | ✅ | **Added for DP-RGB** |
| [rgb_crop_randomizer.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/rgb_crop_randomizer.py) | `CropRandomizer` | ❌ | ✅ | Augmentation for RGB training |
| [rgb_model_getter.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/rgb_model_getter.py) | `get_resnet` | ❌ | ✅ | Backbone factory for ResNet |

---

### 6. [train.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/train.py) — Training Workspace

| Feature | General Policy | In Sim-DP3 | Notes |
|---|---|---|---|
| `TrainDP3Workspace.run()` | ✅ | ✅ | Identical logic, slightly cleaner |
| [compute_loss](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp_rgb_unet_policy.py#255-330) interface | `raw_loss, loss_dict = model.compute_loss(batch)` | ✅ | ✅ | Both use the same two-tuple return |
| DP-RGB training | ❌ | ❌ | **[train.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/train.py) hardcodes `from diffusion_policy_3d.policy.dp3 import DP3`**. To train DP-RGB, the workspace must instantiate [DiffusionUnetImagePolicy](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp_rgb_unet_policy.py#15-330) via hydra cfg. The type hint `self.model: DP3` needs to be generalized or a new `TrainDPRGBWorkspace` created. |
| Rollout during training | Commented out (`# runner_log = env_runner.run(policy, dataset=dataset)`) | Uses `env_runner.run(policy)` (no dataset) | Sim-DP3 runner ignores dataset during rollout |

> [!NOTE]
> In practice, DP-RGB training **will** work via Hydra's dynamic `cfg.policy._target_` even with the [DP3](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/dp3_policy.py#32-51) type annotation, as long as [DiffusionUnetImagePolicy](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp_rgb_unet_policy.py#15-330) exposes the same [compute_loss(batch)](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp_rgb_unet_policy.py#255-330) and [set_normalizer(normalizer)](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp_rgb_unet_policy.py#252-254) APIs (which it does). No code change is strictly needed for training to launch, but the `img` zarr key issue in [rrc_dataset.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/rrc_dataset.py) will be the blocking error.

---

## Blocking Issues Summary

| Priority | Issue | Files Affected | Status |
|---|---|---|---|
| 🔴 **Blocker** | [rrc_dataset.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/rrc_dataset.py) does not load or return `img`/`image` — DP-RGB training will crash at first batch | [dataset/rrc_dataset.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/rrc_dataset.py), [create_zarr.py](file:///home/paddy/rrc/1cross/General-Policy/sim-robot/create_zarr.py) | ❌ Not done |
| 🔴 **Blocker** | [composed_policy.py](file:///home/paddy/rrc/1cross/General-Policy/script/eval_policy_composed_policy.py) not implemented | `policy/composed_policy.py` | ❌ Phase 3, not started |
| 🟡 **For GPC** | [pybullet_runner.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/pybullet_runner.py) lacks [prepare_data()](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#113-145) / [reset_obs()](file:///home/paddy/rrc/1cross/General-Policy/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py#70-72) methods needed by the composed policy loop | [env_runner/pybullet_runner.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/pybullet_runner.py) | ❌ Not done |
| 🟢 **Done** | [replace_submodules](file:///home/paddy/rrc/1cross/General-Policy/policy/Sim-DP3/3D-Diffusion-Policy/diffusion_policy_3d/common/pytorch_util.py#43-76) was missing from Sim-DP3's [pytorch_util.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/common/pytorch_util.py) | [common/pytorch_util.py](file:///home/paddy/rrc/1cross/General-Policy/policy/Diffusion-Policy/diffusion_policy/common/pytorch_util.py) | ✅ Fixed |
