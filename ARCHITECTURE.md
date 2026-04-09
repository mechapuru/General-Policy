# Sim-DP3 Architecture Specification

**Repository:** `policy/Sim-DP3/`
**Goal:** Train DP3 (point cloud) and DP-RGB (image) policies independently on the PyBullet `rrc_sim` task, then compose them into a General Policy at test time.

---

## Repository Layout (Target State)

```
policy/Sim-DP3/
│
├── 3D-Diffusion-Policy/
│   ├── train.py                          ← TrainDP3Workspace  (DP3, exists ✅)
│   ├── train_dp_rgb.py                   ← TrainDPRGBWorkspace (DP-RGB, NEW ❌)
│   │
│   └── diffusion_policy_3d/
│       ├── config/
│       │   ├── dp3.yaml                  ← DP3 top-level config (exists ✅)
│       │   ├── dp_rgb.yaml               ← DP-RGB top-level config (exists ✅)
│       │   └── task/
│       │       ├── rrc_sim.yaml          ← DP3 task (dataset, runner, shape_meta) (exists ✅)
│       │       └── rrc_sim_dp_rgb.yaml   ← DP-RGB task (NEW ❌)
│       │
│       ├── dataset/
│       │   ├── rrc_dataset.py            ← RRCDataset for DP3 (exists ✅)
│       │   └── rrc_image_dataset.py      ← RRCImageDataset for DP-RGB (NEW ❌)
│       │
│       ├── env_runner/
│       │   └── pybullet_runner.py        ← XArm6PyBulletRunner (exists ✅, needs additions ⚠️)
│       │
│       ├── policy/
│       │   ├── dp3.py                    ← DP3 policy (exists ✅)
│       │   ├── dp_rgb_unet_policy.py     ← DP-RGB policy (exists ✅)
│       │   └── composed_policy.py        ← ComposedPolicyWrapper (NEW ❌)
│       │
│       ├── model/
│       │   ├── vision/
│       │   │   ├── pointnet_extractor.py       ← DP3 encoder (✅)
│       │   │   ├── rgb_multi_image_obs_encoder.py ← DP-RGB encoder (✅)
│       │   │   ├── rgb_crop_randomizer.py      ← RGB augmentation (✅)
│       │   │   └── rgb_model_getter.py         ← get_resnet() (✅)
│       │   ├── common/
│       │   │   ├── normalizer.py           (✅)
│       │   │   ├── lr_scheduler.py          (✅)
│       │   │   └── ...
│       │   └── diffusion/
│       │       ├── ema_model.py             (✅)
│       │       └── ...
│       │
│       └── common/
│           ├── pytorch_util.py         ← dict_apply, optimizer_to, replace_submodules (✅)
│           ├── checkpoint_util.py      ← TopKCheckpointManager (✅)
│           ├── replay_buffer.py        (✅)
│           ├── sampler.py              (✅)
│           └── normalize_util.py       ← get_image_range_normalizer (NEW ❌)
│
├── eval_dp3.py                         ← Standalone DP3 eval (NEW ❌)
├── eval_dp_rgb.py                      ← Standalone DP-RGB eval (NEW ❌)
├── eval_composed.py                    ← GPC composed eval (NEW ❌)
│
├── eval_checkpoint.py                  ← Existing basic eval (✅)
├── scripts/
│   ├── train_policy.sh                 ← Launches train.py (✅)
│   └── train_dp_rgb.sh                 ← NEW (❌)
│
└── sim-robot/
    └── create_zarr.py                  ← Data pipeline (✅, --action-mode absolute|delta)
```

---

## 1. Data Zarr Format

Created by `sim-robot/create_zarr.py --action-mode absolute`:

```
rrc_sim_dataset_absolute.zarr/
└── data/
    ├── state        (N, 7)  — 6 arm joints + 1 gripper
    ├── action       (N, 7)  — joints absolute, gripper delta
    ├── point_cloud  (N, 2500, 3)
    └── img          (N, H, W, 3)  — uint8 RGB image
```

DP3 loads: `state`, `action`, `point_cloud`
DP-RGB loads: `state`, `action`, `img`

---

## 2. Dataset Classes

### `RRCDataset` (existing) — for DP3
- **File:** `diffusion_policy_3d/dataset/rrc_dataset.py`
- Loads keys: `['state', 'action', 'point_cloud']`
- Returns `obs = { 'point_cloud': (T,2500,3), 'agent_pos': (T,7) }`

### `RRCImageDataset` (NEW) — for DP-RGB
- **File:** `diffusion_policy_3d/dataset/rrc_image_dataset.py`
- Loads keys: `['state', 'action', 'img']`
- Returns `obs = { 'image': (T,3,H,W), 'agent_pos': (T,7) }`
- Normalizes: `img` → float32, channel-first, `/255.0`
- `get_normalizer()` sets `normalizer['image'] = get_image_range_normalizer()` (identity [0,1])

```python
def _sample_to_data(self, sample):
    agent_pos = sample['state'][:].astype(np.float32)                # (T, 7)
    image = np.moveaxis(sample['img'], -1, 1).astype(np.float32) / 255.0  # (T,H,W,3)→(T,3,H,W)
    return {
        'obs': {
            'image': image,
            'agent_pos': agent_pos,
        },
        'action': sample['action'].astype(np.float32)
    }
```

### `normalize_util.py` (NEW utility)
```python
def get_image_range_normalizer():
    """Returns a normalizer for images already in [0, 1] range — identity."""
    return SingleFieldLinearNormalizer.create_identity()
```

---

## 3. Configs

### `task/rrc_sim.yaml` (DP3 task — existing, update path)
```yaml
name: rrc_sim
shape_meta:
  obs:
    point_cloud: { shape: [2500, 3], type: spatial }
    agent_pos:   { shape: [7],       type: low_dim }
  action:        { shape: [7] }
dataset:
  _target_: diffusion_policy_3d.dataset.rrc_dataset.RRCDataset
  zarr_path: /path/to/rrc_sim_dataset_absolute.zarr
  ...
env_runner:
  _target_: diffusion_policy_3d.env_runner.pybullet_runner.XArm6PyBulletRunner
  ...
```

### `task/rrc_sim_dp_rgb.yaml` (NEW for DP-RGB)
```yaml
name: rrc_sim
shape_meta:
  obs:
    image:     { shape: [3, 224, 224], type: rgb }  # or actual resolution
    agent_pos: { shape: [7],           type: low_dim }
  action:      { shape: [7] }
dataset:
  _target_: diffusion_policy_3d.dataset.rrc_image_dataset.RRCImageDataset
  zarr_path: /path/to/rrc_sim_dataset_absolute.zarr
  ...
env_runner:
  _target_: diffusion_policy_3d.env_runner.pybullet_runner.XArm6PyBulletRunner
  ...
```

### Top-level `dp3.yaml` (exists, points to rrc_sim)
- `policy._target_: diffusion_policy_3d.policy.dp3.DP3`

### Top-level `dp_rgb.yaml` (exists, needs task update)
- `policy._target_: diffusion_policy_3d.policy.dp_rgb_unet_policy.DiffusionUnetImagePolicy`
- `defaults: - task: rrc_sim_dp_rgb`  ← update this line

---

## 4. Training

### DP3 — existing `train.py`
```bash
cd policy/Sim-DP3/3D-Diffusion-Policy
python train.py --config-name=dp3 task=rrc_sim training.seed=42
```

### DP-RGB — new `train_dp_rgb.py`
- Same `TrainDP3Workspace` class structure as `train.py`
- Import: `from diffusion_policy_3d.policy.dp_rgb_unet_policy import DiffusionUnetImagePolicy`
- Type-hint: `self.model: DiffusionUnetImagePolicy`
- Dataset instantiated via hydra as `RRCImageDataset`
- `compute_loss` signature is identical — no other changes needed

```bash
cd policy/Sim-DP3/3D-Diffusion-Policy
python train_dp_rgb.py --config-name=dp_rgb task=rrc_sim_dp_rgb training.seed=42
```

> **Note:** The existing `train.py` _could_ train DP-RGB via Hydra since `cfg.policy._target_` is dynamic. The reason for a separate `train_dp_rgb.py` is dataset routing — `RRCImageDataset` vs `RRCDataset` is specified in the task yaml, so in practice both could share one `train.py`. Decide based on how much difference you anticipate in the training loop (e.g. `freeze_encoder` flag).

---

## 5. Eval Runner Additions (pybullet_runner.py)

The `XArm6PyBulletRunner` currently only has `run(policy, dataset)`. For standalone eval scripts and GPC, add:

```python
def reset_obs(self):
    """Clear obs history between episodes."""
    self._obs_deque.clear()

def update_obs(self, observation: dict):
    """Push a new raw obs dict to the rolling window."""
    self._obs_deque.append(observation)

def get_action(self, policy, observation: dict) -> np.ndarray:
    """One-shot: update obs + predict action. Returns (n_action_steps, action_dim)."""
    ...

def prepare_data(self, policy, observation: dict) -> dict:
    """For GPC: returns intermediate denoising tensors without stepping scheduler.
    Mirrors DPRunner.prepare_data() / RobotRunner.prepare_data() from General Policy."""
    # calls policy.prepare_infer_data(obs_dict_input)
    ...
```

---

## 6. Policy Modules

### `DP3` policy (existing) — `dp3.py`
- Observation: `point_cloud` + `agent_pos`
- Encoder: PointNet
- `predict_action(obs_dict)` → `{'action': ..., 'action_pred': ...}`
- `prepare_infer_data(obs_dict)` → intermediate denoising tensors (needed for GPC)

### `DiffusionUnetImagePolicy` (existing) — `dp_rgb_unet_policy.py`
- Observation: `image` + `agent_pos`
- Encoder: ResNet18 via `MultiImageObsEncoder`
- Same action-level API as DP3
- `prepare_infer_data(obs_dict)` already implemented ✅

### `ComposedPolicyWrapper` (NEW) — `composed_policy.py`
```python
class ComposedPolicyWrapper:
    def __init__(self, dp3: DP3, dp_rgb: DiffusionUnetImagePolicy, w_dp3=0.5, w_dp_rgb=0.5):
        assert type(dp3.noise_scheduler) == type(dp_rgb.noise_scheduler), \
            "Scheduler type mismatch — composition is mathematically unsafe"
        self.dp3 = dp3
        self.dp_rgb = dp_rgb
        self.w_dp3 = w_dp3
        self.w_dp_rgb = w_dp_rgb

    def predict_action(self, obs_dict_dp3: dict, obs_dict_dp_rgb: dict) -> dict:
        """
        obs_dict_dp3:    {'point_cloud': ..., 'agent_pos': ...}
        obs_dict_dp_rgb: {'image': ...,       'agent_pos': ...}
        """
        infer_dp3  = self.dp3.prepare_infer_data(obs_dict_dp3)
        infer_dp_rgb = self.dp_rgb.prepare_infer_data(obs_dict_dp_rgb)

        # Shared denoising loop
        scheduler = infer_dp3['scheduler']
        scheduler.set_timesteps(infer_dp3['num_inference_steps'])

        trajectory = torch.randn_like(infer_dp3['cond_data'])
        for t in scheduler.timesteps:
            # Apply conditioning from both branches
            trajectory[infer_dp3['cond_mask']] = infer_dp3['cond_data'][infer_dp3['cond_mask']]

            eps_dp3  = infer_dp3['model'](trajectory, t,
                            local_cond=infer_dp3['local_cond'],
                            global_cond=infer_dp3['global_cond'])
            eps_rgb  = infer_dp_rgb['model'](trajectory, t,
                            local_cond=infer_dp_rgb['local_cond'],
                            global_cond=infer_dp_rgb['global_cond'])

            # Linear noise blending
            eps_composed = self.w_dp3 * eps_dp3 + self.w_dp_rgb * eps_rgb
            trajectory = scheduler.step(eps_composed, t, trajectory).prev_sample

        # Unnormalize and slice action
        Da = infer_dp3['Da']
        To = infer_dp3['To']
        n  = infer_dp3['n_action_steps']
        naction_pred = trajectory[..., :Da]
        action_pred  = infer_dp3['action_normalizer'].unnormalize(naction_pred)
        action = action_pred[:, To-1 : To-1+n]
        return {'action': action, 'action_pred': action_pred}
```

---

## 7. Evaluation Scripts

All three eval scripts follow the same loop structure:

```
load_policy(checkpoint_path)
for episode in range(n_test):
    runner.reset_obs()
    obs = env.reset()
    while not done:
        action = runner.get_action(policy, obs)  # or composed version
        obs, rew, done, _ = env.step(action)
```

### `eval_dp3.py`
- Loads DP3 checkpoint → `XArm6PyBulletRunner.get_action(dp3_policy, obs)`

### `eval_dp_rgb.py`
- Loads DP-RGB checkpoint → `XArm6PyBulletRunner.get_action(dp_rgb_policy, obs)`
- Observation includes `image` key from environment

### `eval_composed.py`
- Loads both checkpoints
- Calls `ComposedPolicyWrapper.predict_action(obs_dp3, obs_dp_rgb)`
- `w_dp3` and `w_dp_rgb` passed as CLI args
- Saves results to `eval_result/composed/`

---

## 8. Build Order

| # | Task | File(s) | Blocker For |
|---|---|---|---|
| 1 | Add `get_image_range_normalizer` | `common/normalize_util.py` | `RRCImageDataset` |
| 2 | Create `RRCImageDataset` | `dataset/rrc_image_dataset.py` | DP-RGB training |
| 3 | Create `task/rrc_sim_dp_rgb.yaml` | `config/task/` | DP-RGB training |
| 4 | Create `train_dp_rgb.py` | `Sim-DP3/3D-Diffusion-Policy/` | DP-RGB training |
| 5 | Add runner methods: `reset_obs`, `get_action`, `prepare_data` | `env_runner/pybullet_runner.py` | All eval scripts |
| 6 | Create `eval_dp3.py`, `eval_dp_rgb.py` | `Sim-DP3/` | Standalone eval |
| 7 | Create `composed_policy.py` | `policy/composed_policy.py` | GPC |
| 8 | Create `eval_composed.py` | `Sim-DP3/` | GPC eval |
