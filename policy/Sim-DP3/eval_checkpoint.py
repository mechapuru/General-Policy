
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.absolute())
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, '3D-Diffusion-Policy'))

import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
from diffusion_policy_3d.env_runner.base_runner import BaseRunner

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        '3D-Diffusion-Policy', 'diffusion_policy_3d', 'config')),
    config_name='dp3'
)
def main(cfg):
    checkpoint = getattr(cfg, "checkpoint_path", None)
    if checkpoint is None:
        print("Error: Please provide +checkpoint_path=/path/to/checkpoint.ckpt")
        return

    checkpoint_path = pathlib.Path(checkpoint)
    
    if not checkpoint_path.is_file():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Initialize workspace with config form CLI (e.g. task)
    workspace = TrainDP3Workspace(cfg)
    
    # Load checkpoint weights
    payload = workspace.load_checkpoint(path=checkpoint_path)
    print("Checkpoint loaded successfully.")
    
    # Configure env runner using CLI config (allows overriding params like n_test)
    # The output directory will be the hydra run directory for this evaluation
    print(f"Output directory: {workspace.output_dir}")
    env_runner: BaseRunner = hydra.utils.instantiate(
        cfg.task.env_runner, 
        output_dir=workspace.output_dir
    )
    
    # Set model to eval mode
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.eval()
    policy.cuda()
    
    # Run evaluation
    print("Starting evaluation...")
    video_path = getattr(cfg, "video_path", None)
    if video_path:
        print(f"Video will be saved to: {video_path}")
        
    runner_log = env_runner.run(policy, video_path=video_path)
    
    # Print results
    print("---------------- Eval Results --------------")
    for key, value in runner_log.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
