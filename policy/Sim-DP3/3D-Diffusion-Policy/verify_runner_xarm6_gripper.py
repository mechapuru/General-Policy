
import torch
import numpy as np
import os
from diffusion_policy_3d.env_runner.pybullet_runner import XArm6GripperPyBulletRunner
from diffusion_policy_3d.policy.base_policy import BasePolicy

class MockNormalizer:
    def __init__(self):
        self.params_dict = {
            "point_cloud": {},
            "agent_pos": {},
            # "image": {} # Optional
        }

class MockPolicy(BasePolicy):
    def __init__(self, action_dim=7):
        super().__init__()
        # self.device is a property from ModuleAttrMixin
        # self.device = torch.device('cpu') 
        self.normalizer = MockNormalizer()
        self.action_dim = action_dim

    def predict_action(self, obs_dict):
        # Return random actions
        # obs_dict values are (B, To, ...)
        # Output action: (B, Ta, Da)
        # Runner usually expects (B, Ta, Da)
        batch_size = list(obs_dict.values())[0].shape[0]
        # n_action_steps=8 as per runner init
        action = torch.zeros((batch_size, 8, self.action_dim)) 
        return {'action': action}

    def reset(self):
        pass

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

def test_runner():
    print("Initializing Runner...")
    output_dir = "data/outputs/test_runner_xarm6_gripper"
    os.makedirs(output_dir, exist_ok=True)
    
    runner = XArm6GripperPyBulletRunner(
        output_dir=output_dir,
        n_train=0,
        n_test=1, # Just 1 episode
        max_steps=20, # Short episode
        n_obs_steps=2,
        n_action_steps=8,
        fps=10,
        use_gui=False, # Headless
        action_dim=7,
        capture_table=False
    )

    print("Initializing Policy...")
    policy = MockPolicy(action_dim=7)

    print("Running...")
    log_data = runner.run(policy)
    print("Done!")
    print("Log Data keys:", log_data.keys())

if __name__ == "__main__":
    test_runner()
