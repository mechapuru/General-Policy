import pickle
import os
import numpy as np
from PIL import Image

# Path to the data
pkl_path = "/home/paddy/rrc/1cross/General-Policy/data/blocks_stack_hard_L515_pkl/episode0/0.pkl"

if not os.path.exists(pkl_path):
    print(f"Error: File not found at {pkl_path}")
    exit()

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("--- PKL Data Structure ---")
print(f"Main Keys: {list(data.keys())}")

# Print Robot State info
if 'joint_action' in data:
    print(f"\nJoint Action (14 dims): {data['joint_action']}")
if 'endpose' in data:
    print(f"Endpose (6D + Gripper): {data['endpose']}")

# Check PointCloud
if 'pointcloud' in data:
    print(f"\nPointcloud shape: {data['pointcloud'].shape} (Points, XYZ+RGB)")

# Check Images and save one for verification
obs = data.get('observation', {})
head_cam = obs.get('head_camera', {})
if 'rgb' in head_cam:
    rgb_data = head_cam['rgb']
    print(f"\nHead Camera RGB shape: {rgb_data.shape}")
    
    # Save the image using PIL so you can open it
    img = Image.fromarray(rgb_data.astype('uint8'))
    img.save('check_image.png')
    print("Successfully saved 'check_image.png' to the current directory for viewing.")
else:
    print("\nNo RGB data found in head_camera observation.")
