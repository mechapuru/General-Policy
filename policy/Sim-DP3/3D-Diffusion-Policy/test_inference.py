import zarr
import requests
import numpy as np
import json
import argparse
import os

def test_inference(zarr_path, url="http://localhost:8000/predict", obs_steps=2):
    print(f"Opening Zarr file: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    # Check available keys in data
    print(f"Available keys in data: {list(root['data'].keys())}")
    
    # We need to grab a sequence of length obs_steps
    # Let's take the first episode, generating a valid slice
    # Indices 0 to obs_steps
    
    payload_obs = {}
    
    # Define the keys we need based on the config/task
    # Usually 'point_cloud' and 'agent_pos' for DP3
    keys_to_fetch = ['point_cloud', 'agent_pos']
    
    start_idx = 0
    end_idx = start_idx + obs_steps
    
    print(f"Sampling steps {start_idx} to {end_idx}...")
    
    for key in keys_to_fetch:
        if key in root['data']:
            # Get data and convert to list
            data_slice = root['data'][key][start_idx:end_idx]
            
            # Simple cast to list for JSON serialization
            # data_slice is (T, ...) numpy array
            payload_obs[key] = data_slice.tolist()
            print(f"Loaded {key}: shape {data_slice.shape}")
        else:
            print(f"Warning: Key {key} not found in zarr dataset, skipping.")

    request_data = {
        "observation": payload_obs
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            action = np.array(result['action'])
            print("\nSUCCESS!")
            print(f"Received Action Shape: {action.shape}")
            print(f"Action Data (First 2 steps):\n{action[:2]}")
        else:
            print(f"\nFailed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"\nConnection Error: Is the inference server running on {url}?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, 
                        default="/scratch2/cross-emb/DP3_data/data_from_puru_no_eef_binary_gripper.zarr",
                        help="Path to the Zarr dataset file")
    parser.add_argument('--url', type=str, default="http://localhost:8000/predict",
                        help="Inference server URL")
    parser.add_argument('--steps', type=int, default=2, help="Number of observation steps to send")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zarr_path):
        print(f"Error: Zarr file not found at {args.zarr_path}")
    else:
        test_inference(args.zarr_path, args.url, args.steps)
