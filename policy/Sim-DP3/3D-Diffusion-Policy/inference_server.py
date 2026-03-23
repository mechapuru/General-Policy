import os
import sys
import pathlib
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import argparse

# Setup python path to match train.py environment
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

# Import TrainDP3Workspace to load the model
try:
    from train import TrainDP3Workspace
except ImportError:
    # If running from a different directory, try appending the current directory
    sys.path.append(os.getcwd())
    from train import TrainDP3Workspace

from diffusion_policy_3d.common.pytorch_util import dict_apply

app = FastAPI()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKSPACE = None

class InferenceRequest(BaseModel):
    # Expect a dictionary containing "point_cloud" and "agent_pos"
    # values should be lists representing the time sequence
    observation: Dict[str, Any]

@app.on_event("startup")
def load_model():
    global WORKSPACE
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint_path:
        print("Warning: CHECKPOINT_PATH env var not set. Pass --checkpoint argument or set env var.")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        # Load the workspace from checkpoint
        # This restores the model, configuration, and normalizer
        WORKSPACE = TrainDP3Workspace.create_from_checkpoint(checkpoint_path)
        WORKSPACE.model.eval()
        WORKSPACE.model.to(DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # We don't exit here to allow debugging or delayed loading
        
@app.post("/predict")
async def predict(request: InferenceRequest):
    global WORKSPACE
    if WORKSPACE is None:
        raise HTTPException(status_code=500, detail="Model not initialized. Please restart with a valid checkpoint.")

    try:
        # Convert inputs to torch tensors
        # Observation is expected to be a dict of lists or arrays
        # We assume input JSON provides (Time, N_points, 3) for point clouds e.t.c.
        # We assume Batch=1 for inference
        
        obs_dict_input = request.observation
        
        def to_tensor(x):
            # Convert list to numpy if needed
            if isinstance(x, list):
                x = np.array(x)
            # Create tensor
            # Ensure float32 for floating point data
            if x.dtype == np.float64:
                 x = x.astype(np.float32)

            t = torch.from_numpy(x)
            
            # Move to device
            t = t.to(DEVICE)
            
            # Add batch dimension: (T, ...) -> (1, T, ...)
            t = t.unsqueeze(0)
            return t

        obs_dict = dict_apply(obs_dict_input, to_tensor)

        with torch.no_grad():
            # policy.predict_action handles normalization internally using the loaded normalizer
            result = WORKSPACE.model.predict_action(obs_dict)
            action = result['action']
        
        # Action shape: (B, T, D) -> (1, T, D)
        # Return as list, removing batch dimension
        action_list = action[0].cpu().numpy().tolist()
        
        return {"action": action_list}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint file")
    parser.add_argument('--host', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    
    if args.checkpoint:
        os.environ["CHECKPOINT_PATH"] = args.checkpoint
        
    uvicorn.run(app, host=args.host, port=args.port)
