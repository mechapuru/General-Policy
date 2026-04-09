import os
import pickle
import cv2
import numpy as np

def create_video():
    load_dir = './data/pick_and_place_D435_pkl/episode0'
    if not os.path.exists(load_dir):
        print(f"Directory not found: {load_dir}")
        return

    # Find all .pkl files
    pkl_files = [f for f in os.listdir(load_dir) if f.endswith('.pkl')]
    # Sort them numerically
    pkl_files.sort(key=lambda x: int(x.split('.')[0]))

    if not pkl_files:
        print("No .pkl files found.")
        return

    print(f"Found {len(pkl_files)} frames. Generating video...")
    frames = []
    for f_name in pkl_files:
        with open(os.path.join(load_dir, f_name), 'rb') as f:
            data = pickle.load(f)
        
        # Extract head_camera rgb
        img = data['observation']['head_camera']['rgb']
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frames.append(img_bgr)

    if not frames:
        print("No frames extracted.")
        return

    height, width, layers = frames[0].shape
    video_path = 'episode0_video.mp4'
    
    # Save as mp4 at 30 fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()
    print(f"Video saved successfully to {video_path}")

if __name__ == '__main__':
    create_video()
