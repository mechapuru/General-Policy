import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

def create_pcd_video():
    load_dir = './data/pick_and_place_D435_pkl/episode0'
    if not os.path.exists(load_dir):
        print(f"Directory not found: {load_dir}")
        return

    pkl_files = [f for f in os.listdir(load_dir) if f.endswith('.pkl')]
    pkl_files.sort(key=lambda x: int(x.split('.')[0]))

    if not pkl_files:
        print("No .pkl files found.")
        return

    print(f"Found {len(pkl_files)} frames. Generating point cloud video...")
    frames = []
    
    # Use Agg backend for headless rendering
    import matplotlib
    matplotlib.use('Agg')
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    for idx, f_name in enumerate(pkl_files):
        with open(os.path.join(load_dir, f_name), 'rb') as f:
            data = pickle.load(f)
            
        pcd = data['pointcloud']
        points = pcd[:, :3]
        colors = pcd[:, 3:]
        
        # Clip colors just in case they exceed [0, 1] range
        colors = np.clip(colors, 0, 1)
        
        ax.clear()
        
        # Set consistent spatial limits based on table/workspace bounds
        ax.set_xlim([0.15, 0.45])
        ax.set_ylim([-0.2, 0.3])
        ax.set_zlim([0.6, 1.1])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Plot 3D scatter
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=5)
        
        # Render canvas to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frames.append(img_bgr)
        
        print(f"Rendered pcd frame {idx+1}/{len(pkl_files)}", end='\r')
        
    print()
    height, width, layers = frames[0].shape
    video_path = 'episode0_pcd_video.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()
    print(f"Video saved successfully to {video_path}")

if __name__ == '__main__':
    create_pcd_video()
