import os
import pickle
import rerun as rr
import numpy as np

def create_rrd():
    load_dir = './data_2/pick_and_place_D435_pkl/episode0'
    if not os.path.exists(load_dir):
        print(f"Directory not found: {load_dir}")
        return

    pkl_files = [f for f in os.listdir(load_dir) if f.endswith('.pkl')]
    pkl_files.sort(key=lambda x: int(x.split('.')[0]))

    if not pkl_files:
        print("No .pkl files found.")
        return

    print(f"Found {len(pkl_files)} frames. Generating RRD file...")
    
    # Initialize rerun and save to file instead of connecting to a viewer
    rr.init("pick_and_place_episode", spawn=False)
    rrd_path = "episode0_2.rrd"
    rr.save(rrd_path)

    for idx, f_name in enumerate(pkl_files):
        with open(os.path.join(load_dir, f_name), 'rb') as f:
            data = pickle.load(f)
            
        # Update the time timeline for this frame
        rr.set_time_sequence("frame_idx", idx)
        
        # Log Head Camera RGB
        if 'observation' in data and 'head_camera' in data['observation'] and 'rgb' in data['observation']['head_camera']:
            img = data['observation']['head_camera']['rgb']
            rr.log("cameras/head_camera/rgb", rr.Image(img))
            
        # Log Front Camera RGB (Wrist Camera)
        if 'observation' in data and 'front_camera' in data['observation'] and 'rgb' in data['observation']['front_camera']:
            img = data['observation']['front_camera']['rgb']
            rr.log("cameras/front_camera/rgb", rr.Image(img))

        # Log Point Cloud
        if 'pointcloud' in data:
            pcd = data['pointcloud']
            points = pcd[:, :3]
            
            # The colors are likely in [0, 1] range, rerun expects [0, 255] uint8 for Color
            colors = np.clip(pcd[:, 3:], 0, 1)
            colors_uint8 = (colors * 255).astype(np.uint8)
            
            rr.log(
                "world/point_cloud",
                rr.Points3D(
                    positions=points,
                    colors=colors_uint8,
                    radii=0.005
                )
            )

        print(f"Logged frame {idx+1}/{len(pkl_files)}", end='\r')
        
    print(f"\nRRD file saved successfully to {rrd_path}")

if __name__ == '__main__':
    create_rrd()
