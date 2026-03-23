import zarr
import numpy as np

def analyze_episode_sizes(path):
    print(f"Analyzing: {path}")
    z = zarr.open(path, 'r')
    
    episode_ends = z['meta']['episode_ends'][:]
    starts = np.insert(episode_ends[:-1], 0, 0)
    lengths = episode_ends - starts
    
    print("\n=== Duration (Timesteps) ===")
    print(f"Total Episodes: {len(lengths)}")
    print(f"Min Steps: {lengths.min()}")
    print(f"Max Steps: {lengths.max()}")
    print(f"Avg Steps: {lengths.mean():.1f}")
    
    # Calculate data size per step
    bytes_per_step = 0
    print("\n=== Data Size per Step ===")
    for key, val in z['data'].items():
        # Size = itemsize * product of dimensions excluding time
        step_shape = val.shape[1:]
        size = val.dtype.itemsize * np.prod(step_shape)
        bytes_per_step += size
        print(f"  {key:<12}: {size/1024:.2f} KB")
        
    print("-" * 30)
    print(f"  Total/Step  : {bytes_per_step/1024:.2f} KB")
    
    # Calculate episode sizes
    min_size_mb = (lengths.min() * bytes_per_step) / (1024*1024)
    max_size_mb = (lengths.max() * bytes_per_step) / (1024*1024)
    avg_size_mb = (lengths.mean() * bytes_per_step) / (1024*1024)
    
    print("\n=== Total Size per Episode (Uncompressed) ===")
    print(f"Min Size: {min_size_mb:.2f} MB")
    print(f"Max Size: {max_size_mb:.2f} MB")
    print(f"Avg Size: {avg_size_mb:.2f} MB")

if __name__ == "__main__":
    analyze_episode_sizes('/home/cross-emb/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real_robot_demo/drill_40demo_1024.zarr')
