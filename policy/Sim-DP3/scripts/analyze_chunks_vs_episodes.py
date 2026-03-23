import zarr
import numpy as np

def analyze_chunk_overlap(path):
    print(f"Analyzing: {path}")
    z = zarr.open(path, 'r')
    
    episode_ends = z['meta']['episode_ends'][:]
    chunk_size = z['data']['action'].chunks[0]
    
    print(f"Chunk Size: {chunk_size}")
    print(f"Total Episodes: {len(episode_ends)}")
    
    # Calculate start indices for each episode
    episode_starts = np.insert(episode_ends[:-1], 0, 0)
    
    print("\n--- Episode-Chunk Mapping (First 10 Episodes) ---")
    print(f"{'Ep #':<5} | {'Range':<15} | {'Length':<8} | {'Spans Chunks':<20}")
    print("-" * 60)
    
    chunks_with_multiple_episodes = set()
    
    for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        start_chunk = start // chunk_size
        end_chunk = (end - 1) // chunk_size
        
        chunks_involved = list(range(start_chunk, end_chunk + 1))
        
        # Mark chunks that have this episode in them
        # If a chunk sees multiple episodes start/end in it, it contains multiple
        
        if i < 10:
            print(f"{i:<5} | {start}-{end:<10} | {end-start:<8} | {chunks_involved}")

    print("\n--- Detailed Chunk Inspection (First 3 Chunks) ---")
    # For the first few chunks, list which episodes contribute to it
    for chunk_idx in range(3):
        chunk_start = chunk_idx * chunk_size
        chunk_end = (chunk_idx + 1) * chunk_size
        
        # Find episodes that overlap with this chunk
        episodes_in_chunk = []
        for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
            # Episode range is [start, end)
            # Chunk range is [chunk_start, chunk_end)
            if start < chunk_end and end > chunk_start:
                 episodes_in_chunk.append(i)
        
        print(f"Chunk {chunk_idx} (Steps {chunk_start}-{chunk_end}): Contains data from Episodes {episodes_in_chunk}")

if __name__ == "__main__":
    analyze_chunk_overlap('/home/cross-emb/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real_robot_demo/drill_40demo_1024.zarr')
