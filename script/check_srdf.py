import sys
import numpy as np
import mplib

def main():
    urdf_path = "sim-robot/lite-6-updated-urdf/lite_6_new.urdf"
    srdf_path = "sim-robot/lite6.srdf"
    planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group="link6")
    planner.set_base_pose([0, 0, 0.62, 1, 0, 0, 0])
    
    # Qpos from the user's SAPIEN run where it failed
    qpos = np.array([-1.0847810e-02, -1.8874464e-03,  5.2274346e-01, 
                     -5.6767347e-03, -1.9469073e-05,  2.5715628e-03, 
                     -2.0784045e-02, -2.0266034e-02], dtype=np.float64)
                     
    # Target pose directly to absolute space (like base_task)
    target_pose = [0.35, 0.2, 0.8, -0.5, 0.5, -0.5, -0.5]
    
    try:
        col = planner.check_for_self_collision(planner.robot, qpos)
        print(f"Self-Collision at exact start qpos: {col}")
    except Exception as e:
        print(f"Couldn't check collision: {e}")
        
    print(f"\nAttempting plan_screw to absolutely hardcoded Cartesian waypoint:")
    
    result = planner.plan_screw(target_pose, qpos, time_step=1/250)
    print(f"Status: {result['status']}")

    print("\nTesting from SAPIEN un-initialized natural upright pose (all 0's)")
    good_qpos = np.zeros(8, dtype=np.float64)
    res2 = planner.plan_screw(target_pose, good_qpos, time_step=1/250)
    print(f"SAPIEN Natural Start Pose Status: {res2['status']}")
    
if __name__ == "__main__":
    main()
