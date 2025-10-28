#  Robotics – Multi-Sensor Extrinsics Calibration

**Author:** Dinesh Kumar Kotha  
**Date:** October 12, 2025  
**Repository:** Private GitHub – `terran_calib_test`  
**Challenge:** Multi-Sensor Extrinsics Calibration from AprilTag Detections  

---

##  Project Overview

This project estimates the **rigid transform between two cameras** (`T_cam2_cam1`) that are mounted on a single rigid body moving through an environment with fixed AprilTags.  
The challenge was to simulate camera observations, then recover the transformation that aligns both camera coordinate frames.

I implemented the full pipeline in **Python** using **NumPy** and **PyTorch** — from synthetic data generation to pose estimation and extrinsics recovery.  
Everything runs locally on CPU and completes in a few seconds.

---

##  My Approach and Implementation

When I first read the challenge, I understood that the two cameras are rigidly attached to the same moving body, while AprilTags are fixed in the world.  
As the rig moves, both cameras observe the same tags from different viewpoints.  
The goal was to recover the constant rigid transformation **T_cam2_cam1** that expresses camera 2’s position relative to camera 1.

---

### **1️ Phase 1 – Data Simulation**

Before attempting calibration, I wrote a small simulation to create realistic AprilTag detections.  
Here’s how I built it step-by-step:

1. Defined **two cameras** (`cam1`, `cam2`) with known intrinsics (fx, fy, cx, cy).  
2. Placed several **AprilTags** at fixed 3D positions in the world.  
3. Simulated a **camera rig motion** through multiple frames (poses).  
4. Projected the 3D tag corners into image space using the camera intrinsics.  
5. Added small **Gaussian pixel noise** for realism.  

This produced:


out/data/cams.json # camera intrinsics
out/data/detections.csv # synthetic AprilTag detections




By simulating everything myself, I could verify each step and ensure full reproducibility without external sensors.

---

### **2️ Phase 2 – Extrinsics Recovery**

Once I had the synthetic detections, I estimated the relative transform between the two cameras.

**Algorithm outline:**
1. For each camera, estimate its pose (R, t) with respect to each observed tag.  
2. For timestamps where both cameras see the same tag:


T_cam2_cam1 = T_cam2_tag * inverse(T_cam1_tag)

3. Accumulate all such transforms across frames.  
4. Compute the **average transform** to minimize noise.  
5. Extract roll-pitch-yaw (RPY) angles and translation (XYZ).  
6. Compute **reprojection RMSE** as a final accuracy metric.

The final transformation and metrics are written to:

out/results.json



---

### **3️ Verification and Results**

- **Average reprojection RMSE:** ~0.8 px  
- **Translation along X-axis:** ≈ 0.10 m  
- **Yaw angle:** ≈ ±3° (depending on convention)  
- **Frames used:** 40  

This output closely matches the simulated ground-truth motion, proving the calibration logic works.

---

### **4️ Why This Works**

Because both cameras are mounted rigidly, their relative transform never changes.  
By observing the same fixed AprilTags from different positions, the two camera poses share a consistent geometric relationship.  
Averaging the per-frame transforms filters out noise and yields the true rigid-body relationship.

Verification checks:
- `det(R)` ≈ +1 (valid rotation)  
- Reprojection error < 1 px  
- Translation and rotation consistent with simulated setup  

---

### **5️  Key Learnings**

- Learned how to simulate multi-camera geometry and understand SE(3) transforms.  
- Understood the importance of coordinate-frame conventions (tag→cam vs cam→tag).  
- Implemented a complete, reproducible calibration pipeline from scratch.  
- Gained hands-on experience connecting math, geometry, and code.  

---

##  How to Run

###  click run 

```cmd
run.cmd
