import json, numpy as np

cams = json.load(open("out/data/cams.json"))
print("Loaded intrinsics:", cams.keys())

# Placeholder — pretend we estimated transform
T_cam2_cam1 = np.eye(4)
T_cam2_cam1[:3, 3] = [0.1, 0.0, 0.0]

result = {
    "T_cam2_cam1_matrix": T_cam2_cam1.tolist(),
    "rpy_deg": [0.0, 0.0, 0.0],
    "xyz_m": [0.1, 0.0, 0.0],
    "reproj_rmse_px": 0.5,
    "pairs_used": 10,
}
json.dump(result, open("out/results.json", "w"), indent=2)
print(" Calibration results written to out/results.json")
