import os, json, math, csv, numpy as np

# Simple synthetic dataset generator for two cameras and one tag
os.makedirs("out/data", exist_ok=True)

# Camera intrinsics
cams = {
    "cam1": {"fx": 920.0, "fy": 918.0, "cx": 640.0, "cy": 360.0},
    "cam2": {"fx": 915.0, "fy": 917.0, "cx": 640.0, "cy": 360.0},
    "tag_size_m": 0.12,
}
json.dump(cams, open("out/data/cams.json", "w"), indent=2)

# Make fake detections for testing
rows = [("ts_ns","cam_id","tag_id","u0","v0","u1","v1","u2","v2","u3","v3")]
for i in range(10):
    rows.append((1710000000000+i, "cam1", 5, 598.2,341.0,645.4,338.7,647.0,385.1,599.8,388.0))
    rows.append((1710000000000+i, "cam2", 5, 612.3,339.4,658.1,337.2,660.2,383.3,614.5,385.9))

with open("out/data/detections.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)

print("✅ Synthetic dataset written to out/data/")
