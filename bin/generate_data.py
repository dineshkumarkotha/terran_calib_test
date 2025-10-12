import os, json, csv, math, argparse, numpy as np
from src.geometry import se3_from_rt, project_points_xy_plane

def K_from(d):
    return np.array([[d["fx"], 0, d["cx"]],
                     [0, d["fy"], d["cy"]],
                     [0, 0, 1]], dtype=np.float64)

def rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c,-s,0],
                     [ s, c,0],
                     [ 0, 0,1]], dtype=np.float64)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="out/data")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.7, help="pixel noise std")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    cams = {
        "cam1": {"fx": 920.0, "fy": 918.0, "cx": 640.0, "cy": 360.0},
        "cam2": {"fx": 915.0, "fy": 917.0, "cx": 640.0, "cy": 360.0},
        "tag_size_m": 0.12
    }
    json.dump(cams, open(os.path.join(args.out, "cams.json"), "w"), indent=2)

    # Ground truth cam2_from_cam1: +0.10 m on X, +3 deg yaw
    R12 = rot_z(math.radians(3.0))
    t12 = np.array([0.10, 0.00, 0.00], dtype=np.float64)
    T_cam2_cam1 = se3_from_rt(R12, t12)

    # Tag is world frame; its corners in Z=0 plane (CCW)
    s = cams["tag_size_m"] / 2.0
    XY = np.array([[-s, -s], [s, -s], [s, s], [-s, s]], dtype=np.float64)

    K1, K2 = K_from(cams["cam1"]), K_from(cams["cam2"])
    ts0 = 1710000000000
    rows = [("ts_ns","cam_id","tag_id","u0","v0","u1","v1","u2","v2","u3","v3")]

    for i in range(args.frames):
        z = -1.5 + (i / max(1, (args.frames-1))) * 1.0     # -1.5 -> -0.5 m
        yaw = math.radians(-5 + 10 * (i / max(1, (args.frames-1))))
        R1  = rot_z(yaw)
        t1  = np.array([0.0, 0.0, z], dtype=np.float64)

        # world->cam poses
        T_w_c1 = se3_from_rt(R1, t1)
        T_w_c2 = T_w_c1 @ T_cam2_cam1

        # Use world->cam directly for projection (NO inverses)
        for cam_id, K, T in [("cam1", K1, T_w_c1), ("cam2", K2, T_w_c2)]:
            Rcw = T[:3, :3]
            tcw = T[:3, 3]
            uv = project_points_xy_plane(K, Rcw, tcw, XY)
            uv += np.random.normal(0.0, args.noise, uv.shape)
            flat = uv.reshape(-1).tolist()
            rows.append((ts0 + i*100, cam_id, 5, *[round(x, 3) for x in flat]))

    with open(os.path.join(args.out, "detections.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    gt = {
        "T_cam2_cam1_matrix": T_cam2_cam1.tolist(),
        "rpy_deg": [0.0, 0.0, 3.0],
        "xyz_m": t12.tolist()
    }
    json.dump(gt, open(os.path.join(args.out, "gt.json"), "w"), indent=2)
    print(f" Wrote synthetic dataset to {args.out}")

if __name__ == "__main__":
    main()
