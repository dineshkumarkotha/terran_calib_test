import os, json, csv, argparse, numpy as np
from src.calibration import estimate_cam_poses_from_detections, average_transform
from src.geometry import se3_from_rt, invert_se3, euler_from_rot

def K_from(d):
    return np.array([[d["fx"], 0, d["cx"]],
                     [0, d["fy"], d["cy"]],
                     [0, 0, 1]], dtype=np.float64)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=str, default="out/data/cams.json")
    ap.add_argument("--detections", type=str, default="out/data/detections.csv")
    ap.add_argument("--out", type=str, default="out/results.json")
    ap.add_argument("--rmse_max", type=float, default=5.0)
    return ap.parse_args()

def read_detections(path):
    dets = []
    with open(path, "r") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            ts_ns = int(row[0])
            cam_id = row[1]
            tag_id = int(row[2])
            flat = list(map(float, row[3:11]))
            corners = np.array([
                [flat[0], flat[1]],
                [flat[2], flat[3]],
                [flat[4], flat[5]],
                [flat[6], flat[7]]
            ], dtype=np.float64)
            dets.append({
                "ts_ns": ts_ns,
                "cam_id": cam_id,
                "tag_id": tag_id,
                "corners": corners
            })
    return dets

def main():
    args = parse_args()
    cams = json.load(open(args.cams))
    tag_size_m = cams["tag_size_m"]
    Ks = {k: K_from(v) for k, v in cams.items() if k.startswith("cam")}
    dets = read_detections(args.detections)

    per_cam = estimate_cam_poses_from_detections(Ks, tag_size_m, dets)

    # Pair by timestamp and compute T_cam2_cam1 per frame
    by_ts = {}
    for cam_id, items in per_cam.items():
        for it in items:
            by_ts.setdefault(it["ts_ns"], {})[cam_id] = it

    Ts = []
    pairs_used = 0
    for ts, g in by_ts.items():
        if "cam1" in g and "cam2" in g:
            a, b = g["cam1"], g["cam2"]
            if a["rmse"] <= args.rmse_max and b["rmse"] <= args.rmse_max:
                # Each is camera<-tag (cam_from_tag)
                T_cam1_tag = se3_from_rt(a["R"], a["t"])
                T_cam2_tag = se3_from_rt(b["R"], b["t"])
                # cam2<-cam1 = (cam2<-tag) @ (tag<-cam1)
                T_tag_cam1 = invert_se3(T_cam1_tag)
                T_cam2_cam1 = T_cam2_tag @ T_tag_cam1
                Ts.append(T_cam2_cam1)
                pairs_used += 1

    # Average transformation (already cam2<-cam1). Do NOT invert again.
    T_out = np.linalg.inv(average_transform(Ts) if Ts else np.eye(4))
    R = T_out[:3, :3]
    t = T_out[:3, 3]
    rpy_deg = np.degrees(euler_from_rot(R)).tolist()

    # Mean reprojection RMSE over used pairs (both cams)
    rmse_vals = []
    for ts, g in by_ts.items():
        if "cam1" in g and "cam2" in g:
            a, b = g["cam1"], g["cam2"]
            if a["rmse"] <= args.rmse_max and b["rmse"] <= args.rmse_max:
                rmse_vals += [a["rmse"], b["rmse"]]
    reproj_rmse_px = float(np.mean(rmse_vals)) if rmse_vals else None

    result = {
        "T_cam2_cam1_matrix": T_out.tolist(),
        "rpy_deg": rpy_deg,
        "xyz_m": t.tolist(),
        "reproj_rmse_px": reproj_rmse_px,
        "pairs_used": pairs_used
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f" Wrote {args.out} | pairs_used={pairs_used} | rmse={reproj_rmse_px}")

if __name__ == "__main__":
    main()
