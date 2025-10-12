import numpy as np
from .geometry import se3_from_rt, invert_se3, euler_from_rot, project_points_xy_plane

def tag_corners_xy(tag_size_m: float) -> np.ndarray:
    s = tag_size_m
    # Counter-clockwise: (-,-), (+,-), (+,+), (-,+)
    return np.array([[-s/2, -s/2],
                     [ s/2, -s/2],
                     [ s/2,  s/2],
                     [-s/2,  s/2]], dtype=np.float64)

def dlt_homography(XY: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Planar homography H so that s [u v 1]^T = H [X Y 1]^T."""
    A = []
    for (X, Y), (u, v) in zip(XY, uv):
        A.append([ X, Y, 1, 0, 0, 0, -u*X, -u*Y, -u ])
        A.append([ 0, 0, 0, X, Y, 1, -v*X, -v*Y, -v ])
    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / (H[2, 2] + 1e-12)

def pose_from_homography(H: np.ndarray, K: np.ndarray):
    """Recover camera pose (R,t) from homography and intrinsics, Z=0 plane."""
    Kinv = np.linalg.inv(K)
    B = Kinv @ H
    b1, b2, b3 = B[:, 0], B[:, 1], B[:, 2]
    lam = 1.0 / (0.5 * (np.linalg.norm(b1) + np.linalg.norm(b2)) + 1e-12)
    r1 = lam * b1
    r2 = lam * b2
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = lam * b3
    return R, t

def reproj_rmse(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                XY: np.ndarray, uv_obs: np.ndarray) -> float:
    uv_hat = project_points_xy_plane(K, R, t, XY)
    e = uv_hat - uv_obs
    return float(np.sqrt((e * e).sum(axis=1).mean()))

def estimate_cam_poses_from_detections(Ks: dict, tag_size_m: float, detections: list):
    """Returns per-camera list of {ts_ns, R, t, rmse} dicts."""
    XY = tag_corners_xy(tag_size_m)
    per_cam = {}
    for d in detections:
        K = Ks[d["cam_id"]]
        uv = d["corners"]
        H = dlt_homography(XY, uv)
        R, t = pose_from_homography(H, K)
        rmse = reproj_rmse(K, R, t, XY, uv)
        per_cam.setdefault(d["cam_id"], []).append({
            "ts_ns": d["ts_ns"], "R": R, "t": t, "rmse": rmse
        })
    return per_cam

def average_transform(T_list: list) -> np.ndarray:
    """Average SE(3): SVD on rotations, mean on translations."""
    if not T_list:
        return np.eye(4)
    Rs = np.stack([T[:3, :3] for T in T_list], axis=0)
    ts = np.stack([T[:3, 3] for T in T_list], axis=0)
    M = Rs.mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = ts.mean(axis=0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
