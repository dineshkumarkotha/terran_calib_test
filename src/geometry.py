import numpy as np

def se3_from_rt(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def invert_se3(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def euler_from_rot(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-8
    if not singular:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0
    return np.array([roll, pitch, yaw])

def project_points_xy_plane(K, R, t, XY):
    Rt = np.hstack([R, t.reshape(3,1)])
    pts = np.vstack([XY.T, np.zeros((1, XY.shape[0])), np.ones((1, XY.shape[0]))])  # 4xN
    uvw = K @ (Rt @ pts)
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    return np.stack([u, v], axis=1)
