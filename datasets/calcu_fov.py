import open3d as o3d
import numpy as np
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#todo 遍历点云中的每个点 计算出来视场角范围 包括垂直和水平的
def compute_fov(pointcloud: np.ndarray):
    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]

    # 偏航角 yaw: 水平角度 [-pi, pi]
    yaw = np.arctan2(y, x)

    # 俯仰角 pitch: 垂直角度 [-pi/2, pi/2]
    pitch = np.arctan2(z, np.sqrt(x**2 + y**2))

    # 转成角度
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)

    # FoV 范围
    fov_h = (yaw_deg.min(), yaw_deg.max(), yaw_deg.max() - yaw_deg.min())
    fov_v = (pitch_deg.min(), pitch_deg.max(), pitch_deg.max() - pitch_deg.min())

    return fov_h, fov_v



#!  计算一下所有帧的最值
global_fov_h_min = float('inf')
global_fov_h_max = float('-inf')
global_fov_v_min = float('inf')
global_fov_v_max = float('-inf')

scan_dir = '/home/data/ldq/HeRCULES/Library/Library_01_Day/LiDAR/SPVNAS_np8Aeva_plane_segmented/'

for scan_file in tqdm(os.listdir(scan_dir), desc="Processing LiDAR scans"):
    if scan_file.endswith('.bin'):
        scan_path = os.path.join(scan_dir, scan_file)
        pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 5)

        # 计算该帧的 FoV
        fov_h, fov_v = compute_fov(pointcloud)

        # 更新全局极值
        global_fov_h_min = min(global_fov_h_min, fov_h[0])
        global_fov_h_max = max(global_fov_h_max, fov_h[1])
        global_fov_v_min = min(global_fov_v_min, fov_v[0])
        global_fov_v_max = max(global_fov_v_max, fov_v[1])

# 计算最终范围
global_fov_h_range = global_fov_h_max - global_fov_h_min
global_fov_v_range = global_fov_v_max - global_fov_v_min

print(f"全局水平 FoV: {global_fov_h_range:.2f}°  (范围 {global_fov_h_min:.2f}° ~ {global_fov_h_max:.2f}°)")
print(f"全局垂直 FoV: {global_fov_v_range:.2f}°  (范围 {global_fov_v_min:.2f}° ~ {global_fov_v_max:.2f}°)")


# #! 计算单帧的fov范围
# scan_path = '/home/data/ldq/HeRCULES/Library/Library_01_Day/LiDAR/SPVNAS_np8Aeva_plane_segmented/1724811974812895885.bin'
# pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 5)
# # 计算
# fov_h, fov_v = compute_fov(pointcloud)
# print(f"水平 FoV: {fov_h[0]:.2f}° ~ {fov_h[1]:.2f}° (范围 {fov_h[2]:.2f}°)")
# print(f"垂直 FoV: {fov_v[0]:.2f}° ~ {fov_v[1]:.2f}° (范围 {fov_v[2]:.2f}°)")