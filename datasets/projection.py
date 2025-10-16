import numpy as np


class RangeProjection(object):
    '''
    Project the 3D point cloud to 2D data with range projection

    Adapted from Z. Zhuang et al. https://github.com/ICEORY/PMF
    '''

    def __init__(self, fov_up, fov_down, proj_w, proj_h, fov_left=-180, fov_right=180):

        # check params
        assert fov_up >= 0 and fov_down <= 0, 'require fov_up >= 0 and fov_down <= 0, while fov_up/fov_down is {}/{}'.format(
            fov_up, fov_down)
        assert fov_right >= 0 and fov_left <= 0, 'require fov_right >= 0 and fov_left <= 0, while fov_right/fov_left is {}/{}'.format(
            fov_right, fov_left)

        # params of fov angeles
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov_v = abs(self.fov_up) + abs(self.fov_down) #! 转换为弧度制 总垂直视场角

        self.fov_left = fov_left / 180.0 * np.pi
        self.fov_right = fov_right / 180.0 * np.pi
        self.fov_h = abs(self.fov_left) + abs(self.fov_right) #! 转换为弧度制 总水平视场角

        # params of proj img size
        self.proj_w = proj_w
        self.proj_h = proj_h

        self.cached_data = {}

    def doProjection(self, pointcloud: np.ndarray):
        self.cached_data = {}
        # get depth of all points
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1) + 1e-5
        # get point cloud components
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        # get angles of all points
        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / depth)

        # get projection in image coords
        proj_x = (yaw + abs(self.fov_left)) / self.fov_h
        # proj_x = 0.5 * (yaw / np.pi + 1.0) # normalized in [0, 1]
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_v  # normalized in [0, 1]

        # scale to image size using angular resolution
        proj_x *= self.proj_w
        proj_y *= self.proj_h

        px = np.maximum(np.minimum(self.proj_w, proj_x), 0) # or proj_x.copy()
        py = np.maximum(np.minimum(self.proj_h, proj_y), 0) # or proj_y.copy()

        # round and clamp for use as index
        proj_x = np.maximum(np.minimum(
            self.proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)

        proj_y = np.maximum(np.minimum(
            self.proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)

        self.cached_data['uproj_x_idx'] = proj_x.copy()
        self.cached_data['uproj_y_idx'] = proj_y.copy()
        self.cached_data['uproj_depth'] = depth.copy()
        self.cached_data['px'] = px
        self.cached_data['py'] = py

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1] #!  深度越大（离传感器越远）的点排在前面 那么距离更近的点后面就覆盖掉了
        depth = depth[order]
        indices = indices[order]
        pointcloud = pointcloud[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # get projection result
        proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        proj_pointcloud = np.full(
            (self.proj_h, self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32)
        proj_pointcloud[proj_y, proj_x] = pointcloud #! 在这里传入的是整个点云

        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices

        proj_mask = (proj_idx > 0).astype(np.int32)

        return proj_pointcloud, proj_range, proj_idx, proj_mask 
    # proj_pointcloud  (H, W, D) 存储每个像素对应的三维坐标和特征。proj_idx: 存储每个像素对应的原始点云索引。未赋值为 -1
    # proj_range [H, W]存储像素对应的深度信息 全部点对应的 未赋值为 -1 proj_mask [H, W]: 基于 proj_idx 生成一个二值掩码，标记哪些像素有有效的点云投影 未赋值为 -1

class ScanProjection(object):
    '''
    Project the 3D point cloud to 2D data with range projection

    Adapted from A. Milioto et al. https://github.com/PRBonn/lidar-bonnetal
    '''

    def __init__(self, proj_w, proj_h):
        # params of proj img size
        self.proj_w = proj_w
        self.proj_h = proj_h

        self.cached_data = {}

    def doProjection(self, pointcloud: np.ndarray):
        self.cached_data = {}
        # get depth of all points
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        # get point cloud components
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        # get angles of all points
        yaw = -np.arctan2(y, -x)
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        #breakpoint()
        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
        proj_y = np.zeros_like(proj_x)
        proj_y[new_raw] = 1
        proj_y = np.cumsum(proj_y)
        # scale to image size using angular resolution
        proj_x = proj_x * self.proj_w - 0.001

        # print(f'proj_y: [{proj_y.min()} - {proj_y.max()}] - ({(proj_y < self.proj_h).astype(np.int32).sum()} - {(proj_y >= self.proj_h).astype(np.int32).sum()})')

        px = proj_x.copy()
        py = proj_y.copy()

        # round and clamp for use as index
        proj_x = np.maximum(np.minimum(
            self.proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)

        proj_y = np.maximum(np.minimum(
            self.proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)

        self.cached_data['uproj_x_idx'] = proj_x.copy()
        self.cached_data['uproj_y_idx'] = proj_y.copy()
        self.cached_data['uproj_depth'] = depth.copy()
        self.cached_data['px'] = px
        self.cached_data['py'] = py

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        pointcloud = pointcloud[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # get projection result
        proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        proj_pointcloud = np.full(
            (self.proj_h, self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32)
        proj_pointcloud[proj_y, proj_x] = pointcloud

        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices

        proj_mask = (proj_idx > 0).astype(np.int32)

        return proj_pointcloud, proj_range, proj_idx, proj_mask

def save_range_image(proj_range, save_path, fname="range.png"):
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)

    # 处理无效值
    proj_range = np.nan_to_num(proj_range, nan=0.0, posinf=0.0, neginf=0.0)

    # 归一化到 [0, 255]
    proj_range_norm = cv2.normalize(proj_range, None, 0, 255, cv2.NORM_MINMAX)
    proj_range_uint8 = proj_range_norm.astype(np.uint8)

    # 转伪彩色
    proj_range_color = cv2.applyColorMap(proj_range_uint8, cv2.COLORMAP_JET)

    # 保存图片
    out_file = os.path.join(save_path, fname)
    cv2.imwrite(out_file, proj_range_color)
    print(f"✅ 保存成功: {out_file}")

def save_range_image_with_intensity(proj_pointcloud, proj_range, save_path, fname="range.png"):
    """
    使用 proj_pointcloud 的强度（第4列）来为 proj_range 图像赋色
    """
    os.makedirs(save_path, exist_ok=True)

    # 处理无效值
    proj_range = np.nan_to_num(proj_range, nan=0.0, posinf=0.0, neginf=0.0)

    # 提取强度 (假设强度在第4列，索引为3)
    intensity = proj_pointcloud[:, :, 3]
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)

    # 将强度归一化到 [0, 255]
    intensity_norm = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX)
    intensity_uint8 = intensity_norm.astype(np.uint8)

    # 将强度作为颜色映射
    proj_range_color = cv2.applyColorMap(intensity_uint8, cv2.COLORMAP_JET)

    # 保存图片
    out_file = os.path.join(save_path, fname)
    cv2.imwrite(out_file, proj_range_color)
    print(f"✅ 保存成功: {out_file}")


if __name__ == '__main__':
    import open3d as o3d
    import numpy as np
    from omegaconf import OmegaConf, DictConfig
    import matplotlib.pyplot as plt
    import os
    import cv2
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    config =  OmegaConf.load('../cfgs/hercules.yaml')
    
   
    #对RangeProjection的测试
    scan_path = '/home/data/ldq/HeRCULES/Library/Library_03_Day/LiDAR/np8Aeva/1738301782926676285.bin'
    pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 8)
    pointcloud = np.concatenate((pointcloud[:, :3], pointcloud[:, 7:8]), axis=1) # [N, 4]
    pointcloud = np.concatenate((pointcloud, np.zeros((len(pointcloud), 1))), axis=1) # [N, 5]

    projection = RangeProjection(#todo 修改视场角
        fov_up=config.sensors.fov_up, fov_down=config.sensors.fov_down,
        fov_left=config.sensors.fov_left, fov_right=config.sensors.fov_right,
        proj_h=config.sensors.proj_h, proj_w=config.sensors.proj_w,
    )

    proj_pc, proj_range, proj_idx, proj_mask = projection.doProjection(pointcloud)
    print(proj_pc.shape)  # (32, 512, 3)
    print(proj_range.shape)  # (32, 512)
    print(proj_idx.shape)  # (32, 512)
    print(proj_mask.shape)  # (32, 512)
    timestamp = os.path.basename(scan_path).split('.')[0]
    save_range_image_with_intensity(proj_pc, proj_range, "./range_images", fname=f"{timestamp}.png")

    # save_range_image(proj_range, "./range_images", fname=f"{timestamp}.png")


