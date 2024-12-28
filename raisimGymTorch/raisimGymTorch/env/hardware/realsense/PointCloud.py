import pyrealsense2 as rs
import cv2
import numpy as np
import time
from scipy.spatial import KDTree

import matplotlib.pyplot as plt

import os
import open3d as o3d


# KD tree to calculate K-Nearest Neighbors for each point
def remove_outliers(point_cloud, k=5, threshold=1.5):
    tree = KDTree(point_cloud)
    
    # search KNN for each point (including K itself, so it is k+1)
    distances, _ = tree.query(point_cloud, k=k+1)
    
    # mean distance of each KNN point
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    # calculate Median of mean distance and MAD（Mean Absolute Deviation)
    median_distance = np.median(mean_distances)
    mad_distance = np.median(np.abs(mean_distances - median_distance))
    
    # use threshold to ignore outliers
    normalized_distances = np.abs(mean_distances - median_distance) / (mad_distance + 1e-8)
    inliers = normalized_distances < threshold
    
    return point_cloud[inliers]

def depth2xyzmap(depth, K, uvs=None):
    
    invalid_mask = (depth<0.1)
    H,W = depth.shape[:2]
    if uvs is None:
        vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:,0]
        vs = uvs[:,1]
    zs = depth[vs,us]
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H,W,3), dtype=np.float32)
    xyz_map[vs,us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map

# 定义一个函数来计算周围符合条件的像素均值
def calculate_mean_around(img, i, j):
    # 定义周围像素的相对位置
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),         (0, 1),
                 (1, -1), (1, 0), (1, 1)]

    valid_values = []

    # 遍历周围像素
    for di, dj in neighbors:
        ni, nj = i + di, j + dj
        
        # 检查边界条件
        if 0 <= ni < img.shape[0] and 0 <= nj < img.shape[1]:
            # 仅添加符合条件的值
            if img[ni, nj] < 0.4 or img[ni, nj] > 0.8:
                valid_values.append(img[ni, nj])

    # 计算均值，若有效值列表不为空
    if valid_values:
        return np.mean(valid_values)
    else:
        print(f"---error value {i}, {j}")
        return img[i, j]  # 如果没有有效值，返回原值

class Realsense:
    def __init__(self):
        print("test")
        
        self.USE_KD_TREE = False

        # realsense-viewer 软件里的postprocess顺序是：
        """
        decimation_filter --> HDR Merge --> threshold_filter --> Depth to Disparity --> spatial_filter
        --> temporal_filter --> Disparity to Depth
        """
        # 使用rs内部的接口进行点云滤波
        g_rs_downsample_filter = rs.decimation_filter(
            magnitude=2 ** 1,
        )  # 下采样率
        g_rs_thres_filter = rs.threshold_filter(min_dist=0.4, max_dist=0.8)
        g_rs_spatical_filter = rs.spatial_filter(
            magnitude=2,
            smooth_alpha=0.5,
            smooth_delta=20,
            hole_fill=0,
        )
        g_rs_templ_filter = rs.temporal_filter(
            smooth_alpha=0.1,
            smooth_delta=100.,
            persistence_control=0
        )
        g_rs_depth2disparity_trans = rs.disparity_transform(True)
        g_rs_disparity2depth_trans = rs.disparity_transform(False)

        self.g_rs_depth_postprocess_list = [
            #g_rs_downsample_filter,
            g_rs_thres_filter,
            #g_rs_depth2disparity_trans,
            #g_rs_spatical_filter,
            #g_rs_templ_filter,
            #g_rs_disparity2depth_trans
        ]

    def GetPointCloud(self, camK_path):
        log_time1 = time.time()
        # tf matrix
        Tbase2cam = np.loadtxt(camK_path + "/base2cam.txt", delimiter=',')
        Treal2sim = np.array([[  0., 1., 0., 0.],
                            [-1., 0., 0., 0.],
                            [ 0., 0., 1., 0.],
                            [ 0., 0., 0., 1.]])
        Tsim2real = np.linalg.inv(Treal2sim)
        Tsimbase = np.array([[   1., 0., 0., 0.],
                            [ 0., 1., 0., 0.],
                            [ 0., 0., 1., 0.771],
                            [ 0., 0., 0., 1.]])
        
        # https://support.intelrealsense.com/hc/en-us/community/posts/4405875311123-About-make-sure-FOV-specification-of-D435i 
        # tf from RGB to left-IR camera
        Tcamrgb2depth = np.array([[  1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [ 0., 0., 1., 0.],
                            [ 0., 0., 0., 1.]])

        # realsense get depth
        pipeline = rs.pipeline()
        config = rs.config()
        pc = rs.pointcloud()
        with open(camK_path + "/deviceid.txt",'r') as f:
            id=f.read().splitlines()[0]
            config.enable_device(id)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        align = rs.align(align_to)

        cam_K = np.loadtxt(camK_path + "/camK_640x480.txt", delimiter=',')

        wait_cnt = 0
        mask = None
        check_indice = None
        valid = None
        check_array = []
        aligned_depth_frame_table = None
        aligned_depth_frame_object = None
        
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if depth_frame:
                wait_cnt += 1
                if wait_cnt > 10:
                    
                    g_intrinsics = aligned_frames.get_profile().as_video_stream_profile().get_intrinsics()
                    color_intrin = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
                    depth_intrin = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
                    g_color_intrinsics_matrix = np.array([
                        [color_intrin.fx, 0., color_intrin.ppx],
                        [0., color_intrin.fy, color_intrin.ppy],
                        [0, 0, 1.]
                    ])
                    g_depth_intrinsics_matrix = np.array([
                        [depth_intrin.fx, 0., depth_intrin.ppx],
                        [0., depth_intrin.fy, depth_intrin.ppy],
                        [0, 0, 1.]
                    ])

                    break
        
        flat_path = "/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/realsense/flat.ply"
        flat_npy_path = "/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/realsense/flat.npy"
        have_flat = False
        if (os.path.exists(flat_path)): 
            have_flat = True
            flat_pc = o3d.io.read_point_cloud(flat_path)
        if (os.path.exists(flat_npy_path)): 
            have_flat = True
            flat_npy = np.load(flat_npy_path)

        log_time2 = time.time()
        print(f"-------------init time = {log_time2 - log_time1}")
        wait_cnt = 0
        pointcloud_xyz_list = []
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame:
                continue

            wait_cnt += 1

            #for trans in self.g_rs_depth_postprocess_list:
            #    depth_frame = trans.process(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image_scaled = (depth_image * depth_scale).astype(np.float32)
            pointcloud_xyz_list.append(depth_image_scaled)
            if wait_cnt < 60:
                continue
            
            log_time3 = time.time()
            print(f"-------------get depth time = {log_time3 - log_time2}")
            
            nparray = np.array(pointcloud_xyz_list)  # shape is (50, 480, 640)
            
            # rule1: ignore the value smaller then 0.4 and larger then 0.8
            rule1_mask = (nparray < 0.4) | (nparray > 0.8)
            masked_images = np.ma.masked_array(nparray, mask=rule1_mask)
            masked_images = masked_images.filled(np.nan)  
  
            output = np.zeros((480, 640))

            for i in range(480):
                for j in range(640):
                    mask_origin = nparray[:,i,j]
                    tmp = masked_images[:,i,j]
                    valid_values = tmp[~np.isnan(tmp)] # 去除nan
                    mask_data_value = np.sort(valid_values)
                    data_num = len(mask_data_value)
                    if data_num > wait_cnt / 2:
                        mask_max_min = mask_data_value[int(wait_cnt/10):-int(wait_cnt/10)]  # 去掉10%最值
                        max_value = np.max(mask_max_min)
                        min_value = np.min(mask_max_min)
                        diff = max_value - min_value
                        if diff < 0.05:
                            output[i][j] = np.mean(mask_max_min)
                        else:
                            bins = np.linspace(min_value, max_value, num=3) # devide into 2 parts
                            indices = np.digitize(mask_max_min, bins) # put data into 2 parts
                            max_mean = 0.0
                            max_cnt = 0
                            for k in range(1, len(bins)):  # check each parts
                                region_data = mask_max_min[indices == k] # get the data
                                if len(region_data) > 0:  # make sure there is data in this part
                                    diff = np.max(region_data) - np.min(region_data)
                                    if diff < 0.05:
                                        region_mean = np.mean(region_data)
                                        region_count = len(region_data)
                                        if max_cnt < region_count:
                                            max_cnt = region_count
                                            max_mean = region_mean
                            output[i][j] = max_mean
                            # # for debug
                            # if (max_mean > 0.1): 
                            #     plt.cla()
                            #     plt.plot(mask_max_min, 'rd')
                            #     plt.plot(mask_origin, 'bo')
                            #     plt.plot(mask_data_value, 'g*')
                            #     plt.show()
                            #     continue
                                    
                    if self.USE_KD_TREE is False:
                        if abs(output[i][j] - flat_npy[i][j]) < 0.02:
                            output[i][j] = 0.0

            log_time4 = time.time()
            print(f"-------------filter depth time = {log_time4 - log_time3}")

            # get point cloud
            pointcloud_xyz = depth2xyzmap(output, cam_K)
            points1 = pointcloud_xyz.reshape(-1, 3).astype(np.float32)

            log_time5 = time.time()
            print(f"-------------get point cloud time = {log_time5 - log_time4}")
            
            if have_flat:
                if self.USE_KD_TREE:
                    points2 = np.asarray(flat_pc.points)
                    # 使用 cKDTree 进行空间索引： 
                    tree2 = KDTree(points2)

                    # 查找每个点在另一个点云中的最近邻并比较距离： 为了找出不同的点，我们需要查找在另一个点云中没有近邻的点。
                    # 找到 points1 中每个点在 points2 中的最近邻
                    distances1, _ = tree2.query(points1)
                    # 过滤出在阈值范围内没有对应点的点： 这些点即为在另一个点云中没有匹配点的点。
                    # 在 points1 中没有对应点的点
                    unique_points1 = points1[(distances1 >= 0.008) & (distances1 <= 0.4)]

                    # 组合两个点云之间的差异点云，包含在两个点云中没有匹配的点。
                    difference_cloud = unique_points1
                    difference_cloud = remove_outliers(difference_cloud)
                else:
                    difference_cloud = points1

                log_time6 = time.time()
                print(f"-------------diff pointcloud depth time = {log_time6 - log_time5}")
                
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(difference_cloud)
                o3d.visualization.draw_geometries([cloud])
            else:
                output_fill = output.copy()
                for i in range(480):
                    for j in range(640):
                        value = output[i][j]
                        if value < 0.4 or value > 0.8:
                            output_fill[i, j] = calculate_mean_around(value, i, j)

                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points1)
                o3d.visualization.draw_geometries([cloud])
                o3d.io.write_point_cloud(flat_path, cloud, write_ascii=True)
                np.save(flat_npy_path, output)
            
            break

        pipeline.stop()
        
def main() -> None:
    print("test ...")
    test = Realsense()
    test.GetPointCloud("/home/ubuntu/hand/calculate/0_datasets_allegro_hand_topview")

if __name__ == "__main__":
    main()
