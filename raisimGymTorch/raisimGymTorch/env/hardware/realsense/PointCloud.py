import pyrealsense2 as rs
import argparse
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

def depth2xyzmap(depth, K):
    
    invalid_mask = (depth<0.1)
    H,W = depth.shape[:2]
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth[vs,us]
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H,W,3), dtype=np.float32)
    xyz_map[vs,us] = pts
    xyz_map[invalid_mask] = 0

    mask = np.any(xyz_map != 0, axis=-1)
    filtered_points = xyz_map[mask]
    return filtered_points

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
    def __init__(self, camK_path, sample_pc_num, calculate_flag = False):
        print("test")
        
        self.USE_KD_TREE = False
        self.calculate_flag = calculate_flag
        if calculate_flag:
            self.filter_time = 200
            self.downsample = 1
        else:
            self.filter_time = 60
            self.downsample = 1

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
        
        self.sample_pc_num = sample_pc_num
        self.camK_path = camK_path
        self.flat_path = os.path.join(os.path.dirname(__file__), 'flat.ply')
        self.flat_npy_path = os.path.join(os.path.dirname(__file__), 'flat.npy')
        
        self.all_pc = None

    def sample_pc(self):
        pose_pc_idx = np.random.choice(self.all_pc.shape[0], int(self.sample_pc_num*5), replace=False)
        pose_pc = self.all_pc[pose_pc_idx]
        pose = np.mean(pose_pc, axis=0).reshape(1,3)
        filtered_point_cloud = remove_outliers(pose_pc, k=15, threshold=3.0)
        output_pc_idx = np.random.choice(filtered_point_cloud.shape[0], self.sample_pc_num, replace=False)
        return filtered_point_cloud[output_pc_idx], pose
    
    def filter_pc(self, nparray): # shape is (50, 480, 640)
        # rule1: ignore the value smaller then 0.4 and larger then 0.8
        rule1_mask = (nparray < 0.4) | (nparray > 0.8)
        masked_images = np.ma.masked_array(nparray, mask=rule1_mask)
        masked_images = masked_images.filled(np.nan)  
        output = np.zeros((int(480/self.downsample), int(640/self.downsample)))

        # sorted_images = np.sort(masked_images, axis=0)
        # # Calculate the number of valid entries after rule 1
        # valid_counts = (~masked_images.mask).sum(axis=0)

        # # Create a mask for the 5 minimum and 5 maximum values
        # trim_mask = np.zeros_like(sorted_images.mask, dtype=bool)
        # trim_mask[-5:5] = True  # Mask the 5 largest values

        # # Only apply trim_mask where there are more than 10 valid entries
        # trim_mask = np.logical_and(trim_mask, valid_counts > 30)

        # # Apply the trim mask
        # sorted_trimmed_images = np.ma.masked_array(sorted_images, mask=trim_mask)

        # # Step 3 & 4: Calculate the max-min difference and determine output
        # max_values = sorted_trimmed_images.max(axis=0)
        # min_values = sorted_trimmed_images.min(axis=0)
        # max_min_difference = max_values - min_values

        # # Apply Rule 3 & 4
        # within_threshold = max_min_difference <= 0.05
        # output[within_threshold] = sorted_trimmed_images.mean(axis=0)[within_threshold]

        # if self.USE_KD_TREE is False:
        #     if abs(output[i][j] - flat_npy[i][j]) < 0.02:
        #         output[i][j] = 0.0

        for i in range(int(480/self.downsample)):
            for j in range(int(640/self.downsample)):
                #mask_origin = nparray[:,i,j]
                tmp = masked_images[:,i,j]
                valid_values = tmp[~np.isnan(tmp)] # 去除nan
                mask_data_value = np.sort(valid_values)
                data_num = len(mask_data_value)
                if data_num > self.filter_time / 2:
                    mask_max_min = mask_data_value[int(self.filter_time/10):-int(self.filter_time/10)]  # 去掉10%最值
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
                                
                if self.USE_KD_TREE is False and self.calculate_flag is False:
                    if abs(output[i][j] - self.flat_npy[int(i*self.downsample)][int(j*self.downsample)]) < 0.02:
                        output[i][j] = 0.0

        return output
                            
    def GetPointCloud(self):
        log_time1 = time.time()
        
        have_flat = False
        if (os.path.exists(self.flat_path)): 
            have_flat = True
            flat_pc = o3d.io.read_point_cloud(self.flat_path)
        if (os.path.exists(self.flat_npy_path)): 
            have_flat = True
            self.flat_npy = np.load(self.flat_npy_path)
        else:
            if self.calculate_flag is False:
                print(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
                print(" !!!!!!!!!!! need to calculate camera !!!!!!!!!!!!! ")
                print(" !!!!! Please clear the desktop and execute: !!!!!! ")
                print(" !!!!!!!!! python PointCloud.py -c True !!!!!!!!!!! ")
                print(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
                exit(0)

        # tf matrix
        Tbase2cam = np.loadtxt(self.camK_path + "/base2cam.txt", delimiter=',')
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
        with open(self.camK_path + "/deviceid.txt",'r') as f:
            id=f.read().splitlines()[0]
            config.enable_device(id)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        align = rs.align(align_to)
        cam_K = np.loadtxt(self.camK_path + "/camK_640x480.txt", delimiter=',')
        wait_cnt = 0

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
            downsampled_depth_image = depth_image[::self.downsample, ::self.downsample]
            depth_image_scaled = (downsampled_depth_image * depth_scale).astype(np.float32)
            pointcloud_xyz_list.append(depth_image_scaled)
            if wait_cnt < self.filter_time:
                continue
            
            log_time3 = time.time()
            print(f"-------------get depth time = {log_time3 - log_time2}")
            
            output = self.filter_pc(np.array(pointcloud_xyz_list))

            log_time4 = time.time()
            print(f"-------------filter depth time = {log_time4 - log_time3}")

            # get point cloud
            pointcloud_xyz = depth2xyzmap(output, cam_K)
            points1 = pointcloud_xyz.reshape(-1, 3).astype(np.float32)

            log_time5 = time.time()
            print(f"-------------get point cloud time = {log_time5 - log_time4}")
            
            if self.calculate_flag is False:
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
                    self.all_pc = unique_points1
                    self.all_pc = remove_outliers(self.all_pc)
                else:
                    self.all_pc = points1

                log_time6 = time.time()
                print(f"-------------diff pointcloud depth time = {log_time6 - log_time5}")
                
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(self.all_pc)
                o3d.visualization.draw_geometries([cloud])
            else:
                output_fill = output.copy()
                for i in range(int(480)):
                    for j in range(int(640)):
                        value = output[i][j]
                        if value < 0.4 or value > 0.8:
                            output_fill[i, j] = calculate_mean_around(value, i, j)

                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points1)
                o3d.visualization.draw_geometries([cloud])
                o3d.io.write_point_cloud(self.flat_path, cloud, write_ascii=True)
                np.save(self.flat_npy_path, output)
                pipeline.stop()
                return
            
            break

        pipeline.stop()

        mask_pcd_new, mean_pose = self.sample_pc()

        mask_pcd_homogeneous = np.hstack((mask_pcd_new, np.ones((mask_pcd_new.shape[0], 1))))  # (200, 4)
        print (f"depth frame pc = {mask_pcd_homogeneous[0]}")
        mask_pcd_homogeneous_depth = mask_pcd_homogeneous @ Tcamrgb2depth.T
        print (f"RGB frame pc = {mask_pcd_homogeneous_depth[0]}")
        tbase2pcd = mask_pcd_homogeneous_depth @ Tbase2cam.T
        print (f"realbase frame pc = {tbase2pcd[0]}")
        tsim_base2pc = tbase2pcd @ Tsim2real.T
        print (f"simbase frame pc = {tsim_base2pc[0]}")
        tsim2realpc = tsim_base2pc @ Tsimbase.T
        print (f"simworld frame pc = {tsim2realpc[0]}")
        
        pose_homogeneous = np.hstack((mean_pose, np.ones((mean_pose.shape[0], 1))))  # (200, 4)
        print (f"depth frame pose = {pose_homogeneous[0]}")
        mask_pcd_homogeneous_pose = pose_homogeneous @ Tcamrgb2depth.T
        print (f"RGB frame pose = {mask_pcd_homogeneous_pose[0]}")
        tbase2pose = mask_pcd_homogeneous_pose @ Tbase2cam.T
        print (f"realbase frame pose = {tbase2pose[0]}")
        tsim_base2pose = tbase2pose @ Tsim2real.T
        print (f"simbase frame pose = {tsim_base2pose[0]}")
        tsim2realpose = tsim_base2pose @ Tsimbase.T
        print (f"simworld frame pose = {tsim2realpose[0]}")

        return tsim2realpose[:, :3], tsim2realpc[:, :3]
        
def main() -> None:
    print("test ...")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--calculate_flag", help="check the table", type=bool, default=False)
    args = parser.parse_args()
    test = Realsense("/home/ubuntu/hand/calculate/0_datasets_allegro_hand_topview", 200, args.calculate_flag)
    pose, pc = test.GetPointCloud()

if __name__ == "__main__":
    main()
