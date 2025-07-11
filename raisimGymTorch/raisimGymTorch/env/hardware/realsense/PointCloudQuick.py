import pyrealsense2 as rs
import numpy as np
import time
from scipy.spatial import KDTree
import cv2
import threading
import open3d as o3d

class RealsenseQuick:
    def __init__(self, camK_path, sample_pc_num):
        self.debug = False
        self.width = 640
        self.hight = 480
        self.rate = 30
        self.sample_pc_num = sample_pc_num
        self.camK_path = camK_path
        self.rgb_K = None
        
        self.init_hardware()
        self.init_post_filter()

        for i in range(90):
            _, _ = self.get_one_rgbd()

        self.timer = threading.Timer(1.0, self.trigger_save) 
        #self.timer.start()

        self.event = threading.Event()
        self.thread_run_flag = True
        self.b_thread = threading.Thread(target=self.save_rgbd_thread)
        self.b_thread.daemon = True
        self.b_thread.start()

    def trigger_save(self):
        print("save a img..")
        rgb_image, depth_image = self.get_one_rgbd()
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite("/home/ubuntu/continue_rgb.png", rgb_image)
        self.timer = threading.Timer(1.0, self.trigger_save)  # 1秒后调用repeated_task
        self.timer.start()

    def get_one_rgbd(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if depth_frame and color_frame:
                break

        for trans in self.g_rs_depth_postprocess_list:
            depth_frame = trans.process(depth_frame)

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_scaled = (depth_image * self.depth_scale).astype(np.float32)
        H, W = color_image.shape[:2]
        color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)

        return color, depth

    def init_post_filter(self):
        ###################################################################################
        # realsense-viewer 软件里的postprocess顺序是：
        """
        decimation_filter --> HDR Merge --> threshold_filter --> Depth to Disparity --> spatial_filter
        --> temporal_filter --> Hole Filling ---> Disparity to Depth
        """
        # 使用rs内部的接口进行点云滤波
        # 下采样滤波，减少图片细节 2,4,8
        g_rs_downsample_filter = rs.decimation_filter(magnitude=2,)
        # 设置阈值范围
        g_rs_thres_filter = rs.threshold_filter(min_dist=0.3, max_dist=0.8)
        # 空间滤波，深度平滑
        g_rs_spatical_filter = rs.spatial_filter(
            magnitude=2,
            smooth_alpha=0.5,
            smooth_delta=20,
            hole_fill=0,
        )
        # 时间滤波，时间平滑
        g_rs_templ_filter = rs.temporal_filter(
            smooth_alpha=0.1,
            smooth_delta=40.,
            persistence_control=3
        )
        g_rs_depth2disparity_trans = rs.disparity_transform(True)
        g_rs_disparity2depth_trans = rs.disparity_transform(False)
        self.g_rs_depth_postprocess_list = [
            g_rs_downsample_filter,
            g_rs_thres_filter,
            #g_rs_depth2disparity_trans,
            g_rs_spatical_filter,
            #g_rs_templ_filter,
            #g_rs_disparity2depth_trans
        ]
        ###################################################################################


    # KD tree to calculate K-Nearest Neighbors for each point
    def remove_outliers(self, point_cloud, k=5, threshold=1.5):
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

    def depth2xyzmap(self, depth):
        K = self.rgb_K
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

        mask = np.any(xyz_map != 0, axis=-1)
        filtered_points = xyz_map[mask]
        return filtered_points

    def sample_pc(self, all_pc):
        if all_pc.shape[0] > self.sample_pc_num*2:
            replace_flag=False
        else:
            replace_flag=True
        pose_pc_idx = np.random.choice(all_pc.shape[0], int(self.sample_pc_num*2), replace=replace_flag)
        pose_pc = all_pc[pose_pc_idx]
        pose = np.mean(pose_pc, axis=0).reshape(1,3)
        filtered_point_cloud = self.remove_outliers(pose_pc, k=15, threshold=3.0)
        if all_pc.shape[0] > self.sample_pc_num:
            replace_flag=False
        else:
            replace_flag=True
        output_pc_idx = np.random.choice(filtered_point_cloud.shape[0], self.sample_pc_num, replace=replace_flag)
        return filtered_point_cloud[output_pc_idx], pose

    def raisim_frame_tf(self, cam_frame):
    
        Treal2sim = np.array([[  0., 1., 0., 0.],
                            [-1., 0., 0., 0.],
                            [ 0., 0., 1., 0.],
                            [ 0., 0., 0., 1.]])
        Tsim2real = np.linalg.inv(Treal2sim)
        Tsimbase = np.array([[   1., 0., 0., 0.],
                            [ 0., 1., 0., 0.],
                            [ 0., 0., 1., 0.771],
                            [ 0., 0., 0., 1.]])
        
        # tf matrix
        Tbase2cam = np.loadtxt(self.camK_path + "/base2cam.txt", delimiter=',')
        # https://support.intelrealsense.com/hc/en-us/community/posts/4405875311123-About-make-sure-FOV-specification-of-D435i 
        # tf from RGB to left-IR camera
        Tcamrgb2depth = np.array([[  1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [ 0., 0., 1., 0.0031],
                            [ 0., 0., 0., 1.]])

        T_depth = np.hstack((cam_frame, np.ones((cam_frame.shape[0], 1))))  # (N, 4)
        T_rgb = T_depth @ Tcamrgb2depth.T
        T_base = T_rgb @ Tbase2cam.T
        T_simbase = T_base @ Tsim2real.T
        T_simworld = T_simbase @ Tsimbase.T
        
        return T_simworld
    
    def init_hardware(self):
        # realsense get depth
        self.pipeline = rs.pipeline()
        config = rs.config()
        with open(self.camK_path + "/deviceid.txt",'r') as f:
            id=f.read().splitlines()[0]
            config.enable_device(id)
        config.enable_stream(rs.stream.depth, self.width, self.hight, rs.format.z16, self.rate)
        config.enable_stream(rs.stream.color, self.width, self.hight, rs.format.rgb8, self.rate)
        profile = self.pipeline.start(config)
        #profile.get_device().hardware_reset()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            rgb_frame = aligned_frames.get_color_frame()
            if not rgb_frame:
                continue
            color_intrin = aligned_frames.get_profile().as_video_stream_profile().get_intrinsics()
            self.rgb_K = np.array([[color_intrin.fx, 0., color_intrin.ppx], [0., color_intrin.fy, color_intrin.ppy], [0, 0, 1.]])
            break

    def get_mask_rgbd(self, mask):
        depth_image = np.load(self.save_pth + "/tmp.npy")

        output = np.zeros_like(depth_image)
        output[mask] = depth_image[mask]
        # get point cloud
        pointcloud_xyz = self.depth2xyzmap(output)
        all_pc = pointcloud_xyz.reshape(-1, 3).astype(np.float32)
    
        mask_pcd_new, mean_pose = self.sample_pc(all_pc)
        tsim2realpc = self.raisim_frame_tf(mask_pcd_new)
        tsim2realpose = self.raisim_frame_tf(mean_pose)

        if self.debug:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(mask_pcd_new)
            o3d.visualization.draw_geometries([cloud])

        #print(f"-------------get calculate tf time = {log_time5 - log_time4}")
        print(f"-------------point cloud center: camera_frame={mean_pose}, raisim_world_frame={tsim2realpose}")
        return tsim2realpose[:, :3], tsim2realpc[:, :3]
        
    def get_point_from_image(self, color_frame):
        points = []
        def select_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(image_display, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow("Image", image_display)

        # Convert image to numpy array
        image = color_frame
        image_display = image.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", select_points)

        print("Click on the image to select points. Press Enter when done.")

        while True:
            cv2.imshow("Image", image_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break

        print("------------------get point: " + str(points))
        cv2.destroyAllWindows()

        return points

    def async_save_rgbd(self, save_pth):
        self.save_pth = save_pth
        self.thread_run_flag = True
        self.event.set()
         
    def save_rgbd_thread(self):
        while True:
            self.event.wait()
            rgb_image, depth_image = self.get_one_rgbd()
            np.save(self.save_pth + "/tmp.npy", depth_image)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.save_pth+"/rgb.png", rgb_image)
            self.thread_run_flag = False
            self.event.clear()
      
def main() -> None:
    print("test ...")
    test = RealsenseQuick("/home/ubuntu/hand/calculate/0_datasets_allegro_hand_topview", 200) # 0_datasets_allegro_hand_topview 0_datasets_allegro_hand

    for i in range (6):
        rgb_frame, depth_frame = test.get_one_rgbd()
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        bbox = test.get_point_from_image(rgb_frame)
        
        height, width = rgb_frame.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        mask[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] = True
        # mask_visual = mask.astype(np.uint8) * 255
        # cv2.imshow('Mask', mask_visual)
        # cv2.waitKey(0)
        
        test.async_save_rgbd("/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/realsense")
        time.sleep(5)
        _, _ = test.get_mask_rgbd(mask)

if __name__ == "__main__":
    main()
