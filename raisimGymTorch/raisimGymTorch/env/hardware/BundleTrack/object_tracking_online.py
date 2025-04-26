# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2,os,logging,time,sys
import numpy as np
from raisimGymTorch.helper import rotations
from raisimGymTorch.env.hardware.BundleTrack.bundletrack_api import BundleTrackAPI
from raisimGymTorch.env.hardware.BundleTrack.segment_tracking_cv import mask_predict
from raisimGymTorch.env.hardware.BundleTrack.Utils import draw_posed_3d_box
from raisimGymTorch.env.hardware.BundleTrack.PointCloud import Realsense
logging.getLogger().setLevel(logging.ERROR)

import multiprocessing
import ctypes
from filterpy.kalman import KalmanFilter
import csv
from PySide6.QtWidgets import QApplication
import matplotlib
matplotlib.use('Qt5Agg')

class KalmanFilter3D:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=12, dim_z=6)
        # status transition matrix: x,y,z,vx,vy,vz, rx,ry,rz,wx,wy,wz
        self.kf.F = np.eye(12)
        # Position to velocity, Angle to angular velocity
        for i, j in [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]:
            self.kf.F[i, j] = 1  # This will be updated with dt later

        # status observation matrix:
        self.kf.H = np.zeros((6, 12))
        self.kf.H[0, 0] = 1  # x
        self.kf.H[1, 1] = 1  # y
        self.kf.H[2, 2] = 1  # z
        self.kf.H[3, 3] = 1  # RX
        self.kf.H[4, 4] = 1  # RY
        self.kf.H[5, 5] = 1  # RZ

        # 观测噪声协方差矩阵，0.0015m的噪声, 0.01rad噪声
        self.kf.R = np.eye(6) * 0.0015**2
        self.kf.R[3,3] = 0.01**2
        self.kf.R[4,4] = 0.01**2
        self.kf.R[5,5] = 0.01**2
        # 过程噪声协方差矩阵，越大越波动响应速度越快 pose: 1e-3就基本跟随了， 1e-6就基本平滑了, rot:1e-7, 1e-2
        self.kf.Q = np.eye(12) * 1e-10
        for i, j in [(3,3), (4,4), (5,5), (9, 9), (10, 10), (11, 11)]:
            self.kf.Q[i, j] = 1e-10
        # 初始状态协方差矩阵，初始状态不确定性
        self.kf.P *= 10
        # 检测移动
        self.win_size = 5
        self.data_buf = np.zeros((self.win_size, 6), dtype=np.float32)
        self.move_flag = False
        self.abs_dis = 0.0
        self.init_pose = None
        self.safety_flag = True
        self.last_pose = None
        
    def move_detect(self, pose):
        # check safety: 
        if self.init_pose is None:
            self.init_pose = pose
        else:
            self.safety_flag = True
            delta_pose = abs(self.init_pose - pose)
            #if delta_pose[2] > 0.025 or delta_pose[3] > 0.15 or delta_pose[4] > 0.1 or delta_pose[5] > 0.1: # topview
            if delta_pose[2] > 0.025 or delta_pose[3] > 0.45 or delta_pose[4] > 0.4 or delta_pose[5] > 0.3:
                self.safety_flag = False
                print(f"not a safety pose: delta_pose={delta_pose}")
                return

        # move [1,2,3] to [0,1,2] and set [3]
        self.data_buf[0:self.win_size-1, :] = self.data_buf[1:self.win_size, :]
        self.data_buf[self.win_size-1, :] = pose
        mean = np.mean(self.data_buf[0:self.win_size-1, :], axis=0)

        dif = mean - pose
        dis = dif[0]*dif[0] + dif[1]*dif[1]
        self.abs_dis = dis
        
        # check moving: x+y>0.015, rx,ry,rz>0.2rad  0.05m/s, 5HZ, 
        #if dis > 0.005*0.005 or dif[3] > 0.05 or dif[4] > 0.05 or dif[5] > 0.05: # topview
        if dis > 0.005*0.005 or dif[3] > 0.15 or dif[4] > 0.15 or dif[5] > 0.15:
            # moving
            self.move_flag = True
            self.kf.Q = np.eye(12) * 5e-5
            for i, j in [(3,3), (4,4), (5,5), (9, 9), (10, 10), (11, 11)]:
                self.kf.Q[i, j] = 1e-5
        else:
            # stop
            self.move_flag = False
            self.kf.Q = np.eye(12) * 1e-10
            for i, j in [(3,3), (4,4), (5,5), (9, 9), (10, 10), (11, 11)]:
                self.kf.Q[i, j] = 1e-10

    def filter(self, pose, dt):
        self.move_detect(pose)
        if self.safety_flag is False:
            return self.last_pose
        for i, j in [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]:
            self.kf.F[i, j] = dt
        self.kf.predict()
        self.kf.update(pose)
        self.last_pose = self.kf.x[:6]
        return self.kf.x[:6]

def thread(shared_obj_pose_w, shared_init_flag, camK_path, out_folder, log_csv):
    pos_array = np.frombuffer(shared_obj_pose_w.get_obj(), dtype=np.float32).reshape((4, 4))

    debug = False 
    if out_folder is not None:
        debug = True
        out_folder=out_folder
        os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}/pose')

    Treal2sim = np.array([[  0., 1., 0., 0.],
                        [-1., 0., 0., 0.],
                        [ 0., 0., 1., 0.],
                        [ 0., 0., 0., 1.]])
    Tsim2real = np.linalg.inv(Treal2sim)
    Tsimbase = np.array([[   1., 0., 0., 0.],
                        [ 0., 1., 0., 0.],
                        [ 0., 0., 1., 0.771],
                        [ 0., 0., 0., 1.]])
    Tbase2cam = np.loadtxt(camK_path + "/base2cam.txt", delimiter=',')
    
    camera = Realsense(camK_path, 200, False)
    segmenter = mask_predict(None)
    tracker = BundleTrackAPI(cfg_track_dir="/home/ubuntu/hand/foundationpose/BundleSDF/tracking_config_bundletrack.yml")

    K = camera.rgb_K
    W = 640
    H = 480

    np_rgb, np_depth = camera.get_img()
    color = cv2.resize(np_rgb, (W,H), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(np_depth, (W,H), interpolation=cv2.INTER_NEAREST)
        
    input_point = np.array(segmenter.get_point_from_image(color))
    input_label = np.array([1])  # 为分割对象的性质（背景|前景）
    bbox = segmenter.get_point_from_image(color)
    input_box = np.array([bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]])
    mask = segmenter.reset_all(color, input_point, input_label, input_box)
    init_mask_area = np.count_nonzero(mask)

    to_origin = np.eye(4)
    extents = np.array([0.10, 0.06, 0.02])
    show_bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    diff_t = np.zeros(3)
    now_t = np.zeros(4)
    last_t = -1.0
    cnt = 0

    filter_xyz = KalmanFilter3D()
    if log_csv is True:
        csvfile = open(f"/home/ubuntu/filter.csv","w")
        writer = csv.writer(csvfile)
        writer.writerows([['kx', 'x', 'ky', 'y', 'kz', 'z', 'rkx', 'rx', 'rky', 'ry', 'rkz', 'rz', 'dt', 'move', 'dis', 'occlusion', 'safety']])

    print("init finish all !!!")
    shared_init_flag.value = True

    last_pos = None
    while True:
        now_t[0] = time.time()

        np_rgb, np_depth = camera.get_img()
        color = cv2.resize(np_rgb, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(np_depth, (W,H), interpolation=cv2.INTER_NEAREST)

        mask = segmenter.tack_obj(color)
        mask_area = np.count_nonzero(mask)

        now_t[1] = time.time()
        diff_t[0] = diff_t[0] + now_t[1] - now_t[0]
        have_mask = np.any(mask) 
        if have_mask == False:
            continue

        #3大小的图像腐蚀，就是缩小白色区域周围一圈。
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel)

        # cost 0.1 >>> 0.4s for each frame
        occlusion_flag = False
        pose = tracker.run(color, depth, K, str(cnt), mask=mask, occ_mask=None, pose_in_model=np.eye(4))
        if last_pos is None:
            last_pos = pose
        if mask_area < init_mask_area * 0.4:
            print(f"occlusion severely: {mask_area / init_mask_area}")
            occlusion_flag = True
            #continue
            pose = last_pos
        last_pos = pose

        now_t[2] = time.time()
        diff_t[1] = diff_t[1] + now_t[2] - now_t[1]
        
        Tframe = np.ones((1,4))  # (N, 4)
        Tframe[0,:3] = pose[:3, 3]
        T_base = Tframe @ Tbase2cam.T
        T_simbase = T_base @ Tsim2real.T
        T_simworld = T_simbase @ Tsimbase.T
        
        xyz_rxryrz = np.zeros(6)
        xyz_rxryrz[:3] = T_simworld[0, :3]
        xyz_rxryrz[3:6] = rotations.mat2euler(pose[:3,:3])
        
        if last_t < 0.0:
            dt = 0.2
        else:
            dt = now_t[2] - last_t
        last_t = now_t[2]
        new_xyz_rxryrz = filter_xyz.filter(xyz_rxryrz, dt)
        
        if log_csv is True:
            listset = [new_xyz_rxryrz[0][0], xyz_rxryrz[0], new_xyz_rxryrz[1][0], xyz_rxryrz[1], new_xyz_rxryrz[2][0], xyz_rxryrz[2],
                    new_xyz_rxryrz[3][0], xyz_rxryrz[3], new_xyz_rxryrz[4][0], xyz_rxryrz[4], new_xyz_rxryrz[5][0], xyz_rxryrz[5], dt, filter_xyz.move_flag*0.1, filter_xyz.abs_dis, occlusion_flag*0.1, filter_xyz.safety_flag*0.1]
            writer.writerows([listset])
            csvfile.flush()

        now_t[3] = time.time()
        diff_t[2] = diff_t[2] + now_t[3] - now_t[2]

        pos_array[:3,3] = new_xyz_rxryrz[:3,0]
        cnt = cnt + 1
        if cnt % 10 == 0:
            print(f"10 times cost = {(diff_t / 10.0)}, all = {dt}")
            diff_t = np.zeros(3)

        # Tbase = Tbase2cam @ pose
        # Tsimbase = Tsim2real @ Tbase
        # Tsimworld = Tsimbase @ Tsimbase
        

        if debug is True:
            pose = pose@np.linalg.inv(to_origin)
            blue_layer = np.zeros_like(color)
            blue_layer[:] = (111, 166, 222)
            combined_image = np.where(mask[:, :, None] != 0, blue_layer, color)
            vis = draw_posed_3d_box(K, combined_image, ob_in_cam=pose, bbox=show_bbox, line_color=(255,255,0))
            cv2.imwrite(f'{out_folder}/pose/{cnt}.png', vis)
            #imageio.imwrite(f'{video_dir}/pose/{i}.png', vis)


class obj_track:
    def __init__(self, camK_path, out_folder = None, log_csv = False):
        multiprocessing.set_start_method('spawn', force=True)
        self.manager = multiprocessing.Manager()
        self.shared_dict = self.manager.dict()
        obj_pose_w = np.eye(4)
        
        self.shared_obj_pose_w = multiprocessing.Array(ctypes.c_float, obj_pose_w.flatten())
        self.shared_init_flag = multiprocessing.Value(ctypes.c_bool, False)

        self.process = multiprocessing.get_context('spawn').Process(target=thread, args=(self.shared_obj_pose_w, self.shared_init_flag, camK_path, out_folder, log_csv))
        
        self.process.start()

        while True:
            with self.shared_init_flag.get_lock():
                if self.shared_init_flag.value is True:
                    break
            time.sleep(1)
        
        time.sleep(3)
        print("init all finish ~~~~~!!!!!")

    def get_pose(self):
        with self.shared_obj_pose_w.get_lock():
            return np.frombuffer(self.shared_obj_pose_w.get_obj(), dtype=np.float32).reshape((4, 4))

def main():
    track = obj_track("/home/ubuntu/hand/calculate/0_datasets_allegro_hand", "/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/BundleTrack/output", True) # 
    print("init finish !!!!")
    while True:
        pose = track.get_pose()
        time.sleep(2)

if __name__ == '__main__':
    main()