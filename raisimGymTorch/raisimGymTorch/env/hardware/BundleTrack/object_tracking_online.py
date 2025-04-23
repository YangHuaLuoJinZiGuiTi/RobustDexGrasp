# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2,os,logging,time,sys
import numpy as np
from raisimGymTorch.env.hardware.BundleTrack.bundletrack_api import BundleTrackAPI
from raisimGymTorch.env.hardware.BundleTrack.segment_tracking_cv import mask_predict
from raisimGymTorch.env.hardware.BundleTrack.Utils import draw_posed_3d_box
from raisimGymTorch.env.hardware.BundleTrack.PointCloud import Realsense
logging.getLogger().setLevel(logging.ERROR)

import multiprocessing
import ctypes

def thread(shared_obj_pose_w, shared_init_flag, camK_path, out_folder):
    array = np.frombuffer(shared_obj_pose_w.get_obj(), dtype=np.float32).reshape((4, 4))

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

    to_origin = np.eye(4)
    extents = np.array([0.10, 0.06, 0.02])
    show_bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    diff = 0.0
    cnt = 0
    
    print("init finish all !!!")
    shared_init_flag.value = True

    while True:
        start = time.time()

        np_rgb, np_depth = camera.get_img()
        color = cv2.resize(np_rgb, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(np_depth, (W,H), interpolation=cv2.INTER_NEAREST)

        mask = segmenter.tack_obj(color)

        have_mask = np.any(mask) 
        if have_mask == False:
            continue

        #3大小的图像腐蚀，就是缩小白色区域周围一圈。
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel)

        # cost 0.1 >>> 0.4s for each frame
        pose = tracker.run(color, depth, K, str(cnt), mask=mask, occ_mask=None, pose_in_model=np.eye(4))

        diff = diff + time.time() - start
        cnt = cnt + 1
        if cnt % 1 == 0:
            print(f"1 times cost = {(diff / 1.0)}")
            diff = 0.0

        Tbase = Tbase2cam @ pose
        Tsimbase = Tsim2real @ Tbase
        Tsimworld = Tsimbase @ Tsimbase
        
        array[:] = Tsimworld[:]

        if debug is True:
            pose = pose@np.linalg.inv(to_origin)
            blue_layer = np.zeros_like(color)
            blue_layer[:] = (111, 166, 222)
            combined_image = np.where(mask[:, :, None] != 0, blue_layer, color)
            vis = draw_posed_3d_box(K, combined_image, ob_in_cam=pose, bbox=show_bbox, line_color=(255,255,0))
            cv2.imwrite(f'{out_folder}/pose/{cnt}.png', vis)
            #imageio.imwrite(f'{video_dir}/pose/{i}.png', vis)


class obj_track:
    def __init__(self, camK_path, out_folder = None):
        multiprocessing.set_start_method('spawn', force=True)
        self.manager = multiprocessing.Manager()
        self.shared_dict = self.manager.dict()
        obj_pose_w = np.eye(4)
        
        self.shared_obj_pose_w = multiprocessing.Array(ctypes.c_float, obj_pose_w.flatten())
        self.shared_init_flag = multiprocessing.Value(ctypes.c_bool, False)

        self.process = multiprocessing.get_context('spawn').Process(target=thread, args=(self.shared_obj_pose_w, self.shared_init_flag, camK_path, out_folder))
        
        self.process.start()

        while True:
            with self.shared_init_flag.get_lock():
                if self.shared_init_flag.value is True:
                    break
            time.sleep(1)

    def get_pose(self):
        with self.shared_obj_pose_w.get_lock():
            return np.frombuffer(self.shared_obj_pose_w.get_obj(), dtype=np.float32).reshape((4, 4))

def main():
    track = obj_track("/home/ubuntu/hand/calculate/0_datasets_allegro_hand_topview", "/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/BundleTrack/output")
    print("init finish !!!!")
    while True:
        pose = track.get_pose()
        time.sleep(2)

if __name__ == '__main__':
    main()