# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from BundleTrack.scripts.data_reader import YcbineoatReader
from bundletrack_api import BundleTrackAPI
import os
from segment_tracking_cv import mask_predict
import cv2
import numpy as np

import trimesh
import imageio
from Utils import draw_posed_3d_box

import logging
logging.getLogger().setLevel(logging.ERROR)
import time

def main():
    video_dir="/home/ubuntu/hand/foundationpose/realsense/output/realsense/test5_v3mpmin_20hz"
    out_folder="output"

    os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}')

    segmenter = mask_predict(None)
    tracker = BundleTrackAPI(cfg_track_dir="/home/ubuntu/hand/foundationpose/BundleSDF/tracking_config_bundletrack.yml")
    reader = YcbineoatReader(video_dir=video_dir, shorter_side=480)

    last_pose = None
    diff = 0.0
    cnt = 0
    for i in range(0, len(reader.color_files), 6): # 帧间隔是1
        
        start = time.time()
        
        color_file = reader.color_files[i]
        color = cv2.imread(color_file)
        depth = reader.get_depth(i)
        H,W = depth.shape[:2]
        color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)

        if i==0:
            input_point = np.array(segmenter.get_point_from_image(color))
            input_label = np.array([1])  # 为分割对象的性质（背景|前景）
            bbox = segmenter.get_point_from_image(color)
            input_box = np.array([bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]])
            mask = segmenter.reset_all(color, input_point, input_label, input_box)
        else:
            mask = segmenter.tack_obj(color)
        
        have_mask = np.any(mask) 
        if have_mask == False:
            continue
            
        #3大小的图像腐蚀，就是缩小白色区域周围一圈。
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel)

        id_str = reader.id_strs[i]
        pose_in_model = np.eye(4)

        K = reader.K.copy()

        pose = tracker.run(color, depth, K, id_str, mask=mask, occ_mask=None, pose_in_model=pose_in_model)
        if last_pose is not None:
            tf = pose @ np.transpose(last_pose)
        else:
            tf = np.eye(4)
        last_pose = pose            

        if i==0:
            # 转换像素坐标到世界坐标
            world_coords = tracker.pixel_to_world(input_point, 1.0, K, pose)
        
        # 将世界坐标投影回当前帧的像素坐标
        projected_pixel_coords = tracker.world_to_pixel(world_coords, K, pose)

        
        diff = diff + time.time() - start
        cnt = cnt + 1
        if cnt == 10:
            print(f"20 times cost = {(diff / 10.0)}")
            cnt = 0
            diff = 0.0
        if i == 0:
            #mesh = trimesh.load(f'{video_dir}/blue_moon-tmp/textured_mesh.obj')
            #_, extents = trimesh.bounds.oriented_bounds(mesh) # mesh中心坐标，长宽高，
            to_origin = np.eye(4)
            extents = np.array([0.10, 0.06, 0.02])
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        pose = pose@np.linalg.inv(to_origin)
        blue_layer = np.zeros_like(color)
        blue_layer[:] = (111, 166, 222)
        combined_image = np.where(mask[:, :, None] != 0, blue_layer, color)
        vis = draw_posed_3d_box(K, combined_image, ob_in_cam=pose, bbox=bbox, line_color=(255,255,0))
        cv2.imwrite(f'{video_dir}/pose/{i}.png', vis)
        #imageio.imwrite(f'{video_dir}/pose/{i}.png', vis)


if __name__ == '__main__':
    main()