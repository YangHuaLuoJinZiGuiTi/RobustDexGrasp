# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2,os,logging,time
import numpy as np
from bundletrack_api import BundleTrackAPI
from segment_tracking_cv import mask_predict
from Utils import draw_posed_3d_box
from PointCloud import Realsense
logging.getLogger().setLevel(logging.ERROR)

def main():
    camK_path="/home/ubuntu/hand/calculate/0_datasets_allegro_hand_topview"
    out_folder="output"

    os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}/pose')

    camera = Realsense(camK_path, 200, False)
    segmenter = mask_predict(None)
    tracker = BundleTrackAPI(cfg_track_dir="/home/ubuntu/hand/foundationpose/BundleSDF/tracking_config_bundletrack.yml")

    last_pose = None
    diff = 0.0
    cnt = 0
    K = camera.rgb_K
    W = 640
    H = 480

    to_origin = np.eye(4)
    extents = np.array([0.10, 0.06, 0.02])
    show_bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    first_frame = True
    while True:
        start = time.time()

        np_rgb, np_depth = camera.get_img()
        color = cv2.resize(np_rgb, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(np_depth, (W,H), interpolation=cv2.INTER_NEAREST)

        if first_frame:
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

        pose = tracker.run(color, depth, K, str(cnt), mask=mask, occ_mask=None, pose_in_model=np.eye(4))

        if first_frame:
            first_frame = False
            tf = np.eye(4)
        else:
            tf = pose @ np.transpose(last_pose)
            diff = diff + time.time() - start
            cnt = cnt + 1
            if cnt % 10 == 0:
                print(f"10 times cost = {(diff / 10.0)}")
                diff = 0.0

        last_pose = pose
        continue

        pose = pose@np.linalg.inv(to_origin)
        blue_layer = np.zeros_like(color)
        blue_layer[:] = (111, 166, 222)
        combined_image = np.where(mask[:, :, None] != 0, blue_layer, color)
        vis = draw_posed_3d_box(K, combined_image, ob_in_cam=pose, bbox=show_bbox, line_color=(255,255,0))
        cv2.imwrite(f'{out_folder}/pose/{cnt}.png', vis)
        #imageio.imwrite(f'{video_dir}/pose/{i}.png', vis)


if __name__ == '__main__':
    main()