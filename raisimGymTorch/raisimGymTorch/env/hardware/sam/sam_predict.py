# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

import matplotlib.pyplot as plt
import os
from typing import Any, Dict, List
import pyrealsense2 as rs
import numpy as np

class sam_predict:
    def __init__(self):
        print("Loading model...")
        sam = sam_model_registry['default'](checkpoint=os.path.join(os.path.dirname(__file__), 'sam_vit_h_4b8939.pth'))
        _ = sam.to(device="cuda")
        self.generator = SamPredictor(sam)
        self.log = True
    
    # 分割相关
    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    # 仅显示点相关   
    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    #仅显示框相关
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

    def calculate_mask(self, in_img, input_point, input_label, input_box = None):
        in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
    
        if self.log is True:
            plt.figure(figsize=(10,10))
            plt.imshow(in_img)
            self.show_points(input_point, input_label, plt.gca())
            plt.axis('on')
            plt.show() 
        
        self.generator.set_image(in_img)
        masks, scores, logits = self.generator.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :],
            multimask_output=False,
        )

        # method to find the best result (IOU , area% , size)
        if self.log is True:
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10,10))
                plt.imshow(in_img)
                self.show_mask(mask, plt.gca())
                self.show_points(input_point, input_label, plt.gca())
                self.show_box(input_box, plt.gca())
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                plt.show()

        return masks[0]

def main() -> None:
    sam = sam_predict()

    input_point = np.array([[470,185]])  # 为要分割的指定点
    input_label = np.array([1])  # 为分割对象的性质（背景|前景）
    #为单个框
    input_box = np.array([426, 124, 521, 226])

    #加载待处理图片
    image = cv2.imread('/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/test/vlm_planning/test.png')

    sam.calculate_mask(image, input_point, input_label, input_box)

if __name__ == "__main__":
    main()
