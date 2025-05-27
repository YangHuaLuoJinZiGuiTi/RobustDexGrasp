# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

import matplotlib.pyplot as plt
import os
import numpy as np

class sam_predict:
    def __init__(self):
        print("Loading model...")
        sam = sam_model_registry['default'](checkpoint=os.path.join(os.path.dirname(__file__), 'sam_vit_h_4b8939.pth'))
        _ = sam.to(device="cuda")
        self.generator = SamPredictor(sam)
        self.log = False

    def calculate_mask(self, in_img, input_point, input_label, input_box = None):
    
        if self.log is True:
            cv2.circle(in_img, input_point[0], 10, (0,255,0), -1)
            cv2.imshow('Image with Circle', in_img)
            cv2.imwrite("/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/test/vlm_planning/test2.png", in_img)
        
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
                cv2.rectangle(in_img, (input_box[0], input_box[1]),  (input_box[2], input_box[3]), (0,0,255), 2)
                blue_layer = np.zeros_like(in_img)
                blue_layer[:] = (230, 216, 173)
                combined_image = np.where(mask[:, :, None] != 0, blue_layer, in_img)
                cv2.circle(combined_image, input_point[0], 10, (0,255,0), -1)
                cv2.imshow('Image with Circle 2', combined_image)
                cv2.imwrite("/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/test/vlm_planning/test3.png", combined_image)

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
