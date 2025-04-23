# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

import time

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

from segment_anything import sam_model_registry, SamPredictor

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

class mask_predict:
    def __init__(self, log_dir=None):
        print("Loading model...")

        # sam init
        sam = sam_model_registry['default'](checkpoint='/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/sam/sam_vit_h_4b8939.pth')
        _ = sam.to(device="cuda")
        self.predictor = SamPredictor(sam)

        # cutie init
        cutie = get_default_model()
        self.processor = InferenceCore(cutie, cfg=cutie.cfg)
        self.processor.max_internal_size = -1

        print("Loading all finish...")
        # tmp value
        self.cutie_initialized = False

        # debug
        self.log = False
        if log_dir is not None:
            self.log = True
            self.log_dir = log_dir

    def sam_mask(self, in_img, input_point, input_label, input_box = None):
        if self.log is True:
            cv2.circle(in_img, input_point[0], 10, (0,255,0), -1)
            cv2.imwrite(self.log_dir+"origin_img.png", in_img)

        self.predictor.set_image(in_img)
        masks, scores, logits = self.predictor.predict(
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
                blue_layer[:] = (111, 166, 222)
                combined_image = np.where(mask[:, :, None] != 0, blue_layer, in_img)
                cv2.circle(combined_image, input_point[0], 10, (0,255,0), -1)
                cv2.imwrite(self.log_dir+"mask_img.png", combined_image)

        return masks[np.argmax(scores)]

    def reset_all(self, in_img, input_point, input_label, input_box = None):
        self.processor.clear_memory()
        torch.cuda.empty_cache()
        sam_mask_result = self.sam_mask(in_img, input_point, input_label, input_box)

        # Reinitialize Cutie
        self.mask = torch.from_numpy(sam_mask_result.astype('uint8')).cuda()
        self.objects = np.unique(sam_mask_result.astype('uint8'))
        self.objects = self.objects[self.objects != 0].tolist()
        self.cutie_initialized = False  # Reset Cutie initialization flag
        
        return sam_mask_result.reshape(480, 640)

    def tack_obj(self, in_img, cnt = 0):
        with torch.no_grad():
            image_tensor = to_tensor(in_img).cuda().float()
            if self.cutie_initialized == False:
                output_prob = self.processor.step(image_tensor, self.mask, objects=self.objects)
                self.cutie_initialized = True
            else:
                output_prob = self.processor.step(image_tensor)
            current_mask = self.processor.output_prob_to_mask(output_prob)
            current_mask_np = current_mask.cpu().numpy().astype(np.uint8)
            current_mask_np = current_mask_np.reshape(480, 640)
        
        if self.log is True:
            blue_layer = np.zeros_like(in_img)
            blue_layer[:] = (111, 166, 222)
            combined_image = np.where(current_mask_np[:, :, None] != 0, blue_layer, in_img)
            cv2.imwrite(f"{self.log_dir}mask_img{cnt}.png", combined_image)
        return current_mask_np
          
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

def main() -> None:
  
    log_dir = '/home/ubuntu/hand/foundationpose/realsense/output/realsense/hand_move_midobject_move/'
    mask_demo = mask_predict(log_dir)

    #加载待处理图片
    i = 0
    while True:
        image = cv2.imread(f'{log_dir}rgb/{i:04d}.png')
        if i == 0:
            input_point = np.array(mask_demo.get_point_from_image(image))
            input_label = np.array([1])  # 为分割对象的性质（背景|前景）
            bbox = mask_demo.get_point_from_image(image)
            input_box = np.array([bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]])
            mask_demo.reset_all(image, input_point, input_label, input_box)
        else:
            mask_demo.tack_obj(image, i)

        i = i + 2
        if i > 300:
            break

if __name__ == "__main__":
    main()
