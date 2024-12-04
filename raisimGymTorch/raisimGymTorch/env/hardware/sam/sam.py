# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import os
from typing import Any, Dict, List
import pyrealsense2 as rs
import numpy as np

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:

    os.makedirs(path, exist_ok=True)

    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return

def calculate_mask(in_img, generator = None, continue_flag = False):
    if continue_flag == False: 
        cv2.namedWindow("RGB")
        cv2.namedWindow("MASK")

    if generator is None:
        sam = sam_model_registry['default'](checkpoint=os.path.join(os.path.dirname(__file__), 'sam_vit_h_4b8939.pth'))
        _ = sam.to(device="cuda")
        generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

    masks = generator.generate(in_img)
    # method to find the best result (IOU , area% , size)
    H, W = in_img.shape[:2]
    
    # Invert the picture with white background
    area_list = []
    for i, mask_data in enumerate(masks):
        if mask_data["bbox"][2] > W - 40 or mask_data["bbox"][3] > H - 40:
            masks[i]["segmentation"] = ~mask_data["segmentation"]
            masks[i]["segmentation"][0:10, :] = False
            masks[i]["segmentation"][-10:, :] = False
            masks[i]["area"] =  np.count_nonzero(masks[i]["segmentation"])

        area_list.append(masks[i]["area"])        
    area_list = np.array(area_list)

    write_masks_to_folder(masks, os.path.join(os.path.dirname(__file__), 'out'))
    
    max_threshold = 0.
    max_index = 0.
    for i, mask_data in enumerate(masks):
        now_threshold = mask_data["predicted_iou"] + mask_data["area"] / np.max(area_list)
        if max_threshold < now_threshold:
            max_threshold = now_threshold
            max_index = i
    mask = np.array(masks[max_index]["segmentation"] * 255, dtype=np.uint8)

    cv2.imshow("RGB", in_img)
    cv2.imshow("MASK", mask)
    
    if continue_flag == False:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask

def main() -> None:
    print("Loading model...")
    sam = sam_model_registry['default'](checkpoint=os.path.join(os.path.dirname(__file__), 'sam_vit_h_4b8939.pth'))
    _ = sam.to(device="cuda")

    generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

    # realsense get depth
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    for i in range(200):
        frames = pipeline.wait_for_frames()

    cv2.namedWindow("RGB")
    cv2.namedWindow("MASK")

    while True:
        # Wait for a coherent pair of frames: depth and color    
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # Convert image to numpy array
        depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)
        H, W = color_image.shape[:2]
        color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.2) | (depth>=np.inf)] = 0

        calculate_mask(color_image, generator, True)
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter key
            break
        
        print("update once")

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
