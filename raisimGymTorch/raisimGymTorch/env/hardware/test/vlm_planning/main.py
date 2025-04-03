import numpy as np
import cv2

from raisimGymTorch.env.hardware.realsense.PointCloud import Realsense
from raisimGymTorch.env.hardware.planning.vlm_planner import vlm_planner
from raisimGymTorch.env.hardware.sam.sam_predict import sam_predict

def main() -> None:
    rgbd = Realsense("/home/ubuntu/hand/calculate/0_datasets_allegro_hand_topview", 200, False)
    vlm = vlm_planner()
    sam = sam_predict()
    use_vlm = False

    img_pth = "/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/test/vlm_planning/test.png"
    
    while True:
        rgb_frame, depth_frame = rgbd.get_rgbd_frame()
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_pth, rgb_frame)

        if use_vlm is True:
            command = "clean the table"
            objlist = vlm.request_task("decompose_user_prompt", img_pth, command)
            for i in range(len(objlist)):
                bbox_2d = vlm.request_task("mark_bounding_box", img_pth, objlist[i])
                x1, y1, x2, y2 = bbox_2d['bbox_2d']

                input_point = np.array([[int((x1+x2)/2),int((y1+y2)/2)]])  # 为要分割的指定点
                input_label = np.array([1])  # 为分割对象的性质（背景|前景）
                input_box = np.array(bbox_2d['bbox_2d'])
        else:
            input_point = np.array(rgbd.get_point_from_image(rgb_frame))
            input_label = np.array([1])  # 为分割对象的性质（背景|前景）
            bbox = rgbd.get_point_from_image(rgb_frame)
            input_box = np.array([bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]])

        mask = sam.calculate_mask(rgb_frame, input_point, input_label, input_box)
        pc = rgbd.GetPointCloud(mask)

if __name__ == "__main__":
    main()
