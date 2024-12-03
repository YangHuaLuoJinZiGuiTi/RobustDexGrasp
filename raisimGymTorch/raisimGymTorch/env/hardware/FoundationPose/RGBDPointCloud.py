import pyrealsense2 as rs
import cv2
import numpy as np
import time
from scipy.spatial import KDTree

from ..sam.sam import calculate_mask

# KD tree to calculate K-Nearest Neighbors for each point
def remove_outliers(point_cloud, k=5, threshold=1.5):
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

def depth2xyzmap(depth, K, uvs=None):
    
    invalid_mask = (depth<0.1)
    H,W = depth.shape[:2]
    if uvs is None:
        vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:,0]
        vs = uvs[:,1]
    zs = depth[vs,us]
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H,W,3), dtype=np.float32)
    xyz_map[vs,us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map

def get_obj_point_cloud(K, depth, ob_mask, s = 200, valid = None, indices = None):
    xyz_map = depth2xyzmap(depth, K)
    if valid is None:
        valid = (xyz_map[...,2]>=0.1) & (ob_mask>0)
    pc = xyz_map[valid]
    if indices is None:
        indices = np.random.choice(pc.shape[0], size=s, replace=False)
    return pc[indices], valid, indices

def create_mask_auto(in_img):
    return calculate_mask(in_img)

def create_mask_from_align_sensor(color_frame):
    points = []
    mask_path = './mask.png'

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image_display, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("Image", image_display)

    def generate_mask(image, points):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        return mask

    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
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

    mask = generate_mask(image, points)

    # Save the mask image
    cv2.imwrite(mask_path, mask)
    print("------------------write masks success !!!!!!!!!")
    cv2.destroyAllWindows()

    return mask

def create_mask_from_rgb_sensor():
    points = []
    mask_path = './mask.png'

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image_display, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("Image", image_display)

    def generate_mask(image, points):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        return mask

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # Wait for 1 second to allow the camera to warm up
        time.sleep(1)
        # Wait for a coherent pair of frames: depth and color    
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise Exception("Could not capture color frame")

        # Convert image to numpy array
        image = np.asanyarray(color_frame.get_data())
        image_display = image.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", select_points)

        print("Click on the image to select points. Press Enter when done.")

        while True:
            cv2.imshow("Image", image_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break

        mask = generate_mask(image, points)

        # Save the mask image
        cv2.imwrite(mask_path, mask)
        print("------------------write masks success !!!!!!!!!")
        cv2.destroyAllWindows()

        return mask_path

    finally:
        # Stop streaming
        pipeline.stop()

def GetPointCloud(camK_path):
    
    # tf matrix
    Tbase2cam = np.loadtxt(camK_path + "/base2cam.txt", delimiter=',')
    Treal2sim = np.array([[  0., 1., 0., 0.],
                        [-1., 0., 0., 0.],
                        [ 0., 0., 1., 0.],
                        [ 0., 0., 0., 1.]])
    Tsim2real = np.linalg.inv(Treal2sim)
    Tsimbase = np.array([[   1., 0., 0., 0.],
                        [ 0., 1., 0., 0.],
                        [ 0., 0., 1., 0.771],
                        [ 0., 0., 0., 1.]])
    
    # https://support.intelrealsense.com/hc/en-us/community/posts/4405875311123-About-make-sure-FOV-specification-of-D435i 
    # tf from RGB to left-IR camera

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

    cam_K = np.loadtxt(camK_path + "/camK_640x480.txt", delimiter=',')

    wait_cnt = 0
    mask = None
    check_indice = None
    valid = None
    check_array = []
    while True:
        wait_cnt += 1
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame or wait_cnt < 200:
            continue
        depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)

        H, W = color_image.shape[:2]
        color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.1) | (depth>=np.inf)] = 0

        # realsense get rgb mask
        if mask is None:
            #mask = create_mask_from_align_sensor(color)
            mask = create_mask_auto(color)

        if len(mask.shape)==3:
            for c in range(3):
                if mask[...,c].sum()>0:
                    mask = mask[...,c]
                    break
        mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        
        mask_pcd, valid, check_indice = get_obj_point_cloud(cam_K, depth, mask, 400, valid, check_indice)
        check_array.append(mask_pcd)
        if wait_cnt > 225:
            check_array = np.array(check_array) # (502, 200, 3)
            mask_pcd = np.mean(check_array, axis=0) # (200, 3)
        else:
            continue

        filtered_point_cloud = remove_outliers(mask_pcd, k=15, threshold=3.0)
        selected_indices = np.random.choice(filtered_point_cloud.shape[0], 200, replace=False)
        mask_pcd_new = filtered_point_cloud[selected_indices]

        mask_pcd_homogeneous = np.hstack((mask_pcd_new, np.ones((mask_pcd_new.shape[0], 1))))  # (200, 4)
        print (f"camera frame pc = {mask_pcd_homogeneous[0]}")
        tbase2pcd = mask_pcd_homogeneous @ Tbase2cam.T
        print (f"realbase frame pc = {tbase2pcd[0]}")
        tsim_base2pc = tbase2pcd @ Tsim2real.T
        print (f"simbase frame pc = {tsim_base2pc[0]}")
        tsim2realpc = tsim_base2pc @ Tsimbase.T
        print (f"simworld frame pc = {tsim2realpc[0]}")
        
        break

    pipeline.stop()
    return tsim2realpc[:, :3]

