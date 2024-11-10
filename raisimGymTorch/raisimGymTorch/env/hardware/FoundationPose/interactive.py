import pyrealsense2 as rs
from .estimater import *
from .mask import *
import tkinter as tk
import time
import threading
from scipy.spatial.transform import Rotation as R

class FoundationData:
    def __init__(self, mesh_path):
        self.Tsim2realobj = None
        self.lock = threading.Lock()
        self.running = True
        self.mesh_path = mesh_path
        print("---------------- mesh_path = " + mesh_path)
        return

    def euler2rot(self, rx, ry, rz):
        return R.from_euler('xyz', [rx, ry, rz], degrees=True)

    def rot2euler(self, mat, degrees = True):
        r = R.from_matrix(mat)
        return r.as_euler('xyz', degrees=degrees)

    def rot2quar(self, mat):
        r = R.from_matrix(mat)
        return r.as_quat()

    def get_Tmat(self, x, y, z, rx, ry, rz):
        r = self.euler2rot(rx, ry, rz)
        Tmat = np.eye(4)
        Tmat[:3, :3] = r.as_matrix()
        Tmat[:3, 3] = [x, y, z]
        return Tmat
    
    def get_data(self):
        with self.lock:
            return self.Tsim2realobj
    
    def end_thread(self):
        self.running = False

    def start_thread(self):
        USE_ALLEGRO_HAND = True # FALSE MAY USE Inspire hand
        SHOW_IMAGE =  True
        SHOW_LOG = False
        est_refine_iter=4
        track_refine_iter=2

        set_logging_format(logging.WARNING)
        set_seed(0)

        root = tk.Tk()
        root.withdraw()

        mesh = trimesh.load(self.mesh_path)

        # 创建一个坐标系对象，size参数控制坐标轴的长度
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
        # 将trimesh mesh转换为open3d格式
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        # 计算顶点法线
        o3d_mesh.compute_vertex_normals()

        # 可视化
        #o3d.visualization.draw_geometries([o3d_mesh, coordinate_frame])

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        print(f'mesh to_origin xyz={to_origin[:3, 3].reshape((1, 3))},  rpy={self.rot2euler(to_origin[:3, :3])}')

        # from txt
        if USE_ALLEGRO_HAND == True:
            Tbase2cam = np.loadtxt("/home/ubuntu/hand/calculate/0_datasets_allegro_hand_2/base2cam.txt", delimiter=',')
        else:
            Tbase2cam = np.loadtxt("/home/ubuntu/hand/calculate/0_datasets_inspire_hand/base2cam.txt", delimiter=',')
        Tbase2cam[0][3] = Tbase2cam[0][3] * 0.001
        Tbase2cam[1][3] = Tbase2cam[1][3] * 0.001
        Tbase2cam[2][3] = Tbase2cam[2][3] * 0.001
        print(Tbase2cam)

        Treal2sim = np.array([[  0., 1., 0., 0.],
                            [-1., 0., 0., 0.],
                            [ 0., 0., 1., 0.],
                            [ 0., 0., 0., 1.]])

        Tsim2real = np.linalg.inv(Treal2sim)

        Tsimbase = np.array([[   1., 0., 0., 0.55],
                            [ 0., 1., 0., 0.75152],
                            [ 0., 0., 1., 0.771],
                            [ 0., 0., 0., 1.]])

        mask_file_path = create_mask()

        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,glctx=glctx,debug=0,debug_dir='/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/FoundationPose/debug')
        pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clipping_distance_in_meters = 1 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale
        align_to = rs.stream.color
        align = rs.align(align_to)

        i = 0

        mask = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)
        if USE_ALLEGRO_HAND == True:
            cam_K = np.array([[606.541260, 0.000000, 324.194061],[0.000000, 606.598267, 256.890228], [0., 0., 1.]]) # allegro_hand 640*480
        else:
            cam_K = np.array([[605.863, 0.000000, 314.97],[0.000000, 605.804, 254.686], [0., 0., 1.]]) # inspire_hand 640*480

        time.sleep(8)
        # Streaming loop

        test_xyzrpy = []
        start_time = time.time()
        first_flag = True
        try:
            while self.running:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not aligned_depth_frame or not color_frame:
                    continue
                depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
                color_image = np.asanyarray(color_frame.get_data())
                depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)
                if cv2.waitKey(1) == 13:
                    self.running = False
                    break        
                H, W = color_image.shape[:2]
                color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
                depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
                depth[(depth<0.1) | (depth>=np.inf)] = 0
                if i==0:
                    if len(mask.shape)==3:
                        for c in range(3):
                            if mask[...,c].sum()>0:
                                mask = mask[...,c]
                                break
                    mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
                
                    Tcam2obj = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
                else:
                    Tcam2obj = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=track_refine_iter)

                # camera坐标系下。obj的坐标。。
                Tbase2obj = Tbase2cam @ Tcam2obj 
                Tsim_base2obj = Tsim2real @ Tbase2obj
                
                with self.lock:
                    self.Tsim2realobj = Tsimbase @ Tsim_base2obj
                    
                if SHOW_IMAGE == True:
                    center_pose = Tcam2obj@np.linalg.inv(to_origin)
                    vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
                    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    cv2.imshow('1', vis)

                if SHOW_LOG == True:
                    center_pose = Tcam2obj@np.linalg.inv(to_origin)
                    test_xyzrpy.append(np.append(center_pose[:3, 3]*1000.0, self.rot2euler(center_pose[:3,:3])))
                    print(f'base2obj:{Tbase2obj[:3, 3].reshape((1, 3))}, cam2obj:{Tcam2obj[:3, 3].reshape((1, 3))}')
                    if i % 100 == 0:
                        end_time = time.time()
                        print(f'fps = {(end_time - start_time) / 100.0}')
                        start_time = end_time
                        nparray = np.array(test_xyzrpy)
                        print(f"x:  mean={np.mean(nparray[:, 0])},\t var={np.var(nparray[:, 0])},\t max={np.max(nparray[:, 0])},\t min={np.min(nparray[:, 0])}\t\t(mm)")
                        print(f"y:  mean={np.mean(nparray[:, 1])},\t var={np.var(nparray[:, 1])},\t max={np.max(nparray[:, 1])},\t min={np.min(nparray[:, 1])}\t\t(mm)")
                        print(f"z:  mean={np.mean(nparray[:, 2])},\t var={np.var(nparray[:, 2])},\t max={np.max(nparray[:, 2])},\t min={np.min(nparray[:, 2])}\t\t(mm)")
                        print(f"rx: mean={np.mean(nparray[:, 3])},\t var={np.var(nparray[:, 3])},\t max={np.max(nparray[:, 3])},\t min={np.min(nparray[:, 3])}\t\t(rad)")
                        print(f"ry: mean={np.mean(nparray[:, 4])},\t var={np.var(nparray[:, 4])},\t max={np.max(nparray[:, 4])},\t min={np.min(nparray[:, 4])}\t\t(rad)")
                        print(f"rz: mean={np.mean(nparray[:, 5])},\t var={np.var(nparray[:, 5])},\t max={np.max(nparray[:, 5])},\t min={np.min(nparray[:, 5])}\t\t(rad)")
                        test_xyzrpy.clear()

                i += 1
                cv2.waitKey(10)
                
        finally:
            pipeline.stop()