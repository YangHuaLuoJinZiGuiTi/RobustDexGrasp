# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from raisimGymTorch.env.hardware.BundleTrack.Utils import *
from raisimGymTorch.env.hardware.BundleTrack.tool import *
from raisimGymTorch.env.hardware.BundleTrack.BundleTrack.build import my_cpp
from raisimGymTorch.env.hardware.BundleTrack.BundleTrack.scripts.data_reader import *
from raisimGymTorch.env.hardware.BundleTrack.Utils import *
from raisimGymTorch.env.hardware.BundleTrack.loftr_wrapper import LoftrRunner


class BundleTrackAPI:
    def __init__(self, cfg_track_dir=None, translation=None, sc_factor=None):
        with open(cfg_track_dir,'r') as ff:
            self.cfg_track = yaml.load(ff)
        self.debug_dir = self.cfg_track["debug_dir"]
        self.SPDLOG = self.cfg_track["SPDLOG"]
        self.translation = None
        self.sc_factor = None
        if sc_factor is not None:
            self.translation = translation
            self.sc_factor = sc_factor

        yml = my_cpp.YamlLoadFile(cfg_track_dir)
        self.bundler = my_cpp.Bundler(yml)
        self.loftr = LoftrRunner()
        self.cnt = -1
        self.K = None

    def make_frame(self, color, depth, K, id_str, mask=None, occ_mask=None, pose_in_model=np.eye(4)):
        H,W = color.shape[:2]
        roi = [0,W-1,0,H-1]
        frame = my_cpp.Frame(color,depth,roi,pose_in_model,self.cnt,id_str,K,self.bundler.yml)
        if mask is not None:
            frame._fg_mask = my_cpp.cvMat(mask)
        if occ_mask is not None:
            frame._occ_mask = my_cpp.cvMat(occ_mask)
        return frame


    def find_corres(self, frame_pairs):
        is_match_ref = len(frame_pairs)==1 and frame_pairs[0][0]._ref_frame_id==frame_pairs[0][1]._id and self.bundler._newframe==frame_pairs[0][0]

        imgs, tfs, query_pairs = self.bundler._fm.getProcessedImagePairs(frame_pairs)
        imgs = np.array([np.array(img) for img in imgs])

        if len(query_pairs)==0:
            return

        corres = self.loftr.predict(rgbAs=imgs[::2], rgbBs=imgs[1::2])
        for i_pair in range(len(query_pairs)):
            cur_corres = corres[i_pair][:,:4]
            tfA = np.array(tfs[i_pair*2])
            tfB = np.array(tfs[i_pair*2+1])
            cur_corres[:,:2] = transform_pts(cur_corres[:,:2], np.linalg.inv(tfA))
            cur_corres[:,2:4] = transform_pts(cur_corres[:,2:4], np.linalg.inv(tfB))
            self.bundler._fm._raw_matches[query_pairs[i_pair]] = cur_corres.round().astype(np.uint16)

        min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]

        if is_match_ref and len(self.bundler._fm._raw_matches[frame_pairs[0]])<min_match_with_ref:
            self.bundler._fm._raw_matches[frame_pairs[0]] = []
            self.bundler._newframe._status = my_cpp.Frame.FAIL
            logging.info(f'frame {self.bundler._newframe._id_str} mark FAIL, due to no matching')
            return

        self.bundler._fm.rawMatchesToCorres(query_pairs)

        for pair in query_pairs:
            self.bundler._fm.vizCorresBetween(pair[0], pair[1], 'before_ransac')

        self.bundler._fm.runRansacMultiPairGPU(query_pairs)

        for pair in query_pairs:
            self.bundler._fm.vizCorresBetween(pair[0], pair[1], 'after_ransac')



    def process_new_frame(self, frame):
        self.bundler._newframe = frame
        os.makedirs(self.debug_dir, exist_ok=True)

        if frame._id>0:
            ref_frame = self.bundler._frames[list(self.bundler._frames.keys())[-1]]
            frame._ref_frame_id = ref_frame._id
            frame._pose_in_model = ref_frame._pose_in_model
        else:
            self.bundler._firstframe = frame

        frame.invalidatePixelsByMask(frame._fg_mask)
        if frame._id==0 and np.abs(np.array(frame._pose_in_model)-np.eye(4)).max()<=1e-4:
            frame.setNewInitCoordinate()


        n_fg = (np.array(frame._fg_mask)>0).sum()
        if n_fg<100:
            logging.info(f"Frame {frame._id_str} cloud is empty, marked FAIL, roi={n_fg}")
            frame._status = my_cpp.Frame.FAIL;
            self.bundler.forgetFrame(frame)
            return

        if self.cfg_track["depth_processing"]["denoise_cloud"]:
            frame.pointCloudDenoise()

        n_valid = frame.countValidPoints()
        n_valid_first = self.bundler._firstframe.countValidPoints()
        if n_valid<n_valid_first/40.0:
            logging.info(f"frame _cloud_down points#: {n_valid} too small compared to first frame points# {n_valid_first}, mark as FAIL")
            frame._status = my_cpp.Frame.FAIL
            self.bundler.forgetFrame(frame)
            return

        if frame._id==0:
            self.bundler.checkAndAddKeyframe(frame)   # First frame is always keyframe
            self.bundler._frames[frame._id] = frame
            return

        min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]

        self.find_corres([(frame, ref_frame)])
        matches = self.bundler._fm._matches[(frame, ref_frame)]

        if frame._status==my_cpp.Frame.FAIL:
            logging.info(f"find corres fail, mark {frame._id_str} as FAIL")
            self.bundler.forgetFrame(frame)
            return

        matches = self.bundler._fm._matches[(frame, ref_frame)]
        if len(matches)<min_match_with_ref:
            visibles = []
            for kf in self.bundler._keyframes:
                visible = my_cpp.computeCovisibility(frame, kf)
                visibles.append(visible)
            visibles = np.array(visibles)
            ids = np.argsort(visibles)[::-1]
            found = False
            #pdb.set_trace()
            for id in ids:
                kf = self.bundler._keyframes[id]
                ref_frame = kf
                frame._ref_frame_id = kf._id
                frame._pose_in_model = kf._pose_in_model
                self.find_corres([(frame, ref_frame)])

                # self.bundler._fm.findCorres(frame, ref_frame)

                if len(self.bundler._fm._matches[(frame,kf)])>=min_match_with_ref:
                    found = True
                    break

            if not found:
                frame._status = my_cpp.Frame.FAIL
                logging.info(f"frame {frame._id_str} has not suitable ref_frame, mark as FAIL")
                self.bundler.forgetFrame(frame)
                return

        offset = self.bundler._fm.procrustesByCorrespondence(frame, ref_frame)
        frame._pose_in_model = offset@frame._pose_in_model

        window_size = self.cfg_track["bundle"]["window_size"]
        if len(self.bundler._frames)-len(self.bundler._keyframes)>window_size:
            for k in self.bundler._frames:
                f = self.bundler._frames[k]
                isforget = self.bundler.forgetFrame(f)
                if isforget:
                    logging.info(f"exceed window size, forget frame {f._id_str}")
                    break

        self.bundler._frames[frame._id] = frame

        self.bundler.selectKeyFramesForBA()

        local_frames = self.bundler._local_frames

        pairs = self.bundler.getFeatureMatchPairs(self.bundler._local_frames)
        self.find_corres(pairs)
        if frame._status==my_cpp.Frame.FAIL:
            self.bundler.forgetFrame(frame)
            return

        find_matches = False
        self.bundler.optimizeGPU(local_frames, find_matches)

        if frame._status==my_cpp.Frame.FAIL:
            self.bundler.forgetFrame(frame)
            return

        self.bundler.checkAndAddKeyframe(frame)

    def run(self, color, depth, K, id_str, mask=None, occ_mask=None, pose_in_model=np.eye(4)):
        self.cnt += 1

        if self.K is None:
            self.K = K

        # 深度去噪，去掉百分之多少的极大极小值？
        percentile = self.cfg_track['depth_processing']["percentile"]
        if percentile<100:   # Denoise
            valid = (depth>=0.1) & (mask>0)
            valid = np.any(valid) 
            if valid == False:
                return np.eye(4)
            thres = np.percentile(depth[valid], percentile)
            depth[depth>=thres] = 0

        # 构造一个frame
        frame = self.make_frame(color, depth, K, id_str, mask, occ_mask, pose_in_model)

        # 处理一个frame
        #self.bundler.processNewFrame(frame)
        self.process_new_frame(frame)
        
        return np.linalg.inv(frame._pose_in_model)


    def pixel_to_world(self, pixel_coords, depth, K, T_1):
        # 将像素坐标转换到归一化相机坐标
        pixel_coords_homogeneous = np.array([pixel_coords[0][0], pixel_coords[0][1], 1])
        camera_coords = np.linalg.inv(K) @ pixel_coords_homogeneous
        camera_coords *= depth  # 使用深度信息
        
        # 将相机坐标转换到世界坐标
        camera_coords_homogeneous = np.append(camera_coords, 1)
        world_coords = np.linalg.inv(T_1) @ camera_coords_homogeneous
        
        return world_coords[:3]

    def world_to_pixel(self, world_coords, K, T_i):
        # 将世界坐标转换到当前帧的相机坐标
        world_coords_homogeneous = np.append(world_coords, 1)
        camera_coords = T_i @ world_coords_homogeneous
        
        # 将相机坐标投影到像素坐标
        pixel_coords_homogeneous = K @ camera_coords[:3]
        pixel_coords = pixel_coords_homogeneous[:2] / pixel_coords_homogeneous[2]
        
        return pixel_coords