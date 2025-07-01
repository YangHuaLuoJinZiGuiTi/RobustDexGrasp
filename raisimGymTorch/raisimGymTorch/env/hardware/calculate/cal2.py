#https://blog.csdn.net/weixin_43134049/article/details/122826538 

import cv2
import numpy as np
import math
import csv
import os
import time

# camera1
K= None
DISTCOEFF = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


CHESS_BOARD_X_NUM=11#棋盘格x方向格子数
CHESS_BOARD_Y_NUM=8#棋盘格y方向格子数
CHESS_BOARD_LEN=0.02#单位棋盘格长度,m
IMG_NUM=0
PATH='/home/ubuntu/hand/calculate/0_datasets_allegro_hand/'

def rv2rpy(rx,ry,rz):  
    theta = math.sqrt(rx*rx + ry*ry + rz*rz)
    kx = rx/theta
    ky = ry/theta
    kz = rz/theta
    cth = math.cos(theta)
    sth = math.sin(theta)
    vth = 1-math.cos(theta)
    
    r11 = kx*kx*vth + cth
    r12 = kx*ky*vth - kz*sth
    r13 = kx*kz*vth + ky*sth
    r21 = kx*ky*vth + kz*sth
    r22 = ky*ky*vth + cth
    r23 = ky*kz*vth - kx*sth
    r31 = kx*kz*vth - ky*sth
    r32 = ky*kz*vth + kx*sth
    r33 = kz*kz*vth + cth
    
    beta = math.atan2(-r31,math.sqrt(r11*r11+r21*r21))
    
    if beta > math.radians(89.99):
        beta = math.radians(89.99)
        alpha = 0
        gamma = math.atan2(r12,r22)
    elif beta < -math.radians(89.99):
        beta = -math.radians(89.99)
        alpha = 0
        gamma = -math.atan2(r12,r22)
    else:
        cb = math.cos(beta)
        alpha = math.atan2(r21/cb,r11/cb)
        gamma = math.atan2(r32/cb,r33/cb)

    return gamma, beta, alpha

def show_axis(img, rvecs, tvecs, camera_matrix, distortion_coefficients):
    # 假设我们想在(0,0,0)的点显示坐标系
    axis = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, distortion_coefficients)
    start_p = tuple(imgpts[0][0].astype(int))
    endx = tuple(imgpts[1][0].astype(int))
    endy = tuple(imgpts[2][0].astype(int))
    endz = tuple(imgpts[3][0].astype(int))
    img = cv2.line(img, start_p, endx, (255,0,0), 2)
    img = cv2.line(img, start_p, endy, (0,255,0), 2)
    img = cv2.line(img, start_p, endz, (0,0,255), 2)

from scipy.spatial.transform import Rotation as R
def rotation_vector2rot(rx, ry, rz):
    rotation_vector = np.array([rx, ry, rz])
    return R.from_rotvec(rotation_vector)

def rot2euler(mat):
    r = R.from_matrix(mat)
    return r.as_euler('xyz', degrees=True)

# Ta2b =  R=Ra2b, t=a坐标系原点在b坐标系的坐标
def get_Tmat(x, y, z, rx, ry, rz):
    r = rotation_vector2rot(rx, ry, rz)

    Tmat = np.eye(4)
    Tmat[:3, :3] = r.as_matrix()
    Tmat[:3, 3] = [x, y, z]
    return Tmat

def inv_Tmat(T):
    r = R.from_matrix(T[:3, :3])
    inv_R = r.inv()
    inv_T = np.eye(4)
    inv_T[:3, :3] = inv_R.as_matrix()
    inv_T[:3, 3] = -np.dot(inv_T[:3, :3], T[:3, 3])
    return inv_T
    #return np.linalg.inv(T) # the same

#用来从棋盘格图片得到相机外参
def get_RT_from_chessboard(img_path, debug:bool=False):
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 定义标定板大小和格子尺寸,单位为mm
    board_size = (CHESS_BOARD_X_NUM, CHESS_BOARD_Y_NUM)  # 标定板内角点数量
    square_size = (CHESS_BOARD_LEN, CHESS_BOARD_LEN)  # 标定板上每个格子的尺寸

    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if corners is None:
        print(f"------------- {img_path} cannot find corner, ret = {ret}")
        exit(0)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    ##### show image for debug ######
    if debug == True:
        cv2.drawChessboardCorners(img, board_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    
    corner_num = CHESS_BOARD_X_NUM * CHESS_BOARD_Y_NUM
    
    objp = np.zeros((corner_num, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
    #print(objp)

    # rvec/tvec: world frame表示的3D点转换到camera frame
    ret,rvec,tvec,inliers = cv2.solvePnPRansac(objp, corners, K, DISTCOEFF)
    if ret == None:
        print("error check board")
        exit(0)

    test = get_Tmat(tvec[0,0],tvec[1,0],tvec[2,0],rvec[0,0],rvec[1,0],rvec[2,0])
    return test

def get_K_from_chessboard():

    object_points = []
    image_points = []

    # 定义标定板大小和格子尺寸,单位为mm
    board_size = (CHESS_BOARD_X_NUM, CHESS_BOARD_Y_NUM)  # 标定板内角点数量
    square_size = CHESS_BOARD_LEN  # 标定板上每个格子的尺寸
    
    pattern_points = np.zeros((np.prod(board_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(board_size).T.reshape(-1, 2)
    pattern_points *= square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #print(objp)
    for i in range(IMG_NUM):
        img=cv2.imread(PATH+str(i)+'_Color.png')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret is None or corners is None:
            print("------------- error ret")
            continue
        #corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        if len(corners) == np.prod(board_size):
            object_points.append(pattern_points)
            image_points.append(corners)
        else:
            print("error len")
            exit(0)

    ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], K, DISTCOEFF, criteria=criteria)
    
    # Calculate reprojection errors
    mean_error = 0
    for i in range(len(object_points)): # 12
        image_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, DISTCOEFF)
        error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2) / len(image_points2)
        print(f'{i}th error = {error}')
        mean_error += error
    mean_error /= len(object_points)


    for i in range(IMG_NUM):
        img=cv2.imread(PATH+str(i)+'_Color.png')
        point_2d, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, DISTCOEFF)

        for j in range(CHESS_BOARD_X_NUM * CHESS_BOARD_Y_NUM):
            cv2.circle(img, tuple(point_2d[j][0].astype(int)), 2, (0, 0, 255), -1)
        
        show_axis(img, rvecs[i], tvecs[i], K, DISTCOEFF)
        cv2.imshow(f'{i}th img' ,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return

def calculate_all():
    # world frame表示的3D点转换到camera frame
    T_chess2cam=[]
    R_chess2cam=[]
    t_chess2cam=[]
    for i in range(IMG_NUM):
        T_tmp = get_RT_from_chessboard(PATH+str(i)+'_Color.png', False)
        T_chess2cam.append(T_tmp)
        R_chess2cam.append(T_tmp[:3, :3])
        t_chess2cam.append(T_tmp[:3, 3].reshape((3, 1)))
        #print(f'{i}th: chess to cam: xyz={T_tmp[:3, 3].reshape((1, 3))},  rpy={rot2euler(T_tmp[:3, :3])}')

    # 计算end to base变换矩阵
    # 机械臂base坐标系，从后往前看这个机械臂，x向后，y向右，z向上
    # 机械臂eef坐标系，从后往前看这个eef，x向右，y向下，z向外
    T_base2eef=[]
    R_base2eef=[]
    t_base2eef=[]
    with open(PATH+'xyzrotvec.csv', mode='r', encoding='utf-8') as f:
        data = csv.reader(f)
        for row in data:
            data_list = [float(value) for value in row]
            T_tmp = get_Tmat(data_list[0],data_list[1],data_list[2],data_list[3],data_list[4],data_list[5])
            T_tmp_inv = inv_Tmat(T_tmp)
            T_base2eef.append(T_tmp_inv)
            R_base2eef.append(T_tmp_inv[:3, :3])
            t_base2eef.append(T_tmp_inv[:3, 3].reshape((3, 1)))

    R_cam2base, t_cam2base=cv2.calibrateHandEye(R_base2eef, t_base2eef, R_chess2cam, t_chess2cam)#手眼标定，手在眼外的情况
    T_cam2base = np.row_stack((np.column_stack((R_cam2base, t_cam2base)), np.array([0, 0, 0, 1])))

    # for evaluate
    chess2eef_xyzrpy = []
    matrices = []
    for i in range(IMG_NUM):
        T_chess2eef = T_base2eef[i] @ T_cam2base @ T_chess2cam[i]
        matrices.append(T_chess2eef)
        chess2eef_xyzrpy.append(np.append(T_chess2eef[:3, 3], rot2euler(T_chess2eef[:3,:3])))
        #print(f'{i}th evaluate: xyz={T_chess2eef[:3, 3]}, rpy={rot2euler(T_chess2eef[:3,:3])})')
    nparray = np.array(chess2eef_xyzrpy)

    print(f"x:  mean={np.mean(nparray[:, 0])},\t var={np.var(nparray[:, 0])},\t max={np.max(nparray[:, 0])},\t min={np.min(nparray[:, 0])}\t\t(m)")
    print(f"y:  mean={np.mean(nparray[:, 1])},\t var={np.var(nparray[:, 1])},\t max={np.max(nparray[:, 1])},\t min={np.min(nparray[:, 1])}\t\t(m)")
    print(f"z:  mean={np.mean(nparray[:, 2])},\t var={np.var(nparray[:, 2])},\t max={np.max(nparray[:, 2])},\t min={np.min(nparray[:, 2])}\t\t(m)")

    # check the frobenius similarities
    sim_check_array = []
    for i in range(IMG_NUM):
        for j in range(IMG_NUM):
            if i < j:
                sim_check = np.linalg.norm(matrices[i] - matrices[j])
                sim_check_array.append(sim_check)
                if sim_check > 0.15:
                    print(f"frobenius similarities of {i} and {j} is too large = {sim_check}")

    sim_check_array = np.array(sim_check_array)
    print(f"matrix:  mean={np.mean(sim_check_array)},\t var={np.var(sim_check_array)},\t max={np.max(sim_check_array)},\t min={np.min(sim_check_array)}")
    

    #print('base to camera:')
    #print(T_cam2base)
    #print(f'base2cam: xyz={T_cam2base[:3, 3]}, rpy={rot2euler(T_cam2base[:3,:3])})')

    #print('camera to base:')
    #T_base2cam = inv_Tmat(T_cam2base)
    #print(T_base2cam)
    #print(f'cam2base: xyz={T_base2cam[:3, 3]}, rpy={rot2euler(T_base2cam[:3,:3])})')
    
    np.savetxt(PATH+'base2cam.txt', T_cam2base, delimiter=',')   # X is an array
    

def check_one_img(check_num):

    object_points = []
    image_points = []

    # 定义标定板大小和格子尺寸,单位为mm
    board_size = (CHESS_BOARD_X_NUM, CHESS_BOARD_Y_NUM)  # 标定板内角点数量
    square_size = CHESS_BOARD_LEN  # 标定板上每个格子的尺寸
    
    pattern_points = np.zeros((np.prod(board_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(board_size).T.reshape(-1, 2)
    pattern_points *= square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img=cv2.imread(PATH+str(check_num)+'_Color.png')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if ret is None or corners is None:
        print("------------- error ret no corner find !!!!!")
        exit(0)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    if len(corners) == np.prod(board_size):
        object_points.append(pattern_points)
        image_points.append(corners)
    else:
        print(f"error len ::: corner={len(corners)} is not the same board_size{np.prod(board_size)}")
        cv2.drawChessboardCorners(img, board_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        exit(0)

    ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], K, DISTCOEFF, criteria=criteria)

    # Calculate reprojection errors
    mean_error = 0
    image_points2, _ = cv2.projectPoints(object_points[0], rvecs[0], tvecs[0], K, DISTCOEFF)
    error = cv2.norm(image_points[0], image_points2, cv2.NORM_L2) / len(image_points2)
    print(f' error = {error}')
    mean_error += error
    mean_error /= len(object_points)
    

    img=cv2.imread(PATH+str(check_num)+'_Color.png')
    point_2d, _ = cv2.projectPoints(object_points[0], rvecs[0], tvecs[0], K, DISTCOEFF)

    for j in range(CHESS_BOARD_X_NUM * CHESS_BOARD_Y_NUM):
        cv2.circle(img, tuple(point_2d[j][0].astype(int)), 2, (0, 0, 255), -1)
    
    #show_axis(img, rvecs[0], tvecs[0], camera_matrix, distortion_coefficients)
    #cv2.imwrite('check.png' ,img)
    return

def auto_save_and_check_one_img(check_num):
    from rtde_receive import RTDEReceiveInterface as RTDEReceive

    rtde_frequency = 500.0
    ur_ip = "192.168.56.101"

    rtde_r = RTDEReceive(ur_ip, rtde_frequency)
    # Actual Cartesian coordinates of the tool: (x,y,z,rx,ry,rz), where rx, ry and rz is a rotation vector representation of the tool orientation
    actuaTCPpose = rtde_r.getActualTCPPose()

    rows = []
    with open(PATH + 'xyzrotvec.csv', mode='r', encoding='utf-8') as f:
        csv_file = csv.reader(f)
        for row in csv_file:
            rows.append(row)
        if check_num < len(rows):
            rows[check_num] = actuaTCPpose
        else:
            while len(rows) < check_num:
                rows.append([])
            rows.append(actuaTCPpose)

    with open(PATH + 'xyzrotvec.csv', mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        
    check_one_img(check_num)

if __name__ == '__main__':
    # get the camera param
    K = np.loadtxt(PATH+"camK_1920x1080.txt", delimiter=',')

    # get the number of png file
    files = os.listdir(PATH)
    png_files = [file for file in files if file.lower().endswith('.png')]
    IMG_NUM = len(png_files)

    #auto_save_and_check_one_img(14)
    calculate_all()#
    #get_K_from_chessboard()
