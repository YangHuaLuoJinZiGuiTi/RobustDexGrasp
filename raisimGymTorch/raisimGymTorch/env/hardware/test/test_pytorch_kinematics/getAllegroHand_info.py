import os
import argparse
import pytorch_kinematics as pk
from pytorch_kinematics import build_chain_from_urdf
import numpy as np
import math
import joblib
import transforms3d
import torch
# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

def calculate_PalmLinkOrientation_in_UR5baselinkFrame(handjoints,TCPpose,chain):
    # urdf_file = "/home/leech/raisim/dgrasp/rsc/mano/mano_mean.urdf"
    # base_link_name = "base_link"
    end_link_names = ["base_link",
        'link_1.0', 'link_2.0', 'link_3.0', 'link_3.0_tip',
        'link_5.0', 'link_6.0', 'link_7.0', 'link_7.0_tip',
        'link_9.0', 'link_10.0', 'link_11.0', 'link_11.0_tip',
        'link_13.0', 'link_14.0', 'link_15.0', 'link_15.0_tip']    
    joint_names = [
        'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 
        'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 
        'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 
        'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'
    ]
    wrist_joint_names = [
        'x_joint', 'y_joint', 'z_joint',
        'x_rotation_joint', 'y_rotation_joint', 'z_rotation_joint'
    ]     

    joint_angles = {}

    # chain = build_chain_from_urdf(open(urdf_file).read())

    rpy = rv2rpy(*(TCPpose[3:]))
    # print("------------------------")
    # print("TCPpose = ", TCPpose)
    # print("------------------------")
    # print('rpy-----------------------------')
    # print(rpy)
    # rot_transform3d = rv2rm(*(TCPpose[3:]))
    rot_transform3d = np.array(transforms3d.euler.euler2mat(*(rpy), axes='sxyz')).astype('float32')
    # rot_transform3d = np.array(transforms3d.euler.euler2mat(*(TCPpose[3:]), axes='sxyz')).astype('float32')
    euler_correct = transforms3d.euler.mat2euler(rot_transform3d, axes='rxyz')
    euler_correct = list(euler_correct)
    euler_correct[2] += 1.57
    # print("euler_correct = ", euler_correct)
    rot_transform3d_palmlink = np.array(transforms3d.euler.euler2mat(*(euler_correct), axes='rxyz')).astype('float32')

    return rot_transform3d_palmlink

# can‘t not deal with theta == 0, return rpy is sxyz or rxyz
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
    
    rpy = [0.0,0.0,0.0]
    rpy[0]= gamma
    rpy[1]= beta
    rpy[2]= alpha
    
    return rpy

# 旋转矢量转旋转矩阵
def rv2rm(rx, ry, rz):
    theta = np.linalg.norm([rx, ry, rz])
    kx = rx / theta
    ky = ry / theta
    kz = rz / theta

    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c

    R = np.zeros((3, 3))
    R[0][0] = kx * kx * v + c
    R[0][1] = kx * ky * v - kz * s
    R[0][2] = kx * kz * v + ky * s

    R[1][0] = ky * kx * v + kz * s
    R[1][1] = ky * ky * v + c
    R[1][2] = ky * kz * v - kx * s

    R[2][0] = kz * kx * v - ky * s
    R[2][1] = kz * ky * v + kx * s
    R[2][2] = kz * kz * v + c

    return R

def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler




# ly check: 来自UR5的TCPpose相对于机座坐标系的欧拉角是 rxyz 还是 sxyz, ans: 是sxyz
# 注意这个TCP pose需要是setTCP_offset之后的，因为allegro hand base link不在机械臂末端而在稍微上一点的地方

urdf_file = "/home/ubuntu/hand/dgrasp-main/rsc/mano/mano_mean_full_effort.urdf"
chain = build_chain_from_urdf(open(urdf_file).read())

def calculate_bodyPartPosition_in_baselinkFrame(handjoints,TCPpose,chain):
    # urdf_file = "/home/leech/raisim/dgrasp/rsc/mano/mano_mean.urdf"
    # base_link_name = "base_link"
    end_link_names = ["base_link",
        'link_1.0', 'link_2.0', 'link_3.0', 'link_3.0_tip',
        'link_5.0', 'link_6.0', 'link_7.0', 'link_7.0_tip',
        'link_9.0', 'link_10.0', 'link_11.0', 'link_11.0_tip',
        'link_13.0', 'link_14.0', 'link_15.0', 'link_15.0_tip']    
    joint_names = [
        'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 
        'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 
        'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 
        'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'
    ]
    wrist_joint_names = [
        'x_joint', 'y_joint', 'z_joint',
        'x_rotation_joint', 'y_rotation_joint', 'z_rotation_joint'
    ]     

    joint_angles = {}

    for name, joint_angle in zip(joint_names, handjoints):
        joint_angles[name] = joint_angle    

    # print("joint_angles = ", joint_angles)
    # origin
    # ret = chain.forward_kinematics(joint_angles)
    # ly fix
    joint_angles_torch = {k: torch.tensor(v) for k, v in joint_angles.items()}
    ret = chain.forward_kinematics(joint_angles_torch)

    joint_coords = []
    joint_eulers = []
    for end_link_name in end_link_names:
        th = ret[end_link_name]
        m = th.get_matrix()
        xyz = m[:, :3, 3]
        rot = pk.matrix_to_quaternion(m[:, :3, :3])
        xyz_transform = xyz.numpy().ravel()
        # ly test 1-17
        # xyz_transform = xyz_transform @ rot_transform3d + np.array(TCPpose[:3])   # 与 joint_angles[name] = 0 配套使用
        joint_coords.append(xyz_transform)
        joint_euler = np.array([mat2euler(mat) for mat in m[:, :3, :3].numpy()]).ravel()
        joint_eulers.append(joint_euler)
    # ly predict: 按道理应该是（17，3）
    # print("np.array(joint_coords).shape = ", np.array(joint_coords).shape)
    # print("np.array(joint_eulers).shape = ", np.array(joint_eulers).shape)
    return np.array(joint_coords), TCPpose

