# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
import cv2
import imageio
import trimesh
import os
import copy
import math


def draw_registration_result(src, tat, coarse_trans, final_trans):
    """
    可视化配准结果，并返回变换后的点云和坐标轴
    """
    ori_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    trans_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    src_coarse = copy.deepcopy(src)
    src_temp = copy.deepcopy(src)
    tat_temp = copy.deepcopy(tat)
    src_coarse.paint_uniform_color([0, 1, 0])
    src_temp.paint_uniform_color([1, 0, 0])
    tat_temp.paint_uniform_color([0, 0.6, 0.9])
    src_coarse.transform(coarse_trans)
    src_temp.transform(final_trans)
    trans_axis.transform(final_trans)
    o3d.visualization.draw_geometries([src_temp, src_coarse, tat_temp, trans_axis])
    combined_pc = src_temp + tat_temp
    return combined_pc, trans_axis


def rotation_base_selection(candidates, align_axis='z'):
    #align_axis:跟世界坐标z轴对齐的零件坐标轴
    assert (align_axis=='x' or align_axis=='y' or align_axis=='z')
    best_idx = -1
    best_score = -1
    score = []
    # ---------- 1、计算 ICP 匹配得分
    correspondence = []
    for i in range(len(candidates['icp_result'])):
        correspondence.append(np.asarray(candidates['icp_result'][i].correspondence_set).shape[0]) # correspondence_set 代表 ICP 计算出的对应点数量，即点云匹配质量的一个指标。
    
    print(f"correspondence: {correspondence}")
    # ---------- 2、选择最佳对齐的目标：选取 对齐程度最高的目标。
    if (len(candidates)>2):
        sort_fitness_idx = sorted(range(len(candidates['icp_result'])), key=lambda x: correspondence[x], reverse=True)
        #pick top 3
        for i in sort_fitness_idx[:3]:
            if (align_axis=='x'):
                score = abs(candidates['icp_result'][i].transformation[2,0])  # 表示目标在 z 轴上的 x 旋转量
            elif (align_axis=='y'):
                score = abs(candidates['icp_result'][i].transformation[2,1])  # 表示目标在 z 轴上的 y 旋转量
            elif (align_axis=='z'):
                score = abs(candidates['icp_result'][i].transformation[2,2])  # 表示目标在 z 轴上的 z 旋转量
            if(score > best_score):
                best_score = score
                best_idx = i
    # --------- 3、处理两个候选目标的情况：如果只有 2 个候选目标，直接比较 旋转对齐程度 选择最佳的。
    else:
        assert (len(candidates['icp_result'])==2)
        if (align_axis=='x'):
                score0 = abs(candidates['icp_result'][0].transformation[2,0])
                score1 = abs(candidates['icp_result'][1].transformation[2,0])
        elif (align_axis=='y'):
                score0 = abs(candidates['icp_result'][0].transformation[2,1])
                score0 = abs(candidates['icp_result'][1].transformation[2,1])
        elif (align_axis=='z'):
                score0 = abs(candidates['icp_result'][0].transformation[2,2])
                score1 = abs(candidates['icp_result'][1].transformation[2,2])
        if (score0 > score1):
            best_score = score0
            best_idx = 0
        else:
            best_score = score1
            best_idx = 1
    # debug
    # add bias
    # --------- 4、添加变换偏移（修正位姿）
    # 加上额外的平移和旋转：平移修正 translate_0matrix (0.015m 在 x 轴, 0.04m 在 y 轴)
    # 来源: 通常是来自于实际的硬件调试，例如通过测试得出一些小的修正值。平移矩阵，表示在 x 轴上平移 0.015m 和 y 轴上平移 0.04m。
    translate_0matrix = np.array([
        [1, 0, 0, 0.015],
        [0, 1, 0, 0.04],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # 旋转修正 rotation_z90matrix (绕 z 轴旋转 -90°)。
    # 来源: 通常是来自于实际的硬件调试，例如通过测试得出一些小的修正值。旋转矩阵，表示绕 z 轴旋转 -90°。(逆时针旋转90度，表示与目标物体方向对齐)
    theta = -math.pi/2
    rotation_z90matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    debug_trans = candidates['icp_result'][best_idx].transformation
    #with bias
    debug_trans1 = np.dot(debug_trans,translate_0matrix)    
    debug_trans1 = np.dot(debug_trans1,rotation_z90matrix)
    debug_axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    debug_axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #no bias
    debug_trans1 = debug_trans

    # cam to ee pose
    # ---------- 5、计算相机到机器人基座的变换 
    # e_T_c: 末端到相机：相机坐标系在机器人末端坐标系下的变换，用于将相机坐标转换到末端坐标系
    # 来源：这种变换通常来源于实际 相机标定 和 机器人坐标系 的预设或测量值。
    e_T_c =np.array([[-0.999313, -0.0281012, 0.0241576, 0.168315],
                     [0.0299463,-0.996365, 0.0797554, -0.0612298],
                     [0.0218286, 0.0804241, 0.996522, 0.172745],
                     [0, 0, 0, 1]])

    # robot arm ee pose
    # b_T_e: 基座到末端：机器人末端坐标系在基座坐标系下的变换， 用于将末端坐标转换到基座坐标系。
    # 来源：这些变换矩阵的值通常来自 机器人臂的 DH 参数（Denavit-Hartenberg） 或 工厂校准。这些是事先定义的变换，用于将 末端执行器坐标系 转换为 基座坐标系。
    # 这里的旋转矩阵由 R.from_euler 给出，旋转角度（例如绕 x 轴旋转180度）。平移量来自 手动测量 或 CAD 模型。
    rot = R.from_euler('xyz', [3.14159, 0, 0])
    rot_m = rot.as_matrix()
    b_T_e = np.zeros((4,4))
    b_T_e[:3,:3] = rot_m
    b_T_e[0,3] = 0.142006
    b_T_e[1,3] = 0.452899
    b_T_e[2,3] = 0.588427
    b_T_e[3,3] = 1.0

    # b_T_c: 基座到相机：相机坐标系在机器人基座坐标系下的变换， b_T_c = np.dot(b_T_e, e_T_c) 这意味着 先从相机坐标转换到末端坐标，再转换到基座坐标。
    b_T_c = np.dot(b_T_e, e_T_c)
    # b_T_o: 基座到目标: 目标物体坐标系在机器人基座坐标系下的变换。debug_trans1 是目标点云在相机坐标系下的变换（通常是 icp_result 计算出的变换）
    b_T_o = np.dot(b_T_c, debug_trans1)
    debug_axis1.transform(b_T_o)

    # ------- 6：转换为欧拉角，输出最终位姿
    # 提取 b_T_o 的旋转矩阵 并转换为欧拉角 (xyz 轴旋转)
    rot_final = R.from_matrix(b_T_o[:3,:3])
    rot_final = rot_final.as_euler('xyz')
    # 转换为度数 (°) 输出最终位姿
    rot_final = rot_final/np.pi *180

    print("--------final out ---------")
    print(b_T_o)
    print(rot_final)
    """
    --------final out ---------
    [[ 0.27893391  0.95954971  0.03821153  0.29455047]
    [-0.94674163  0.28144036 -0.15643727  0.48012974]
    [-0.16086353  0.00745914  0.98694879 -0.00539537]
    [ 0.          0.          0.          1.        ]]
    [  0.43302463   9.25702073 -73.58369482]    
    """

    # ------ 7：可视化最终点云
    pc_vis = copy.deepcopy(candidates['debug_pc'][best_idx])
    # 转换点云 到机器人基座坐标系 b_T_c。
    pc_vis.transform(b_T_c)
    #print(candidates['icp_result'][best_idx].transformation)
    #o3d.visualization.draw_geometries([candidates['debug_pc'][best_idx], candidates['debug_trans'][best_idx], debug_axis1,debug_axis2])
    # 绘制坐标系（debug_axis1 和 debug_axis2），可视化最终目标点云
    o3d.visualization.draw_geometries([pc_vis, debug_axis1,debug_axis2])
    return best_idx, best_score


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  # mustard0_cad_test  textured_simple
  # fdkj_air_fat  120_m
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/fdkj_air_fat/mesh/120_m.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/fdkj_air_fat')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_aircon/debug_aircon_cad')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  downscale = 1
  reader = YcbineoatReader(video_dir=args.test_scene_dir, downscale=downscale, shorter_side=None, zfar=np.inf)
  print(f"--- reader: {reader.color_files}")

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    print(f"color: {color.shape}")  # 1080 x 1440; 错误，应该读取彩色三通道图像 1080 x 1440 x 3 rgb, 修改YcbineoatReader类
    depth = reader.get_depth(i)
    print(f"depth: {depth.shape}")  # 1080 x 1440

    # rgb_path = args.test_scene_dir + "/rgb/000000.png"    
    # print(f"------------- rgb_path: {rgb_path}")
    # color = cv2.imread(rgb_path)
    # depth = cv2.imread(rgb_path.replace('rgb','depth'),-1)/1e3
    # print(f"color_ori: {color.shape}, depth_ori: {depth.shape}")

    # ------------ 1、粗匹配
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      print(f"-------- depth: {depth.shape}, mask: {mask.shape}")
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      trans = pose.reshape(4,4)
      print(f"trans: {trans}") # 4 x 4 matrix
      """ Dr.Li 原有代码结果 
      trans: [[-0.23898166 -0.970123   -0.04182552  0.02670384]
            [-0.90239674  0.23779097 -0.35935447  0.00145889]
            [ 0.35856366 -0.04813593 -0.93226343  0.42379868]
            [ 0.          0.          0.          1.        ]]
      本份代码结果: 不一致，原因是downscale=1, 原有代码 YcbineoatReader 的 downscale=0.5
      trans: [[-0.237795   -0.9702371  -0.04575697  0.02684642]
            [-0.9107593   0.23909548 -0.3366762   0.00153804]
            [ 0.33759603 -0.03838624 -0.94050807  0.4238183 ]
            [ 0.          0.          0.          1.        ]]
      """
      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      # cv2.imshow('1', vis[...,::-1])
      # cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)


    # ---------- 2、精匹配
    # CAD 模型点云
    pc_cad = o3d.io.read_point_cloud(f'./demo_data/fdkj_air_fat/120_m.ply') # 圆管 , 建模与 obj 一起给过来

    # 深度图生成点云
    scale = 1 / downscale
    print(f"scale: {scale}")
    # 使用mask, 提取有效深度区域
    masked_depth = cv2.bitwise_and(depth, depth, mask=mask.astype(np.uint8))
    print(f"masked_depth: {masked_depth.shape}, min: {np.min(masked_depth)}, max: {np.max(masked_depth)}")
    # cv2.imwrite(f'{debug_dir}/masked_depth.png', masked_depth)
    # 构造 Open3D 的相机参数
    o3d_intri = o3d.camera.PinholeCameraIntrinsic(
      width=masked_depth.shape[1],  # 深度图的宽度
      height=masked_depth.shape[0],  # 深度图的高度
      fx=reader.K[0,0]*scale,  # 焦距 fx, 考虑缩放比例
      fy=reader.K[1,1]*scale,  # 焦距 fy, 考虑缩放比例
      cx=reader.K[0,2]*scale,  # 光心 cx, 考虑缩放比例
      cy=reader.K[1,2]*scale   # 光心 cy, 考虑缩放比例
    )
    # 深度图转为Open3D的点云
    o3d_depth = masked_depth.astype('float32')
    o3d_depth = o3d.geometry.Image(o3d_depth)
    # 通过深度图生成点云
    pc_cam = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth,
        o3d_intri,
        depth_scale=1.0 # depth 单位是毫米，必须设为 1000.0
        )
    
    # 保存点云
    o3d.io.write_point_cloud(f'{debug_dir}/pcd_cam.ply', pc_cam)

    threshold = 3 / 1000  # 3mm

    # 精匹配
    # 使用ICP算法进行配准
    reg_p2p = o3d.pipelines.registration.registration_icp(
      pc_cad, 
      pc_cam, 
      threshold, 
      trans, 
      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
      o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    print(reg_p2p)
    print(f"Transformation is: {reg_p2p.transformation}")
    """
    [[-2.46879484e-01 -9.67481683e-01 -5.50441449e-02  2.59693042e-02]
    [-9.38199806e-01  2.52852608e-01 -2.36319194e-01  4.34009330e-04]
    [ 2.42552552e-01 -6.69986481e-03 -9.70115136e-01  4.21943035e-01]
    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
    """

    # ------ 计算变换矩阵并转换到 机器人基座坐标系
    candidates = dict()
    candidates['pointcloud'] = []
    candidates['icp_result'] = []
    candidates['debug_pc'] = []
    candidates['debug_trans'] = []

    ombined_pc, trans_axis = draw_registration_result(pc_cad, pc_cam, trans, reg_p2p.transformation)

    candidates['pointcloud'].append(pc_cam)
    candidates['icp_result'].append(reg_p2p)
    candidates['debug_pc'].append(ombined_pc)
    candidates['debug_trans'].append(trans_axis)

    print(f"candidates: {candidates}")
    if(len(candidates)>1):
        best_idx, best_score = rotation_base_selection(candidates, align_axis='y')
