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

  reader = YcbineoatReader(video_dir=args.test_scene_dir, downscale=0.5, shorter_side=None, zfar=np.inf)
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

    if i==0:
      print(f"-------- depth: {depth.shape}")
      mask = reader.get_mask(0).astype(bool)
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

