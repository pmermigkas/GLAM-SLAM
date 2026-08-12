#!/usr/bin/env python3
import os.path
os.environ['GLOG_minloglevel'] = '2'
import time
import cv2
import threading
import torch
import torch.nn as nn 
import numpy as np
from collections import Counter
import random
import orbslam2
from types import SimpleNamespace
from gsplat.cuda._wrapper import fully_fused_projection
import yaml
from munch import munchify
from scipy.spatial.transform import Rotation
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import uuid
import caffe

import datetime
import queue
import sys
import json

from utils.graphics_utils import BasicPointCloud, getProjectionMatrix, getWorld2View2, focal2fov
from src.gaussian_scene import GaussianKeyframe, GaussianScene
from src.base_model import GaussianModelScaffold
from utils.loss_utils import ssim
from utils.image_utils import mse, psnr
from gaussian_renderer import render

BOLD_GREEN_START = "\033[1m\033[92m"
RESET_FORMATTING = "\033[0m"



def choose_keyframe(total_frames, threshold, beta):
    keyframe_indices = np.arange(total_frames)
    current_keyframe = total_frames - 1

    if total_frames <= threshold:
        normalized_probabilities = 1/total_frames * np.ones_like(keyframe_indices)
    else:
        normalized_probabilities = np.ones(total_frames, dtype=float)
        normalized_probabilities[:total_frames-threshold] = ((1 - beta) / (total_frames - threshold)) * np.ones(total_frames-threshold)
        normalized_probabilities[total_frames-threshold:total_frames] = (beta / threshold) * np.ones(threshold)

    selected_keyframe_index = np.random.choice(keyframe_indices, p=normalized_probabilities)
    
    return selected_keyframe_index





def load_gaussian_config(config_path):
    """
    Loads and parses a YAML configuration file.
    
    Returns:
        A dictionary of Munch objects for model, pipeline, and optimization parameters.
    """
    try:
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)
            ip = SimpleNamespace(**cfg.get('input_params', {}))
            lp = SimpleNamespace(**cfg.get('model_params', {}))
            pp = SimpleNamespace(**cfg.get('pipeline_params', {}))
            op = SimpleNamespace(**cfg.get('optim_params', {}))
            return ip, lp, pp, op
    except FileNotFoundError:
        print(f"[GaussianMapper] Error: The configuration file was not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"[GaussianMapper] Error parsing YAML file: {e}")
        return None

def load_orb_config(config_path):
    try:
        with open(config_path, 'r') as file:
            lines = file.readlines()
            
            # Join all lines except the first one into a single string
            content = "".join(lines[1:])

            # Pass the modified content string to the YAML parser
            config = yaml.safe_load(content)

            return munchify({
                "camera_fx": config.get("Camera.fx", 0.0),
                "camera_fy": config.get("Camera.fy", 0.0),
                "camera_cx": config.get("Camera.cx", 0.0),
                "camera_cy": config.get("Camera.cy", 0.0),
                "camera_k1": config.get("Camera.k1", 0.0),
                "camera_k2": config.get("Camera.k2", 0.0),
                "camera_p1": config.get("Camera.p1", 0.0),
                "camera_p2": config.get("Camera.p2", 0.0),
                "camera_image_height": config.get("Camera.image_height", 0),
                "camera_image_width": config.get("Camera.image_width", 0),
            })
    except FileNotFoundError:
        print(f"[GaussianMapper] Error: The ORB-SLAM2 configuration file was not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"[GaussianMapper] Error parsing ORB-SLAM2 YAML file: {e}")
        return None    






def skew(x):
    """Return the skew-symmetric matrix of a 3-vector."""
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def compute_fundamental_from_projections(P1, P2):
    """
    Compute the fundamental matrix F from projection matrices P1 and P2.
    """
    # Compute camera center C1 (null space of P1)
    U, S, Vt = np.linalg.svd(P1)
    C1_hom = Vt[-1]
    C1 = C1_hom / C1_hom[3]  # in homogeneous coords

    # Project C1 into second image
    e2 = P2 @ C1  # epipole in image 2
    e2 = e2 / e2[2]

    # Compute F = [e2]_x * P2 * P1_pinv
    e2_skew = skew(e2[:3])
    P1_pinv = np.linalg.pinv(P1)
    F = e2_skew @ P2 @ P1_pinv

    return F / F[2, 2]

def compute_extra_3d_points_from_keyframes_and_flow(keyframe1, keyframe2, flow, K, num_samples=5000, threshold=0.5):

    R_mat1 = Rotation.from_quat(keyframe1.pose_quaternion).as_matrix()
    t1 = keyframe1.pose_translation
    pose1 = np.hstack((R_mat1, t1.reshape(3, 1)))

    R_mat2 = Rotation.from_quat(keyframe2.pose_quaternion).as_matrix()
    t2 = keyframe2.pose_translation
    pose2 = np.hstack((R_mat2, t2.reshape(3, 1)))

    # Compute projection matrices
    P1 = K @ pose1
    P2 = K @ pose2

    F = compute_fundamental_from_projections(P1, P2)


    # start_time = time.time()

    # Ensure images are the same size
    assert keyframe1.original_image.shape[:2] == keyframe2.original_image.shape[:2] == flow.shape[:2]

    # Randomly sample "num_samples" unique pixel coordinates from image B
    ys = np.random.randint(0, keyframe2.original_image.shape[0], size=num_samples)
    xs = np.random.randint(0, keyframe2.original_image.shape[1], size=num_samples)

    # Get flow at sampled points
    points_B = np.stack((xs, ys), axis=1)                   # shape (N, 2)
    flow_at_points = flow[ys, xs]                           # shape (N, 2)
    points_A = points_B + flow_at_points                    # shape (N, 2)

    # Homogeneous coordinates
    points_B_h = np.hstack((points_B, np.ones((num_samples, 1))))  # (N, 3)
    points_A_h = np.hstack((points_A, np.ones((num_samples, 1))))  # (N, 3)

    # Compute epipolar constraint: x2^T * F * x1
    Fx1 = points_A_h @ F.T
    x2tFx1 = np.sum(points_B_h * Fx1, axis=1)

    # Apply threshold
    valid_mask = np.abs(x2tFx1) < threshold

    # Filter valid correspondences
    valid_points_A = points_A[valid_mask]
    valid_points_B = points_B[valid_mask]
    # Transpose to shape (2, N) as required by OpenCV
    pts1 = valid_points_A.T.astype(np.float32)  # shape: (2, N)
    pts2 = valid_points_B.T.astype(np.float32)  # shape: (2, N)

    # Vectorized triangulation
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)  # shape: (4, N)
    
    points_3d = (points_4d_hom[:3, :] / points_4d_hom[3, :]).T  # shape: (N, 3)

    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # (N, 4)

    points_cam1_hom = (pose1 @ points_3d_h.T).T  # shape: (N, 4)

    # Keep only points with positive Z (in front of the camera)
    mask = points_cam1_hom[:, 2] > 0

    # Get the corresponding original 3D points (before transformation)
    points_3d_visible = points_3d[mask]

    # Convert to numpy array
    points_3d_visible_np = np.array(points_3d_visible)

    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print(f"[GaussianMapper] Triangulation (from optical flow) execution time: {elapsed_time:.4f} seconds")

    return points_3d_visible_np


def compute_heading(keyframe1, keyframe2):
    # Poses are saved in the GaussianScene as Tcw
    Rcw1 = Rotation.from_quat(keyframe1.pose_quaternion).as_matrix()
    Rcw2 = Rotation.from_quat(keyframe2.pose_quaternion).as_matrix()

    
    # Extract forward (camera z-axis) direction
    # If the pose is Tcw, the forward direction is the 3rd row of R
    v1 = Rcw1[2, :]  # shape (1, 3)
    v2 = Rcw2[2, :]  # shape (1, 3)


    # Compute incremental yaw change
    # Project onto ground plane (ignore height = y direction)
    v1_xy = v1[[0, 2]]
    v2_xy = v2[[0, 2]]
    v1_xy /= np.linalg.norm(v1_xy)
    v2_xy /= np.linalg.norm(v2_xy)
    cos_angle = np.clip(np.dot(v1_xy, v2_xy), -1.0, 1.0)
    angle_mag = np.degrees(np.arccos(cos_angle))
    sign = np.sign(np.cross(np.append(v1_xy, 0), np.append(v2_xy, 0))[2])
    angle = angle_mag * sign

    heading = keyframe1.heading + angle

    return heading



class GaussianMapper:
    """
    This class is responsible for creating and optimizing a map
    represented by 3D Gaussians. It runs in a separate thread from the
    main ORB-SLAM tracking thread.
    """
    def __init__(self, slam_system, gaussian_config_path, orb_config_path, output_dir, path_to_sequence, device='cpu'):
        """
        Initializes the GaussianMapper.

        Args:
            slam_system: The running ORB_SLAM2::System object.
            gaussian_config_path (str): Path to the GaussianMapper configuration file.
            orb_config_path (str): Path to the ORB-SLAM settings file (YAML).
            output_dir (str): Directory to save outputs.
            path_to_sequence (str): Path to the image sequence or dataset directory.
            device (str, optional): Compute device to use (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.slam_system = slam_system
        self.gaussian_config_path = gaussian_config_path
        self.orb_config_path = orb_config_path
        self.output_dir = output_dir
        self.path_to_sequence = path_to_sequence
        self.device_type = torch.device(device)


        self.stop_requested = False
        self.slam_ended = False
        


        # --- Gaussian Model ---
        # Load parameters from the config file
        self.ip, self.mp, self.pp, self.op = load_gaussian_config(self.gaussian_config_path)
        if not (self.mp and self.op and self.pp):
            raise FileNotFoundError(f"Failed to load config from {self.gaussian_config_path}")
        
        # Scaffold-GS
        model_config = self.mp.model_config
        self.gaussian_model = GaussianModelScaffold(**model_config['kwargs'])
        self.gaussian_scene = GaussianScene()

        self.start_gaussian_training = self.op.start_gaussian_training

        # Keyframe optimization selection policy
        self.optimize_num_last_keyframes = getattr(self.mp, 'optimize_num_last_keyframes', 25)
        self.optimize_prob_last_keyframes = getattr(self.mp, 'optimize_prob_last_keyframes', 0.7)
        print(f"{BOLD_GREEN_START}[GaussianMapper] Keyframe optimization selection policy: Pick last optimize_num_last_keyframes={self.optimize_num_last_keyframes} with optimize_prob_last_keyframes={self.optimize_prob_last_keyframes}{RESET_FORMATTING}")



        # Optical Flow usage flag
        self.use_optical_flow = getattr(self.mp, 'use_optical_flow', True)
        self.optical_flow_interval = getattr(self.mp, 'optical_flow_interval', 7)
        if self.use_optical_flow:
            print(f"{BOLD_GREEN_START}[GaussianMapper] Optical Flow: ENABLED with optical_flow_interval={self.optical_flow_interval}{RESET_FORMATTING}")
        else:
            print(f"{BOLD_GREEN_START}[GaussianMapper] Optical Flow: DISABLED{RESET_FORMATTING}")



        # --- ORB Model ---
        # Load parameters from the config file
        self.orb_config = load_orb_config(self.orb_config_path)
        if not self.orb_config:
            raise FileNotFoundError(f"Failed to load config from {self.orb_config_path}")
        


        self.background = torch.tensor([0.0, 0.0, 0.0], device=self.device_type)
        
        # Configuration (load from gaussian_config_path or set defaults)
        self.iteration = 0
        self.processed_kf_ids = set()
        self.l1_loss = nn.L1Loss()
        
        # Camera parameters
        self.K = np.array([
            [self.orb_config.camera_fx, 0, self.orb_config.camera_cx],
            [0, self.orb_config.camera_fy , self.orb_config.camera_cy],
            [0,  0,  1]
        ])
        self.fovx = focal2fov(self.orb_config.camera_fx, self.orb_config.camera_image_width)
        self.fovy = focal2fov(self.orb_config.camera_fy, self.orb_config.camera_image_height)
        
        
        # Load LiteFlowNet3 Optical Flow Network
        self.fpath_prototxt = getattr(self.mp, 'fpath_prototxt', '/home/')
        self.fpath_caffemodel = getattr(self.mp, 'fpath_caffemodel', '/home/')
        self.net = caffe.Net(self.fpath_prototxt, self.fpath_caffemodel, caffe.TEST)
        self.mapping_operations_queue = queue.Queue()





    def request_stop(self):
        """
        Requests the mapper to stop its execution.
        """
        self.stop_requested = True

    def is_stopped(self):
        return self.stop_requested

    def has_met_initial_mapping_conditions(self):
        """Check if the SLAM map is ready for initial mapping."""
        # Wait for a stable map with a few keyframes
        mapping_operations_queue_size = self.slam_system.get_mapping_operation_queue_size()
        if  mapping_operations_queue_size >= self.start_gaussian_training:
            print(f"[GaussianMapper] Initial mapping condition met ({mapping_operations_queue_size} mapping operations found in the queue).")
            return True
        return False
    
    def add_gaussian_keyframes(self, kf_mnIds, kf_kfIds, kf_poses):
        # The pose_element is a flattened list of floats from C++
        # Convert to numpy array and reshape to (4, 4)
        se3_poses = []
        for pose_element in kf_poses:
            # Convert the list of floats to a numpy array
            np_pose = np.array(pose_element, dtype=np.float32)
            
            # Reshape the 16-element array into a 4x4 matrix
            pose = np.reshape(np_pose, (4, 4))
            se3_poses.append(pose)
    
        # Create the initial keyframe objects
        for kf_ind in range(len(kf_mnIds)):
            se3_pose = se3_poses[kf_ind]
            kf_pose_quaternion = Rotation.from_matrix(se3_pose[:3, :3]).as_quat()
            kf_pose_translation = se3_pose[:3, 3]
            mn_Id = kf_mnIds[kf_ind]
            kf_Id = kf_kfIds[kf_ind]
            initial_keyframe = GaussianKeyframe(kf_pose_quaternion, kf_pose_translation, mn_Id, kf_Id, self.path_to_sequence, self.device_type)
            self.gaussian_scene.add_keyframe(initial_keyframe)
            print("[GaussianMapper] Added a new keyframe to the Gaussian map with kfId", kf_Id)

    def psnr_average(self):
        psnr_values_per_kf = []
        for kf in self.gaussian_scene.get_keyframes():
            psnr_values_per_kf.append(kf.psnr_value)
        
        return sum(psnr_values_per_kf) / len(psnr_values_per_kf)
    
    def lpips_average(self):
        lpips_values_per_kf = []
        for kf in self.gaussian_scene.get_keyframes():
            lpips_values_per_kf.append(kf.lpips_value)
        
        return sum(lpips_values_per_kf) / len(lpips_values_per_kf)

    def psnr_report(self):
        keyframes = self.gaussian_scene.get_keyframes()
        print("Times each keyframe was optimized ", end='')
        for kf in keyframes:
            print(kf.train_counter, end=' ')
        print("")
        print("Number of anchors ", self.gaussian_model._anchor.size()[0], end=' ')
        print("Number of Gaussians ", self.gaussian_model._anchor.size()[0]* self.gaussian_model.n_offsets)
        print("PSNRs for all KFs: ", end='')
        psnr_values_per_kf = []
        for kf in self.gaussian_scene.get_keyframes():
            print("{:.2f}".format(kf.psnr_value), end=' ')
            psnr_values_per_kf.append(kf.psnr_value)
        average_psnr = sum(psnr_values_per_kf) / len(psnr_values_per_kf)
        print(f"Average {BOLD_GREEN_START}{average_psnr:.4f}{RESET_FORMATTING}")

    def ssim_report(self):
        print("SSIMs for all KFs: ", end='')
        ssim_values_per_kf = []
        for kf in self.gaussian_scene.get_keyframes():
            print("{:.2f}".format(kf.ssim_value), end=' ')
            ssim_values_per_kf.append(kf.ssim_value)
        average_ssim = sum(ssim_values_per_kf) / len(ssim_values_per_kf)
        print(f"Average {BOLD_GREEN_START}{average_ssim:.4f}{RESET_FORMATTING}")

    def lpips_report(self):
        print("LPIPSs for all KFs: ", end='')
        lpips_values_per_kf = []
        for kf in self.gaussian_scene.get_keyframes():
            print("{:.2f}".format(kf.lpips_value), end=' ')
            lpips_values_per_kf.append(kf.lpips_value)
        average_lpips = sum(lpips_values_per_kf) / len(lpips_values_per_kf)

        print(f"Average {BOLD_GREEN_START}{average_lpips:.4f}{RESET_FORMATTING}")

    def ate_report(self, output_filepath):        
        trajectory_data = []
        for keyframe in self.gaussian_scene.get_keyframes():
            kf_entry = {
                "KF_id": keyframe.kfId,
                "F_id": keyframe.mnId,
                "translation": keyframe.pose_t_wc.tolist() if hasattr(keyframe.pose_t_wc, 'tolist') else list(keyframe.pose_t_wc),
                "rotation": keyframe.pose_R_wc.tolist() if hasattr(keyframe.pose_R_wc, 'tolist') else list(keyframe.pose_R_wc)
            }
            trajectory_data.append(kf_entry)
        
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        with open(output_filepath, "w") as json_file:
            json.dump(trajectory_data, json_file, indent=4)
        print(f"[GaussianMapper] Trajectory ATE saved to {output_filepath}")



    def render_keyframe(self, kf_image, kf_pose_quaternion, kf_pose_translation, kfId):
        # Reconstruct the 4x4 pose matrix (Tcw, camera_to_world_transform)
        # from the quaternion and translation vector
        rotation_matrix = Rotation.from_quat(kf_pose_quaternion).as_matrix()
        Tcw = np.eye(4, dtype=np.float32)
        Tcw[:3, :3] = rotation_matrix
        Tcw[:3, 3] = kf_pose_translation

        # The camera center is the translation part of the world_to_camera_transform (Twc)
        Twc = np.linalg.inv(Tcw)
        camera_center = Twc[:3, 3]
    


        kf_image_tensor = torch.from_numpy(np.transpose(kf_image, (2, 0, 1))).float().to(self.device_type)

        image_height, image_width = kf_image.shape[:2]

        world_view_transform = torch.tensor(getWorld2View2(rotation_matrix, kf_pose_translation)).to(self.device_type).T
        projmatrix = getProjectionMatrix(self.ip.z_near, self.ip.z_far, self.fovx, self.fovy).to(self.device_type).T


        viewpoint_camera = munchify({
            'FoVx' : self.fovx,
            'FoVy' : self.fovy,
            'cx' : self.orb_config.camera_cx,
            'cy' : self.orb_config.camera_cy,
            'image_height': image_height,
            'image_width': image_width,
            'image': kf_image_tensor, # 3D-GS expects it to be (C, H, W) (conversion from (H,W,C))
            'kfId': kfId, # This is passed to the render method, so that the correct MLP is chosen in generate_neural_gaussians
            'world_view_transform': world_view_transform,
            'full_proj_transform': world_view_transform @ projmatrix,
            'camera_center': torch.from_numpy(camera_center).float().to(self.device_type),
            'resolution_scale': 1.0
        })


        render_output = render(viewpoint_camera, self.gaussian_model, self.pp, self.background, self.iteration, self.mp.render_mode)

        return render_output, viewpoint_camera










    def compute_dense_flow(self, img0, img1):
        # Caffe requires images in BGR format
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

        # Caffe requires images in [0, 255]
        img0 = np.clip(img0 * 255.0, 0, 255).astype(np.uint8)
        img1 = np.clip(img1 * 255.0, 0, 255).astype(np.uint8)

        _, c, h, w = self.net.blobs['img0'].data.shape

        # Resize and normalize images
        def preprocess(img):
            img = cv2.resize(img, (w, h))       # Resize to model input size
            img = img.transpose(2, 0, 1)        # Convert to (C, H, W)
            return img

        img0 = preprocess(img0)
        img1 = preprocess(img1)

        # Feed data
        self.net.blobs['img0'].data[...] = img0
        self.net.blobs['img1'].data[...] = img1

        # Forward pass
        self.net.forward()

        # Retrieve optical flow (assuming 'predict_flow_final' is the output blob name)
        flow = self.net.blobs['predict_flow_final'].data[0]  # Shape: (2, H, W)

        # Convert to H x W x 2 format
        flow = flow.transpose(1, 2, 0)

        return flow
    


    def render_final_images(self):
        keyframes = self.gaussian_scene.get_keyframes()
        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(self.device_type)

        rgb_dir = os.path.join(self.output_dir, "image_rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        rendered_dir = os.path.join(self.output_dir, "image_rendered")
        os.makedirs(rendered_dir, exist_ok=True)

        for keyframe in keyframes:
            # Save original RGB image
            filename_rgb = os.path.join(rgb_dir, f"image_rgb_{keyframe.kfId}.png")
            image_rgb = np.clip(keyframe.original_image * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(filename_rgb, image_rgb[:, :, ::-1])
            

            # Render keyframe
            render_output, _ = self.render_keyframe(keyframe.original_image, keyframe.pose_quaternion, keyframe.pose_translation, keyframe.kfId)
            rendered_image = render_output["render"]
            

            # Compute PSNR, SSIM, LPIPS metrics
            psnr_tensor = psnr(rendered_image, torch.from_numpy(np.transpose(keyframe.original_image, (2, 0, 1))).float().to(self.device_type))
            ssim_tensor = ssim((rendered_image).permute(1, 2, 0).unsqueeze(0), torch.from_numpy(keyframe.original_image).to(device=self.device_type).unsqueeze(0))
            lpips_tensor = cal_lpips(rendered_image.unsqueeze(0), torch.from_numpy(keyframe.original_image).to(device=self.device_type).permute(2, 0, 1).unsqueeze(0))

            keyframe.psnr_value = psnr_tensor.item()
            keyframe.ssim_value = ssim_tensor.item()
            keyframe.lpips_value = lpips_tensor.item()



            # Save rendered RGB image
            rendered_image = rendered_image.cpu()
            rendered_image = rendered_image.permute(1, 2, 0) # Convert from (C, H, W) to (H, W, C)
            rendered_image = (rendered_image * 255).clamp(0, 255).byte() # Convert pixel values from [0.0, 1.0] to [0, 255]

            rendered_image_np = rendered_image.numpy()
            rendered_image_bgr = cv2.cvtColor(rendered_image_np, cv2.COLOR_RGB2BGR)

            filename_rendered = os.path.join(rendered_dir, f"image_rendered_{keyframe.kfId}.png")
            cv2.imwrite(filename_rendered, rendered_image_bgr)




    def correct_anchors_after_loop_closure(self, corrected_keyframe_kfIds, corrected_keyframes_poses, min_visible_anchors=20):
        """
        Corrects the positions of Gaussian anchors after a loop closure event in SLAM.

        This function iterates through keyframes that have been updated by the SLAM backend
        (e.g., ORB-SLAM's pose graph optimization). For each keyframe, it identifies the
        Gaussian anchors visible within its frustum and applies the keyframe's pose
        correction to them. A mask is used to ensure that each anchor is corrected only
        once per loop closure event, preventing redundant transformations.

        Args:
            corrected_keyframe_kfIds: Keyframe IDs that ORB-SLAM2 has updated during LC.
            corrected_keyframes_poses: The corresponding updated poses.
            min_visible_anchors: The minimum number of anchors that must 
                be visible in a keyframe for the correction to be applied.
        """

        print(f"[GaussianMapper] Starting anchor correction for {len(corrected_keyframe_kfIds)} keyframes.")
    
        # Initialize a mask to track which anchors have already been updated in this batch.
        # This is crucial to prevent applying transformations multiple times to the same anchor
        # if it's visible in multiple corrected keyframes.
        num_anchors = self.gaussian_model.get_anchor.shape[0]
        updated_anchors_mask = torch.zeros(num_anchors, dtype=torch.bool, device=self.device_type)

        # Pre-fetch all anchor properties from the Gaussian model
        all_anchors = self.gaussian_model.get_anchor
        all_anchors_quats = self.gaussian_model.get_rotation
        all_anchors_scales = self.gaussian_model.get_scaling[:, :3]

        
        # Iterate through all keyframes that have been corrected by the loop closure
        for index, kfId in enumerate(corrected_keyframe_kfIds):
            old_keyframe = self.gaussian_scene.get_keyframes()[kfId]
            
            world_view_transform = torch.tensor(getWorld2View2(old_keyframe.pose_R_cw, old_keyframe.pose_t_cw)).to(self.device_type).T
            viewmats = world_view_transform.transpose(0, 1).unsqueeze(0)


            np_new_pose = np.array(corrected_keyframes_poses[index], dtype=np.float32)
            # Reshape the 16-element array into a 4x4 matrix
            new_pose_cw = np.reshape(np_new_pose, (4, 4))


            K = torch.tensor([
                    [self.orb_config.camera_fx, 0, self.orb_config.camera_cx],
                    [0, self.orb_config.camera_fy, self.orb_config.camera_cy],
                    [0, 0, 1],
                ],device=self.device_type,)[None]

            # Use gsplat's projection function to find out which anchors are in the frustum.
            # This function projects the 3D Gaussians (here, our anchors) onto the 2D image plane.
            proj_results = fully_fused_projection(
                means=all_anchors,
                covars=None,   # covars
                quats=all_anchors_quats,
                scales=all_anchors_scales,
                viewmats=viewmats,
                Ks=K,
                width=int(old_keyframe.original_image.shape[1]),
                height=int(old_keyframe.original_image.shape[0]),
                eps2d=0.3, # Standard epsilon value
                packed=False,
                near_plane=0.01,
                far_plane=1.0,
                radius_clip=0.0,
                sparse_grad=False,
                calc_compensations=False,
            )
            radii = proj_results[0]
            # A Gaussian is considered visible if its projected radius is greater than 0.
            # gsplat >= 1.5 returns radii as (C, N, 2), so we take the max along the last dim.
            visible_in_kf_mask = torch.max(radii.squeeze(0), dim=1).values > 0

            # Identify anchors that are visible AND have not been updated yet
            anchors_to_correct_mask = visible_in_kf_mask & ~updated_anchors_mask

            num_visible = anchors_to_correct_mask.sum().item()
            if num_visible < min_visible_anchors:
                # print(f"[GaussianMapper] Skipping KF {old_keyframe.kfId}: Found only {num_visible} anchors to correct.")
                continue

            print(f"[GaussianMapper] Processing KF {old_keyframe.kfId}: Found {num_visible} anchors to correct.")

            # Calculate the transformation delta
            # Create transformation 4x4 matrix from old_pose_quaternion and old_pose_translation
            old_pose_cw = np.eye(4, dtype=np.float32)
            old_pose_cw[:3, :3] = old_keyframe.pose_R_cw
            old_pose_cw[:3, 3] = old_keyframe.pose_t_cw
            old_pose_cw_tensor = torch.from_numpy(old_pose_cw).to(self.device_type)
            new_pose_cw_tensor = torch.from_numpy(new_pose_cw).to(self.device_type)
            transform_delta = torch.inverse(new_pose_cw_tensor) @ old_pose_cw_tensor
            print("[GaussianMapper] Transform delta is ", transform_delta)

            # Convert to homogeneous coordinates (x, y, z, 1) for matrix multiplication
            anchors_to_correct = all_anchors[anchors_to_correct_mask]
            num_points = anchors_to_correct.shape[0]
            homogeneous_anchors = torch.cat([anchors_to_correct, torch.ones(num_points, 1, device=self.device_type)], dim=1)

            # Apply the transformation: [4, 4] @ [4, N] -> [4, N], then transpose back to [N, 4]
            transformed_homogeneous = (transform_delta @ homogeneous_anchors.T).T

            # Convert back to 3D coordinates by dropping the last component
            corrected_xyz = transformed_homogeneous[:, :3]
            with torch.no_grad():
                new_anchor = self.gaussian_model._anchor.clone()
                new_anchor[anchors_to_correct_mask] = corrected_xyz
                self.gaussian_model._anchor.data.copy_(new_anchor)


            # Mark the affected anchors as corrected
            updated_anchors_mask |= anchors_to_correct_mask
        
            # Check if all anchors have been updated
            if updated_anchors_mask.all():
                print("[GaussianMapper] All anchors have been corrected. Exiting early.")
                break

        total_updated = updated_anchors_mask.sum().item()
        print(f"[GaussianMapper] Anchor correction finished. Total of {total_updated} / {num_anchors} anchors were updated.")









































    def train_for_one_iteration(self):
        """Performs a single iteration of Gaussian optimization."""
        self.iteration += 1
        

        # start = time.perf_counter()
        
        all_keyframes = self.gaussian_scene.get_keyframes()
        if not all_keyframes:
            print("[GaussianMapper] No keyframes available for training.")
            return

        # Select a random keyframe
        random_keyframe_index = choose_keyframe(len(all_keyframes), threshold=self.optimize_num_last_keyframes, beta=self.optimize_prob_last_keyframes)
        random_keyframe = all_keyframes[random_keyframe_index]
        
    
        # Extract elements from the random keyframe
        kf_image = random_keyframe.original_image
        render_output, viewpoint_camera = self.render_keyframe(kf_image, random_keyframe.pose_quaternion, random_keyframe.pose_translation, random_keyframe.kfId)

        rendered_image = render_output["render"]
        # scaling = render_output["scaling"]
        # viewspace_point_tensor = render_output["viewspace_points"]
        # visibility_filter = render_output["visibility_filter"]
        # visible_mask = render_output["visible_mask"]
        # selection_mask = render_output["selection_mask"]
        # opacity = render_output["opacity"]
        # depth_image = render_output["render_depth"]
        # radii = render_output["radii"]


        # Calculate PSNR value
        gt_image = viewpoint_camera.image.to(self.device_type)
        random_keyframe.psnr_value = psnr(rendered_image, gt_image).item()



        # Calculate loss
        # start_loss = time.perf_counter()
        Ll1 = self.l1_loss(rendered_image, gt_image)
        Ldssim = (1.0 - ssim(rendered_image, gt_image))
        # scaling_reg = scaling.prod(dim=1).mean()

        # Backpropagation
        loss = (1.0 - self.op.lambda_dssim) * Ll1 + self.op.lambda_dssim * Ldssim # + 0.01 * scaling_reg
        loss.backward()
        # end_loss = time.perf_counter()

        

        # Check for densification conditions and perform densification/pruning
        with torch.no_grad():
            if self.iteration < self.op.densify_until_iter and self.iteration > self.op.densification_stat_start:
                # statis
                self.gaussian_model.training_statis(render_output, rendered_image.shape[2], rendered_image.shape[1])
                
                # densification
                if self.op.densification and self.iteration > self.op.densify_from_iter and self.iteration % self.op.densification_interval == 0:
                    self.gaussian_model.run_densify(self.iteration, self.op)
            
            elif self.iteration == self.op.densify_until_iter:
                self.gaussian_model.clean()
            
            # # Reset opacity periodically
            # if (self.op.opacity_reset_interval and self.iteration % self.op.opacity_reset_interval == 0):
            #     self.gaussian_model.reset_opacity()


            # Optimizer step
            self.gaussian_model.optimizer.step()
            self.gaussian_model.optimizer.zero_grad(set_to_none=True)
        random_keyframe.train_counter += 1

        # end = time.perf_counter()






        # elapsed_time_loss = end_loss - start_loss
        # elapsed_time_iteration = end - start
        # print(f"average psnr: {self.psnr_average():.2f} @ {self.iteration} @ {all_keyframes[-1].kfId} KF id= {random_keyframe.kfId} F id= {random_keyframe.mnId} last psnr: {random_keyframe.psnr_value:.2f}, (loss+backward): {elapsed_time_loss:.6f} seconds, (train_for_one_iteration): {elapsed_time_iteration:.6f} seconds")











































    def run(self):
        """
        Main loop for the GaussianMapper. This function will continuously
        process data from the SLAM system to build the Gaussian map.
        """
        print("[GaussianMapper] GaussianMapper thread started.")

        caffe.set_mode_gpu()
        caffe.set_device(0)
        
        start_total = time.perf_counter()
        while not self.is_stopped():
            if self.has_met_initial_mapping_conditions():
                print("\n[GaussianMapper] Starting initial map creation from sparse SLAM map.")

                keyframe_list = self.slam_system.get_all_keyframes_data()

                mappoints = []
                imagepoints = []
                kfIds = []
                initial_keyframe_mnIds = []
                initial_keyframe_kfIds = []
                initial_keyframe_poses = []
                for (mnId, kfId, pose, mps, _) in keyframe_list:
                    mappoints.extend(mps)
                    kfIds.extend([kfId] * len(mps)) 
                    initial_keyframe_mnIds.append(mnId)
                    initial_keyframe_kfIds.append(kfId)
                    initial_keyframe_poses.append(pose)
                
                
                

                if not mappoints:
                    print("[GaussianMapper] No map points found yet. Waiting...")
                    time.sleep(0.2)
                    continue

                self.add_gaussian_keyframes(initial_keyframe_mnIds, initial_keyframe_kfIds, initial_keyframe_poses)


                # Mark initial keyframes as processed
                self.processed_kf_ids = []
                for kf in self.gaussian_scene.get_keyframes():
                    self.processed_kf_ids.append(kf.kfId)
                    



                
                if (self.use_optical_flow):
                    # Define which images to run dense Optical Flow for
                    keyframe1 = self.gaussian_scene.get_keyframes()[0]
                    keyframe2 = self.gaussian_scene.get_keyframes()[1]

                    img1 = keyframe1.original_image
                    img2 = keyframe2.original_image


                    # Optical Flow computation
                    start_time_flow = time.perf_counter()
                    flow = self.compute_dense_flow(img2, img1) # We want to compute inverse flow here (assumption for the following)
                    end_time_flow = time.perf_counter()
                    elapsed_time_flow = end_time_flow - start_time_flow
                    print(f"[GaussianMapper] Optical Flow Computation, elapsed time: {elapsed_time_flow:.6f} seconds")

                    # Triangulate points using Optical Flow
                    flow_points_3d_np = compute_extra_3d_points_from_keyframes_and_flow(keyframe1, keyframe2, flow, self.K, num_samples=5000, threshold=0.5)
                
                    points = np.concatenate((flow_points_3d_np, np.array(mappoints)), axis=0)
                    
                    print(f"[GaussianMapper] Initialized {len(mappoints)} Gaussians from ORB-SLAM2 and {len(flow_points_3d_np)} Gaussians from Optical Flow on frames 0-1")
                else:
                    points = np.array(mappoints)
                    print(f"[GaussianMapper] Initialized {len(mappoints)} Gaussians from ORB-SLAM2")


                pcd = BasicPointCloud(points, colors=None, normals=np.zeros_like(points))



                self.cameras_extent = 1.0
                self.gaussian_model.create_from_pcd(pcd, self.cameras_extent) # self.train_cameras, self.resolution_scales



                self.gaussian_model.set_coarse_interval(self.op)
                self.gaussian_model.training_setup(self.op)
                self.gaussian_model.update_learning_rate(self.iteration)






                


                print("\n[GaussianMapper] Initial mapping finished. Transitioning to incremental mapping loop.\n")
                break # Exit the first loop


            else:
                print("[GaussianMapper] Waiting for initial mapping conditions...")
                time.sleep(0.1)


        # Remove the mapping operations that correspond to the initialization keyframes
        assert(self.slam_system.get_mapping_operation_queue_size() >= len(initial_keyframe_kfIds))
        for _ in range(len(initial_keyframe_mnIds)):
            self.slam_system.get_and_pop_mapping_operation()

        # Perform some initial 3DGS iterations 
        for i in range(10):
            self.train_for_one_iteration()
            self.psnr_report()

        # Second loop: Incremental Gaussian mapping
        while not self.is_stopped():
            # If there is mapping operation perform these extra operations (in every case there will be a train_for_one_iteration() every time)
            if (self.slam_system.has_mapping_operation()):
                
                start_mapping_operation = time.perf_counter()

                opr = self.slam_system.get_and_pop_mapping_operation()

                # Get keyframes that participated in the Local Bundle adjustement or loop closing optimization
                associated_keyframe_mnIds, associated_keyframe_kfIds, associated_keyframe_poses = opr.get_associated_keyframes()

                if opr.meOperationType == orbslam2.OprType.LocalMappingBA:
                    print(f"[GaussianMapper] Received a MappingOperation of type \033[92m{opr.meOperationType}\033[0m | Associated keyframes: {associated_keyframe_kfIds}")
                elif opr.meOperationType == orbslam2.OprType.LoopClosingBA:
                    print(f"[GaussianMapper] Received a MappingOperation of type \033[92m{opr.meOperationType}\033[0m | Associated keyframes: {associated_keyframe_kfIds}")



                # Add NEW keyframes first, then process to update OLD keyframes AFTER we move the 3D points to new locations (otherwise we lose the old pose)
                new_kfs = []
                for idx in range(len(associated_keyframe_kfIds)):
                    # If the keyframe is a new one, add it in a list
                    if associated_keyframe_kfIds[idx] not in self.processed_kf_ids:
                        new_kf = (associated_keyframe_mnIds[idx], associated_keyframe_kfIds[idx], associated_keyframe_poses[idx])
                        new_kfs.append(new_kf)
                new_kfs.sort(key=lambda x: x[1]) # Sort new keyframes so that kfIds are in order
                
                
                for associated_keyframe_mnId, associated_keyframe_kfId, associated_keyframe_pose in new_kfs:
                    self.add_gaussian_keyframes([associated_keyframe_mnId], [associated_keyframe_kfId], [associated_keyframe_pose])
                    self.processed_kf_ids.append(associated_keyframe_kfId)



                    if (self.gaussian_model.use_multiple_mlp):
                        keyframe1 = self.gaussian_scene.get_keyframes()[-2]
                        keyframe2 = self.gaussian_scene.get_keyframes()[-1]

                        keyframe2.heading = compute_heading(keyframe1, keyframe2)

                        delta = keyframe2.heading - self.gaussian_model.turn_start_heading

                        # Detect completed turns
                        if not self.gaussian_model.turn_in_progress and abs(delta) > self.gaussian_model.turn_threshold_detect:
                            # Turn started
                            self.gaussian_model.turn_in_progress = True
                        elif self.gaussian_model.turn_in_progress and abs(keyframe2.heading - keyframe1.heading) < self.gaussian_model.turn_threshold_end:
                            # Heading stabilized again (turn ended)
                            self.gaussian_model.turn_in_progress = False
                            self.gaussian_model.turn_start_heading = keyframe2.heading

                            turns = self.gaussian_model.turns
                            if (not turns and keyframe2.kfId > self.gaussian_model.turn_min_keyframes) or (turns and (keyframe2.kfId - turns[-1]) > self.gaussian_model.turn_min_keyframes):

                                print("[GaussianMapper] Turn detected. List of turn keyframe IDs: ", self.gaussian_model.turns)
                                self.gaussian_model.turns.append(keyframe2.kfId)
                                
                                print("[GaussianMapper] Adding MLPs. keyframe id is: ", associated_keyframe_kfId)
                                self.gaussian_model.add_mlps(self.op)
                    


                    



                # Add new map_points if it is a Local Mapping Operation or correct positions if it is a Loop Closure Operation
                if (opr.meOperationType == orbslam2.OprType.LocalMappingBA):

                    # Get Mappoints that were added due to the new keyframe
                    associated_mappoints, _, _ = opr.get_associated_mappoints_and_imagepoints()
                    if not associated_mappoints:
                        print("[GaussianMapper] No map points found yet. Waiting...")
                        time.sleep(0.05)
                        continue


                    if (self.use_optical_flow):
                        if (self.gaussian_scene.get_keyframes()[-1].kfId % self.optical_flow_interval == 0):
                            # Define which images to run dense Optical Flow for
                            keyframe1 = self.gaussian_scene.get_keyframes()[-2]
                            keyframe2 = self.gaussian_scene.get_keyframes()[-1]

                            img1 = keyframe1.original_image
                            img2 = keyframe2.original_image


                            # Optical Flow computation
                            start_time_flow = time.perf_counter()
                            flow = self.compute_dense_flow(img2, img1) # We want to compute inverse flow here (assumption for the following)
                            end_time_flow = time.perf_counter()
                            elapsed_time_flow = end_time_flow - start_time_flow
                            print(f"[GaussianMapper] Optical Flow Computation, elapsed time: {elapsed_time_flow:.6f} seconds")

                            # Triangulate points using Optical Flow
                            flow_points_3d_np = compute_extra_3d_points_from_keyframes_and_flow(keyframe1, keyframe2, flow, self.K, num_samples=2000, threshold=2)

                            points = np.concatenate((flow_points_3d_np, np.array(associated_mappoints)), axis=0)

                            print(f"[GaussianMapper] Added {len(associated_mappoints)} Gaussians from ORB-SLAM2 and {len(flow_points_3d_np)} Gaussians from Optical Flow on frames {keyframe1.kfId}-{keyframe2.kfId}")
                        else:
                            points = np.array(associated_mappoints)
                            print(f"[GaussianMapper] Added {len(associated_mappoints)} Gaussians from ORB-SLAM2")
                    else:
                        points = np.array(associated_mappoints)


                    # Create a BasicPointCloud object for the Gaussian Model
                    pcd = BasicPointCloud(points, colors=None, normals=np.zeros_like(points))
                    # Extend the Gaussian Model with new gaussians
                    self.gaussian_model.extend_from_pcd(pcd)




                elif (opr.meOperationType == orbslam2.OprType.LoopClosingBA):
                        self.correct_anchors_after_loop_closure(associated_keyframe_kfIds, associated_keyframe_poses, min_visible_anchors=20)
                
                



                # Update poses in both cases (Local Mapping/Loop Closure)
                for idx in range(len(associated_keyframe_kfIds)):
                    # Convert the list of floats to a numpy array
                    np_pose = np.array(associated_keyframe_poses[idx], dtype=np.float32)
                    # Reshape the 16-element array into a 4x4 matrix
                    se3_pose = np.reshape(np_pose, (4, 4))
                    # Get the keyframes quaternion and translation
                    kf_pose_quaternion = Rotation.from_matrix(se3_pose[:3, :3]).as_quat()
                    kf_pose_translation = se3_pose[:3, 3]

                    target_keyframe = self.gaussian_scene.get_keyframes()[associated_keyframe_kfIds[idx]]
                    target_keyframe.set_pose(kf_pose_quaternion, kf_pose_translation)

                

                end_mapping_operation = time.perf_counter()
                elapsed_time_mapping_operation = end_mapping_operation - start_mapping_operation
                print(f"[GaussianMapper] Finished processing of the MappingOperation, elapsed time: {elapsed_time_mapping_operation:.6f} seconds\033[0m")


            # Perform one optimization step
            self.train_for_one_iteration()
            
            # Check for termination conditions
            if not self.slam_system.is_running():
                print("[GaussianMapper] SLAM system has ended. Finalizing mapping...")
                self.slam_ended = True

            
            if (self.iteration % 100 == 0 or self.slam_ended):
                print("\n================================================================================================================")
                self.psnr_report()
                print("================================================================================================================\n")

            
            if self.slam_ended or self.iteration >= self.op.max_num_iterations:
                print("[GaussianMapper] Termination condition met. Exiting mapping loop.")
                break
        

            
        
        end_total = time.perf_counter()
        elapsed_time_total = end_total - start_total
        print(f"[GaussianMapper] Finished at iteration {self.iteration}. Elapsed time: {BOLD_GREEN_START}{elapsed_time_total:.3f}{RESET_FORMATTING} and mean fps: {BOLD_GREEN_START}{self.gaussian_scene.get_keyframes()[-1].mnId/elapsed_time_total:.3f}{RESET_FORMATTING}")
        
        self.render_final_images()


        print("\n================================================================================================================")
        self.psnr_report()
        self.ssim_report()
        self.lpips_report()
        print("================================================================================================================\n")
        self.ate_report(os.path.join(self.output_dir, "ate_report.json"))