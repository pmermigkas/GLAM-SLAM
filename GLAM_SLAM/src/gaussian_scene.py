import numpy as np
import cv2
import os.path
import time
from scipy.spatial.transform import Rotation

# A class that will hold information about keyframes, like pose, image etc
class GaussianKeyframe():
    def __init__(self, pose_quaternion, pose_translation, mnId, kfId, path_to_sequence, device_type):
        self.set_pose(pose_quaternion, pose_translation)

        self.filename = f"{mnId:06d}.png"
        full_image_path = os.path.join(path_to_sequence, 'image_2', self.filename)
        image = cv2.imread(full_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_image = image.astype(np.float32) / 255.0
        # self.image_height, self.image_width = self.original_image.shape[:2]
        self.mnId = mnId # Id of the frame in ORB-SLAM2
        self.kfId = kfId # Id of the keyframe
        self.device_type = device_type
        
        self.train_counter = 0 # How many times they keyframe has been optimized

        # Used for automatic detection of turns (multiple MLPs)
        self.heading = 0.0 # Cumulative direction of the camera w.r.t. the world frame

        self.psnr_value = 0.0
        self.ssim_value = 0.0
        self.lpips_value = 0.0

    def set_pose(self, pose_quaternion, pose_translation):
        """Updates the base pose and all dependent matrices."""
        self.pose_quaternion = pose_quaternion
        self.pose_translation = pose_translation
        self.pose_R_cw = Rotation.from_quat(self.pose_quaternion).as_matrix()
        self.pose_t_cw = self.pose_translation
        self.pose_R_wc = self.pose_R_cw.T
        self.pose_t_wc = -self.pose_R_wc @ self.pose_t_cw





# A class that holds information about the scene. Initially it will have information only about keyframes
class GaussianScene():
    def __init__(self):
        self.keyframes = []


    def get_keyframes(self):
        return self.keyframes
    
    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)