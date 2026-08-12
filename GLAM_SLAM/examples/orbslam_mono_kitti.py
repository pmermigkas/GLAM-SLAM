#!/usr/bin/env python3
import sys
import os.path
import time
import cv2
import threading
import orbslam2
import importlib.util

# This block allows the script to be run from anywhere
# by adding the project root to the Python path.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.GaussianMapper import GaussianMapper



def main(vocab_path, settings_path, sequence_path, gaussian_config_path):

    image_filenames, timestamps = load_images(sequence_path)
    num_images = len(image_filenames)
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.MONOCULAR)
    slam.set_use_viewer(True)
    slam.initialize()

    # --- Create and start the GaussianMapper ---
    gaussian_mapper = GaussianMapper(slam, gaussian_config_path, settings_path, output_dir, sys.argv[3], device='cuda')
    mapper_thread = threading.Thread(target=gaussian_mapper.run)
    mapper_thread.start()
    counter = 0
    try:
        times_track = [0 for _ in range(num_images)]
        print('[ORB-SLAM2 system] Start processing sequence ...')
        print('[ORB-SLAM2 system] Images in the sequence: {0}'.format(num_images))

        for idx in range(num_images):
            image = cv2.imread(image_filenames[idx], cv2.IMREAD_UNCHANGED)
            tframe = timestamps[idx]

            if image is None:
                print("[ORB-SLAM2 system] Failed to load image at {0}".format(image_filenames[idx]))
                return 1

            t1 = time.time()
            slam.process_image_mono(image, tframe)
            t2 = time.time()

            ttrack = t2 - t1
            times_track[idx] = ttrack

            t = 0
            if idx < num_images - 1:
                t = timestamps[idx + 1] - tframe
            elif idx > 0:
                t = tframe - timestamps[idx - 1]

            if ttrack < t:
                time.sleep(t - ttrack)
            counter = idx
        
    except KeyboardInterrupt:
        print("\n\n[ORB-SLAM2 system] Ctrl+C interrupt requested.")

        gaussian_mapper.request_stop()

        mapper_thread.join()
        print("[ORB-SLAM2 system] Gaussian Mapper thread exited successfully.")


    save_trajectory(slam.get_trajectory_points(), os.path.join(output_dir, 'trajectory.txt'))
    print("[ORB-SLAM2 system] Saved ORB-SLAM2 Trajectory in trajectory.txt")

    slam.shutdown()

    return 0


def load_images(path_to_sequence):
    timestamps = []
    with open(os.path.join(path_to_sequence, 'times.txt')) as times_file:
        for line in times_file:
            if len(line) > 0:
                timestamps.append(float(line))

    return [
        os.path.join(path_to_sequence, 'image_2', "{0:06}.png".format(idx))
        for idx in range(len(timestamps))
    ], timestamps


def save_trajectory(trajectory, filename):
    with open(filename, 'w') as traj_file:
        traj_file.writelines('{time} {r00} {r01} {r02} {t0} {r10} {r11} {r12} {t1} {r20} {r21} {r22} {t2}\n'.format(
            time=repr(t),
            r00=repr(r00),
            r01=repr(r01),
            r02=repr(r02),
            t0=repr(t0),
            r10=repr(r10),
            r11=repr(r11),
            r12=repr(r12),
            t1=repr(t1),
            r20=repr(r20),
            r21=repr(r21),
            r22=repr(r22),
            t2=repr(t2)
        ) for t, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 in trajectory)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: ./orbslam_mono_kitti path_to_vocabulary path_to_settings path_to_sequence path_to_gaussian_config')
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
