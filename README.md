<!-- Header layout adapted from image_2e6da5.png -->

<div align="center">
  <h1>GLAM-SLAM: Real-time Gaussian Large-scale Mapping via Flow Densification and Spatial Decomposition</h1>
  <p>
    Panagiotis Mermigkas ·
    Argyris Manetas ·
    Petros Maragos
  </p>
  
  <h3>IROS 2026</h3>
  
  <p>
    <a href="https://arxiv.org/abs/2607.21416">Paper</a> |
    <!-- <a href="#">Video</a> | -->
    <a href="https://glamslam.github.io/">Project Page</a>
  </p>
</div>

---





## 🛠️ Installation Guide

### 1. [ORB-SLAM2](https://github.com/raulmur/orb_slam2) and [ORB-SLAM2-PythonBindings](https://github.com/jskinn/ORB_SLAM2-PythonBindings) Setup

We have modified the original codes and provide the instructions to build the versions we use.

Before setting up the project, ensure you have the required system-level dependencies installed:

*   **CUDA Toolkit (12.8.1):** Download and install the appropriate `deb (local)` package for Ubuntu 24.04 from the [NVIDIA Archive](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local).
*   **Eigen3:** 
    ```bash
    sudo apt update
    sudo apt install libeigen3-dev
    ```
*   **OpenCV:** Version 4.12
*   **Pangolin (v0.5 branch):** 
    Clone the repository, checkout the `0.5` branch, and disable FFMPEG during the CMake configuration to prevent build errors:
    ```bash
    git clone https://github.com/stevenlovegrove/Pangolin.git
    cd Pangolin
    git checkout 0.5
    mkdir build && cd build
    cmake .. -DBUILD_PANGOLIN_FFMPEG=OFF
    make -j$(nproc)
    ```


Update `./build_glam_slam.sh` script (inside build_project() function) to include:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPangolin_DIR=/path/to/Pangolin/build
```
so that it explicitly points to your Pangolin build directory.

Then, run the provided script
```bash
./build_glam_slam.sh Release
```

### 2. Conda Environment Creation and Python Requirements (`requirements.yml`)
Create and activate a dedicated Python environment:
```bash
conda create --name glam_slam python=3.10.12 -y
conda activate glam_slam
```




Install the provided `requirements.yml` using:
```bash
conda env update -f requirements.yml
```

### 3. LiteFlowNet3 Setup
We use LiteFlowNet3 for optical flow computation. 

1. Follow the official instructions (Prerequisite and Compiling) from [LiteFlowNet](https://github.com/twhui/LiteFlowNet) (v1)
2. Link the compiled Caffe library to your Conda environment:
   ```bash
   # Copy the python package
   cp -r /path/to/LiteFlowNet/python/caffe $CONDA_PREFIX/lib/python3.10/site-packages
   
   # Copy the shared objects
   cp -P /path/to/LiteFlowNet/build/lib/libcaffe.so* $CONDA_PREFIX/lib
   
   # Map the library path persistently in Conda
   conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
   
   # Reactivate environment to apply changes
   conda deactivate
   conda activate glam_slam
   ```
3. Download the model weights `LiteFlowNet3-S-ft-kitti.caffemodel` from [LiteFlowNet3](https://github.com/twhui/LiteFlowNet3/tree/master/models/testing) and place it in /path/to/GLAM_SLAM/repo/GLAM_SLAM/cfg/liteflow_models/.

### 4. Build GLAM-SLAM Dependencies

Navigate to `/path/to/GLAM_SLAM/repo/GLAM_SLAM/submodules` and install the three Python submodules without build isolation to ensure they link correctly with your environment variables:
```bash
cd diff_gaussian_rasterization
pip install . --no-build-isolation
cd ../fused-ssim
pip install . --no-build-isolation
cd ../simple-knn
pip install . --no-build-isolation
```

### 5. Execute GLAM-SLAM

```bash
clear && cd /path/to/GLAM_SLAM/repo/ && python GLAM_SLAM/examples/orbslam_mono_kitti.py ORB_SLAM2/Vocabulary/ORBvoc.txt ORB_SLAM2/Examples/Monocular/KITTI00-02.yaml /path/to/KITTI/dataset/sequences/00 GLAM_SLAM/cfg/gaussian_scaffold/kitti_mono/kitti_mono.yaml
```

> **⚠️ Note on gsplat:**  
> The first time you run GLAM-SLAM, `gsplat` will compile its CUDA kernels using `MAX_JOBS=10`. This process may take a few minutes. Allow it to complete fully before actually running the system.



### GLAM-SLAM parameters
#### Localized MLP initialization
*   **`use_multiple_mlp`** (`bool`): Enables the Localized MLP initialization
*   **`turn_threshold_detect`** / **`turn_threshold_end`** (`float`): Degrees of angular change required to trigger (e.g., `45.0`) or conclude (e.g., `1.0`) a new turn
*   **`turn_min_keyframes`** (`int`): Minimum number of frames before a new turn can be introduced

#### Flow densification
*   **`use_optical_flow`** (`bool`): Enables Flow densification
*   **`fpath_prototxt`** / **`fpath_caffemodel`** (`string`): Absolute paths to the network's Caffe architecture and weights files
*   **`optical_flow_interval`** (`int`): Computes flow every *N* frames (e.g., `7`)

#### Keyframe selection policy
*   **`optimize_num_last_keyframes`** (`int`): The number of the most recent keyframes forming the "local window" for 3DGS optimization
*   **`optimize_prob_last_keyframes`** (`float`): Probability (e.g., `0.7` = 70%) of sampling keyframes from the local window instead of earlier ones