#!/bin/bash
ROOT_PATH="$PWD"

# Check if an argument was provided and if it's either Debug or Release
if [ -z "$1" ] || [[ "$1" != "Debug" && "$1" != "Release" ]]; then
  echo "Usage: $0 <build_type>"
  echo "Build types: Debug or Release"
  exit 1
fi

BUILD_TYPE="$1"

# Define a function to handle build directories and commands
build_project() {
    local project_path=$1
    local build_dir="build"

    echo "Configuring and building $project_path with type: $BUILD_TYPE"
    cd "$project_path"
    mkdir -p "build_$BUILD_TYPE"
    cd "build_$BUILD_TYPE"
    cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCMAKE_CXX_FLAGS="-w" -DPangolin_DIR=/path/to/Pangolin/build
    make -j4
    cd $ROOT_PATH > /dev/null
}



# Clone ORB_SLAM2
cd "$ROOT_PATH"

# Clone ORB_SLAM2 only if the directory does not exist
if [ ! -d "ORB_SLAM2" ]; then
    echo "Cloning original ORB-SLAM2..."
    git clone https://github.com/raulmur/ORB_SLAM2.git
else
    echo "ORB_SLAM2 directory already exists. Skipping clone."
fi

ORB_SLAM_COMMIT="f2e6f51"
cd ORB_SLAM2

# Apply patch only if our hidden marker file does not exist
if [ ! -f ".patch_applied" ]; then
    echo "Locking ORB-SLAM2 to commit $ORB_SLAM_COMMIT..."
    git checkout "$ORB_SLAM_COMMIT"

    echo "Applying patch to ORB_SLAM2..."
    git apply ../ORB_SLAM2_patch/custom_orbslam2.patch
    
    echo "Copying additional ORB-SLAM2 config files..."
    cp ../ORB_SLAM2_patch/*.yaml Examples/Monocular/

    # Create the marker file so the script knows to skip this next time
    touch .patch_applied
    echo "Patch applied successfully."
else
    echo "Patch has already been applied. Skipping."
fi

# --- Build ORB_SLAM2 Third-party Libraries ---
cd $ROOT_PATH
build_project "ORB_SLAM2/Thirdparty/DBoW2"
cd $ROOT_PATH
build_project "ORB_SLAM2/Thirdparty/g2o"


# --- Uncompress vocabulary ---
echo "Uncompress vocabulary ..."
cd $ROOT_PATH/ORB_SLAM2/Vocabulary
tar -xf ORBvoc.txt.tar.gz


# --- Build ORB_SLAM2 ---
cd $ROOT_PATH
build_project "ORB_SLAM2"
# --- Install ORB_SLAM2 ---
echo "Installing ORB_SLAM2 with sudo make install ..."
cd ORB_SLAM2/"build_$BUILD_TYPE"
sudo make install




# --- Build ORB_SLAM2-PythonBindings ---
cd $ROOT_PATH
build_project "ORB_SLAM2-PythonBindings"
# --- Install ORB_SLAM2-PythonBindings ---
echo "Installing ORB_SLAM2-PythonBindings..."
cd ORB_SLAM2-PythonBindings/"build_$BUILD_TYPE"
sudo make install

# ldconfig so that Linux Dynamic Linker sees the installed .so files in usr/local/lib
sudo ldconfig


echo "Build and installation complete."

cd $ROOT_PATH > /dev/null
