# prepare for py environment

```Shell
conda create -n robust_dex_grasp python=3.8.10
conda activate robust_dex_grasp
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
## install torch from https://pytorch.org/get-started/previous-versions/
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
## install torch3D linux-64/pytorch3d-0.7.7-py38_cu118_pyt231.tar.bz2 from https://anaconda.org/pytorch3d/pytorch3d/files?page=4
conda install pytorch3d-0.7.7-py38_cu118_pyt231.tar.bz2
```

# python library

```Shell
pip install empy catkin_pkg scipy scikit-learn opencv-python opencv-python pyrealsense2 open3d numpy==1.23.1 kornia transformations psutil imageio segment-anything pynput SpeechRecognition json_repair openai ruamel.yaml omegaconf h5py

git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .
```
    
# system library

```Shell
sudo apt install curl gnupg2 lsb-release
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -
echo "deb http://packages.ros.org/ros/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/ros-latest.list

sudo add-apt-repository ppa:ros-for-jammy/noetic
sudo apt update
sudo apt install ros-noetic-desktop 
export LIB_PKGS_PTH=<your_own_path>
```


## install hpp-fcl
```Shell
cd ${LIB_PKGS_PTH}
git clone https://github.com/leggedrobotics/hpp-fcl.git
cd hpp-fcl && mkdir build && cd build && cmake .. && make -j && sudo make install
```

## install casadi
```Shell
cd ${LIB_PKGS_PTH}
git clone https://github.com/casadi/casadi.git
cd casadi && mkdir build && cd build && cmake .. && make -j && sudo make install
```

## install pinocchio
```Shell
cd ${LIB_PKGS_PTH}
git clone --recursive https://github.com/stack-of-tasks/pinocchio
mkdir pinocchio/build && mkdir pinocchio/install && cd pinocchio/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_WITH_LIBPYTHON=OFF -DCMAKE_INSTALL_PREFIX=../install && make -j && make install
```

## install rtde
```Shell
sudo add-apt-repository ppa:sdurobotics/ur-rtde
sudo apt-get update
sudo apt install librtde librtde-dev
```

## install allegro
```Shell
sudo apt install libpopt-dev libxmlrpcpp-dev ros-noetic-libcan librospack-dev librosconsole-dev

# https://www.peak-system.com/fileadmin/media/linux/index.php 
# https://www.peak-system.com/fileadmin/media/linux/can-version-history.php 8.20 is ok
# regist the gcc and switch 12 to build driver
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
# sudo update-alternatives --config gcc

cd ${LIB_PKGS_PTH}
tar -xzvf peak-linux-driver-x.x.tar.gz
cd peak-linux-driver-x.x && make NET=NO && sudo make install && sudo modprobe pcan

# https://www.peak-system.com/fileadmin/media/linux/can-pcan-basic.php
cd ${LIB_PKGS_PTH}
tar -xzvf PCAN_Basic_Linux-x.x.x.tar.gz
cd PCAN_Basic_Linux-x.x.x/pcanbasic && make && sudo make install

# https://www.allegrohand.com/v4-main/grasping-library-for-linux
cd ${LIB_PKGS_PTH}
unzip LibBHand_64.zip
cd LibBHand_64 && sudo make install

sudo vim /etc/udev/rules.d/97-allegro-usb.rules
SUBSYSTEMS=="usb",ATTRS{idVendor}=="0c72", ATTRS{idProduct}=="000c", MODE:="0777", SYMLINK+="allegro"
sudo vim /etc/udev/rules.d/96-leaphand-usb.rules
SUBSYSTEMS=="usb", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", MODE:="0777", SYMLINK+="leaphand"
sudo wget -O /etc/udev/rules.d/99-realsense-libusb.rules https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules

```

# compile
change the cmake flag in raisimGymTorch/CMakeLists.txt[this folder](./raisimGymTorch/CMakeLists.txt).  to build the hardware layer

set(BUILD_UR5_REAL          ON)

set(BUILD_ALLEGRO_REAL      ON)

set(BUILD_PINOCCHIO         ON)

# evaluate
run the code of real world evaluation
```Shell
cd raisimGymTorch 
python raisimGymTorch/env/envs/allegro_real/real.py
```


# for ubuntu22.04
```Shell
sudo ln -s /usr/lib/x86_64-linux-gnu/libdl.so.2 /usr/lib/x86_64-linux-gnu/libdl.so
```