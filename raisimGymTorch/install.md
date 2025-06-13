# prepare for py environment
    conda create -n robust_dex_grasp python=3.8.10
    conda activate robust_dex_grasp
    ## install cuda11.8 in ubuntu20.04
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    ## install torch from https://pytorch.org/get-started/previous-versions/
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
    ## install torch3D linux-64/pytorch3d-0.7.7-py38_cu118_pyt231.tar.bz2 from https://anaconda.org/pytorch3d/pytorch3d/files?page=4
    conda install pytorch3d-0.7.7-py38_cu121_pyt231.tar.bz2

# python library
    pip install empy catkin_pkg

# system library
    sudo apt install ros-noetic-hpp-fcl 

    ## install casadi
    git clone git@github.com:casadi/casadi.git
    cd casadi && mkdir build && cd build && cmake .. && make -j && sudo make install

    ## install pinocchio and do not build python in this source
    git clone --recursive https://github.com/stack-of-tasks/pinocchio
    mkdir pinocchio/build && mkdir pinocchio/install && cd pinocchio/build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_WITH_LIBPYTHON=OFF -DCMAKE_INSTALL_PREFIX=../install && make -j && make install

    ## install rtde
    sudo add-apt-repository ppa:sdurobotics/ur-rtde
    sudo apt-get update
    sudo apt install librtde librtde-dev