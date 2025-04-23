####### prepare
export install_pth=/home/ubuntu/pkgs/bundlesdf_lib

# glog 0.4.0

# opencv
cd $install_pth && git clone -b 3.4.15 https://github.com/opencv/opencv opencv-3.4.15 && mkdir opencv-3.4.15/build opencv-3.4.15/install
cd $install_pth && git clone -b 3.4.15 https://github.com/opencv/opencv_contrib.git opencv_contrib-3.4.15
cd $install_pth/opencv-3.4.15/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA_STUBS=OFF -DBUILD_DOCS=OFF -DWITH_MATLAB=OFF -Dopencv_dnn_BUILD_TORCH_IMPORTE=OFF -DCUDA_FAST_MATH=ON  -DMKL_WITH_OPENMP=ON -DOPENCV_ENABLE_NONFREE=ON -DWITH_OPENMP=ON -DWITH_QT=ON -WITH_OPENEXR=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_opencv_cudacodec=OFF -DINSTALL_PYTHON_EXAMPLES=OFF  -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.11.0/modules -DCMAKE_CXX_FLAGS=-std=c++11 -DENABLE_CXX11=OFF  -DBUILD_opencv_xfeatures2d=ON -DOPENCV_DNN_OPENCL=OFF -DWITH_CUDA=ON -DWITH_OPENCL=OFF -DPYTHON_EXECUTABLE=/home/ubuntu/anaconda3/envs/grasp/bin/python3.8 -DPYTHON3_EXECUTABLE=/home/ubuntu/anaconda3/envs/grasp/bin/python3.8 -DPYTHON3_LIBRARY=/home/ubuntu/anaconda3/envs/grasp/lib/python3.8 -DPYTHON3_INCLUDE_DIR=/home/ubuntu/anaconda3/envs/grasp/include/python3.8 -DBUILD_opencv_python3=ON -DPYTHON3_PACKAGES_PATH=/home/ubuntu/anaconda3/envs/grasp/lib/python3.8/site-packages -DCMAKE_INSTALL_PREFIX="../install" && make -j18 && make install

# pcl
cd $install_pth && git clone -b pcl-1.10.0 https://github.com/PointCloudLibrary/pcl pcl-1.10.0 && mkdir pcl-1.10.0/build pcl-1.10.0/install
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_apps=OFF -DBUILD_GPU=OFF  -DBUILD_CUDA=OFF -DBUILD_examples=OFF -DBUILD_global_tests=OFF -DBUILD_simulation=OFF -DCUDA_BUILD_EMULATION=OFF -DCMAKE_CXX_FLAGS=-std=c++11 -DPCL_ENABLE_SSE=ON -DPCL_SHARED_LIBS=ON -DWITH_VTK=OFF -DCMAKE_INSTALL_PREFIX="../install" && make -j16 && make install

# pybind11
cd $install_pth && git clone -b v2.10.0 https://github.com/pybind/pybind11 pybind11-2.10.0 && mkdir pybind11-2.10.0/build pybind11-2.10.0/install
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF -DCMAKE_INSTALL_PREFIX="../install" && make -j16 && make install

# yaml-cpp
cd $install_pth && git clone -b yaml-cpp-0.7.0 https://github.com/jbeder/yaml-cpp yaml-cpp-0.7.0 && mkdir yaml-cpp-0.7.0/build yaml-cpp-0.7.0/install
cmake .. -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DINSTALL_GTEST=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_BUILD_SHARED_LIBS=ON  -DCMAKE_INSTALL_PREFIX="../install" && make -j6 && make install

#kaolin
cd $install_pth && git clone --recursive https://github.com/NVIDIAGameWorks/kaolin && cd $install_pth/kaolin && pip install -e .

# bundletrack
cd $install_pth && git clone https://github.com/NVlabs/BundleSDF.git && cd $install_pth/BundleSDF/BundleTrack && rm -rf build bin && mkdir build bin && cd build && cmake .. && make -j10

# build file
cp -rf $install_pth/BundleSDF/BundleTrack $pwd
