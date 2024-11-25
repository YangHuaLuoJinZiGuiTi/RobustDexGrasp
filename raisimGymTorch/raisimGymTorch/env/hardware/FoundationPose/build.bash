# CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build.bash
DIR=$(pwd)

cd $DIR/mycpp1/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11

cd ${DIR}