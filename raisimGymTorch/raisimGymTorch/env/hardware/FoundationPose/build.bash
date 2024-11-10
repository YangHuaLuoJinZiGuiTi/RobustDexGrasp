DIR=$(pwd)

cd $DIR/mycpp1/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11

cd ${DIR}