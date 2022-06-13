rm -rf checkpoints/
rm -rf CMake*
rm -rf __pycache__
rm -rf *.npy
rm -rf *.png
rm -rf cmake_install.cmake
rm -rf Makefile
rm -rf app

cmake ../

make

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1