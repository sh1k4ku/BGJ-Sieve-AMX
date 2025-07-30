NTL_INSTALL_PREFIX=$(pwd)/dep/ntl
GMP_INSTALL_PREFIX=$(pwd)/dep/gmp
NUM_CORES=8

if [ -f ${NTL_INSTALL_PREFIX}/lib/libntl.a ] && [ -d ${NTL_INSTALL_PREFIX}/include/NTL ]; then
	echo "it seems that ntl has been installed correctly."
	exit 0
fi

cd dep
mkdir ntl
wget https://libntl.org/ntl-9.1.0.tar.gz
cd ..

if [ ! -f ${NTL_INSTALL_PREFIX}/../ntl-9.1.0.tar.gz ]; then
	echo "Error: archive corrupted"
	exit 1
fi

rm -rf ${NTL_INSTALL_PREFIX}/*
rm -rf ${NTL_INSTALL_PREFIX}/../ntl-9.1.0

cd dep
tar -zxvf ntl-9.1.0.tar.gz
cd ntl-9.1.0/src
./configure PREFIX=${NTL_INSTALL_PREFIX} GMP_PREFIX=${GMP_INSTALL_PREFIX} "CXX=clang++" "CXXFLAGS=-g -O3 -ftree-vectorize -mpclmul -march=native -Wno-error=register -stdlib=libc++ -pthread -L${GMP_INSTALL_PREFIX}/lib -lgmp" NTL_THREADS=on NTL_GF2X_LIB=off
make -j ${NUM_CORES}
make install

