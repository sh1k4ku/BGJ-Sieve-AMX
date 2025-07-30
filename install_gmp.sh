GMP_INSTALL_PREFIX=$(pwd)/dep/gmp
NUM_CORES=8

if [ -f ${GMP_INSTALL_PREFIX}/lib/libgmp.a ] && [ -f ${GMP_INSTALL_PREFIX}/include/gmp.h ]; then
	echo "it seems that gmp has been installed correctly."
	exit 0
fi

cd dep
mkdir gmp
wget https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz
cd ..


if [ ! -f ${GMP_INSTALL_PREFIX}/../gmp-6.2.1.tar.xz ]; then
	echo "Error: archive corrupted"
	exit 1
fi

rm -rf ${GMP_INSTALL_PREFIX}/*
rm -rf ${GMP_INSTALL_PREFIX}/../gmp-6.2.1

cd dep
tar -xf gmp-6.2.1.tar.xz
cd gmp-6.2.1
./configure --prefix=${GMP_INSTALL_PREFIX}
make -j ${NUM_CORES}
make install
