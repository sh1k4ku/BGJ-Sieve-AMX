OBJ = config.o pool_epi8.o bucket_epi8.o bgj_epi8.o 3Sieve_parallel.o naivedh_epi8.o naivesh_epi8.o fcSieve.o bgj1.o bgjf32.o dual.o
OBJ += vec.o quad.o utils.o lattice.o lll.o UidHashTable.o pool.o 3Sieve.o svp.o bkz.o sampler.o lsfdh_epi8.o lsfsh_epi8.o
OBJ += bgj_amx.o bucket_amx.o booster_amx.o
CXX = clang++
CFLAGS = -O3 -g -fopenmp=libomp -march=native -pthread -stdlib=libc++ -Wno-deprecated-declarations
LIBS = -L../dep/ntl/lib -lntl -L../dep/gmp/lib -lgmp -lm
LIBDIR = 
INC_DIR = -I../include/ -I../dep/ntl/include
ON_CYGWIN = false

ifeq ($(ON_CYGWIN), true)
	CFLAGS += -D__ON_CYGWIN
endif

all: libllib.a

libllib.a: $(OBJ)
	ar rcs libllib.a $(OBJ)

%.o: %.cpp 
	$(CXX) $(CFLAGS) $^ $(INC_DIR) $(LIBDIR) -c

clean:
	-rm $(OBJ) libllib.a

