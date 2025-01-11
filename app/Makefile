BIN = bgj_epi8 svp_solver pump_red svp_tool svp_check
CXX = clang++
CFLAGS = -O3 -g -funroll-loops -ftree-vectorize -fopenmp=libomp -march=native -stdlib=libc++ -pthread -Wno-deprecated-declarations
LIBS = -L../src -lllib -L../dep/ntl/lib -lntl -L../dep/gmp/lib -lgmp -lm
LIBDIR =
INC_DIR = -I../include/ -I../dep/ntl/include

ON_CYGWIN = false

ifeq ($(ON_CYGWIN), true)
	CFLAGS += -D__ON_CYGWIN
endif

all: $(BIN)

clean:
	-rm $(BIN)

%: bin_%.cpp ../src/libllib.a
	$(CXX) $(CFLAGS) $^ $(INC_DIR) $(LIBDIR) $(LIBS) -o $*
