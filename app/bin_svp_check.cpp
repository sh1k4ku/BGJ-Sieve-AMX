#include <iostream>
#include <fstream>

#include "NTL/LLL.h"

int main(int argc, char **argv) {
    if (argc == 1) {
        printf("Usage: %s <raw_file> <basis_file>\n", argv[0]);
    }
    NTL::Mat<NTL::ZZ> B, R;
    std::ifstream raw_file(argv[1]);
    std::ifstream basis_file(argv[2]);
    raw_file >> R;
    basis_file >> B;
    raw_file.close();
    basis_file.close();
    if (R.NumRows() != B.NumRows() || R.NumCols() != B.NumCols()) {
        printf("failed: dimension mismatch\n");
        return -1;
    }
    long n = R.NumRows();
    long m = R.NumCols();
    for (long i = 0; i < n; i++) {
        NTL::ZZ res;
        res = B[i][0];

        for (long j = 1; j < m; j++) {
            res -= R[j][0] * B[i][j];
        }

        if (res % R[0][0] != 0) {
            printf("failed\n");
            return -1;
        }
    }
    printf("success\n");
    return 0;
}