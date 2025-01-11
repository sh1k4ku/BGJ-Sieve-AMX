/**  
*   \file   bin_pump_red.cpp
*   \brief  A general lattice reduction tool, use bkz with 
*           bgjf left progressive sieve based pump. lsf
*           based hash insert is disabled.
*/

#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>



#include "../include/lattice.h"
#include "../include/svp.h"


#if COMPILE_POOL_EPI8_160
#define MAX_PUMP_DIM 160
long (*_pump_red_epi8[104])(Lattice_QP*, long) = {
    _pump_red_epi8_60, _pump_red_epi8_61, _pump_red_epi8_62, _pump_red_epi8_63, _pump_red_epi8_64, _pump_red_epi8_65, _pump_red_epi8_66, _pump_red_epi8_67, _pump_red_epi8_68, _pump_red_epi8_69,
    _pump_red_epi8_70, _pump_red_epi8_71, _pump_red_epi8_72, _pump_red_epi8_73, _pump_red_epi8_74, _pump_red_epi8_75, _pump_red_epi8_76, _pump_red_epi8_77, _pump_red_epi8_78, _pump_red_epi8_79,
    _pump_red_epi8_80, _pump_red_epi8_81, _pump_red_epi8_82, _pump_red_epi8_83, _pump_red_epi8_84, _pump_red_epi8_85, _pump_red_epi8_86, _pump_red_epi8_87, _pump_red_epi8_88, _pump_red_epi8_89,
    _pump_red_epi8_90, _pump_red_epi8_91, _pump_red_epi8_92, _pump_red_epi8_93, _pump_red_epi8_94, _pump_red_epi8_95, _pump_red_epi8_96, _pump_red_epi8_97, _pump_red_epi8_98, _pump_red_epi8_99,
    _pump_red_epi8_100, _pump_red_epi8_101, _pump_red_epi8_102, _pump_red_epi8_103, _pump_red_epi8_104, _pump_red_epi8_105, _pump_red_epi8_106, _pump_red_epi8_107, _pump_red_epi8_108, _pump_red_epi8_109,
    _pump_red_epi8_110, _pump_red_epi8_111, _pump_red_epi8_112, _pump_red_epi8_113, _pump_red_epi8_114, _pump_red_epi8_115, _pump_red_epi8_116, _pump_red_epi8_117, _pump_red_epi8_118, _pump_red_epi8_119,
    _pump_red_epi8_120, _pump_red_epi8_121, _pump_red_epi8_122, _pump_red_epi8_123, _pump_red_epi8_124, _pump_red_epi8_125, _pump_red_epi8_126, _pump_red_epi8_127, _pump_red_epi8_128, _pump_red_epi8_129,
    _pump_red_epi8_130, _pump_red_epi8_131, _pump_red_epi8_132, _pump_red_epi8_133, _pump_red_epi8_134, _pump_red_epi8_135, _pump_red_epi8_136, _pump_red_epi8_137, _pump_red_epi8_138, _pump_red_epi8_139,
    _pump_red_epi8_140, _pump_red_epi8_141, _pump_red_epi8_142, _pump_red_epi8_143, _pump_red_epi8_144, _pump_red_epi8_145, _pump_red_epi8_146, _pump_red_epi8_147, _pump_red_epi8_148, _pump_red_epi8_149,
    _pump_red_epi8_150, _pump_red_epi8_151, _pump_red_epi8_152, _pump_red_epi8_153, _pump_red_epi8_154, _pump_red_epi8_155, _pump_red_epi8_156, _pump_red_epi8_157, _pump_red_epi8_158, _pump_red_epi8_159, _pump_red_epi8_160
};
#elif COMPILE_POOL_EPI8_128
#define MAX_PUMP_DIM 128
long (*_pump_red_epi8[104])(Lattice_QP*, long) = {
    _pump_red_epi8_60, _pump_red_epi8_61, _pump_red_epi8_62, _pump_red_epi8_63, _pump_red_epi8_64, _pump_red_epi8_65, _pump_red_epi8_66, _pump_red_epi8_67, _pump_red_epi8_68, _pump_red_epi8_69,
    _pump_red_epi8_70, _pump_red_epi8_71, _pump_red_epi8_72, _pump_red_epi8_73, _pump_red_epi8_74, _pump_red_epi8_75, _pump_red_epi8_76, _pump_red_epi8_77, _pump_red_epi8_78, _pump_red_epi8_79,
    _pump_red_epi8_80, _pump_red_epi8_81, _pump_red_epi8_82, _pump_red_epi8_83, _pump_red_epi8_84, _pump_red_epi8_85, _pump_red_epi8_86, _pump_red_epi8_87, _pump_red_epi8_88, _pump_red_epi8_89,
    _pump_red_epi8_90, _pump_red_epi8_91, _pump_red_epi8_92, _pump_red_epi8_93, _pump_red_epi8_94, _pump_red_epi8_95, _pump_red_epi8_96, _pump_red_epi8_97, _pump_red_epi8_98, _pump_red_epi8_99,
    _pump_red_epi8_100, _pump_red_epi8_101, _pump_red_epi8_102, _pump_red_epi8_103, _pump_red_epi8_104, _pump_red_epi8_105, _pump_red_epi8_106, _pump_red_epi8_107, _pump_red_epi8_108, _pump_red_epi8_109,
    _pump_red_epi8_110, _pump_red_epi8_111, _pump_red_epi8_112, _pump_red_epi8_113, _pump_red_epi8_114, _pump_red_epi8_115, _pump_red_epi8_116, _pump_red_epi8_117, _pump_red_epi8_118, _pump_red_epi8_119,
    _pump_red_epi8_120, _pump_red_epi8_121, _pump_red_epi8_122, _pump_red_epi8_123, _pump_red_epi8_124, _pump_red_epi8_125, _pump_red_epi8_126, _pump_red_epi8_127, _pump_red_epi8_128
};
#else
#define MAX_PUMP_DIM 96
long (*_pump_red_epi8[104])(Lattice_QP*, long) = {
    _pump_red_epi8_60, _pump_red_epi8_61, _pump_red_epi8_62, _pump_red_epi8_63, _pump_red_epi8_64, _pump_red_epi8_65, _pump_red_epi8_66, _pump_red_epi8_67, _pump_red_epi8_68, _pump_red_epi8_69,
    _pump_red_epi8_70, _pump_red_epi8_71, _pump_red_epi8_72, _pump_red_epi8_73, _pump_red_epi8_74, _pump_red_epi8_75, _pump_red_epi8_76, _pump_red_epi8_77, _pump_red_epi8_78, _pump_red_epi8_79,
    _pump_red_epi8_80, _pump_red_epi8_81, _pump_red_epi8_82, _pump_red_epi8_83, _pump_red_epi8_84, _pump_red_epi8_85, _pump_red_epi8_86, _pump_red_epi8_87, _pump_red_epi8_88, _pump_red_epi8_89,
    _pump_red_epi8_90, _pump_red_epi8_91, _pump_red_epi8_92, _pump_red_epi8_93, _pump_red_epi8_94, _pump_red_epi8_95, _pump_red_epi8_96
};
#endif

long get_continue(int argc, char *argv[]){
    long cont = 0;
    for (long i = 1; i < argc; i++){
        if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "-continue")) cont = 1;
    }
    return cont;
}
long get_num_threads(int argc, char *argv[]){
    long num_threads = 0;
    for (long i = 1; i < argc; i++){
        if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "-threads") || !strcmp(argv[i], "-thread")){
            if (i + 1 < argc) num_threads = atol(argv[i+1]);
        }
    }
    return num_threads;
}

int main (int argc, char *argv[]) {
    if (!get_continue(argc, argv)){
        Lattice_QP Ls(argv[1]);
        Ls.store("L_tmp.txt");
        std::ofstream sround(".cround", std::ios::trunc);
        std::ofstream sindex(".cindex", std::ios::trunc);
        std::ofstream sstop(".cstop", std::ios::trunc);
        std::ofstream sblocksize(".cblocksize", std::ios::trunc);
        std::ofstream sadjust(".cadjust", std::ios::trunc);
        sadjust << "0" << std::endl;
        sround << "0" << std::endl;
        sindex << "-8" << std::endl;
        sstop << Ls.NumRows() << std::endl;
        sblocksize << "0" << std::endl;
    }
    if (get_num_threads(argc, argv)){
        std::ofstream snumthreads(".cnumthreads", std::ios::trunc);
        snumthreads << get_num_threads(argc, argv) << std::endl;
    }
    Lattice_QP L("L_tmp.txt");
    std::ifstream cround_(".cround", std::ios::in);
    std::ifstream cindex_(".cindex", std::ios::in);
    long cindex, cround;
    cindex_ >> cindex;
    cround_ >> cround;

    while (true) {
        long dim = L.pump_red_msd();
        std::ifstream cblocksize_(".cblocksize", std::ios::in);
        std::ifstream cstop_(".cstop", std::ios::in);
        std::ifstream cnumthreads_(".cnumthreads", std::ios::in);
        std::ifstream cadjust_(".cadjust", std::ios::in);
        long cadjust = 0;
        long cnumthreads = 0;
        long cblocksize = 0;
        long cstop = L.NumRows();
        cstop_ >> cstop;
        cblocksize_ >> cblocksize;
        cnumthreads_ >> cnumthreads;
        cadjust_ >> cadjust;
        dim += cadjust;
        if (cnumthreads == 0) cnumthreads = 8;
        if (dim < cblocksize) dim = cblocksize;
        if (dim < 60) dim = 60;
        if (dim > MAX_PUMP_DIM) {
            std::cerr << "[Warning] expect sieving dimension is " << dim << ", use dim = " << MAX_PUMP_DIM << " instead\n";
            dim = MAX_PUMP_DIM;
        }
        std::cout << "================current dim " << dim  << "================"<< std::endl;
        long blocksize = dim + (((dim) + 8) / 16 + 13);
        L.BKZ_tour(blocksize, cnumthreads, _pump_red_epi8[dim - 60], cindex, cstop, cindex, cround);
    }
	return 0;
}