#include "../include/lattice.h"
#include "../include/utils.h"
#include "../include/config.h"
#include "../include/svp.h"
#include <fstream>
#include <string>
#include <sys/time.h>


double Lattice_QP::BKZ_tour(long blocksize, long num_threads, long (*red)(Lattice_QP*, long), long ind_l, long ind_r, long& cind, long& crnd){
    long current_index = max(ind_l, -blocksize);
    long current_index_l = max(current_index, 0);
    long current_index_r = min(current_index_l + GSO_BLOCKSIZE, n);
    long gso_step = GSO_BLOCKSIZE - blocksize;
    double total_time = 0.0;
    double red_time = 0.0;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    double pot = this->Pot();
    //main loop
    while (current_index_l < min(ind_r + blocksize, n) - GSO_BLOCKSIZE){
        Lattice_QP *L_loc = this->b_loc_QP(current_index_l, current_index_r);
        while (current_index + blocksize < current_index_r){
            long index_l = max(current_index-current_index_l, 0);
            long index_r = current_index - current_index_l + blocksize;
            Lattice_QP *L_loc_loc = L_loc->b_loc_QP(index_l, index_r);
            struct timeval red_start, red_end;
	        gettimeofday(&red_start, NULL);
            std::cerr << "working on index " << current_index << ", ";
            long step = (*red)(L_loc_loc, num_threads);
            gettimeofday(&red_end, NULL);
            red_time += (red_end.tv_sec-red_start.tv_sec)+(double)(red_end.tv_usec-red_start.tv_usec)/1000000.0;
            L_loc->trans_to(index_l, index_r, L_loc_loc);
            L_loc->compute_gso_QP();
            L_loc->size_reduce();
            L_loc->LLL_QP();
            L_loc->compute_gso_QP();
            current_index += step;
        }
        this->trans_to(current_index_l, current_index_r, L_loc);
        this->compute_gso_QP();
        this->size_reduce();
        this->LLL_QP();
        this->compute_gso_QP();
        this->show_dist_vec();

        //store
        do {
            this->store(("L_tmp" + std::to_string(crnd) +"_" + std::to_string(current_index) + ".txt").c_str());
            this->store("L_tmp.txt");
            std::ofstream hout (".cindex", std::ios::trunc);
            hout << current_index << std::endl;
        } while (0);
        current_index_r = min(current_index_r + gso_step, n);
        current_index_l = max(current_index_r - GSO_BLOCKSIZE, 0);
    }

    //tail processing
    do {
        Lattice_QP *L_loc = this->b_loc_QP(current_index_l, current_index_r);
        while (current_index < min(ind_r, n - 20)) {
            long index_l = max(current_index-current_index_l, 0);
            long index_r = min(current_index-current_index_l+blocksize, current_index_r-current_index_l);
            Lattice_QP *L_loc_loc = L_loc->b_loc_QP(index_l, index_r);
            struct timeval red_start, red_end;
	        gettimeofday(&red_start, NULL);
            std::cerr << "working on index " << current_index << ", ";
            
            
            long step = (*red)(L_loc_loc, num_threads);
            gettimeofday(&red_end, NULL);
            red_time += (red_end.tv_sec-red_start.tv_sec)+(double)(red_end.tv_usec-red_start.tv_usec)/1000000.0;
            L_loc->trans_to(index_l, index_r, L_loc_loc);
            L_loc->compute_gso_QP();
            L_loc->size_reduce();
            L_loc->LLL_QP();
            L_loc->compute_gso_QP();
            current_index += step;
        }
        this->trans_to(current_index_l, current_index_r, L_loc);
        this->compute_gso_QP();
        this->size_reduce();
        this->LLL_QP();
        this->compute_gso_QP();
        this->show_dist_vec();
    } while (0);

    gettimeofday(&end, NULL);
	total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    std::cerr << "BKZ_tour done: total_time = "<< (total_time + (num_threads - 1) * red_time) <<"s, red_time = ";
    std::cerr << (num_threads * red_time) << "s, dPot = " << (pot-this->Pot()) << "\n";
    //store
    do {
        this->store(("L_tmp" + std::to_string(crnd) + ".txt").c_str());
        this->store("L_tmp.txt");
        crnd++;
        cind = -8;
        std::ofstream hout (".cindex", std::ios::trunc);
        hout << (-8) << std::endl;
        std::ofstream gout (".cround", std::ios::trunc);
        gout << crnd << std::endl;
    } while (0);
    return (pot - this->Pot())/(total_time + (num_threads - 1) * red_time);
}

// the svp subroutine used in pump based bkz
long _sred(Lattice_QP *L, long msd, long d4f, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 40) {
        __hkz_red(L, num_threads);
        if (L->NumRows() < 32) return 32;
        return JUMPING_STEP;
    }
    long _msd = msd;
    long _d4f = d4f;
    if (msd < 40) _msd = 40;
    _d4f = msd + d4f - _msd; 
    __pump_red(L, num_threads, 1.10, _msd, _d4f, 24, 0, 24);
    return JUMPING_STEP;
}


long _pump_epi8_red(Lattice_QP *L, long msd, long d4f, long num_threads, long log_level, long minsd){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < minsd) {
        __hkz_red(L, num_threads);
        if (L->NumRows() < 32) return 32;
        return JUMPING_STEP;
    }
    long _msd = msd;
    long _d4f = d4f;
    if (msd < minsd) _msd = minsd;
    _d4f = msd + d4f - _msd; 
    __pump_red_epi8(L, num_threads, 1.10, _msd, _d4f, 24, 0, 24, log_level, 1, minsd);
    return JUMPING_STEP;
}

double Lattice_QP::BKZ_tour(long msd, long d4f, long num_threads, long ind_l, long ind_r){
    const long blocksize = d4f + msd;
    long current_index = max(ind_l, -blocksize);
    long current_index_l = max(current_index, 0);
    long current_index_r = min(current_index_l + GSO_BLOCKSIZE, n);
    long gso_step = GSO_BLOCKSIZE - blocksize;
    double total_time = 0.0;
    double red_time = 0.0;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    double pot = this->Pot();
    //main loop
    while (current_index_l < min(ind_r + blocksize, n) - GSO_BLOCKSIZE){
        Lattice_QP *L_loc = this->b_loc_QP(current_index_l, current_index_r);
        while (current_index + blocksize < current_index_r){
            long index_l = max(current_index-current_index_l, 0);
            long index_r = current_index - current_index_l + blocksize;
            Lattice_QP *L_loc_loc = L_loc->b_loc_QP(index_l, index_r);
            struct timeval red_start, red_end;
	        gettimeofday(&red_start, NULL);
            std::cout << "working on index " << current_index << ", ";
            long step = _sred(L_loc_loc, msd, d4f, num_threads);
            gettimeofday(&red_end, NULL);
            red_time += (red_end.tv_sec-red_start.tv_sec)+(double)(red_end.tv_usec-red_start.tv_usec)/1000000.0;
            L_loc->trans_to(index_l, index_r, L_loc_loc);
            L_loc->compute_gso_QP();
            L_loc->size_reduce();
            L_loc->LLL_QP();
            L_loc->compute_gso_QP();
            current_index += step;
        }
        this->trans_to(current_index_l, current_index_r, L_loc);
        this->compute_gso_QP();
        this->size_reduce();
        this->LLL_QP();
        this->compute_gso_QP();
        //this->show_dist_vec();

        current_index_r = min(current_index_r + gso_step, n);
        current_index_l = max(current_index_r - GSO_BLOCKSIZE, 0);
    }

    //tail processing
    do {
        Lattice_QP *L_loc = this->b_loc_QP(current_index_l, current_index_r);
        while (current_index < min(ind_r, n - 20)) {
            long index_l = max(current_index-current_index_l, 0);
            long index_r = min(current_index-current_index_l+blocksize, current_index_r-current_index_l);
            Lattice_QP *L_loc_loc = L_loc->b_loc_QP(index_l, index_r);
            struct timeval red_start, red_end;
	        gettimeofday(&red_start, NULL);
            std::cout << "working on index " << current_index << ", ";
            
            long step = _sred(L_loc_loc, msd, d4f, num_threads);
            gettimeofday(&red_end, NULL);
            red_time += (red_end.tv_sec-red_start.tv_sec)+(double)(red_end.tv_usec-red_start.tv_usec)/1000000.0;
            L_loc->trans_to(index_l, index_r, L_loc_loc);
            L_loc->compute_gso_QP();
            L_loc->size_reduce();
            L_loc->LLL_QP();
            L_loc->compute_gso_QP();
            current_index += step;
        }
        this->trans_to(current_index_l, current_index_r, L_loc);
        this->compute_gso_QP();
        this->size_reduce();
        this->LLL_QP();
        this->compute_gso_QP();
        //this->show_dist_vec();
    } while (0);

    gettimeofday(&end, NULL);
	total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    std::cout << "BKZ_tour done: total_time = "<< (total_time + (num_threads - 1) * red_time) <<"s, red_time = ";
    std::cout << (num_threads * red_time) << "s, dPot = " << (pot-this->Pot()) << "\n";
    std::cout << std::flush;
    return (pot - this->Pot())/(total_time + (num_threads - 1) * red_time);
}

double Lattice_QP::BKZ_tour_pump_epi8(long msd, long d4f, long num_threads, long ind_l, long ind_r, long log_level, long minsd) {
    const long blocksize = d4f + msd;
    long current_index = max(ind_l, -blocksize);
    long current_index_l = max(current_index, 0);
    long current_index_r = min(current_index_l + GSO_BLOCKSIZE, n);
    long gso_step = GSO_BLOCKSIZE - blocksize;
    double total_time = 0.0;
    double red_time = 0.0;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    double pot = this->Pot();
    // main loop
    while (current_index_l < min(ind_r + blocksize, n) - GSO_BLOCKSIZE){
        Lattice_QP *L_loc = this->b_loc_QP(current_index_l, current_index_r);
        while (current_index + blocksize < current_index_r){
            long index_l = max(current_index-current_index_l, 0);
            long index_r = current_index - current_index_l + blocksize;
            Lattice_QP *L_loc_loc = L_loc->b_loc_QP(index_l, index_r);
            struct timeval red_start, red_end;
	        gettimeofday(&red_start, NULL);
            if (log_level >= 2) std::cout << "working on index " << current_index << ", ";
            long step = _pump_epi8_red(L_loc_loc, msd, d4f, num_threads, log_level, minsd);
            gettimeofday(&red_end, NULL);
            red_time += (red_end.tv_sec-red_start.tv_sec)+(double)(red_end.tv_usec-red_start.tv_usec)/1000000.0;
            L_loc->trans_to(index_l, index_r, L_loc_loc);
            L_loc->compute_gso_QP();
            L_loc->size_reduce();
            L_loc->LLL_QP();
            L_loc->compute_gso_QP();
            current_index += step;
        }
        this->trans_to(current_index_l, current_index_r, L_loc);
        this->compute_gso_QP();
        this->size_reduce();
        this->LLL_QP();
        this->compute_gso_QP();
        if (log_level >= 2) this->show_dist_vec();

        current_index_r = min(current_index_r + gso_step, n);
        current_index_l = max(current_index_r - GSO_BLOCKSIZE, 0);
    }

    //tail processing
    do {
        Lattice_QP *L_loc = this->b_loc_QP(current_index_l, current_index_r);
        while (current_index < min(ind_r, n - 20)) {
            long index_l = max(current_index-current_index_l, 0);
            long index_r = min(current_index-current_index_l+blocksize, current_index_r-current_index_l);
            Lattice_QP *L_loc_loc = L_loc->b_loc_QP(index_l, index_r);
            struct timeval red_start, red_end;
	        gettimeofday(&red_start, NULL);
            if (log_level >= 2) std::cout << "working on index " << current_index << ", ";
            
            long step = _pump_epi8_red(L_loc_loc, msd, d4f, num_threads, log_level, minsd);
            gettimeofday(&red_end, NULL);
            red_time += (red_end.tv_sec-red_start.tv_sec)+(double)(red_end.tv_usec-red_start.tv_usec)/1000000.0;
            L_loc->trans_to(index_l, index_r, L_loc_loc);
            L_loc->compute_gso_QP();
            L_loc->size_reduce();
            L_loc->LLL_QP();
            L_loc->compute_gso_QP();
            current_index += step;
        }
        this->trans_to(current_index_l, current_index_r, L_loc);
        this->compute_gso_QP();
        this->size_reduce();
        this->LLL_QP();
        this->compute_gso_QP();
        if (log_level >= 2) this->show_dist_vec();
    } while (0);

    if (log_level >= 1) {
        gettimeofday(&end, NULL);
        total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
        std::cout << "BKZ_tour done: total_time = "<< (total_time + (num_threads - 1) * red_time) <<"s, red_time = ";
        std::cout << (num_threads * red_time) << "s, dPot = " << (pot-this->Pot()) << "\n";
        std::cout << std::flush;
    }
    
    return (pot - this->Pot())/(total_time + (num_threads - 1) * red_time);
}
