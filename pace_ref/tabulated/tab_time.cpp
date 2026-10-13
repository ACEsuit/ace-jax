// Per-atom time of ML-PACE's ACERecursiveEvaluator (pair_style pace: recursive or product) on a rattled
// diamond Si supercell, serial, as one MPI rank of the CPU benchmark does it.
//   tab_time MODEL.yace recursive|product [nrep] [ncell]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "ace-evaluator/ace_c_basis.h"
#include "ace-evaluator/ace_recursive.h"

int main(int argc, char **argv) {
    std::string mode = argv[2];
    int nrep = argc > 3 ? atoi(argv[3]) : 5, nc = argc > 4 ? atoi(argv[4]) : 4;
    ACECTildeBasisSet basis;
    basis.load_yaml(argv[1]);
    ACERecursiveEvaluator ev;
    ev.set_recursive(mode == "recursive");
    ev.element_type_mapping.init(1);
    ev.element_type_mapping(0) = 0;
    ev.set_basis(basis, 1);
    const double a = 5.43, rc = basis.cutoffmax;
    const double fr[8][3] = {{0,0,0},{0,.5,.5},{.5,0,.5},{.5,.5,0},{.25,.25,.25},{.25,.75,.75},{.75,.25,.75},{.75,.75,.25}};
    std::mt19937 rng(1);
    std::normal_distribution<double> nd(0, 0.1);
    std::vector<double> pos;
    for (int i = 0; i < nc; i++) for (int j = 0; j < nc; j++) for (int k = 0; k < nc; k++) for (auto &f: fr) {
        pos.push_back((i + f[0]) * a + nd(rng)); pos.push_back((j + f[1]) * a + nd(rng)); pos.push_back((k + f[2]) * a + nd(rng));
    }
    int N = pos.size() / 3;
    double L = nc * a;
    int ni = (int) std::ceil(rc / L);
    std::vector<double> all(pos);
    for (int x = -ni; x <= ni; x++) for (int y = -ni; y <= ni; y++) for (int z = -ni; z <= ni; z++) {
        if (!x && !y && !z) continue;
        for (int i = 0; i < N; i++) { all.push_back(pos[3*i] + x*L); all.push_back(pos[3*i+1] + y*L); all.push_back(pos[3*i+2] + z*L); }
    }
    int M = all.size() / 3;
    std::vector<double *> X(M);
    for (int j = 0; j < M; j++) X[j] = &all[3 * j];
    std::vector<SPECIES_TYPE> type(M, 0);
    std::vector<std::vector<int>> jl(N);
    size_t nn = 0, maxj = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (j == i) continue;
            double r2 = 0;
            for (int d = 0; d < 3; d++) r2 += (X[j][d] - X[i][d]) * (X[j][d] - X[i][d]);
            if (r2 < rc * rc) jl[i].push_back(j);
        }
        nn += jl[i].size(); maxj = std::max(maxj, jl[i].size());
    }
    ev.resize_neighbours_cache(maxj);
    double E = 0, F2 = 0;
    for (int i = 0; i < N; i++) {   // warm-up, and a force checksum
        ev.compute_atom(i, X.data(), type.data(), jl[i].size(), jl[i].data()); E += ev.e_atom;
        for (size_t jj = 0; jj < jl[i].size(); jj++) for (int d = 0; d < 3; d++) F2 += ev.neighbours_forces(jj, d) * ev.neighbours_forces(jj, d);
    }
#ifdef FINE_TIMING
    for (ACETimer *t: {&ev.A_construction_timer, &ev.energy_calc_timer, &ev.forces_calc_loop_timer,
                       &ev.forces_calc_neighbour_timer, &ev.per_atom_calc_timer}) t->init();
#endif
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < nrep; r++)
        for (int i = 0; i < N; i++) ev.compute_atom(i, X.data(), type.data(), jl[i].size(), jl[i].data());
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%s %s: %d atoms, %.1f neighbours/atom, E/atom %.12f, |f_ij|^2 %.12e, %.2f us/atom (serial)\n", argv[1], mode.c_str(), N,
           (double) nn / N, E / N, F2, 1e6 * s / (nrep * N));
#ifdef FINE_TIMING
    double tot = ev.per_atom_calc_timer.as_microseconds();
    printf("  A construction %.0f%%, energy (products) %.0f%%, forces loop %.0f%% (of which neighbour forces %.0f%%)\n",
           100 * ev.A_construction_timer.as_microseconds() / tot, 100 * ev.energy_calc_timer.as_microseconds() / tot,
           100 * ev.forces_calc_loop_timer.as_microseconds() / tot, 100 * ev.forces_calc_neighbour_timer.as_microseconds() / tot);
#endif
}
