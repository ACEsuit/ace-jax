// E and F of configurations from a .yace with ML-PACE's C-tilde evaluator (brute-force periodic
// neighbour list; forces as LAMMPS pair_pace accumulates them).
// stdin: nconf, then per configuration: N; 3x3 cell (rows); pbc (3 ints); N lines t x y z, with t the
// atom's element index in the .yace `elements` list.
// stdout: per configuration: E, then N lines fx fy fz.
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
#include "ace-evaluator/ace_c_basis.h"
#include "ace-evaluator/ace_evaluator.h"

int main(int argc, char **argv) {
    ACECTildeBasisSet basis;
    basis.load_yaml(argv[1]);
    ACECTildeEvaluator ev;
    ev.set_basis(basis);
    ev.element_type_mapping.init(basis.nelements);
    for (int t = 0; t < basis.nelements; t++) ev.element_type_mapping(t) = t;
    const double rc = basis.cutoffmax;
    int nconf;
    std::cin >> nconf;
    std::cout.precision(17);
    for (int c = 0; c < nconf; c++) {
        int N;
        std::cin >> N;
        double cell[3][3];
        int pbc[3];
        for (auto &row: cell) for (double &v: row) std::cin >> v;
        for (int &p: pbc) std::cin >> p;
        std::vector<double> pos(3 * N);
        std::vector<SPECIES_TYPE> t0(N);
        for (int i = 0; i < N; i++) std::cin >> t0[i] >> pos[3 * i] >> pos[3 * i + 1] >> pos[3 * i + 2];
        // image range per direction: rc over the plane spacing
        double vol = 0, cr[3][3];
        for (int k = 0; k < 3; k++) {
            const double *a = cell[(k + 1) % 3], *b = cell[(k + 2) % 3];
            cr[k][0] = a[1] * b[2] - a[2] * b[1];
            cr[k][1] = a[2] * b[0] - a[0] * b[2];
            cr[k][2] = a[0] * b[1] - a[1] * b[0];
        }
        for (int d = 0; d < 3; d++) vol += cell[0][d] * cr[0][d];
        int nimg[3];
        for (int k = 0; k < 3; k++) {
            double area = std::sqrt(cr[k][0] * cr[k][0] + cr[k][1] * cr[k][1] + cr[k][2] * cr[k][2]);
            nimg[k] = pbc[k] ? (int) std::ceil(rc / (std::fabs(vol) / area)) : 0;
        }
        std::vector<double> all(pos);      // owned first, then ghosts
        std::vector<int> owner;
        for (int i = 0; i < N; i++) owner.push_back(i);
        for (int a = -nimg[0]; a <= nimg[0]; a++)
            for (int b = -nimg[1]; b <= nimg[1]; b++)
                for (int g = -nimg[2]; g <= nimg[2]; g++) {
                    if (!a && !b && !g) continue;
                    for (int i = 0; i < N; i++) {
                        for (int d = 0; d < 3; d++)
                            all.push_back(pos[3 * i + d] + a * cell[0][d] + b * cell[1][d] + g * cell[2][d]);
                        owner.push_back(i);
                    }
                }
        int M = owner.size();
        std::vector<double *> x(M);
        for (int j = 0; j < M; j++) x[j] = &all[3 * j];
        std::vector<SPECIES_TYPE> type(M);
        for (int j = 0; j < M; j++) type[j] = t0[owner[j]];
        std::vector<double> F(3 * N, 0);
        double E = 0;
        for (int i = 0; i < N; i++) {
            std::vector<int> jl;
            for (int j = 0; j < M; j++) {
                if (j == i) continue;
                double r2 = 0;
                for (int d = 0; d < 3; d++) r2 += (x[j][d] - x[i][d]) * (x[j][d] - x[i][d]);
                if (r2 < rc * rc) jl.push_back(j);
            }
            ev.resize_neighbours_cache(jl.size());
            ev.compute_atom(i, x.data(), type.data(), jl.size(), jl.data());
            E += ev.e_atom;
            for (size_t jj = 0; jj < jl.size(); jj++)
                for (int d = 0; d < 3; d++) {
                    F[3 * i + d] += ev.neighbours_forces(jj, d);
                    F[3 * owner[jl[jj]] + d] -= ev.neighbours_forces(jj, d);
                }
        }
        std::cout << E << "\n";
        for (int i = 0; i < N; i++) std::cout << F[3 * i] << " " << F[3 * i + 1] << " " << F[3 * i + 2] << "\n";
    }
}
