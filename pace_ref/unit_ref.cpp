// Dumps ML-PACE's own radial / core / embedding / Ylm values as JSON, so the
// JAX ports in src/ace_jax/eval/pace_radial.py are checked against the C++,
// not against a re-derivation.  Regenerate: see pace_ref/README.md.
#include "ace-evaluator/ace_radial.h"
#include "ace-evaluator/ace_abstract_basis.h"
#include "ace-evaluator/ace_spherical_cart.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
using namespace std;

static bool first = true;
static void emit(const string &key, const vector<double> &v, int ncol) {
    printf("%s\"%s\": {\"ncol\": %d, \"data\": [", first ? "" : ",\n", key.c_str(), ncol);
    first = false;
    for (size_t i = 0; i < v.size(); i++) printf("%s%.17g", i ? "," : "", v[i]);
    printf("]}");
}

int main() {
    const int K = 8;
    vector<double> rs;
    for (int i = 0; i < 600; i++) rs.push_back(0.005 + 0.01 * i);
    printf("{\n");
    emit("r", rs, 1);

    for (string name : {"ChebExpCos", "ChebPow", "ChebLinear", "SBessel"}) {
        for (string inner : {"density", "distance", "zbl"}) {
            ACERadialFunctions rf(K, 0, 1, 0.001, 1, {{name}});
            rf.inner_cutoff_type = inner;
            vector<double> out;
            for (double r : rs) {
                rf.radbase(2.5, 5.0, 0.3, name, r, 1.2, 0.4);
                for (int k = 0; k < K; k++) out.push_back(rf.gr(k));
            }
            emit("g_" + name + "_" + inner, out, K);
        }
    }
    {   // zbl quirk: cut_in = (dcut_in == 0)
        ACERadialFunctions rf(K, 0, 1, 0.001, 1, {{"ChebExpCos"}});
        rf.inner_cutoff_type = "zbl";
        vector<double> out;
        for (double r : rs) {
            rf.radbase(2.5, 5.0, 0.3, "ChebExpCos", r, 1.2, 0.0);
            for (int k = 0; k < K; k++) out.push_back(rf.gr(k));
        }
        emit("g_ChebExpCos_zbl_dcutin0", out, K);
    }
    for (string inner : {"density", "distance"}) {
        ACERadialFunctions rf(K, 0, 1, 0.001, 1, {{"ChebExpCos"}});
        rf.inner_cutoff_type = inner;
        vector<double> out;
        for (double r : rs) {
            double cr, dcr;
            rf.radcore(r, 3.0, 0.7, 5.0, cr, dcr, 1.2, 0.4);
            out.push_back(cr);
        }
        emit("cr_" + inner, out, 1);
    }
    {
        vector<double> out;
        for (double r : rs) {
            double cr, dcr;
            ACERadialFunctions::ZBL(r, 14, 32, cr, dcr, 5.0, 4.7, 1.0);
            out.push_back(cr);
        }
        emit("cr_zbl", out, 1);
    }
    {
        vector<double> out;
        for (double r : rs) {
            double fc, dfc;
            cutoff_func_poly(r, 3.0, 1.0, fc, dfc);
            out.push_back(fc);
        }
        emit("fcpoly", out, 1);
    }
    {
        vector<double> xs = {0.0, 1e-12, -1e-12, 1e-9, 1e-7, -1e-7, 2e-6, 3e-6, 4e-6, -5e-6,
                             1e-5, 1e-3, -1e-3, 0.1, -0.1, 0.5, 1.0, -2.0, 7.3, -40.0};
        emit("fx", xs, 1);
        for (double m : {0.5, 1.0, 2.0}) {
            vector<double> a, b;
            for (double x : xs) {
                double F, DF;
                Fexp(x, m, F, DF); a.push_back(F);
                FexpShiftedScaled(x, m, F, DF); b.push_back(F);
            }
            char buf[32];
            snprintf(buf, sizeof buf, "%g", m);
            emit(string("fexp_m") + buf, a, 1);
            emit(string("fexpss_m") + buf, b, 1);
        }
    }
    {
        const int L = 6;
        ACECartesianSphericalHarmonics sh(L);
        mt19937 gen(7);
        normal_distribution<double> nd;
        vector<double> dirs, re, im;
        for (int s = 0; s < 50; s++) {
            double x = nd(gen), y = nd(gen), z = nd(gen), n = sqrt(x * x + y * y + z * z);
            x /= n; y /= n; z /= n;
            dirs.insert(dirs.end(), {x, y, z});
            sh.compute_ylm(x, y, z, L);
            for (int l = 0; l <= L; l++)
                for (int m = -l; m <= l; m++) {
                    int am = abs(m);
                    double sgn = (am % 2 == 0) ? 1.0 : -1.0;
                    double yr = sh.ylm(l, am).real, yi = sh.ylm(l, am).img;
                    if (m < 0) { yr = sgn * yr; yi = -sgn * yi; }
                    re.push_back(yr);
                    im.push_back(yi);
                }
        }
        emit("ydir", dirs, 3);
        emit("ylm_re", re, (L + 1) * (L + 1));
        emit("ylm_im", im, (L + 1) * (L + 1));
    }
    printf("\n}\n");
    return 0;
}
