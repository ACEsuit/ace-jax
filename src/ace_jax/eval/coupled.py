"""The product basis of a linear ACE in coupled form (docs/dev/specs/2026-11-19-coupled-product-basis.md).

`ctilde` has the rotational coupling multiplied out: the monomials of one multiset of (radial column, l) blocks
of A are a few invariants.  For correlation order <= 4 the site energy is exactly

    z = [A; X],   X^L_{P,M} = sum T[m1, m2, M] A_{b1 m1} A_{b2 m2}   (P = (b1, b2), T = cg.real_cg(l1, l2, L))
    E = lin . A + 1/2 z^T W z

with W sparse and, between two blocks of one L, w times the identity in M.  Order 2 is a weight between two A
blocks, order 3 between an X block and an A block, order 4 between two X blocks.  `build` chooses which blocks of
each multiset are coupled (so that multisets share X blocks) and fits the weights as a linear map of ctilde,
`Plan.weights`; the fit is exact, and `check` holds it to that.  NumPy only: the kernels' plan (`ffi`) and the
tests' reference (`evaluate`) both read this."""
import dataclasses

import numpy as np

from .cg import real_cg

MAX_ORDER = 4
BITS = 15                                             # of an A index in a monomial's key
PAIR_COST = (1.0, 0.5)                                # of an X block: per value, per CG term (`build`)
RESIDUAL = 1e-10                                      # relative, of the fit (`Plan.check`)


class NoPlan(ValueError):
    """The model has no coupled form here: `ffi` keeps the DAG."""


@dataclasses.dataclass(frozen=True)
class Plan:
    """A in the model's order.  z blocks: the A blocks (ids 0..), then the X blocks."""
    n_a: int
    n_aa: int
    mem: tuple                # per A block, its 2l + 1 A indices, m = -l..l
    bl: np.ndarray            # (n_blocks,) l of each A block
    xb: np.ndarray            # (n_xb, 3) X blocks: A block b1 <= b2, L
    lin: np.ndarray           # (n_lin,) A index of each linear weight
    quad: np.ndarray          # (n_quad, 2) z blocks U <= V of each quadratic weight
    P: object                 # scipy CSR (n_lin + n_quad, n_AA): the weights of a readout

    @property
    def n_x(self):
        return int((2 * self.xb[:, 2] + 1).sum())

    @property
    def x_base(self):         # each X block's first index in z
        return self.n_a + np.r_[0, np.cumsum(2 * self.xb[:, 2] + 1)][:-1]

    def zl(self, U):          # L of z block U
        nb = len(self.bl)
        return int(self.bl[U]) if U < nb else int(self.xb[U - nb, 2])

    def weights(self, ctilde):
        """(n_lin + n_quad, NZ) for ctilde (n_AA, NZ)."""
        return np.asarray(self.P @ np.asarray(ctilde, np.float64))

    def cg_terms(self):
        """(x, a, b, c): X_x = sum c A_a A_b, x an index into z."""
        out = []
        for (b1, b2, L), x0 in zip(self.xb.tolist(), self.x_base.tolist()):
            T = real_cg(int(self.bl[b1]), int(self.bl[b2]), L)
            i, j, M = np.nonzero(np.abs(T) > 1e-14)
            out.append(np.stack([x0 + M, np.asarray(self.mem[b1])[i], np.asarray(self.mem[b2])[j]], 1))
            out.append(T[i, j, M])
        if not out:
            return np.zeros((0, 3), np.int64), np.zeros(0)
        return np.concatenate(out[0::2]), np.concatenate(out[1::2])

    def stats(self):
        return {"n_a": self.n_a, "n_x": self.n_x, "cg_terms": len(self.cg_terms()[1]),
                "weights": len(self.quad), "linear": len(self.lin), "x_blocks": len(self.xb)}


def _key(q):
    """A monomial (..., k) of A indices as one int64, whatever its order of factors."""
    q = np.sort(np.asarray(q, np.int64), axis=-1)
    out = np.zeros(q.shape[:-1], np.int64)
    for i in range(q.shape[-1]):
        out = (out << BITS) | q[..., i]
    return out


def _blocks(ar, ay):
    """(block of each A, per block its A indices by m, per block l); every block must be whole."""
    ar, ay = np.asarray(ar, np.int64), np.asarray(ay, np.int64)
    l = np.floor(np.sqrt(ay + 0.5)).astype(np.int64)
    m = ay - l * l - l
    ids, blk = np.unique(np.stack([ar, l], 1), axis=0, return_inverse=True)
    blk = blk.ravel()
    mem = []
    for b, (_, lb) in enumerate(ids.tolist()):
        q = np.flatnonzero(blk == b)
        q = q[np.argsort(m[q])]
        if len(q) != 2 * lb + 1 or not np.array_equal(m[q], np.arange(-lb, lb + 1)):
            raise NoPlan(f"A block (radial {ids[b, 0]}, l = {lb}) is not whole")
        mem.append(q)
    return blk, tuple(mem), ids[:, 1].copy()


def _options(g, bl):
    """The ways to couple multiset g (sorted A blocks, order 3 or 4): [(pairing, X blocks)], X block (b1, b2, L)."""
    def xs(p, Ls):
        a, b = sorted(p)
        return [(a, b, L) for L in Ls if abs(bl[a] - bl[b]) <= L <= bl[a] + bl[b] and not (a == b and L % 2)]
    out, seen = [], set()
    if len(g) == 3:
        for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            x = xs((g[i], g[j]), [int(bl[g[k]])])
            if x and (x[0], g[k]) not in seen:
                seen.add((x[0], g[k]))
                out.append(((i, j, k), x))
    else:
        for (i, j), (k, n) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            p, q = tuple(sorted((g[i], g[j]))), tuple(sorted((g[k], g[n])))
            if (min(p, q), max(p, q)) in seen:
                continue
            seen.add((min(p, q), max(p, q)))
            lo = max(abs(bl[p[0]] - bl[p[1]]), abs(bl[q[0]] - bl[q[1]]))
            hi = min(bl[p[0]] + bl[p[1]], bl[q[0]] + bl[q[1]])
            Ls = [L for L in range(int(lo), int(hi) + 1)
                  if not (p[0] == p[1] and L % 2) and not (q[0] == q[1] and L % 2)]
            if Ls:
                out.append((((i, j), (k, n)), sorted(set(xs(p, Ls) + xs(q, Ls)))))
    return out


def _nnz(l1, l2, L):
    return int(np.count_nonzero(np.abs(real_cg(l1, l2, L)) > 1e-14))


def choose_pairings(groups, bl, cost=PAIR_COST, ilp_seconds=0.0):
    """One of `_options` per multiset, to make the X blocks all of them need cheap: greedy (two passes), then an
    integer program for `ilp_seconds` if that is positive.  cost: per X value and per CG term of a block."""
    opts = [_options(g, bl) for g in groups]
    wx, wc = cost
    cst = {}

    def c(x):
        if x not in cst:
            cst[x] = wx * (2 * x[2] + 1) + wc * _nnz(int(bl[x[0]]), int(bl[x[1]]), x[2])
        return cst[x]
    use = {}                                           # X block -> multisets that need it
    sel = [None] * len(groups)
    order = sorted(range(len(groups)), key=lambda i: len(opts[i]))
    for _ in range(2):
        for i in order:
            if not opts[i]:
                continue
            if sel[i] is not None:
                for x in opts[i][sel[i]][1]:
                    use[x] -= 1
            sel[i] = min(range(len(opts[i])), key=lambda o: sum(c(x) for x in opts[i][o][1] if not use.get(x)))
            for x in opts[i][sel[i]][1]:
                use[x] = use.get(x, 0) + 1
    if ilp_seconds > 0:
        sel = _ilp(opts, sel, c, ilp_seconds)
    return [opts[i][s][0] if s is not None else None for i, s in enumerate(sel)]


def _ilp(opts, sel, c, seconds):
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    free = [i for i, o in enumerate(opts) if len(o) > 1]
    if not free:
        return sel
    fixed = {x for i, o in enumerate(opts) if len(o) == 1 for x in o[0][1]}
    xs = sorted({x for i in free for o in opts[i] for x in o[1]} - fixed)
    xi = {x: k for k, x in enumerate(xs)}
    var = [(i, o) for i in free for o in range(len(opts[i]))]
    nv = len(var)
    r1, c1, r2, c2, v2 = [], [], [], [], []
    row = {i: k for k, i in enumerate(free)}
    for v, (i, o) in enumerate(var):
        r1.append(row[i]); c1.append(v)
        for x in opts[i][o][1]:
            if x in xi:
                r2 += [len(r2) // 2] * 2; c2 += [v, nv + xi[x]]; v2 += [1.0, -1.0]
    n = nv + len(xs)
    A1 = coo_matrix((np.ones(len(r1)), (r1, c1)), (len(free), n)).tocsr()
    A2 = coo_matrix((v2, (r2, c2)), (len(r2) // 2, n)).tocsr()
    obj = np.r_[np.zeros(nv), [c(x) for x in xs]]
    res = milp(obj, constraints=[LinearConstraint(A1, 1, 1), LinearConstraint(A2, -np.inf, 0)],
               integrality=np.ones(n), bounds=Bounds(0, 1), options={"time_limit": float(seconds)})
    if res.x is None:
        return sel
    new = list(sel)
    for v, (i, o) in enumerate(var):
        if res.x[v] > 0.5:
            new[i] = o

    def total(s):
        return sum(c(x) for x in {x for i, o in enumerate(opts) if s[i] is not None for x in o[s[i]][1]})
    return new if total(new) < total(sel) else sel


def _independent(B, tol=1e-9):
    """The columns of B, in order, that add to its span (modified Gram-Schmidt)."""
    Q, keep = [], []
    for j in range(B.shape[1]):
        v = B[:, j].copy()
        n0 = np.linalg.norm(v)
        if n0 == 0:
            continue
        for _ in range(2):
            for q in Q:
                v -= (q @ v) * q
        n1 = np.linalg.norm(v)
        if n1 > tol * n0:
            Q.append(v / n1)
            keep.append(j)
    return keep


def build(ar, ay, specs, cost=PAIR_COST, ilp_seconds=0.0):
    """The coupled `Plan` of the product basis `specs` (per order an (n_v, k) array of A indices, their rows the
    columns of ctilde in order) over A with radial column ar and harmonic ay.  Raises `NoPlan` for an order above
    4, an A block that is not whole, or more than 2^15 A."""
    from scipy.sparse import coo_matrix
    specs = [np.asarray(s, np.int64) for s in specs]
    n_a = len(np.asarray(ar))
    if n_a >= 1 << BITS:
        raise NoPlan(f"{n_a} A entries")
    if any(s.size and s.shape[1] > MAX_ORDER for s in specs):
        raise NoPlan(f"correlation order above {MAX_ORDER}")
    blk, mem, bl = _blocks(ar, ay)
    nb = len(bl)
    off, items = 0, []                                 # (order, multiset, its columns, their keys)
    for s in specs:
        if s.size:
            k = s.shape[1]
            gb = np.sort(blk[s], axis=1)
            keys = _key(s)
            u, inv = np.unique(gb, axis=0, return_inverse=True)
            inv = inv.ravel()
            o = np.argsort(inv, kind="stable")
            cut = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(u)))]
            for gi in range(len(u)):
                rows = o[cut[gi]:cut[gi + 1]]
                items.append((k, tuple(u[gi].tolist()), off + rows, keys[rows]))
        off += s.shape[0]
    n_aa = off
    hi = [i for i, it in enumerate(items) if it[0] >= 3]
    choice = dict(zip(hi, choose_pairings([items[i][1] for i in hi], bl, cost, ilp_seconds)))

    xid, xb = {}, []

    def xblock(x):
        if x not in xid:
            xid[x] = len(xb)
            xb.append(x)
        return nb + xid[x]
    M_ = [np.asarray(q) for q in mem]

    def T(a, b, L):
        return real_cg(int(bl[a]), int(bl[b]), L)
    lin, quad, pr, pc, pv = [], [], [], [], []
    for it, (k, g, cols, keys) in enumerate(items):
        cands = []                                     # (weight's place, monomial keys, values)
        if k == 1:
            if bl[g[0]] != 0:
                raise NoPlan("a linear term with l > 0")
            cands.append((("lin", int(M_[g[0]][0])), _key(M_[g[0]][:1, None]), np.ones(1)))
        elif k == 2:
            if bl[g[0]] != bl[g[1]]:
                raise NoPlan("an order-2 term of two l")
            cands.append((("quad", g[0], g[1]), _key(np.stack([M_[g[0]], M_[g[1]]], 1)), np.ones(len(M_[g[0]]))))
        elif choice[it] is None:
            raise NoPlan(f"no coupling for the blocks {g}")
        elif k == 3:
            i, j, n = choice[it]
            a, b = sorted((g[i], g[j]))
            t = T(a, b, int(bl[g[n]]))
            q = np.stack(np.broadcast_arrays(M_[a][:, None, None], M_[b][None, :, None], M_[g[n]][None, None, :]), -1)
            cands.append((("x", (a, b, int(bl[g[n]])), g[n]), _key(q).ravel(), t.ravel()))
        else:
            (i, j), (m, n) = choice[it]
            p, q = tuple(sorted((g[i], g[j]))), tuple(sorted((g[m], g[n])))
            if q < p:
                p, q = q, p
            idx = _key(np.stack(np.broadcast_arrays(M_[p[0]][:, None, None, None], M_[p[1]][None, :, None, None],
                                                    M_[q[0]][None, None, :, None], M_[q[1]][None, None, None, :]),
                                -1)).ravel()
            lo = max(abs(bl[p[0]] - bl[p[1]]), abs(bl[q[0]] - bl[q[1]]))
            top = min(bl[p[0]] + bl[p[1]], bl[q[0]] + bl[q[1]])
            for L in range(int(lo), int(top) + 1):
                if (p[0] == p[1] or q[0] == q[1]) and L % 2:
                    continue
                cands.append((("xx", p + (L,), q + (L,)), idx,
                              np.einsum("ijM,knM->ijkn", T(*p, L), T(*q, L)).ravel()))
        tk, tinv = np.unique(keys, return_inverse=True)
        rows = np.unique(np.concatenate([tk] + [c[1] for c in cands]))
        B = np.zeros((len(rows), len(cands)))
        for c, (_, ck, cv) in enumerate(cands):
            np.add.at(B[:, c], np.searchsorted(rows, ck), cv)
        B[np.abs(B) < 1e-13] = 0.0
        keep = _independent(B)
        if not keep:
            raise NoPlan(f"no invariant for the blocks {g}")
        pinv = np.linalg.pinv(B[:, keep])[:, np.searchsorted(rows, tk)]   # (kept, target monomials)
        for c, row in zip(keep, pinv):
            where = cands[c][0]
            if where[0] == "lin":
                w = ("l", len(lin)); lin.append(where[1])
            else:
                if where[0] == "quad":
                    U, V = where[1], where[2]
                elif where[0] == "x":
                    U, V = where[2], xblock(where[1])
                else:
                    U, V = xblock(where[1]), xblock(where[2])
                w = ("q", len(quad)); quad.append((min(U, V), max(U, V)))
            nzr = np.abs(row[tinv]) > 0
            pr.append(np.full(int(nzr.sum()), -1 - w[1] if w[0] == "l" else w[1]))
            pc.append(cols[nzr]); pv.append(row[tinv][nzr])
    n_lin = len(lin)
    r = np.concatenate(pr) if pr else np.zeros(0, np.int64)
    r = np.where(r < 0, -1 - r, r + n_lin)
    P = coo_matrix((np.concatenate(pv) if pv else np.zeros(0), (r, np.concatenate(pc) if pc else r)),
                   (n_lin + len(quad), n_aa)).tocsr()
    return Plan(n_a=n_a, n_aa=n_aa, mem=mem, bl=bl, xb=np.asarray(xb, np.int64).reshape(-1, 3),
                lin=np.asarray(lin, np.int64), quad=np.asarray(quad, np.int64).reshape(-1, 2), P=P)


def evaluate(plan, w, A):
    """(E, dE/dA) of sites A (n, n_a) for one centre species' weights w (n_lin + n_quad,): the reference."""
    A = np.asarray(A, np.float64)
    n, nb = len(A), len(plan.bl)
    z = np.concatenate([A, np.zeros((n, plan.n_x))], 1)
    xt, xc = plan.cg_terms()
    np.add.at(z, (slice(None), xt[:, 0]), xc * A[:, xt[:, 1]] * A[:, xt[:, 2]])
    xbase = plan.x_base
    span = lambda U: (np.asarray(plan.mem[U]) if U < nb
                      else xbase[U - nb] + np.arange(2 * plan.xb[U - nb, 2] + 1))
    n_lin = len(plan.lin)
    g = np.zeros_like(z)
    for (U, V), wq in zip(plan.quad.tolist(), w[n_lin:]):
        su, sv = span(U), span(V)
        g[:, su] += (2 * wq if U == V else wq) * z[:, sv]
        if U != V:
            g[:, sv] += wq * z[:, su]
    dA = g[:, :plan.n_a].copy()
    dA[:, plan.lin] += w[:n_lin]
    E = A[:, plan.lin] @ w[:n_lin] + 0.5 * np.einsum("nz,nz->n", z, g)
    np.add.at(dA, (slice(None), xt[:, 1]), xc * g[:, xt[:, 0]] * A[:, xt[:, 2]])
    np.add.at(dA, (slice(None), xt[:, 2]), xc * g[:, xt[:, 0]] * A[:, xt[:, 1]])
    return E, dA


def monomial_energy(specs, ct, A):
    """sum_c ct_c prod A over the monomials: what the plan must equal."""
    A = np.asarray(A, np.float64)
    E, off = np.zeros(len(A)), 0
    for s in specs:
        s = np.asarray(s, np.int64)
        if s.size:
            E += np.prod(A[:, s], axis=2) @ ct[off:off + s.shape[0]]
        off += s.shape[0]
    return E


def check(plan, specs, ctilde, w=None, sites=3, seed=0, tol=RESIDUAL):
    """The largest relative difference between the plan and the monomials at random A, over the centre species;
    raises `NoPlan` above `tol` (a readout outside the invariants' span, or a bug)."""
    ct = np.asarray(ctilde, np.float64)
    w = plan.weights(ct) if w is None else w
    A = np.random.default_rng(seed).normal(size=(sites, plan.n_a))
    worst = 0.0
    for zc in range(ct.shape[1]):
        ref = monomial_energy(specs, ct[:, zc], A)
        scale = monomial_energy(specs, np.abs(ct[:, zc]), np.abs(A))
        worst = max(worst, float(np.max(np.abs(evaluate(plan, w[:, zc], A)[0] - ref) / np.maximum(scale, 1e-300))))
    if not worst <= tol:
        raise NoPlan(f"the coupled form differs from the product basis by {worst:.1e} (relative)")
    return worst
