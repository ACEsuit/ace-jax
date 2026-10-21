"""The interactive version of plot_cost.py's figure for the docs Performance page: a Plotly figure spec (JSON:
data + layout), rendered in the page by plotly.js (docs/user/javascripts/plotly-figures.js); no Python plotting
library is needed, here or in the docs build.

Panels: test force RMSE and QoI error, against each --row's cost (A100, then the CPU node) and the fitted-parameter
count -- plot_cost.py's panels A/B, D/E, G/H.  The cheapest-model panels (C, F, I) are left out: the page's
table lists those models.  One legend entry per family (`legendgroup`): a click shows or hides it in every
panel.  Buttons (drawn by the page from layout.meta) select a preset (PRESETS); families outside it stay in the legend, hidden.  Data, labels,
colours and markers are plot_cost.py's (`collect`, `FAM`), so the PNG and this figure cannot disagree.

    python bench/si_qoi/plot_cost_web.py --row "A100 GPU, µs/atom-step|cost.json" ... --out fig.json
"""
import argparse, json, math, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import plot_cost as P  # noqa: E402

SYMBOL = {"o": "circle", "D": "diamond", "d": "diamond-tall", "s": "square", "^": "triangle-up", "v": "triangle-down",
          "*": "star", "X": "x", "P": "cross", "p": "pentagon", "h": "hexagon"}
DASH = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}
# presets: the families shown (the rest stay in the legend, hidden); the first is the page's initial view
PRESETS = {
    "Headline": ("ace", "ace_tabfork", "pace", "mace_L0", "mace_L1", "mace_xl_L0", "mace_xl_L1", "nep", "gap", "grace"),
    "ACE variants": ("ace", "ace_E10", "ace_f32", "ace_lr", "ace_pshape", "ace_pshape_lr", "ace_pshape_E10"),
    "Same model, other code": ("ace", "ace_tab", "ace_tabfork", "ace_trim", "ace_tab_mixed", "mace_L0", "mace_L1",
                               "mace_xl_L0", "mace_xl_L1"),
    "All": None,
}
GAP_X, GAP_Y, ROW_H = 0.08, 0.1, 380          # subplot gaps (paper fraction) and the height of one row (px)
QOI_TICKS = [0.3, 0.5, 1, 2, 3]


def _num(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def timed_with(codes, fam, size):
    """The code a cost was measured with (make_cost.py's sidecar, its file tag dropped), or "" (the parameter row)."""
    return codes.get(P.cost_name(fam, size), "").split(" [")[0]


def traces(data, key, xa, ya, xlabel, shown, codes):
    """One scatter per family with points on these axes: plot_cost's style, seed min-max as error bars; the hover
    names the code each cost was measured with (it says "dag:greedy" where plot_cost.py draws a centre dot)."""
    out = []
    for fam, pts in data.items():
        lab, col, mk, ls, filled, _ = P.FAM[fam]
        p = [q for q in pts if q[key] is not None and q["x"] is not None]
        if not p:
            continue
        t = {"type": "scatter", "mode": "lines+markers", "name": lab, "legendgroup": fam, "showlegend": False,
             "xaxis": xa, "yaxis": ya, "x": [q["x"] for q in p], "y": [q[key] for q in p],
             "line": {"color": col, "width": 2, "dash": DASH[ls]},
             "marker": {"symbol": SYMBOL[mk] + ("" if filled else "-open"), "size": 9, "color": col,
                        "line": {"color": col, "width": 1.5}},
             "customdata": [[q["size"], q["n"], q["rmse_F"], _num(q["qoi"]), timed_with(codes, fam, q["size"])] for q in p],
             "hovertemplate": (f"<b>{lab}</b> %{{customdata[0]}}<br>{xlabel}: %{{x:.3g}}<br>"
                               "force RMSE %{customdata[2]:.4f} eV/Å<br>QoI error %{customdata[3]:.2f}<br>"
                               "parameters %{customdata[1]:,}<br>%{customdata[4]}<extra></extra>"),
             "visible": True if shown is None or fam in shown else "legendonly", "meta": fam}
        if all(key + "_lo" in q for q in p):
            t["error_y"] = {"type": "data", "symmetric": False, "color": col, "thickness": 1.2, "width": 3,
                            "array": [q[key + "_hi"] - q[key] for q in p],
                            "arrayminus": [q[key] - q[key + "_lo"] for q in p]}
        out.append(t)
    return out


def figure(rows):
    """rows: [(xlabel, cost dict or None, codes dict)]; a None cost is the fitted-parameter row.  Returns the Plotly
    spec."""
    n = len(rows)
    first = next(iter(PRESETS.values()))
    data, layout, shapes, notes = [], {}, [], []
    col_w, row_h = (1 - GAP_X) / 2, (1 - (n - 1) * GAP_Y) / n
    for r, (xlabel, cost, codes) in enumerate(rows):
        d, ref = P.collect(cost)
        y1 = 1 - r * (row_h + GAP_Y)
        for c, (key, title) in enumerate((("rmse_F", "force error"), ("qoi", "silicon QoIs vs DFT"))):
            k = r * 2 + c + 1
            xa, ya = ("x", "y") if k == 1 else (f"x{k}", f"y{k}")
            layout[f"xaxis{'' if k == 1 else k}"] = {
                "type": "log", "domain": [c * (col_w + GAP_X), c * (col_w + GAP_X) + col_w], "anchor": ya,
                "title": {"text": xlabel, "standoff": 4}, "showline": True, "ticks": "outside", "zeroline": False,
                "dtick": 1}                                  # log axis: decade labels only
            yax = {"domain": [y1 - row_h, y1], "anchor": xa, "showline": True, "ticks": "outside", "zeroline": False,
                   "title": {"text": "test force RMSE (eV/Å)" if key == "rmse_F" else "QoI error (lower is better)"}}
            if key == "qoi":
                yax.update(type="log", tickvals=QOI_TICKS, ticktext=[f"{v:g}" for v in QOI_TICKS])
            layout[f"yaxis{'' if k == 1 else k}"] = yax
            data += traces(d, key, xa, ya, xlabel.split(",")[0], first, codes)
            lines = [(ref[key], "GAP-18", "dot")] + ([(P.TARGET_QOI, "target", "dash")] if key == "qoi" else [])
            for yv, lab, dash in lines:
                shapes.append({"type": "line", "xref": f"{xa} domain", "yref": ya, "x0": 0, "x1": 1,
                               "y0": yv, "y1": yv, "line": {"dash": dash, "width": 1.2, "color": "#888"}})
                notes.append({"xref": f"{xa} domain", "yref": ya, "x": 1 if lab == "target" else 0,
                              "xanchor": "right" if lab == "target" else "left", "y": yv, "yanchor": "bottom",
                              "text": lab, "showarrow": False, "font": {"size": 10, "color": "#888"}})
            notes.append({"xref": f"{xa} domain", "yref": f"{ya} domain", "x": 0, "y": 1.02, "xanchor": "left",
                          "yanchor": "bottom", "showarrow": False, "text": f"<b>{title}</b>", "font": {"size": 13}})
    seen = set()
    for t in data:                                      # one legend entry per family: its first trace
        if t["legendgroup"] not in seen:
            seen.add(t["legendgroup"]); t["showlegend"] = True
    # the presets go to the page as data (layout.meta): plotly-figures.js draws them as buttons above the plot, so
    # the legend can sit just above the panels and Plotly's margin auto-expansion fits it at any page width
    layout.update(
        height=n * ROW_H + 80, margin={"l": 60, "r": 20, "t": 30, "b": 50}, hovermode="closest",
        legend={"orientation": "h", "x": 0, "y": 1.04, "yanchor": "bottom", "groupclick": "togglegroup",
                "font": {"size": 11}, "itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        meta={"presets": [{"label": name, "families": None if fs is None else list(fs)} for name, fs in PRESETS.items()]},
        shapes=shapes, annotations=notes, font={"family": "Roboto, Helvetica, Arial, sans-serif", "size": 12},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return {"data": data, "layout": layout}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", type=P._row_arg, default=[],
                    help='"LABEL|COST.json[|CODES.json]" as plot_cost.py (CODES: the hover\'s code line); the parameter row comes last')
    ap.add_argument("--out", default=str(P.ROOT / "docs/user/assets/benchmarks/si_accuracy_cost.json"))
    a = ap.parse_args()
    P.SKIP.update(f for f in P.FAM if f.startswith("ace_wl_"))      # as the main PNG: the wL ladder has its own
    rows = [(lab, json.load(open(c)), json.load(open(k[0])) if k else {}) for lab, c, *k in a.row] + [("fitted parameters", None, {})]
    fig = figure(rows)
    pathlib.Path(a.out).write_text(json.dumps(fig, separators=(",", ":"), allow_nan=False, default=float) + "\n")   # newline: end-of-file-fixer
    print("wrote", a.out, f"({len(fig['data'])} traces)")


if __name__ == "__main__":
    main()
