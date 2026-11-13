// Interactive figures: each <div class="plotly-figure"> holds a fallback image (no JavaScript, print); the Plotly
// spec beside it (same name, .json; bench/si_qoi/plot_cost_web.py) replaces it on screen.  plotly.js is loaded
// only on pages that have such a figure.  Colours follow the Material palette, also when the reader switches it.
const PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js";
let plotlyReady = null;

function loadPlotly() {
  if (!plotlyReady) {
    plotlyReady = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = PLOTLY_JS; s.onload = () => resolve(window.Plotly); s.onerror = reject;
      document.head.appendChild(s);
    });
  }
  return plotlyReady;
}

// Material's palette colours are nested custom properties (slate: hsla(var(--md-hue), ...)), which Plotly cannot
// parse: resolve each through a probe element's computed colour, which the browser gives as rgb()/rgba().
function theme() {
  const probe = document.createElement("span");
  document.body.appendChild(probe);
  const v = (name) => { probe.style.color = `var(${name})`; return getComputedStyle(probe).color; };
  const t = { fg: v("--md-default-fg-color"), muted: v("--md-default-fg-color--light"),
              grid: v("--md-default-fg-color--lightest") };
  probe.remove();
  return t;
}

function themed(layout) {
  const t = theme(), axis = { color: t.muted, gridcolor: t.grid, linecolor: t.muted };
  const upd = { "font.color": t.fg, "legend.font.color": t.fg };
  (layout.annotations || []).forEach((a, i) => {          // panel titles in the text colour, reference labels muted
    upd[`annotations[${i}].font.color`] = /<b>/.test(a.text) ? t.fg : t.muted;
  });
  for (const k of Object.keys(layout).filter((k) => /^[xy]axis\d*$/.test(k))) {
    for (const [p, c] of Object.entries(axis)) upd[`${k}.${p}`] = c;
  }
  return upd;
}

// Buttons for the figure's presets (layout.meta.presets: {label, families: [legendgroup...] | null for all}): a
// preset shows its families and leaves the rest in the legend, hidden.  The first preset is the initial view.
function presetBar(presets, plot, traces) {
  const bar = document.createElement("div");
  bar.className = "plotly-figure__presets";
  presets.forEach((p, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = p.label;
    b.className = "md-button" + (i === 0 ? " md-button--primary" : "");
    b.addEventListener("click", () => {
      const shown = p.families && new Set(p.families);
      window.Plotly.restyle(plot, { visible: traces.map((t) => (!shown || shown.has(t.legendgroup) ? true : "legendonly")) });
      bar.querySelectorAll("button").forEach((x) => x.classList.toggle("md-button--primary", x === b));
    });
    bar.appendChild(b);
  });
  return bar;
}

async function render(el) {
  const img = el.querySelector("img");
  if (!img || el.dataset.rendered) return;
  el.dataset.rendered = "1";
  try {
    const [Plotly, fig] = await Promise.all([loadPlotly(),
      fetch(img.src.replace(/\.png$/, ".json")).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })]);
    const plot = document.createElement("div");
    plot.className = "plotly-figure__plot";
    const presets = (fig.layout.meta && fig.layout.meta.presets) || [];
    if (presets.length) el.appendChild(presetBar(presets, plot, fig.data));
    el.appendChild(plot);
    await Plotly.newPlot(plot, fig.data, fig.layout, { responsive: true, displaylogo: false,
                                                      modeBarButtonsToRemove: ["select2d", "lasso2d"] });
    await Plotly.relayout(plot, themed(fig.layout));
    el.classList.add("plotly-figure--live");                    // CSS hides the fallback image on screen
    new MutationObserver(() => Plotly.relayout(plot, themed(fig.layout)))
      .observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  } catch (e) {
    console.warn("plotly-figure: keeping the static image", e);  // offline, blocked CDN, missing spec
  }
}

document$.subscribe(() => document.querySelectorAll("div.plotly-figure").forEach(render));
