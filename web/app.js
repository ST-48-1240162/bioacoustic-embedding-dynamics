const COLAB = {
  demo: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Demo_Colab.ipynb",
  a: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Route_A_Bacpipe_Colab.ipynb",
  b: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Route_B_BMZ_Colab.ipynb",
};

const MATH = {
  pca: [
    String.raw`\tilde x_{id}=(x_{id}-\mu_d)/\sigma_d,\quad \mathbf{z}_i=W_{:2}^{\top}\tilde{\mathbf{x}}_i`,
  ],
  centroid: [
    String.raw`w_i=\dfrac{(c_i)_{+}}{\sum_{j\in b}(c_j)_{+}},\quad \mathbf{c}_b=\sum_{i\in b}w_i\mathbf{z}_i`,
    String.raw`\bar{c}_b=\dfrac{1}{|b|}\sum_{i\in b}c_i`,
  ],
  rate: [String.raw`r_b=n_b/\Delta t,\quad \Delta t=60\,\mathrm{s}`],
  peltBoth: [
    String.raw`y\in\{r_b,\,c_b^{(1)}\}`,
    String.raw`\min_{\tau}\sum_k C_{\mathrm{rbf}}(y_{\tau_{k-1}:\tau_k})+\beta|\tau|`,
  ],
  peltRate: [
    String.raw`y_b=r_b,\ \beta=3`,
    String.raw`\min_{\tau}\sum_k C_{\mathrm{rbf}}(y_{\tau_{k-1}:\tau_k})+\beta|\tau|`,
  ],
  peltPc1: [
    String.raw`y_b=c_b^{(1)},\ \beta=2.5`,
    String.raw`\min_{\tau}\sum_k C_{\mathrm{rbf}}(y_{\tau_{k-1}:\tau_k})+\beta|\tau|`,
  ],
  hmm: [
    String.raw`\mathbf{x}_t\mid s_t\sim\mathcal{N}(\boldsymbol{\mu}_{s_t},\mathrm{diag}(\boldsymbol{\sigma}^{2}_{s_t}))`,
    String.raw`\mathbf{x}^{\mathrm{emb}}_t=(c_b^{(1)},c_b^{(2)}),\quad \mathbf{x}^{\mathrm{act}}_t=(r_b,R_b,\bar{c}_b)`,
    String.raw`N_{\mathrm{sw}}=\sum_t\mathbf{1}[s_t\neq s_{t-1}]`,
  ],
  shuffle: [String.raw`\mathrm{start}'_i=\mathrm{start}_{\pi(i)}`],
};

const COPY = {
    kicker: "colab walkthrough",
    lede: "Runtime, Run all. The notebook clones this repo, runs BirdNET on a short real clip (or your manifest), and plots embedding dynamics for that recording.",
    ask_label: "what it is doing",
    ask_h: "Embeddings during the recording",
    ask_p: "BirdNET already names the species and emits a 1024-d vector for each detection. The usual next step is to count those calls. This notebook keeps the vectors and watches whether that cloud moves during the file. Demo, Route A, and Route B all use real BirdNET embeddings on short test audio (~1 min).",
    walk_label: "pipeline",
    keys: "Left and right arrows change the step. The drawing on the left is a cartoon. It is not the PNG Colab exports.",
    prev: "prev",
    next: "next",
    fig_label: "reports/",
    fig_h: "Output figures",
    fig_note: "Demo uses 15 s bins on the bacpipe test wav (~1 min, 22 BirdNET detections, five occupied bins). Route A and Route B site bundles only include PCA and UMAP (default 60 s bins on that clip).",
    fig_missing: "This PNG is not in the site bundle. Colab still writes it on Run all.",
    runs: [
      { id: "demo", label: "Demo" },
      { id: "bmz", label: "Route B" },
      { id: "bacpipe", label: "Route A" },
    ],
    nb_label: "notebooks",
    nb_h: "Which notebook",
    steps: [
      {
        chip: "detections",
        file: "JSONL manifest",
        title: "One JSON line per detection",
        body: "Each line has start_s, species, confidence, and an optional embedding. The demo runs BMZ BirdNET on the bacpipe bundled test wav (~1 min). Empty bins stay on the timeline as rate 0.",
      },
      {
        chip: "vectors",
        file: "BirdNET or demo mapper",
        title: "Where the 1024-d numbers come from",
        body: "Route A (bacpipe) and Route B (BMZ) attach BirdNET embeddings. Pass --make-sample only if you want a synthetic JSONL for offline testing. Mixing real vectors with missing ones is an error.",
      },
      {
        chip: "geometry",
        file: "pca_species.png and umap_species.png",
        title: "PCA and UMAP",
        body: "Same detections, two maps. PCA is linear. UMAP pulls nearby points into islands. Color is species. The axis numbers are not Hertz or meters. Demo blobs look tidy because the prototypes were written in. BirdNET clouds usually look messier.",
        math: MATH.pca,
      },
      {
        chip: "trajectory",
        file: "trajectory_pca.png",
        title: "Centroid of each minute",
        body: "Calls go into time bins (15 s in the demo notebook, 60 s by default). The bin mean is weighted by confidence, so weak detections pull less. The plot is that mean in PC1 and PC2. Color is the embedding HMM state. Arrows still follow time.",
        math: MATH.centroid,
      },
      {
        chip: "breaks",
        file: "changepoints.png and trajectory_changepoints.png",
        title: "Change-points on rate and on PC1",
        body: "PELT on detection rate (empty bins count as 0) finds when calling gets busier or quieter. PELT on PC1 finds when the average vector jumps. A red line on the rate plot only means more or fewer calls. Rate uses \u03b2 = 3, PC1 uses \u03b2 = 2.5.",
        math: [...MATH.rate, ...MATH.peltBoth],
      },
      {
        chip: "HMM",
        file: "hmm_regimes.png",
        title: "Two HMMs",
        body: "The upper row is fit on PC1/PC2 centroids, so the states live in embedding space. The lower row is fit on call rate, species count, and mean confidence. Features are standardized first. 0, 1, and 2 are just labels.",
        math: MATH.hmm,
      },
      {
        chip: "shuffle",
        file: "shuffle_null.png",
        title: "Shuffle start_s",
        body: "Only the times are permuted. Species and embeddings stay on the same rows. If the PC1 breaks and HMM flips depended on order, the shuffled side should look noisier. Compare hmm_n_switches with shuffle_hmm_n_switches in summary.json. The fit still uses a fixed number of states, so the right-hand HMM will not go flat.",
        math: MATH.shuffle,
      },
    ],
    nbs: [
      { name: "Demo", meta: "CPU, about 5-10 min. BMZ BirdNET on the bacpipe test wav, 15 s bins.", href: COLAB.demo },
      { name: "Route B", meta: "CPU, about 5-10 min. BMZ BirdNET, 1024-d. Uses bacpipe test wav if /content/audio is empty.", href: COLAB.b },
      { name: "Route A", meta: "T4 if you have one. bacpipe BirdNET on the bundled test wavs. The first run downloads weights.", href: COLAB.a },
    ],
};

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generate() {
  const rng = mulberry32(42);
  const species = ["A", "B", "C", "D"];
  const centers = {
    0: { A: [0.18, 0.72], B: [0.28, 0.55] },
    1: { A: [0.42, 0.48], B: [0.52, 0.38], C: [0.62, 0.58] },
    2: { C: [0.72, 0.32], D: [0.82, 0.22] },
  };
  const points = [];
  for (let i = 0; i < 96; i++) {
    const t = i / 95;
    const regime = t < 1 / 3 ? 0 : t < 2 / 3 ? 1 : 2;
    const pool = Object.keys(centers[regime]);
    const sp = pool[Math.floor(rng() * pool.length)];
    const c = centers[regime][sp];
    points.push({
      t,
      regime,
      sp,
      x: c[0] + (rng() - 0.5) * 0.11,
      y: c[1] + (rng() - 0.5) * 0.11,
      conf: 0.55 + rng() * 0.4,
    });
  }
  const nBins = 24;
  const bins = [];
  for (let b = 0; b < nBins; b++) {
    const t0 = b / nBins;
    const t1 = (b + 1) / nBins;
    const members = points.filter((p) => p.t >= t0 && p.t < t1);
    const tMid = (t0 + t1) / 2;
    if (!members.length) {
      bins.push({ t: tMid, x: null, y: null, rate: 0, state: 0 });
      continue;
    }
    let wsum = 0;
    let x = 0;
    let y = 0;
    for (const m of members) {
      wsum += m.conf;
      x += m.x * m.conf;
      y += m.y * m.conf;
    }
    x /= wsum;
    y /= wsum;
    const state = tMid < 1 / 3 ? 0 : tMid < 2 / 3 ? 1 : 2;
    bins.push({ t: tMid, x, y, rate: members.length, state });
  }
  const shuf = points.map((p, i) => ({ ...p, t: points[(i * 17) % points.length].t }));
  shuf.sort((a, b) => a.t - b.t);
  const shufBins = [];
  for (let b = 0; b < nBins; b++) {
    const t0 = b / nBins;
    const t1 = (b + 1) / nBins;
    const members = shuf.filter((p) => p.t >= t0 && p.t < t1);
    const tMid = (t0 + t1) / 2;
    if (!members.length) {
      shufBins.push({ t: tMid, x: null, y: null, rate: 0, state: (b % 3) });
      continue;
    }
    let x = 0;
    let y = 0;
    for (const m of members) {
      x += m.x;
      y += m.y;
    }
    shufBins.push({
      t: tMid,
      x: x / members.length,
      y: y / members.length,
      rate: members.length,
      state: b % 3,
    });
  }
  return { points, bins, shufBins, species };
}

const FIG_DEFS = [
  { id: "pca_species.png", title: "species in PCA", body: "One point per detection. Color is species. Axes are the first two principal components of the scaled embedding. Useful if you want to see whether species sit apart. The tick labels are not physical units.", math: MATH.pca },
  { id: "umap_species.png", title: "species in UMAP", body: "Same points, nonlinear map. Handy if PCA is a blob but local groups still exist. A distance on this plot is not a PCA distance." },
  { id: "trajectory_pca.png", title: "minute centroids", body: "Each marker is one minute's confidence-weighted centroid in PC space. Color is HMM state. Needs many occupied bins over a long recording; two points is just a line.", math: MATH.centroid },
  { id: "changepoints.png", title: "call rate", body: "Call rate against minutes, with empty bins at 0. Red dashed lines in the Colab PNG are PELT breaks in how often animals called.", math: [...MATH.rate, ...MATH.peltRate] },
  { id: "trajectory_changepoints.png", title: "centroid PC1", body: "PC1 of the centroid against minutes. Breaks here are shifts in the average vector. Put changepoints.png next to it.", math: MATH.peltPc1 },
  { id: "hmm_regimes.png", title: "two HMMs", body: "Top: HMM on embedding centroids. Bottom: HMM on activity stats. They do not have to agree.", math: MATH.hmm },
  { id: "shuffle_null.png", title: "shuffle times", body: "Original times on the left, shuffled start_s on the right. The test is whether the ordered structure survives the permutation.", math: MATH.shuffle },
];

const RUN_FIG_IDS = {
  demo: [
    "pca_species.png",
    "umap_species.png",
    "trajectory_pca.png",
    "changepoints.png",
    "trajectory_changepoints.png",
    "hmm_regimes.png",
    "shuffle_null.png",
  ],
  bmz: ["pca_species.png", "umap_species.png"],
  bacpipe: ["pca_species.png", "umap_species.png"],
};

function figsForRun(run) {
  const ids = RUN_FIG_IDS[run] || RUN_FIG_IDS.demo;
  return ids.map((id) => FIG_DEFS.find((f) => f.id === id)).filter(Boolean);
}

const STEP_FIG_ID = {
  2: "pca_species.png",
  3: "trajectory_pca.png",
  4: "changepoints.png",
  5: "hmm_regimes.png",
  6: "shuffle_null.png",
};

const DATA = generate();
const SPECIES_COLOR = {
  A: "var-nacht",
  B: "var-gold",
  C: "var-rubric",
  D: "var-fg",
};

let step = 0;
let figIdx = 0;
let runId = "demo";

function cssColor(name) {
  const shell = document.querySelector(".thoth-page-shell");
  const map = {
    "var-nacht": "--lib-nacht",
    "var-gold": "--lib-gold",
    "var-rubric": "--lib-rubric",
    "var-fg": "--fg",
  };
  return getComputedStyle(shell).getPropertyValue(map[name]).trim();
}

function draw() {
  const canvas = document.getElementById("viz");
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || 720;
  const cssH = canvas.clientHeight || 352;
  if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(cssH * dpr)) {
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = cssW;
  const h = cssH;
  ctx.clearRect(0, 0, w, h);
  const pad = 28;
  const plotH = step === 4 || step === 6 ? h - 88 : h - 36;
  const plotW = w - pad * 2;

  function sx(x) {
    return pad + x * plotW;
  }
  function sy(y) {
    return pad + (1 - y) * (plotH - pad);
  }

  const faint = getComputedStyle(document.body).getPropertyValue("--fg-dim").trim();
  const ink = getComputedStyle(document.body).getPropertyValue("--fg").trim();
  const rubric = cssColor("var-rubric");
  const nacht = cssColor("var-nacht");
  const gold = cssColor("var-gold");
  const stateCol = [nacht, gold, rubric];

  ctx.font = "11px Manrope, system-ui, sans-serif";
  ctx.fillStyle = faint;
  if (step <= 2) {
    ctx.fillText("PC2", 8, 18);
    ctx.fillText("PC1", w - 36, plotH + 8);
  }

  if (step === 0) {
    DATA.points.forEach((p) => {
      ctx.globalAlpha = 0.22;
      ctx.fillStyle = cssColor(SPECIES_COLOR[p.sp]);
      ctx.fillRect(sx(p.x) - 2, sy(p.y) - 2, 4, 4);
    });
    ctx.globalAlpha = 1;
    ctx.fillStyle = ink;
    ctx.font = "600 14px Manrope, system-ui, sans-serif";
    ctx.fillText("just the detections so far", pad, h - 14);
    return;
  }

  if (step === 1) {
    DATA.points.forEach((p) => {
      ctx.fillStyle = cssColor(SPECIES_COLOR[p.sp]);
      ctx.globalAlpha = 0.85;
      ctx.fillRect(pad + p.t * plotW, 40 + p.regime * 70, 3, 36);
    });
    ctx.globalAlpha = 1;
    ctx.fillStyle = faint;
    ctx.fillText("t", w - 24, 210);
    ["0", "1", "2"].forEach((r, i) => {
      ctx.fillStyle = faint;
      ctx.fillText(`regime ${r}`, pad, 36 + i * 70);
    });
    return;
  }

  DATA.points.forEach((p) => {
    ctx.globalAlpha = step >= 3 ? 0.22 : 0.8;
    ctx.fillStyle = cssColor(SPECIES_COLOR[p.sp]);
    ctx.fillRect(sx(p.x) - 2, sy(p.y) - 2, 4, 4);
  });
  ctx.globalAlpha = 1;

  const useBins = step === 6 ? DATA.shufBins : DATA.bins;
  const occ = useBins.filter((b) => b.x != null);

  if (step >= 3) {
    ctx.beginPath();
    occ.forEach((b, i) => {
      const x = sx(b.x);
      const y = sy(b.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = faint;
    ctx.lineWidth = 1;
    ctx.stroke();
    occ.forEach((b) => {
      ctx.fillStyle = step >= 5 ? stateCol[b.state] : ink;
      ctx.fillRect(sx(b.x) - 3, sy(b.y) - 3, 6, 6);
    });
  }

  if (step === 4 || step === 6) {
    const base = h - 52;
    const maxR = Math.max(...useBins.map((b) => b.rate), 1);
    useBins.forEach((b, i) => {
      const x = pad + (i / useBins.length) * plotW;
      const bh = (b.rate / maxR) * 28;
      ctx.fillStyle = faint;
      ctx.fillRect(x, base - bh, plotW / useBins.length - 1, bh);
    });
    const breaks = step === 6 ? [] : [8, 16];
    breaks.forEach((i) => {
      const x = pad + (i / useBins.length) * plotW;
      ctx.strokeStyle = rubric;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(x, pad);
      ctx.lineTo(x, h - 18);
      ctx.stroke();
      ctx.setLineDash([]);
    });
    ctx.fillStyle = faint;
    ctx.fillText("calls per bin, empty bins included", pad, h - 8);
  }
}

function renderMath(el, latexList) {
  if (!el) return;
  el.replaceChildren();
  if (!latexList || !latexList.length || typeof katex === "undefined") {
    el.hidden = true;
    return;
  }
  latexList.forEach((tex) => {
    const line = document.createElement("div");
    katex.render(tex, line, { displayMode: true, throwOnError: false });
    el.appendChild(line);
  });
  el.hidden = false;
}

function renderCopy() {
  const c = COPY;
  document.querySelectorAll("[data-i]").forEach((el) => {
    const key = el.getAttribute("data-i");
    if (typeof c[key] === "string") el.textContent = c[key];
  });
  document.getElementById("prev-btn").textContent = c.prev;
  document.getElementById("next-btn").textContent = c.next;

  const chips = document.getElementById("step-chips");
  chips.replaceChildren();
  c.steps.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "lite-chip-btn" + (i === step ? " is-on" : "");
    b.textContent = s.chip;
    b.setAttribute("aria-selected", i === step ? "true" : "false");
    b.addEventListener("click", () => setStep(i));
    chips.appendChild(b);
  });
  const s = c.steps[step];
  document.getElementById("step-file").textContent = s.file;
  document.getElementById("step-title").textContent = s.title;
  document.getElementById("step-body").textContent = s.body;
  renderMath(document.getElementById("step-math"), s.math);

  const runChips = document.getElementById("run-chips");
  runChips.replaceChildren();
  c.runs.forEach((r) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "lite-chip-btn" + (r.id === runId ? " is-on" : "");
    b.textContent = r.label;
    b.addEventListener("click", () => {
      runId = r.id;
      renderCopy();
    });
    runChips.appendChild(b);
  });

  const figs = figsForRun(runId);
  if (figIdx >= figs.length) figIdx = 0;

  const figList = document.getElementById("fig-list");
  figList.replaceChildren();
  figs.forEach((f, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "fig-item" + (i === figIdx ? " is-on" : "");
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", i === figIdx ? "true" : "false");
    const name = document.createElement("span");
    name.className = "fig-item-name";
    name.textContent = f.title;
    b.appendChild(name);
    if (i === figIdx) {
      const body = document.createElement("span");
      body.className = "fig-item-body";
      body.textContent = f.body;
      b.appendChild(body);
    }
    b.addEventListener("click", () => {
      figIdx = i;
      renderCopy();
    });
    figList.appendChild(b);
  });

  const chosen = figs[figIdx];
  document.getElementById("fig-file").textContent = chosen.id;
  document.getElementById("fig-body").textContent = chosen.body;
  renderMath(document.getElementById("fig-math"), chosen.math);

  const img = document.getElementById("fig-img");
  const frame = document.getElementById("fig-frame");
  const miss = document.getElementById("fig-missing");
  img.src = `./reports/${runId}/${chosen.id}`;
  img.alt = chosen.title;
  frame.classList.remove("is-empty");
  miss.hidden = true;

  const nb = document.getElementById("nb-list");
  nb.replaceChildren();
  c.nbs.forEach((n) => {
    const row = document.createElement("div");
    row.className = "nb-row";
    row.innerHTML = `<span class="nb-name">${n.name}</span><span class="nb-meta">${n.meta}</span>`;
    const a = document.createElement("a");
    a.className = "lite-chip-btn";
    a.href = n.href;
    a.textContent = "Colab";
    row.appendChild(a);
    nb.appendChild(row);
  });
}

function setStep(i) {
  step = Math.max(0, Math.min(COPY.steps.length - 1, i));
  const figId = STEP_FIG_ID[step];
  if (figId != null) {
    const figs = figsForRun(runId);
    const idx = figs.findIndex((f) => f.id === figId);
    if (idx >= 0) figIdx = idx;
  }
  renderCopy();
  draw();
}

document.getElementById("prev-btn").addEventListener("click", () => setStep(step - 1));
document.getElementById("next-btn").addEventListener("click", () => setStep(step + 1));
document.getElementById("theme-btn").addEventListener("click", () => {
  const body = document.body;
  const dark = body.getAttribute("data-theme") !== "dark";
  body.setAttribute("data-theme", dark ? "dark" : "light");
  const btn = document.getElementById("theme-btn");
  btn.setAttribute("aria-pressed", dark ? "true" : "false");
  btn.setAttribute("aria-label", dark ? "Switch to light" : "Switch to dark");
  draw();
});
window.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") setStep(step + 1);
  if (e.key === "ArrowLeft") setStep(step - 1);
});

renderCopy();
draw();
window.addEventListener("resize", draw);
