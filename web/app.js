const COLAB = {
  demo: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Demo_Colab.ipynb",
  a: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Route_A_Bacpipe_Colab.ipynb",
  b: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Route_B_BMZ_Colab.ipynb",
};

const MATH = {
  goal: [
    String.raw`\text{question: does the cloud } \{\mathbf{x}_i\} \text{ move along the recording?}`,
    String.raw`t \mapsto \mathbf{c}(t)=\text{confidence-weighted mean in PC space}`,
  ],
  embedding: [
    String.raw`\mathbf{x}_i\in\mathbb{R}^{1024},\quad c_i\in[0,1],\quad \text{species}_i`,
    String.raw`\text{BirdNET: } \mathbf{x}_i \leftarrow f_{\mathrm{BN}}(\text{audio window at } \mathrm{start\_s}_i)`,
  ],
  manifest: [
    String.raw`\text{one JSON line per detection: } (\mathrm{start\_s},\mathrm{end\_s},\text{species},c_i,\mathbf{x}_i)`,
  ],
  scale: [
    String.raw`\tilde x_{id}=\dfrac{x_{id}-\mu_d}{\sigma_d},\quad \mu_d=\mathrm{mean}_i(x_{id}),\ \sigma_d=\mathrm{std}_i(x_{id})`,
  ],
  pca: [
    String.raw`\mathbf{z}_i=W_{:2}^{\top}\tilde{\mathbf{x}}_i,\quad W\in\mathbb{R}^{D\times 2}`,
    String.raw`\mathrm{PC1},\mathrm{PC2}=\operatorname*{arg\,max}\text{ variance of }\{\tilde{\mathbf{x}}_i\}`,
  ],
  umap: [
    String.raw`\mathbf{u}_i=\mathrm{UMAP}(\tilde{\mathbf{x}}_i)\in\mathbb{R}^2,\quad n_{\mathrm{neighbors}}=15`,
    String.raw`\text{keeps local neighbors; distances on the plot are not PCA distances}`,
  ],
  cosine: [
    String.raw`d_{\cos}(i,j)=1-\dfrac{\mathbf{x}_i\cdot\mathbf{x}_j}{\lVert\mathbf{x}_i\rVert\,\lVert\mathbf{x}_j\rVert}`,
    String.raw`\text{compare intra-species vs inter-species means in } summary.json`,
  ],
  bin: [
    String.raw`b(i)=\left\lfloor\dfrac{\mathrm{start\_s}(i)-t_0}{\Delta t}\right\rfloor,\quad \Delta t=\texttt{bin\_s}`,
    String.raw`t_0=\min_i\mathrm{start\_s}(i),\quad t_1=\max_i\mathrm{start\_s}(i)`,
  ],
  centroid: [
    String.raw`w_i=\dfrac{(c_i)_{+}}{\sum_{j\in b}(c_j)_{+}},\quad (c)_{+}=\max(c,0)`,
    String.raw`\mathbf{c}_b=\sum_{i\in b}w_i\mathbf{z}_i,\quad c_b^{(1)}=\mathbf{c}_b\cdot e_1`,
    String.raw`\bar{c}_b=\dfrac{1}{|b|}\sum_{i\in b}c_i,\quad r_b=\dfrac{n_b}{\Delta t}`,
  ],
  rate: [
    String.raw`r_b=\dfrac{n_b}{\Delta t},\quad n_b=\#\{\text{detections in bin } b\}`,
    String.raw`\text{empty bins stay on the timeline with } r_b=0`,
  ],
  peltBoth: [
    String.raw`y_b\in\{r_b,\,c_b^{(1)}\},\quad \text{segment } y_{1:T}`,
    String.raw`\min_{\tau}\ \sum_{k=1}^{|\tau|+1} C_{\mathrm{rbf}}\!\left(y_{\tau_{k-1}:\tau_k}\right)+\beta\,|\tau|`,
  ],
  peltRate: [
    String.raw`y_b=r_b,\quad \beta=3`,
    String.raw`\min_{\tau}\ \sum_k C_{\mathrm{rbf}}(y_{\tau_{k-1}:\tau_k})+\beta|\tau|`,
  ],
  peltPc1: [
    String.raw`y_b=c_b^{(1)},\quad \beta=2.5`,
    String.raw`\min_{\tau}\ \sum_k C_{\mathrm{rbf}}(y_{\tau_{k-1}:\tau_k})+\beta|\tau|`,
  ],
  hmm: [
    String.raw`P(s_t\mid s_{t-1})=\mathbf{A}_{s_{t-1},s_t},\quad \mathbf{x}_t\mid s_t\sim\mathcal{N}(\boldsymbol{\mu}_{s_t},\mathrm{diag}(\boldsymbol{\sigma}^{2}_{s_t}))`,
    String.raw`\mathbf{x}^{\mathrm{emb}}_t=(c_b^{(1)},c_b^{(2)}),\quad \mathbf{x}^{\mathrm{act}}_t=(r_b,R_b,\bar{c}_b)`,
    String.raw`N_{\mathrm{sw}}=\sum_t\mathbf{1}[s_t\neq s_{t-1}]`,
  ],
  shuffle: [
    String.raw`\mathrm{start}'_i=\mathrm{start}_{\pi(i)},\quad \pi\text{ random permutation}`,
    String.raw`\text{species and } \mathbf{x}_i \text{ unchanged; only time order breaks}`,
  ],
};

const COPY = {
    kicker: "colab walkthrough",
    lede: "Takes a JSONL of bioacoustic detections (optional BirdNET embeddings) and runs PCA, UMAP, a binned centroid trajectory, change-point detection, and Gaussian HMMs on how the embedding cloud changes over time.",
    intro_label: "intro",
    intro_tabs: [
      { id: "overview", label: "overview" },
      { id: "concepts", label: "concepts" },
    ],
    ask_h: "Do embeddings move?",
    ask_p: [
      "BirdNET gives each detection window a species label and a $1024$-d vector $\\mathbf{x}_i$. Most pipelines stop at species counts.",
      "We keep the vectors. Each detection is a point in a large space. Does that cloud drift, jump, or settle into recurring shapes as the recording plays?",
      "Demo, Route A, and Route B all embed the bacpipe test wav ($\\sim 1\\,\\mathrm{min}$) with real BirdNET. The demo bins at $\\Delta t=15\\,\\mathrm{s}$ so a one-minute clip still has several points on the timeline.",
    ],
    ask_math: MATH.goal,
    concepts_h: "Notation",
    concepts_lede: "Same symbols as the code and summary.json. Pipeline steps and figures repeat the formulas when they apply.",
    concepts: [
      {
        title: "Detection embedding",
        body: [
          "BirdNET reads a short audio window and returns a species label, confidence $c_i$, and embedding $\\mathbf{x}_i\\in\\mathbb{R}^{1024}$. Similar calls sit close in that space even when the labels disagree.",
        ],
        math: MATH.embedding,
      },
      {
        title: "JSONL manifest",
        body: [
          "One JSON object per line. $\\mathrm{start\\_s}$ and $\\mathrm{end\\_s}$ place the call on the timeline. $\\text{species}$ and $c_i$ come from BirdNET. $\\mathbf{x}_i$ feeds PCA, UMAP, and trajectories.",
        ],
        math: MATH.manifest,
      },
      {
        title: "Scaling and PCA",
        body: [
          "Columns are standardized ($\\tilde x_{id}$) so one loud dimension does not dominate. PCA is linear: $\\mathrm{PC1}$ and $\\mathrm{PC2}$ are the top variance directions after scaling.",
        ],
        math: [...MATH.scale, ...MATH.pca],
      },
      {
        title: "UMAP",
        body: [
          "UMAP maps the same points into 2D with a nonlinear fit. Local clusters can separate here when PCA smears them. Axis numbers are arbitrary; trust neighborhoods, not distances.",
        ],
        math: MATH.umap,
      },
      {
        title: "Time bins and centroids",
        body: [
          "Detections fall into fixed-width bins from first to last call. Empty bins stay on the timeline at rate zero. Occupied bins get a confidence-weighted mean in PC space; that sequence is the trajectory.",
        ],
        math: [...MATH.bin, ...MATH.centroid],
      },
      {
        title: "Change-points and HMM",
        body: [
          "PELT cuts the binned series where a split beats one long segment, with penalty $\\beta|\\tau|$. Two HMMs label regimes: embedding centroids, and activity $(r_b,R_b,\\bar{c}_b)$.",
        ],
        math: [...MATH.peltBoth, ...MATH.hmm],
      },
    ],
    walk_label: "pipeline",
    keys: "Arrow keys change the step. The canvas is a sketch for each stage, not the Colab PNG.",
    prev: "prev",
    next: "next",
    fig_label: "reports/",
    fig_h: "figures",
    fig_note: "All three runs use the same bacpipe test wav ($\\sim 1\\,\\mathrm{min}$, $22$ detections). Demo has the full set at $\\Delta t=15\\,\\mathrm{s}$. Route A and Route B on this site stop at PCA and UMAP.",
    fig_footnote:
      "Demo and Route B show the same PCA and UMAP files. Same wav, same BMZ BirdNET run, analysis seed $42$. Demo also has trajectory, changepoint, HMM, and shuffle plots ($\\Delta t=15\\,\\mathrm{s}$, five occupied bins). Route B Colab matches that setup; the site just omits the extra PNGs. Route A runs bacpipe's own BirdNET on the same wav, so its two plots differ.",
    fig_missing: "Not included for this run. Try Demo, or run Colab.",
    step_fig_gap: "Only Demo includes this figure ($\\Delta t=15\\,\\mathrm{s}$). Route A and Route B stop at PCA and UMAP on the site.",
    runs: [
      { id: "demo", label: "Demo", href: COLAB.demo },
      { id: "bacpipe", label: "Route A", href: COLAB.a },
      { id: "bmz", label: "Route B", href: COLAB.b },
    ],
    steps: [
      {
        chip: "detections",
        file: "JSONL manifest",
        title: "One JSON line per detection",
        body: [
          "Each line is one BirdNET window: $\\mathrm{start\\_s}$, $\\text{species}$, $c_i$, and optional $\\mathbf{x}_i$.",
          "The demo runs BMZ BirdNET on the bacpipe test wav that ships with the repo ($\\sim 1\\,\\mathrm{min}$, $22$ detections on the site). Swap in your own wav or manifest if you want.",
        ],
        math: MATH.manifest,
      },
      {
        chip: "vectors",
        file: "BirdNET or demo mapper",
        title: "Where the 1024-d numbers come from",
        body: [
          "Route A (bacpipe) and Route B (bioacoustics-model-zoo) call BirdNET and write $\\mathbf{x}_i$ into the manifest. Each vector covers that window, not the whole file.",
          "Pass --make-sample only for offline tests with synthetic $128$-d vectors. Do not mix real embeddings with missing ones on the same run.",
        ],
        math: MATH.embedding,
      },
      {
        chip: "geometry",
        file: "pca_species.png and umap_species.png",
        title: "PCA and UMAP",
        body: [
          "Same detections, two 2D views. Color is species. Axes are abstract coordinates, not seconds or kHz.",
          "PCA is linear and easy to read: do species separate along $\\mathrm{PC1}$? UMAP pulls local neighborhoods apart; do not compare its distances to PCA.",
          "summary.json lists mean intra- vs inter-species cosine distance $d_{\\cos}$ as a quick separation read.",
        ],
        math: [...MATH.scale, ...MATH.pca, ...MATH.umap, ...MATH.cosine],
      },
      {
        chip: "trajectory",
        file: "trajectory_pca.png",
        title: "Centroid through time",
        body: [
          "Detections land in bins of width $\\texttt{bin\\_s}$ ($\\Delta t=15\\,\\mathrm{s}$ in the demo notebook, $60\\,\\mathrm{s}$ by default). The confidence-weighted centroid $\\mathbf{c}_b$ in PC space marks each occupied bin; arrows follow time.",
          "$\\bar{c}_b$ is a plain average per bin, not confidence-weighted. You need several occupied bins before the path is anything more than a stub.",
        ],
        math: [...MATH.bin, ...MATH.centroid],
      },
      {
        chip: "breaks",
        file: "changepoints.png and trajectory_changepoints.png",
        title: "Change-points on rate and on PC1",
        body: [
          "PELT hunts break locations that minimize segment cost plus $\\beta|\\tau|$.",
          "On detection rate $r_b$, a break means calling got busier or quieter (empty bins count as zero). On $c_b^{(1)}$, a break means the average embedding shifted. Defaults: $\\beta=3$ for rate, $\\beta=2.5$ for $\\mathrm{PC1}$.",
        ],
        math: [...MATH.rate, ...MATH.peltBoth],
      },
      {
        chip: "HMM",
        file: "hmm_regimes.png",
        title: "Two HMMs",
        body: [
          "A diagonal Gaussian HMM assigns each bin a hidden state $s_t$. Transitions are Markov; emissions are Gaussian in the chosen features.",
          "Top row: states on $(c_b^{(1)},c_b^{(2)})$ centroids. Bottom row: states on standardized $(r_b,R_b,\\bar{c}_b)$. State IDs are arbitrary ($2$–$3$ states depending on bin count).",
          "$N_{\\mathrm{sw}}$ in summary.json counts embedding-HMM switches along the timeline.",
        ],
        math: MATH.hmm,
      },
      {
        chip: "shuffle",
        file: "shuffle_null.png",
        title: "Shuffle start_s",
        body: [
          "Only $\\mathrm{start\\_s}$ is permuted; species and $\\mathbf{x}_i$ stay on their rows. Timeline order breaks; the point cloud does not.",
          "If changepoints and HMM switches track real time structure, the shuffled run should look noisier and switch less. Compare $N_{\\mathrm{sw}}$ with shuffle $N_{\\mathrm{sw}}$.",
        ],
        math: MATH.shuffle,
      },
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
    const x = c[0] + (rng() - 0.5) * 0.11;
    const y = c[1] + (rng() - 0.5) * 0.11;
    const umapNudge = { A: [0, -0.06], B: [0.07, 0.04], C: [-0.05, 0.08], D: [0.06, -0.05] }[sp] || [0, 0];
    points.push({
      t,
      regime,
      sp,
      x,
      y,
      umapX: Math.min(0.94, Math.max(0.06, x + umapNudge[0] + 0.06 * Math.sin(t * 13))),
      umapY: Math.min(0.94, Math.max(0.06, y + umapNudge[1] + 0.05 * Math.cos(t * 10))),
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
      bins.push({ t: tMid, x: null, y: null, rate: 0, state: 0, activityState: 0 });
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
    const activityState = members.length >= 5 ? 2 : members.length >= 2 ? 1 : 0;
    bins.push({ t: tMid, x, y, rate: members.length, state, activityState });
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
      shufBins.push({ t: tMid, x: null, y: null, rate: 0, state: b % 3, activityState: b % 3 });
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
      activityState: (b + 1) % 3,
    });
  }
  return { points, bins, shufBins, species };
}

function drawArrowhead(ctx, x0, y0, x1, y1, color) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len < 8) return;
  const ux = dx / len;
  const uy = dy / len;
  const ax = x1 - ux * 7;
  const ay = y1 - uy * 7;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(ax - uy * 4, ay + ux * 4);
  ctx.lineTo(ax + uy * 4, ay - ux * 4);
  ctx.closePath();
  ctx.fill();
}

function drawStateRibbon(ctx, bins, y0, hRow, label, stateKey, stateCol, pad, plotW) {
  const faint = getComputedStyle(document.body).getPropertyValue("--fg-dim").trim();
  ctx.font = "10px Manrope, system-ui, sans-serif";
  ctx.fillStyle = faint;
  ctx.fillText(label, pad, y0 + hRow * 0.72);
  bins.forEach((b, i) => {
    const x = pad + (i / bins.length) * plotW;
    const wCell = plotW / bins.length - 1;
    ctx.fillStyle = stateCol[b[stateKey] ?? 0] || faint;
    ctx.globalAlpha = b.x == null && stateKey === "state" ? 0.15 : 0.85;
    ctx.fillRect(x, y0, wCell, hRow);
  });
  ctx.globalAlpha = 1;
}

const FIG_DEFS = [
  {
    id: "pca_species.png",
    title: "species in PCA",
    body: [
      "One point per detection after column scaling ($\\tilde{\\mathbf{x}}_i$). Color is BirdNET species.",
      "$\\mathrm{PC1}$ and $\\mathrm{PC2}$ are the leading variance directions. Read overlap between species clouds, not absolute timbre.",
    ],
    math: [...MATH.scale, ...MATH.pca, ...MATH.cosine],
  },
  {
    id: "umap_species.png",
    title: "species in UMAP",
    body: [
      "Same scaled embeddings, UMAP with default $n_{\\mathrm{neighbors}}=15$. Useful for local structure; distances are not calibrated like PCA.",
      "Below four detections, the code falls back to PCA.",
    ],
    math: MATH.umap,
  },
  {
    id: "trajectory_pca.png",
    title: "binned centroids",
    body: [
      "Each marker is a bin's confidence-weighted centroid $\\mathbf{c}_b$ in PC space. Color is embedding HMM state $s_t$. The polyline follows bin order.",
      "Demo has five occupied bins on $\\sim 1\\,\\mathrm{min}$ with $\\texttt{bin\\_s}=15\\,\\mathrm{s}$. At default $\\Delta t=60\\,\\mathrm{s}$ on a short clip you often get two bins and a flat line.",
    ],
    math: [...MATH.bin, ...MATH.centroid],
  },
  {
    id: "changepoints.png",
    title: "call rate",
    body: [
      "$r_b=n_b/\\Delta t$ on the full bin grid, including empty bins at zero.",
      "Red dashed lines are PELT breaks ($\\beta=3$) where calling intensity shifts.",
    ],
    math: [...MATH.rate, ...MATH.peltRate],
  },
  {
    id: "trajectory_changepoints.png",
    title: "centroid PC1",
    body: [
      "$c_b^{(1)}$ of the binned centroid versus time. Breaks ($\\beta=2.5$) mark jumps in the average embedding, not changes in call count.",
      "Pair with changepoints.png: $r_b$ can spike while $\\mathbf{c}_b$ stays put, or the other way around.",
    ],
    math: [...MATH.centroid, ...MATH.peltPc1],
  },
  {
    id: "hmm_regimes.png",
    title: "two HMMs",
    body: [
      "Top: hidden states $s_t$ on embedding centroids $(c_b^{(1)},c_b^{(2)})$. Bottom: states on activity features after standardization.",
      "The two rows need not agree. Loud mixed-species calling and a pure embedding shift are different signals.",
    ],
    math: MATH.hmm,
  },
  {
    id: "shuffle_null.png",
    title: "shuffle times",
    body: [
      "Left: original timeline. Right: $\\mathrm{start\\_s}$ permuted, same $\\mathbf{x}_i$. Compare changepoint counts and $N_{\\mathrm{sw}}$ in summary.json.",
      "Big drops on the shuffled side suggest the original order carried something real.",
    ],
    math: MATH.shuffle,
  },
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
let introTab = "overview";

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
  const plotH = step === 4 || step === 6 ? h - 88 : step === 5 ? h - 118 : h - 36;
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

  function setVizLabels(show, yTex, xTex) {
    const vizLabels = document.getElementById("viz-labels");
    const yEl = document.getElementById("viz-label-y");
    const xEl = document.getElementById("viz-label-x");
    if (!vizLabels || !yEl || !xEl) return;
    vizLabels.hidden = !show;
    yEl.hidden = !yTex;
    xEl.hidden = !xTex;
    yEl.replaceChildren();
    xEl.replaceChildren();
    if (yTex) renderInline(yEl, yTex);
    if (xTex) renderInline(xEl, xTex);
  }

  setVizLabels(false, null, null);

  function drawScatter(alpha) {
    DATA.points.forEach((p) => {
      ctx.globalAlpha = alpha;
      ctx.fillStyle = cssColor(SPECIES_COLOR[p.sp]);
      ctx.fillRect(sx(p.x) - 2, sy(p.y) - 2, 4, 4);
    });
    ctx.globalAlpha = 1;
  }

  function drawCentroidPath(occ, { arrows = false, colorMarkers = false } = {}) {
    if (occ.length < 1) return;
    ctx.beginPath();
    occ.forEach((b, i) => {
      const x = sx(b.x);
      const y = sy(b.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = faint;
    ctx.lineWidth = colorMarkers ? 1 : 1.5;
    ctx.stroke();
    if (arrows) {
      for (let i = 1; i < occ.length; i++) {
        drawArrowhead(ctx, sx(occ[i - 1].x), sy(occ[i - 1].y), sx(occ[i].x), sy(occ[i].y), faint);
      }
    }
    occ.forEach((b) => {
      if (colorMarkers) {
        ctx.fillStyle = stateCol[b.state];
        ctx.beginPath();
        ctx.arc(sx(b.x), sy(b.y), 5, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle = ink;
        ctx.fillRect(sx(b.x) - 4, sy(b.y) - 4, 8, 8);
      }
    });
  }

  if (step === 0) {
    setVizLabels(true, null, "\\mathrm{start\\_s}");
    const laneY = h * 0.42;
    DATA.points.forEach((p) => {
      ctx.fillStyle = cssColor(SPECIES_COLOR[p.sp]);
      ctx.globalAlpha = 0.9;
      ctx.fillRect(pad + p.t * plotW, laneY, 3, 36);
    });
    ctx.globalAlpha = 1;
    ctx.fillStyle = faint;
    ctx.font = "11px Manrope, system-ui, sans-serif";
    ctx.fillText("JSONL manifest: one mark per detection (color = species)", pad, h - 10);
    return;
  }

  if (step === 1) {
    setVizLabels(true, null, "t");
    const laneY = h * 0.26;
    const hi = Math.floor(DATA.points.length * 0.45);
    DATA.points.forEach((p, i) => {
      ctx.fillStyle = cssColor(SPECIES_COLOR[p.sp]);
      ctx.globalAlpha = i === hi ? 1 : 0.32;
      ctx.fillRect(pad + p.t * plotW, laneY, 3, 34);
    });
    ctx.globalAlpha = 1;
    const hp = DATA.points[hi];
    const hx = pad + hp.t * plotW + 1;
    const boxY = h * 0.5;
    ctx.strokeStyle = faint;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(hx, laneY + 36);
    ctx.lineTo(hx, boxY - 6);
    ctx.stroke();
    ctx.strokeRect(hx - 30, boxY, 60, 40);
    ctx.fillStyle = ink;
    ctx.font = "600 11px Manrope, system-ui, sans-serif";
    ctx.fillText("x_i", hx - 8, boxY + 16);
    ctx.fillStyle = faint;
    ctx.font = "10px Manrope, system-ui, sans-serif";
    ctx.fillText("1024-d", hx - 14, boxY + 30);
    ctx.font = "11px Manrope, system-ui, sans-serif";
    ctx.fillText("one BirdNET window -> embedding column in JSONL", pad, h - 10);
    return;
  }

  if (step === 2) {
    setVizLabels(false, null, null);
    const mid = pad + plotW * 0.5;
    const halfW = plotW * 0.46;
    const sxPca = (x) => pad + x * halfW;
    const sxU = (x) => mid + 10 + x * halfW;
    const plotTop = pad;
    const plotBottom = h - 36;
    const syPlot = (y) => plotTop + (1 - y) * (plotBottom - plotTop);

    ctx.strokeStyle = faint;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(mid, plotTop);
    ctx.lineTo(mid, plotBottom);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = faint;
    ctx.font = "10px Manrope, system-ui, sans-serif";
    ctx.fillText("PCA (linear)", pad + 4, plotTop + 12);
    ctx.fillText("UMAP (nonlinear)", mid + 12, plotTop + 12);
    ctx.fillText("PC2", pad + 2, plotTop + 2);
    ctx.fillText("PC1", pad + halfW - 22, plotBottom + 14);
    ctx.fillText("UMAP-2", mid + 8, plotTop + 2);
    ctx.fillText("UMAP-1", mid + halfW - 8, plotBottom + 14);

    DATA.points.forEach((p) => {
      ctx.fillStyle = cssColor(SPECIES_COLOR[p.sp]);
      ctx.globalAlpha = 0.82;
      ctx.fillRect(sxPca(p.x) - 2, syPlot(p.y) - 2, 4, 4);
      ctx.globalAlpha = 0.82;
      ctx.fillRect(sxU(p.umapX) - 2, syPlot(p.umapY) - 2, 4, 4);
    });
    ctx.globalAlpha = 1;
    ctx.fillStyle = faint;
    ctx.font = "11px Manrope, system-ui, sans-serif";
    ctx.fillText("same detections, two 2D maps (species color)", pad, h - 10);
    return;
  }

  const useBins = step === 6 ? DATA.shufBins : DATA.bins;
  const occ = useBins.filter((b) => b.x != null);

  if (step === 3) {
    setVizLabels(true, "\\mathrm{PC2}", "\\mathrm{PC1}");
    drawScatter(0.22);
    drawCentroidPath(occ, { arrows: true });
    ctx.fillStyle = faint;
    ctx.font = "11px Manrope, system-ui, sans-serif";
    ctx.fillText("confidence-weighted centroid path (time order)", pad, h - 10);
    return;
  }

  if (step === 5) {
    setVizLabels(true, "\\mathrm{PC2}", "\\mathrm{PC1}");
    drawScatter(0.12);
    occ.forEach((b) => {
      ctx.fillStyle = stateCol[b.state];
      ctx.beginPath();
      ctx.arc(sx(b.x), sy(b.y), 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = ink;
      ctx.font = "9px Manrope, system-ui, sans-serif";
      ctx.fillText(String(b.state), sx(b.x) - 3, sy(b.y) + 3);
    });
    stateCol.forEach((col, i) => {
      const lx = pad + i * 52;
      const ly = h - 102;
      ctx.fillStyle = col;
      ctx.fillRect(lx, ly, 10, 10);
      ctx.fillStyle = faint;
      ctx.font = "10px Manrope, system-ui, sans-serif";
      ctx.fillText(`embed ${i}`, lx + 14, ly + 9);
    });
    drawStateRibbon(ctx, useBins, h - 88, 14, "embedding HMM", "state", stateCol, pad, plotW);
    drawStateRibbon(ctx, useBins, h - 68, 14, "activity HMM", "activityState", stateCol, pad, plotW);
    ctx.fillStyle = faint;
    ctx.font = "11px Manrope, system-ui, sans-serif";
    ctx.fillText("disks = HMM state on PC centroid; ribbons = states over bins", pad, h - 10);
    return;
  }

  if (step === 4) {
    drawScatter(0.18);
    drawCentroidPath(occ);
    const base = h - 52;
    const maxR = Math.max(...useBins.map((b) => b.rate), 1);
    useBins.forEach((b, i) => {
      const x = pad + (i / useBins.length) * plotW;
      const bh = (b.rate / maxR) * 28;
      ctx.fillStyle = faint;
      ctx.fillRect(x, base - bh, plotW / useBins.length - 1, bh);
    });
    [8, 16].forEach((i) => {
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
    ctx.font = "11px Manrope, system-ui, sans-serif";
    ctx.fillText("top: PC centroid path; bottom: r_b with PELT breaks (red)", pad, h - 8);
    setVizLabels(true, "r_b", "t");
    return;
  }

  if (step === 6) {
    const mid = pad + plotW * 0.5;
    const halfW = plotW * 0.46;
    const sxL = (x) => pad + x * halfW;
    const sxR = (x) => mid + 10 + x * halfW;

    ctx.strokeStyle = faint;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(mid, pad);
    ctx.lineTo(mid, h - 52);
    ctx.stroke();
    ctx.setLineDash([]);

    drawScatter(0.12);

    function drawHalfCentroids(occ, sxFn) {
      if (!occ.length) return;
      ctx.beginPath();
      occ.forEach((b, i) => {
        const px = sxFn(b.x);
        const py = sy(b.y);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = faint;
      ctx.lineWidth = 1;
      ctx.stroke();
      occ.forEach((b) => {
        ctx.fillStyle = ink;
        ctx.fillRect(sxFn(b.x) - 3, sy(b.y) - 3, 6, 6);
      });
    }

    const occOrig = DATA.bins.filter((b) => b.x != null);
    const occShuf = DATA.shufBins.filter((b) => b.x != null);
    drawHalfCentroids(occOrig, sxL);
    drawHalfCentroids(occShuf, sxR);

    ctx.fillStyle = faint;
    ctx.font = "10px Manrope, system-ui, sans-serif";
    ctx.fillText("original times", pad + 4, pad + 12);
    ctx.fillText("shuffled start_s", mid + 10, pad + 12);

    const base = h - 52;
    const maxR = Math.max(...DATA.bins.map((b) => b.rate), ...DATA.shufBins.map((b) => b.rate), 1);
    DATA.bins.forEach((b, i) => {
      const x = pad + (i / DATA.bins.length) * halfW;
      const bh = (b.rate / maxR) * 28;
      ctx.fillStyle = faint;
      ctx.fillRect(x, base - bh, halfW / DATA.bins.length - 1, bh);
    });
    DATA.shufBins.forEach((b, i) => {
      const x = mid + 10 + (i / DATA.shufBins.length) * halfW;
      const bh = (b.rate / maxR) * 28;
      ctx.fillStyle = faint;
      ctx.fillRect(x, base - bh, halfW / DATA.shufBins.length - 1, bh);
    });

    ctx.fillStyle = faint;
    ctx.font = "11px Manrope, system-ui, sans-serif";
    ctx.fillText("same embeddings; permuted times on the right", pad, h - 8);
    setVizLabels(true, "r_b", "t");
    return;
  }
}

function katexReady() {
  return typeof katex !== "undefined" && typeof katex.render === "function";
}

function renderMath(el, latexList) {
  if (!el) return;
  el.replaceChildren();
  if (!latexList || !latexList.length) {
    el.hidden = true;
    return;
  }
  if (!katexReady()) {
    latexList.forEach((tex) => {
      const pre = document.createElement("pre");
      pre.className = "math-fallback";
      pre.textContent = tex;
      el.appendChild(pre);
    });
    el.hidden = false;
    return;
  }
  latexList.forEach((tex) => {
    const line = document.createElement("div");
    try {
      katex.render(tex, line, { displayMode: true, throwOnError: false, strict: "ignore" });
    } catch {
      line.textContent = tex;
    }
    el.appendChild(line);
  });
  el.hidden = false;
}

function renderInline(el, tex) {
  if (!el || !tex) return;
  el.replaceChildren();
  if (!katexReady()) {
    el.textContent = tex;
    return;
  }
  try {
    katex.render(tex, el, { displayMode: false, throwOnError: false, strict: "ignore" });
  } catch {
    el.textContent = tex;
  }
}

function appendInlineMath(parent, text) {
  if (!text) return;
  if (!katexReady()) {
    parent.appendChild(document.createTextNode(text));
    return;
  }
  const re = /\$([^$]+)\$/g;
  let last = 0;
  let match;
  while ((match = re.exec(text)) !== null) {
    if (match.index > last) {
      parent.appendChild(document.createTextNode(text.slice(last, match.index)));
    }
    const span = document.createElement("span");
    span.className = "math-inline";
    try {
      katex.render(match[1], span, { displayMode: false, throwOnError: false, strict: "ignore" });
    } catch {
      span.textContent = match[1];
    }
    parent.appendChild(span);
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    parent.appendChild(document.createTextNode(text.slice(last)));
  }
}

function setRichText(el, text) {
  if (!el) return;
  el.replaceChildren();
  appendInlineMath(el, text);
}

function setParagraphs(el, body) {
  if (!el) return;
  el.replaceChildren();
  const parts = Array.isArray(body) ? body : body ? [body] : [];
  parts.forEach((text) => {
    const p = document.createElement("p");
    appendInlineMath(p, text);
    el.appendChild(p);
  });
}

function renderIntroTabs() {
  const tabs = document.getElementById("intro-tabs");
  const overview = document.getElementById("intro-panel-overview");
  const concepts = document.getElementById("intro-panel-concepts");
  if (!tabs || !overview || !concepts) return;

  tabs.replaceChildren();
  COPY.intro_tabs.forEach((t) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "lite-chip-btn" + (t.id === introTab ? " is-on" : "");
    b.textContent = t.label;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", t.id === introTab ? "true" : "false");
    b.setAttribute("aria-controls", `intro-panel-${t.id}`);
    b.addEventListener("click", () => {
      introTab = t.id;
      renderCopy();
    });
    tabs.appendChild(b);
  });

  const showOverview = introTab === "overview";
  overview.hidden = !showOverview;
  concepts.hidden = showOverview;
  overview.setAttribute("aria-hidden", showOverview ? "false" : "true");
  concepts.setAttribute("aria-hidden", showOverview ? "true" : "false");
}

function renderConcepts() {
  const grid = document.getElementById("concept-grid");
  if (!grid) return;
  grid.replaceChildren();
  COPY.concepts.forEach((item) => {
    const card = document.createElement("article");
    card.className = "concept-card";
    const h = document.createElement("h3");
    h.textContent = item.title;
    card.appendChild(h);
    item.body.forEach((text) => {
      const p = document.createElement("p");
      appendInlineMath(p, text);
      card.appendChild(p);
    });
    if (item.math && item.math.length) {
      const math = document.createElement("div");
      math.className = "math-block";
      renderMath(math, item.math);
      card.appendChild(math);
    }
    grid.appendChild(card);
  });
}

function renderCopy() {
  const c = COPY;
  document.querySelectorAll("[data-i]").forEach((el) => {
    const key = el.getAttribute("data-i");
    if (typeof c[key] === "string") setRichText(el, c[key]);
  });
  renderIntroTabs();
  setParagraphs(document.getElementById("ask-body"), c.ask_p);
  renderMath(document.getElementById("ask-math"), c.ask_math);
  renderConcepts();
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
  setParagraphs(document.getElementById("step-body"), s.body);
  renderMath(document.getElementById("step-math"), s.math);

  const stepFigId = STEP_FIG_ID[step];
  const stepGap = document.getElementById("step-fig-gap");
  const figsForStepCheck = figsForRun(runId);
  if (stepGap) {
    const missingStepFig = stepFigId && !figsForStepCheck.some((f) => f.id === stepFigId);
    stepGap.hidden = !missingStepFig;
    if (missingStepFig) setRichText(stepGap, c.step_fig_gap);
  }

  const runChips = document.getElementById("run-chips");
  runChips.replaceChildren();
  c.runs.forEach((r) => {
    const group = document.createElement("div");
    group.className = "run-chip-group";
    const b = document.createElement("button");
    b.type = "button";
    b.className = "lite-chip-btn" + (r.id === runId ? " is-on" : "");
    b.textContent = r.label;
    b.addEventListener("click", () => {
      runId = r.id;
      renderCopy();
    });
    group.appendChild(b);
    if (r.href) {
      const a = document.createElement("a");
      a.className = "lite-chip-btn run-colab-link";
      a.href = r.href;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = "Colab";
      group.appendChild(a);
    }
    runChips.appendChild(group);
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
      const parts = Array.isArray(f.body) ? f.body : [f.body];
      appendInlineMath(body, parts.join(" "));
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
  setParagraphs(document.getElementById("fig-body"), chosen.body);
  renderMath(document.getElementById("fig-math"), chosen.math);

  const img = document.getElementById("fig-img");
  const frame = document.getElementById("fig-frame");
  const miss = document.getElementById("fig-missing");
  img.alt = chosen.title;
  img.onload = () => {
    img.hidden = false;
    frame.classList.remove("is-empty");
    miss.hidden = true;
  };
  img.onerror = () => {
    img.hidden = true;
    frame.classList.add("is-empty");
    miss.hidden = false;
    setRichText(miss, c.fig_missing);
  };
  img.hidden = false;
  frame.classList.remove("is-empty");
  miss.hidden = true;
  img.src = `./reports/${runId}/${chosen.id}?v=${chosen.id}`;
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

function boot() {
  renderCopy();
  draw();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
window.addEventListener("resize", draw);
