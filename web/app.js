const COLAB = {
  demo: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Demo_Colab.ipynb",
  a: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Route_A_Bacpipe_Colab.ipynb",
  b: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Route_B_BMZ_Colab.ipynb",
};

const I18N = {
  en: {
    kicker: "colab walkthrough",
    lede: "Runtime, Run all. The notebook clones the repo, writes or loads a detection manifest, then turns embedding vectors into geometry, a path through time, and a check that the path is not an accident.",
    ask_label: "the question",
    ask_h: "Does the soundscape move in embedding space?",
    ask_p: "BirdNET already assigns a species and a 1024-d vector to each detection. Ordinary counts stop at how often something called. This Colab keeps the vectors and asks whether that cloud changes as the recording proceeds: a different neighborhood, a break in the path, a new HMM state. Demo data plants three regimes so the methods have something known to find. Route A/B swap in real BirdNET vectors.",
    walk_label: "pipeline",
    keys: "Left / Right to step. The sketch is a toy 2-d version of the same idea, not the Colab PNG.",
    prev: "prev",
    next: "next",
    fig_label: "reports/",
    fig_h: "Figures the notebook writes",
    fig_note: "From Colab verification runs. Click a run, then a file. shuffle_null.png is newer than this dump, so it has no PNG here.",
    fig_missing: "No PNG in this dump. The current notebook still writes it on Run all.",
    runs: [
      { id: "demo", label: "Demo" },
      { id: "bmz", label: "Route B" },
      { id: "bacpipe", label: "Route A" },
    ],
    nb_label: "notebooks",
    nb_h: "What to open",
    steps: [
      {
        chip: "detections",
        file: "JSONL manifest",
        title: "A row is one call, not a spectrogram",
        body: "Each line has start_s, species, confidence, and optionally embedding. Demo synthesizes a 3-hour fake site with three successive species pools. Empty minutes stay in the timeline later; they are rate 0, not deleted.",
      },
      {
        chip: "vectors",
        file: "BirdNET or demo mapper",
        title: "The object of study is the vector",
        body: "Route A (bacpipe) and Route B (BMZ) attach 1024-d BirdNET embeddings. If the column is missing, a small PyTorch mapper fills it for the demo only. Mixed real/missing rows are refused.",
      },
      {
        chip: "geometry",
        file: "pca_species.png · umap_species.png",
        title: "Same points, two projections",
        body: "PCA is the linear view; UMAP exaggerates neighborhoods. Color is species. Axes have no physical unit. Demo clusters look clean because prototypes were planted. Real BirdNET clouds are messier.",
      },
      {
        chip: "trajectory",
        file: "trajectory_pca.png",
        title: "Average the minute, then walk",
        body: "Detections are binned (default 60 s). The centroid is confidence-weighted so weak calls pull less. The path is that centroid in PC1/PC2. Color on the PNG is HMM state; arrows still mark time.",
      },
      {
        chip: "breaks",
        file: "changepoints.png · trajectory_changepoints.png",
        title: "Count breaks vs content breaks",
        body: "PELT on detection rate (empty bins = 0) marks when calling gets denser or quieter. PELT on PC1 marks when the average vector jumps. A red line on only one of the two means activity changed without the sound content changing, or the reverse.",
      },
      {
        chip: "HMM",
        file: "hmm_regimes.png",
        title: "Two HMMs, stacked on purpose",
        body: "The top row is fit on the embedding centroids (PC1/PC2). That is the regime in representation space. The bottom row is fit on rate, richness, and mean confidence: activity structure. State numbers have no names.",
      },
      {
        chip: "shuffle",
        file: "shuffle_null.png",
        title: "Break time, keep the vectors",
        body: "Start times are permuted; species and embeddings stay on the same rows. If PC1 breaks and HMM switches were about chronology, they should get noisier. Look at hmm_n_switches vs shuffle_hmm_n_switches in summary.json. The shuffle still fits the same number of states, so the state count itself will not vanish.",
      },
    ],
    figs: [
      { id: "pca_species.png", title: "pca_species.png", body: "Each point is a detection. Color is species. The two axes are principal components of the scaled embedding. Read cluster separation, not the numeric ticks." },
      { id: "umap_species.png", title: "umap_species.png", body: "Same detections, nonlinear map. Useful when PCA is a smear and local islands still exist. Do not compare UMAP distances to PCA distances." },
      { id: "trajectory_pca.png", title: "trajectory_pca.png", body: "Minute centroids in PC space. Color is the embedding HMM state. A long arrow is a minute whose average vector moved far from the last one." },
      { id: "changepoints.png", title: "changepoints.png", body: "Detection rate vs minutes, including silent bins at 0. Crimson in the Colab PNG is a PELT break in how often things called, not in what they were." },
      { id: "trajectory_changepoints.png", title: "trajectory_changepoints.png", body: "PC1 of the centroid vs minutes. Breaks here are content shifts. Compare against changepoints.png on the same time axis." },
      { id: "hmm_regimes.png", title: "hmm_regimes.png", body: "Upper: HMM on embedding centroids. Lower: HMM on activity stats. They can disagree. That disagreement is the point." },
      { id: "shuffle_null.png", title: "shuffle_null.png", body: "Left original, right time-shuffled. The control is whether structure tied to order survives a permutation of start_s." },
    ],
    nbs: [
      { name: "Demo", meta: "CPU, about 2-3 min. Synthetic 128-d vectors, planted regimes.", href: COLAB.demo },
      { name: "Route B", meta: "CPU, about 5-10 min. BMZ BirdNET 1024-d. Writes a 120 s wav if /content/audio is empty.", href: COLAB.b },
      { name: "Route A", meta: "T4 recommended. bacpipe BirdNET on bundled test wavs. First run downloads weights.", href: COLAB.a },
    ],
  },
  zh: {
    kicker: "colab 说明",
    lede: "Runtime，Run all。Notebook 会 clone 仓库，生成或读入检测清单，再把 embedding 做成几何、时间轨迹，以及一条“这条轨迹不是碰巧”的对照。",
    ask_label: "在问什么",
    ask_h: "声景在 embedding 空间里有没有在走？",
    ask_p: "BirdNET 已经给每次检测一个物种和 1024 维向量。普通分析停在叫了多少次。这套 Colab 把向量留下来，问这团点有没有随录音往前而换邻域、断路径、换 HMM 状态。Demo 人为埋了三档，用来看方法能不能找回来。Route A/B 换成真正的 BirdNET 向量。",
    walk_label: "流程",
    keys: "左右方向键切换。左侧是同一思路的 2 维示意，不是 Colab 里的 PNG。",
    prev: "上一步",
    next: "下一步",
    fig_label: "reports/",
    fig_h: "Notebook 写出的图",
    fig_note: "来自 Colab 验证跑次。先选 Demo / Route B / Route A，再点文件名。shuffle_null.png 是后来加的，这批 dump 里没有图。",
    fig_missing: "这批 dump 里没有这张 PNG。现在的 notebook Run all 仍会写。",
    runs: [
      { id: "demo", label: "Demo" },
      { id: "bmz", label: "Route B" },
      { id: "bacpipe", label: "Route A" },
    ],
    nb_label: "notebook",
    nb_h: "打开哪一份",
    steps: [
      { chip: "检测", file: "JSONL manifest", title: "一行是一次叫声，不是一张频谱图", body: "每行有 start_s、物种、置信度，以及可选的 embedding。Demo 造一段约 3 小时、三档物种池的假样地。后面的时间轴会保留空分钟，记成速率 0，而不是删掉。" },
      { chip: "向量", file: "BirdNET 或 demo mapper", title: "真正在分析的是向量", body: "Route A（bacpipe）和 Route B（BMZ）挂上 1024 维 BirdNET embedding。缺这一列时，才用小的 PyTorch mapper 填 Demo。真假向量混在同一张表里会直接报错。" },
      { chip: "几何", file: "pca_species.png · umap_species.png", title: "同一批点，两种投影", body: "PCA 是线性视图，UMAP 会把邻域拉得更夸张。颜色是物种。轴没有物理单位。Demo 的团块干净，是因为原型是埋进去的。真 BirdNET 云会糊很多。" },
      { chip: "轨迹", file: "trajectory_pca.png", title: "按分钟平均，再连成路", body: "默认 60 秒一箱。质心按置信度加权，低分检测少拉。路径是该质心在 PC1/PC2 上的位置。PNG 上的颜色是 embedding HMM 状态，箭头仍表示时间。" },
      { chip: "断裂", file: "changepoints.png · trajectory_changepoints.png", title: "次数断了，还是内容断了", body: "检测率上的 PELT（空箱 = 0）标的是叫得密还是疏。PC1 上的 PELT 标的是平均向量跳了没有。只在一张图上有红线，就是“更勤但还是同一类东西”，或反过来。" },
      { chip: "HMM", file: "hmm_regimes.png", title: "两个 HMM，故意叠在一起", body: "上行吃 embedding 质心（PC1/PC2），这是表征空间里的档。下行吃速率、物种数、平均置信度，这是活动结构。状态编号没有名字。" },
      { chip: "打乱", file: "shuffle_null.png", title: "打乱时间，向量留在原检测上", body: "只置换 start_s。若 PC1 断裂和 HMM 跳变真的依赖时间顺序，打乱后应变吵。看 summary.json 里 hmm_n_switches 和 shuffle_hmm_n_switches。状态个数仍会拟合，所以不要指望右边变成一条平线。" },
    ],
    figs: [
      { id: "pca_species.png", title: "pca_species.png", body: "每个点一次检测，颜色是物种，轴是标准化 embedding 的主成分。看团块分不分得开，不要读刻度当物理量。" },
      { id: "umap_species.png", title: "umap_species.png", body: "同一批点的非线性投影。PCA 糊成一片、UMAP 仍有岛时有用。不要拿 UMAP 距离去和 PCA 比。" },
      { id: "trajectory_pca.png", title: "trajectory_pca.png", body: "每分钟质心在 PC 平面上的路。颜色是 embedding HMM 状态。长箭头表示这一分钟的平均向量离上一分钟很远。" },
      { id: "changepoints.png", title: "changepoints.png", body: "检测率对分钟，空箱为 0。Colab PNG 里的红虚线是叫声次数的 PELT 断裂，不是内容断裂。" },
      { id: "trajectory_changepoints.png", title: "trajectory_changepoints.png", body: "质心 PC1 对分钟。这里的断裂是内容在变。和 changepoints.png 同一时间轴对读。" },
      { id: "hmm_regimes.png", title: "hmm_regimes.png", body: "上：embedding 质心 HMM。下：活动统计 HMM。两者可以不一致，不一致才有信息。" },
      { id: "shuffle_null.png", title: "shuffle_null.png", body: "左原时间，右打乱 start_s。对照的是：和顺序绑在一起的结构，置换之后还在不在。" },
    ],
    nbs: [
      { name: "Demo", meta: "CPU，约 2-3 分钟。128 维合成向量，三档是埋进去的。", href: COLAB.demo },
      { name: "Route B", meta: "CPU，约 5-10 分钟。BMZ BirdNET 1024 维。/content/audio 为空时会写 120 秒合成 wav。", href: COLAB.b },
      { name: "Route A", meta: "建议 T4。bacpipe BirdNET，自带测试 wav。第一次会下权重。", href: COLAB.a },
    ],
  },
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

const HAS_PNG = new Set([
  "pca_species.png",
  "umap_species.png",
  "trajectory_pca.png",
  "changepoints.png",
  "trajectory_changepoints.png",
  "hmm_regimes.png",
]);
const STEP_FIG = { 2: 0, 3: 2, 4: 3, 5: 5, 6: 6 };

const DATA = generate();
const SPECIES_COLOR = {
  A: "var-nacht",
  B: "var-gold",
  C: "var-rubric",
  D: "var-fg",
};

let lang = "en";
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
    ctx.fillText(lang === "zh" ? "一团检测，还没有时间方向" : "a cloud of detections, no time axis yet", pad, h - 14);
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
      ctx.fillText(lang === "zh" ? `档 ${r}` : `regime ${r}`, pad, 36 + i * 70);
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
    ctx.fillText(lang === "zh" ? "检测率（含空箱）" : "detection rate (empty bins kept)", pad, h - 8);
  }
}

function t() {
  return I18N[lang];
}

function renderCopy() {
  const c = t();
  document.documentElement.lang = lang === "zh" ? "zh-Hans" : "en";
  document.querySelectorAll("[data-i]").forEach((el) => {
    const key = el.getAttribute("data-i");
    if (typeof c[key] === "string") el.textContent = c[key];
  });
  document.getElementById("lang-btn").textContent = lang === "en" ? "中文" : "EN";
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

  const figChips = document.getElementById("fig-chips");
  figChips.replaceChildren();
  c.figs.forEach((f, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "fig-thumb" + (i === figIdx ? " is-on" : "");
    if (HAS_PNG.has(f.id)) {
      const im = document.createElement("img");
      im.src = `./reports/${runId}/${f.id}`;
      im.alt = "";
      b.appendChild(im);
    }
    const cap = document.createElement("span");
    cap.textContent = f.id;
    b.appendChild(cap);
    b.addEventListener("click", () => {
      figIdx = i;
      renderCopy();
    });
    figChips.appendChild(b);
  });
  document.getElementById("fig-title").textContent = c.figs[figIdx].title;
  document.getElementById("fig-body").textContent = c.figs[figIdx].body;

  const chosen = c.figs[figIdx];
  const img = document.getElementById("fig-img");
  const frame = img.parentElement;
  const miss = document.getElementById("fig-missing");
  if (HAS_PNG.has(chosen.id)) {
    img.src = `./reports/${runId}/${chosen.id}`;
    img.alt = chosen.id;
    frame.classList.remove("is-empty");
    miss.hidden = true;
  } else {
    img.removeAttribute("src");
    img.alt = "";
    frame.classList.add("is-empty");
    miss.hidden = false;
    miss.textContent = c.fig_missing;
  }

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
  step = Math.max(0, Math.min(t().steps.length - 1, i));
  if (STEP_FIG[step] != null) figIdx = STEP_FIG[step];
  renderCopy();
  draw();
}

document.getElementById("prev-btn").addEventListener("click", () => setStep(step - 1));
document.getElementById("next-btn").addEventListener("click", () => setStep(step + 1));
document.getElementById("lang-btn").addEventListener("click", () => {
  lang = lang === "en" ? "zh" : "en";
  renderCopy();
  draw();
});
document.getElementById("theme-btn").addEventListener("click", () => {
  const body = document.body;
  const dark = body.getAttribute("data-theme") !== "dark";
  body.setAttribute("data-theme", dark ? "dark" : "light");
  document.getElementById("theme-btn").textContent = dark ? "light" : "dark";
  document.getElementById("theme-btn").setAttribute("aria-pressed", dark ? "true" : "false");
  draw();
});
window.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") setStep(step + 1);
  if (e.key === "ArrowLeft") setStep(step - 1);
});

renderCopy();
draw();
window.addEventListener("resize", draw);
