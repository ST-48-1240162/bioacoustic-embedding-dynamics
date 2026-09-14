const COLAB = {
  demo: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Demo_Colab.ipynb",
  a: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Route_A_Bacpipe_Colab.ipynb",
  b: "https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Route_B_BMZ_Colab.ipynb",
};

const I18N = {
  en: {
    kicker: "colab walkthrough",
    lede: "Runtime, Run all. The notebook clones this repo, writes a sample detection JSONL or loads yours, and plots the embedding vectors for the length of the recording.",
    ask_label: "what it is doing",
    ask_h: "Embeddings during the recording",
    ask_p: "BirdNET already names the species and emits a 1024-d vector for each detection. The usual next step is to count those calls. This notebook keeps the vectors and watches whether that cloud moves during the file. The demo plants three regimes so you can see if the plots respond. Route A and Route B use real BirdNET embeddings.",
    walk_label: "pipeline",
    keys: "Left and right arrows change the step. The drawing on the left is a cartoon. It is not the PNG Colab exports.",
    prev: "prev",
    next: "next",
    fig_label: "reports/",
    fig_h: "Output figures",
    fig_note: "PNGs from Colab test runs. Pick Demo, Route B, or Route A, then a file. shuffle_null.png was added later, so it is missing from this set.",
    fig_missing: "This set has no PNG for that file. A current Run all still writes it.",
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
        body: "Each line has start_s, species, confidence, and an optional embedding. The demo writes a fake three-hour site whose species pool changes twice. Minutes with no calls stay on the timeline as rate 0.",
      },
      {
        chip: "vectors",
        file: "BirdNET or demo mapper",
        title: "Where the 1024-d numbers come from",
        body: "Route A (bacpipe) and Route B (BMZ) attach BirdNET embeddings. If that column is empty, a small PyTorch mapper fills it, and only for the demo. Mixing real vectors with missing ones is an error.",
      },
      {
        chip: "geometry",
        file: "pca_species.png and umap_species.png",
        title: "PCA and UMAP",
        body: "Same detections, two maps. PCA is linear. UMAP pulls nearby points into islands. Color is species. The axis numbers are not Hertz or meters. Demo blobs look tidy because the prototypes were written in. BirdNET clouds usually look messier.",
      },
      {
        chip: "trajectory",
        file: "trajectory_pca.png",
        title: "Centroid of each minute",
        body: "Calls go into 60 s bins by default. The bin mean is weighted by confidence, so weak detections pull less. The plot is that mean in PC1 and PC2. Color is the embedding HMM state. Arrows still follow time.",
      },
      {
        chip: "breaks",
        file: "changepoints.png and trajectory_changepoints.png",
        title: "Change-points on rate and on PC1",
        body: "PELT on detection rate (empty bins count as 0) finds when calling gets busier or quieter. PELT on PC1 finds when the average vector jumps. A red line on the rate plot only means more or fewer calls.",
      },
      {
        chip: "HMM",
        file: "hmm_regimes.png",
        title: "Two HMMs",
        body: "The upper row is fit on PC1/PC2 centroids, so the states live in embedding space. The lower row is fit on call rate, species count, and mean confidence. 0, 1, and 2 are just labels.",
      },
      {
        chip: "shuffle",
        file: "shuffle_null.png",
        title: "Shuffle start_s",
        body: "Only the times are permuted. Species and embeddings stay on the same rows. If the PC1 breaks and HMM flips depended on order, the shuffled side should look noisier. Compare hmm_n_switches with shuffle_hmm_n_switches in summary.json. The fit still uses a fixed number of states, so the right-hand HMM will not go flat.",
      },
    ],
    figs: [
      { id: "pca_species.png", title: "pca_species.png", body: "One point per detection. Color is species. Axes are the first two principal components of the scaled embedding. Useful if you want to see whether species sit apart. The tick labels are not physical units." },
      { id: "umap_species.png", title: "umap_species.png", body: "Same points, nonlinear map. Handy if PCA is a blob but local groups still exist. A distance on this plot is not a PCA distance." },
      { id: "trajectory_pca.png", title: "trajectory_pca.png", body: "Each marker is one minute's confidence-weighted centroid in PC space. Color is HMM state. A long arrow means that minute's average vector moved a long way." },
      { id: "changepoints.png", title: "changepoints.png", body: "Call rate against minutes, with empty bins at 0. Red dashed lines in the Colab PNG are PELT breaks in how often animals called." },
      { id: "trajectory_changepoints.png", title: "trajectory_changepoints.png", body: "PC1 of the centroid against minutes. Breaks here are shifts in the average vector. Put changepoints.png next to it." },
      { id: "hmm_regimes.png", title: "hmm_regimes.png", body: "Top: HMM on embedding centroids. Bottom: HMM on activity stats. They do not have to agree." },
      { id: "shuffle_null.png", title: "shuffle_null.png", body: "Original times on the left, shuffled start_s on the right. The test is whether the ordered structure survives the permutation." },
    ],
    nbs: [
      { name: "Demo", meta: "CPU, about 2-3 min. Fake 128-d vectors with three planted regimes.", href: COLAB.demo },
      { name: "Route B", meta: "CPU, about 5-10 min. BMZ BirdNET, 1024-d. If /content/audio is empty it writes a 120 s wav.", href: COLAB.b },
      { name: "Route A", meta: "T4 if you have one. bacpipe BirdNET on the bundled test wavs. The first run downloads weights.", href: COLAB.a },
    ],
  },
  zh: {
    kicker: "colab 说明",
    lede: "Runtime，Run all。Notebook 会 clone 这个仓库，写一份示例检测 JSONL 或读你的，然后把这段录音里的 embedding 画出来。",
    ask_label: "在干什么",
    ask_h: "向量有没有随时间在动",
    ask_p: "BirdNET 每次检测已经给出物种和 1024 维向量。后面通常是数叫声。这个 notebook 把向量留下来，看整段文件里这团点会不会挪。Demo 里埋了三档，方便看图会不会跟着变。Route A 和 Route B 用真的 BirdNET 向量。",
    walk_label: "流程",
    keys: "左右方向键切步骤。左边是示意，不是 Colab 导出的 PNG。",
    prev: "上一步",
    next: "下一步",
    fig_label: "reports/",
    fig_h: "输出的图",
    fig_note: "图来自 Colab 试跑。先选 Demo、Route B 或 Route A，再点文件。shuffle_null.png 是后来加的，这批文件里没有。",
    fig_missing: "这批文件里没有这张图。现在的 notebook Run all 还是会写。",
    runs: [
      { id: "demo", label: "Demo" },
      { id: "bmz", label: "Route B" },
      { id: "bacpipe", label: "Route A" },
    ],
    nb_label: "notebook",
    nb_h: "开哪本",
    steps: [
      { chip: "检测", file: "JSONL manifest", title: "一行一次检测", body: "字段是 start_s、物种、置信度，embedding 可有可无。Demo 写一段大约三小时的假样地，物种池改两次。没有叫声的分钟还在时间轴上，速率记 0。" },
      { chip: "向量", file: "BirdNET 或 demo mapper", title: "1024 维从哪来", body: "Route A 走 bacpipe，Route B 走 BMZ，都挂 BirdNET embedding。缺这一列时，只有 Demo 会用一个小的 PyTorch 网络填。真假向量混在一张表里会报错。" },
      { chip: "几何", file: "pca_species.png 和 umap_species.png", title: "PCA 和 UMAP", body: "同一批点，两张图。PCA 是线性的。UMAP 会把近邻拉成岛。颜色是物种。轴上的数字不是赫兹也不是米。Demo 的团看起来整齐，因为原型是写进去的。真 BirdNET 通常更糊。" },
      { chip: "轨迹", file: "trajectory_pca.png", title: "每分钟一个质心", body: "默认 60 秒一箱。箱子里的平均按置信度加权，低分的少拉一点。图画的是这个平均在 PC1、PC2 上的位置。颜色是 embedding HMM 的状态。箭头还是时间方向。" },
      { chip: "断裂", file: "changepoints.png 和 trajectory_changepoints.png", title: "检测率和 PC1 上的 change-point", body: "检测率上的 PELT（空箱当 0）找叫得更密还是更稀。PC1 上的 PELT 找平均向量有没有跳。红线只出现在检测率图上，多半只是叫声变多或变少。" },
      { chip: "HMM", file: "hmm_regimes.png", title: "两个 HMM", body: "上行用 PC1/PC2 质心，状态在 embedding 空间里。下行用叫的频率、物种数、平均置信度。0、1、2 只是编号。" },
      { chip: "打乱", file: "shuffle_null.png", title: "打乱 start_s", body: "只打乱时间。物种和向量还在原来的行上。如果 PC1 的断裂和 HMM 跳变真靠时间顺序，打乱之后应该更乱。去 summary.json 里比 hmm_n_switches 和 shuffle_hmm_n_switches。状态个数还是会拟合，右边不会变成一条平线。" },
    ],
    figs: [
      { id: "pca_species.png", title: "pca_species.png", body: "一个点一次检测。颜色是物种。轴是标准化 embedding 的前两个主成分。用来看物种分不分得开。刻度不是物理量。" },
      { id: "umap_species.png", title: "umap_species.png", body: "同一批点，非线性投影。PCA 糊成一团、局部还有分组时有用。这张图上的距离不能当成 PCA 距离。" },
      { id: "trajectory_pca.png", title: "trajectory_pca.png", body: "每个标记是一分钟里按置信度加权的质心。颜色是 HMM 状态。箭头很长，说明这一分钟的平均向量走得远。" },
      { id: "changepoints.png", title: "changepoints.png", body: "每分钟的叫声速率，空箱是 0。Colab PNG 里的红虚线是 PELT 切在叫的次数上。" },
      { id: "trajectory_changepoints.png", title: "trajectory_changepoints.png", body: "质心 PC1 对分钟。这里的断裂是平均向量在变。旁边放 changepoints.png 一起看。" },
      { id: "hmm_regimes.png", title: "hmm_regimes.png", body: "上：embedding 质心上的 HMM。下：活动统计上的 HMM。两行可以对不上。" },
      { id: "shuffle_null.png", title: "shuffle_null.png", body: "左边原时间，右边打乱 start_s。看跟顺序绑在一起的结构，置换之后还在不在。" },
    ],
    nbs: [
      { name: "Demo", meta: "CPU，大约 2-3 分钟。假的 128 维向量，三档是写进去的。", href: COLAB.demo },
      { name: "Route B", meta: "CPU，大约 5-10 分钟。BMZ 的 BirdNET，1024 维。/content/audio 是空的就会写 120 秒 wav。", href: COLAB.b },
      { name: "Route A", meta: "有 T4 再用。bacpipe 的 BirdNET，用自带的测试 wav。第一次会下权重。", href: COLAB.a },
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
    ctx.fillText(lang === "zh" ? "还只是一堆点" : "just the detections so far", pad, h - 14);
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
    ctx.fillText(lang === "zh" ? "每箱叫声数，空箱也算" : "calls per bin, empty bins included", pad, h - 8);
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
