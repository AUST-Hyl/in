/**
 * 绝缘子破损检测前端
 */

const API = {
  health: "/api/health",
  models: "/api/models",
  detect: "/api/detect",
  detectBatch: "/api/detect/batch",
};

const state = {
  viewMode: "result",
  isDetecting: false,
  selectedWeights: "",
};

const $ = (id) => document.getElementById(id);

const els = {
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  modelSelect: $("modelSelect"),
  confSlider: $("confSlider"),
  confValue: $("confValue"),
  confBrokenSlider: $("confBrokenSlider"),
  confBrokenValue: $("confBrokenValue"),
  iouSlider: $("iouSlider"),
  iouValue: $("iouValue"),
  selectBtn: $("selectBtn"),
  clearBtn: $("clearBtn"),
  fileInput: $("fileInput"),
  dropzone: $("dropzone"),
  dropOverlay: $("dropOverlay"),
  progressWrap: $("progressWrap"),
  progressLabel: $("progressLabel"),
  progressCount: $("progressCount"),
  progressBar: $("progressBar"),
  summaryPanel: $("summaryPanel"),
  statsGrid: $("statsGrid"),
  resultsPanel: $("resultsPanel"),
  resultsGrid: $("resultsGrid"),
  resultCardTemplate: $("resultCardTemplate"),
};

function showToast(message, isError = false) {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("show");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => toast.classList.remove("show"), 3500);
}

function setStatus(text, level = "ok") {
  els.statusText.textContent = text;
  els.statusDot.className = `status-dot ${level}`;
}

function getParams() {
  return {
    conf: parseFloat(els.confSlider.value),
    conf_broken: parseFloat(els.confBrokenSlider.value),
    iou: parseFloat(els.iouSlider.value),
    weights: state.selectedWeights || els.modelSelect.value || "",
  };
}

function buildFormData(files, params) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  form.append("conf", params.conf);
  form.append("conf_broken", params.conf_broken);
  form.append("iou", params.iou);
  if (params.weights) {
    form.append("weights", params.weights);
  }
  return form;
}

function buildSingleFormData(file, params) {
  const form = new FormData();
  form.append("file", file);
  form.append("conf", params.conf);
  form.append("conf_broken", params.conf_broken);
  form.append("iou", params.iou);
  if (params.weights) {
    form.append("weights", params.weights);
  }
  return form;
}

function filterImageFiles(fileList) {
  const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
  const files = Array.from(fileList).filter((f) => {
    if (f.type && allowed.includes(f.type)) return true;
    return /\.(jpe?g|png|webp|bmp)$/i.test(f.name);
  });
  if (files.length === 0) {
    showToast("请拖入 JPG / PNG / WEBP 图片", true);
  }
  return files;
}

function setProgress(current, total, label = "正在检测...") {
  els.progressWrap.classList.remove("hidden");
  els.progressLabel.textContent = label;
  els.progressCount.textContent = `${current} / ${total}`;
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  els.progressBar.style.width = `${pct}%`;
}

function hideProgress() {
  els.progressWrap.classList.add("hidden");
  els.progressBar.style.width = "0%";
}

function setDetecting(busy) {
  state.isDetecting = busy;
  els.selectBtn.disabled = busy;
  els.dropzone.classList.toggle("disabled", busy);
  els.dropzone.style.pointerEvents = busy ? "none" : "";
  els.dropzone.style.opacity = busy ? "0.65" : "";
}

async function fetchHealth() {
  try {
    const res = await fetch(API.health);
    const data = await res.json();
    if (data.missing_deps && data.missing_deps.length > 0) {
      setStatus(`缺少依赖: ${data.missing_deps.join(", ")}`, "error");
      showToast(`请先安装: pip install ${data.missing_deps.join(" ")}`, true);
      return data;
    }
    const device = data.device === "cuda" ? "GPU" : "CPU";
    if (data.weights_loaded) {
      const name = (data.weights || "").split(/[/\\]/).slice(-3, -2)[0] || "已加载";
      setStatus(`${device} · ${name}`, "ok");
    } else {
      setStatus(`${device} · 等待模型`, "warn");
    }
    return data;
  } catch {
    setStatus("服务未连接", "error");
    return null;
  }
}

async function fetchModels() {
  try {
    const res = await fetch(API.models);
    const data = await res.json();
    const models = data.models || [];
    els.modelSelect.innerHTML = "";

    if (models.length === 0) {
      els.modelSelect.innerHTML =
        '<option value="">未找到权重（请先训练模型）</option>';
      showToast("未找到 runs/**/best.pt，请先训练或指定权重启动服务", true);
      return;
    }

    models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.path;
      opt.textContent = m.name;
      els.modelSelect.appendChild(opt);
    });

    const defaultPath = data.default || models[0].path;
    els.modelSelect.value = defaultPath;
    state.selectedWeights = defaultPath;
  } catch {
    els.modelSelect.innerHTML = '<option value="">加载失败</option>';
  }
}

function renderSummary(data) {
  els.summaryPanel.classList.remove("hidden");
  const isBatch = Array.isArray(data.results);
  const totalImages = isBatch ? data.total_images : 1;
  const totalDetections = isBatch
    ? data.total_detections
    : data.detection_count;
  const damageImages = isBatch
    ? data.damage_images
    : data.has_damage
      ? 1
      : 0;

  els.statsGrid.innerHTML = `
    <div class="stat-card">
      <div class="value">${totalImages}</div>
      <div class="label">检测图片</div>
    </div>
    <div class="stat-card">
      <div class="value">${totalDetections}</div>
      <div class="label">目标总数</div>
    </div>
    <div class="stat-card danger">
      <div class="value">${damageImages}</div>
      <div class="label">疑似破损</div>
    </div>
  `;
}

function applyViewMode(card) {
  const media = card.querySelector(".card-media");
  const resultImg = card.querySelector(".result-image");
  const originalImg = card.querySelector(".original-image");

  media.classList.remove("compare");
  resultImg.classList.remove("hidden");
  originalImg.classList.add("hidden");

  if (state.viewMode === "original") {
    resultImg.classList.add("hidden");
    originalImg.classList.remove("hidden");
  } else if (state.viewMode === "compare") {
    media.classList.add("compare");
    originalImg.classList.remove("hidden");
  }
}

function renderResultItem(item) {
  const node = els.resultCardTemplate.content.cloneNode(true);
  const card = node.querySelector(".result-card");
  const resultImg = card.querySelector(".result-image");
  const originalImg = card.querySelector(".original-image");
  const badge = card.querySelector(".card-badge");
  const title = card.querySelector(".card-title");
  const meta = card.querySelector(".card-meta");
  const list = card.querySelector(".detection-list");

  resultImg.src = `data:image/jpeg;base64,${item.image_result}`;
  originalImg.src = `data:image/jpeg;base64,${item.image_original}`;

  title.textContent = item.filename;
  meta.textContent = `${item.width} × ${item.height} · 检出 ${item.detection_count} 个目标（破损 ${item.broken_count ?? 0} 处）`;

  if (item.has_damage) {
    badge.textContent = "疑似破损";
    badge.classList.add("damage");
  } else {
    badge.textContent = "未检出破损";
    badge.classList.add("safe");
  }

  if (item.detections.length === 0) {
    list.innerHTML = '<p class="detection-empty">未检测到目标</p>';
  } else {
    item.detections.forEach((d) => {
      const row = document.createElement("div");
      row.className = "detection-item";
      const isBroken = d.class_id === 1;
      row.innerHTML = `
        <span class="name ${isBroken ? "broken" : ""}">${d.class_name_zh}</span>
        <span class="conf">${(d.confidence * 100).toFixed(1)}%</span>
      `;
      list.appendChild(row);
    });
  }

  applyViewMode(card);
  els.resultsGrid.appendChild(card);
}

function renderResults(data) {
  els.resultsPanel.classList.remove("hidden");
  els.clearBtn.disabled = false;
  els.resultsGrid.innerHTML = "";

  const items = data.results || [data];
  items.forEach(renderResultItem);
  renderSummary(data);
}

async function detectFiles(files) {
  if (state.isDetecting || files.length === 0) return;

  setDetecting(true);
  const params = getParams();
  setProgress(0, files.length);

  try {
    let data;

    if (files.length === 1) {
      setProgress(0, 1, "正在检测单张图片...");
      const res = await fetch(API.detect, {
        method: "POST",
        body: buildSingleFormData(files[0], params),
      });
      data = await res.json();
      if (!res.ok) throw new Error(data.detail || "检测失败");
      setProgress(1, 1, "检测完成");
    } else {
      setProgress(0, files.length, `正在批量检测 ${files.length} 张...`);
      const res = await fetch(API.detectBatch, {
        method: "POST",
        body: buildFormData(files, params),
      });
      data = await res.json();
      if (!res.ok) throw new Error(data.detail || "批量检测失败");
      setProgress(files.length, files.length, "批量检测完成");
    }

    renderResults(data);
    const count = data.results ? data.results.length : 1;
    showToast(`检测完成：${count} 张图片`);
    fetchHealth();
  } catch (err) {
    showToast(err.message || "检测请求失败", true);
  } finally {
    setDetecting(false);
    setTimeout(hideProgress, 600);
  }
}

function handleFiles(fileList) {
  const files = filterImageFiles(fileList);
  if (files.length) detectFiles(files);
}

function clearResults() {
  els.resultsGrid.innerHTML = "";
  els.summaryPanel.classList.add("hidden");
  els.resultsPanel.classList.add("hidden");
  els.clearBtn.disabled = true;
}

function setupDragDrop() {
  const zone = els.dropzone;

  ["dragenter", "dragover"].forEach((evt) => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove("drag-over");
    });
  });

  zone.addEventListener("drop", (e) => {
    handleFiles(e.dataTransfer.files);
  });

  zone.addEventListener("click", () => {
    if (!state.isDetecting) els.fileInput.click();
  });

  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.fileInput.click();
    }
  });
}

function setupEvents() {
  els.confSlider.addEventListener("input", () => {
    els.confValue.textContent = els.confSlider.value;
  });

  els.confBrokenSlider.addEventListener("input", () => {
    els.confBrokenValue.textContent = Number(els.confBrokenSlider.value).toFixed(2);
  });

  els.iouSlider.addEventListener("input", () => {
    els.iouValue.textContent = els.iouSlider.value;
  });

  els.modelSelect.addEventListener("change", () => {
    state.selectedWeights = els.modelSelect.value;
  });

  els.selectBtn.addEventListener("click", () => els.fileInput.click());

  els.fileInput.addEventListener("change", (e) => {
    handleFiles(e.target.files);
    e.target.value = "";
  });

  els.clearBtn.addEventListener("click", clearResults);

  document.querySelectorAll(".toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".toggle-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.viewMode = btn.dataset.view;
      document.querySelectorAll(".result-card").forEach(applyViewMode);
    });
  });

  setupDragDrop();
}

async function init() {
  setupEvents();
  await fetchModels();
  await fetchHealth();
}

init();
