const state = {
  modules: [],
  recipes: [],
  sidebarTab: 'modules',
  searchQuery: '',
  currentRecipeId: null,
  currentModuleId: null,
  currentUnitId: null,
  unitData: null,
  tpuStatus: null,
  completedUnits: new Set(),
  viewMode: localStorage.getItem('tpu_lab_view_mode') || 'editorial',
  isRunningLab: false,
  quizSelections: {},
  widgetState: {
    hwGen: 'v6e',
    tensorShape: '4096x4096',
    tensorDtype: 'bfloat16',
    bitFormat: 'bfloat16',
    bitSample: '1.0',
    mxuMode: 'default',
    syncMode: 'accum',
    rooflineHw: 'v6e',
    rooflineBatch: 16,
    pallasBlock: '128x128',
    spmdMode: 'fsdp',
    hfModel: 'qwen06b',
    hfTuneMode: 'lora',
    hfLossMode: 'chunked'
  }
};

const API = {
  async getCurriculum() {
    const res = await fetch('/api/curriculum');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  },
  async getRecipes() {
    const res = await fetch('/api/recipes');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  },
  async getUnit(moduleId, unitId) {
    const res = await fetch(`/api/unit?module=${encodeURIComponent(moduleId)}&unit=${encodeURIComponent(unitId)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  },
  async getTpuStatus(zone) {
    const res = await fetch(`/api/tpu/status?zone=${encodeURIComponent(zone || 'us-east1-d')}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  },
  async runTpuAction(action, zone, accelerator_type, node_id) {
    const res = await fetch('/api/tpu/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, zone, accelerator_type, node_id })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  },
  async saveProgress(payload) {
    const res = await fetch('/api/progress', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  },
  async runCode(code, target, zone, node_id, capture_hlo) {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, target, zone, node_id, capture_hlo })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }
};

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function unitKey(moduleId, unitId) {
  return `${moduleId}/${unitId}`;
}

function isUnitCompleted(moduleId, unitId) {
  return state.completedUnits.has(unitKey(moduleId, unitId));
}

function updateCourseProgressUI() {
  let totalUnits = 0;
  let completedCount = 0;
  for (const mod of state.modules) {
    for (const u of mod.units || []) {
      totalUnits += 1;
      if (isUnitCompleted(mod.id, u.id)) completedCount += 1;
    }
  }
  const pct = totalUnits > 0 ? Math.round((completedCount / totalUnits) * 100) : 0;
  const fillEl = document.getElementById('course-progress-fill');
  const summaryEl = document.getElementById('progress-summary');
  if (fillEl) fillEl.style.width = `${pct}%`;
  if (summaryEl) summaryEl.textContent = `${completedCount} / ${totalUnits}`;
}

function setViewMode(mode) {
  state.viewMode = mode === 'split' ? 'split' : 'editorial';
  localStorage.setItem('tpu_lab_view_mode', state.viewMode);
  const workspace = document.getElementById('workspace-grid');
  const btnEditorial = document.getElementById('mode-editorial-btn');
  const btnSplit = document.getElementById('mode-split-btn');
  if (workspace) {
    workspace.classList.toggle('mode-editorial', state.viewMode === 'editorial');
    workspace.classList.toggle('mode-split', state.viewMode === 'split');
  }
  if (btnEditorial) btnEditorial.classList.toggle('active', state.viewMode === 'editorial');
  if (btnSplit) btnSplit.classList.toggle('active', state.viewMode === 'split');
}

function updateEditorGutter() {
  const editor = document.getElementById('code-editor');
  const gutter = document.getElementById('editor-gutter');
  if (!editor || !gutter) return;
  const lineCount = Math.max(1, (editor.value || '').split('\n').length);
  const nums = [];
  for (let i = 1; i <= lineCount; i++) nums.push(i);
  gutter.textContent = nums.join('\n');
}

function setSidebarTab(tab) {
  state.sidebarTab = tab === 'recipes' ? 'recipes' : 'modules';
  const modBtn = document.getElementById('sidebar-tab-modules');
  const recBtn = document.getElementById('sidebar-tab-recipes');
  const modTree = document.getElementById('curriculum-tree');
  const recTree = document.getElementById('recipes-tree');
  if (modBtn) modBtn.classList.toggle('active', state.sidebarTab === 'modules');
  if (recBtn) recBtn.classList.toggle('active', state.sidebarTab === 'recipes');
  if (modTree) modTree.classList.toggle('hidden', state.sidebarTab !== 'modules');
  if (recTree) recTree.classList.toggle('hidden', state.sidebarTab !== 'recipes');
}

function renderRecipesSidebar() {
  const tree = document.getElementById('recipes-tree');
  const recTabBtn = document.getElementById('sidebar-tab-recipes');
  if (recTabBtn) recTabBtn.textContent = `Recipes (${state.recipes.length})`;
  if (!tree) return;
  tree.innerHTML = '';

  const q = (state.searchQuery || '').trim().toLowerCase();
  state.recipes.forEach((rec) => {
    if (q && !rec.filename.toLowerCase().includes(q) && !rec.title.toLowerCase().includes(q)) {
      return;
    }
    const item = document.createElement('div');
    const isActive = state.currentRecipeId === rec.id;
    item.className = `recipe-sidebar-item${isActive ? ' active' : ''}`;
    item.innerHTML = `
      <div class="recipe-sidebar-code">${escapeHtml(rec.filename)}</div>
      <div class="recipe-sidebar-title">${escapeHtml(rec.title)}</div>
    `;
    item.addEventListener('click', () => selectRecipe(rec.id));
    tree.appendChild(item);
  });
}

function selectRecipe(recipeId) {
  const rec = state.recipes.find((r) => r.id === recipeId);
  if (!rec) return;
  state.currentRecipeId = rec.id;
  renderRecipesSidebar();

  const editor = document.getElementById('code-editor');
  const labLabel = document.getElementById('lab-filename-label');
  const inlineFilename = document.getElementById('inline-lab-filename');
  if (editor) {
    editor.value = rec.code || '';
    updateEditorGutter();
  }
  if (labLabel) labLabel.textContent = rec.filename;
  if (inlineFilename) inlineFilename.textContent = rec.filename;
  setViewMode('split');
}

function renderSidebar() {
  const tree = document.getElementById('curriculum-tree');
  const modTabBtn = document.getElementById('sidebar-tab-modules');
  if (modTabBtn) modTabBtn.textContent = `Modules (${state.modules.length})`;
  if (!tree) return;
  tree.innerHTML = '';

  const q = (state.searchQuery || '').trim().toLowerCase();

  state.modules.forEach((mod, modIdx) => {
    const units = (mod.units || []).filter((u) => {
      if (!q) return true;
      return u.title.toLowerCase().includes(q) || mod.title.toLowerCase().includes(q);
    });
    if (q && units.length === 0) return;

    const block = document.createElement('div');
    block.className = 'module-block';

    const allUnits = mod.units || [];
    const doneInMod = allUnits.filter((u) => isUnitCompleted(mod.id, u.id)).length;
    const allDone = allUnits.length > 0 && doneInMod === allUnits.length;

    const headerBtn = document.createElement('button');
    headerBtn.type = 'button';
    headerBtn.className = 'module-header-btn';
    headerBtn.innerHTML = `
      <div class="module-header-text">${modIdx}. ${escapeHtml(mod.title.replace(/^Module\s+\d+:\s*/i, ''))}</div>
      <span class="module-count-pill${allDone ? ' all-done' : ''}">${doneInMod}/${allUnits.length}</span>
    `;
    block.appendChild(headerBtn);

    const list = document.createElement('ul');
    list.className = 'unit-list';

    units.forEach((unit) => {
      const li = document.createElement('li');
      const isActive = mod.id === state.currentModuleId && unit.id === state.currentUnitId;
      const isDone = isUnitCompleted(mod.id, unit.id);
      li.className = `unit-item${isActive ? ' active' : ''}`;

      const leftWrap = document.createElement('div');
      leftWrap.className = 'unit-item-left';

      const check = document.createElement('span');
      check.className = `unit-check-circle${isDone ? ' done' : ''}`;
      check.textContent = '✓';

      const label = document.createElement('span');
      label.textContent = unit.title;

      leftWrap.appendChild(check);
      leftWrap.appendChild(label);

      li.appendChild(leftWrap);
      li.addEventListener('click', () => {
        state.currentRecipeId = null;
        renderRecipesSidebar();
        selectUnit(mod.id, unit.id);
      });
      list.appendChild(li);
    });

    headerBtn.addEventListener('click', () => {
      list.style.display = list.style.display === 'none' ? 'block' : 'none';
    });

    block.appendChild(list);
    tree.appendChild(block);
  });
}

function updateCompleteButtonsUI() {
  if (!state.currentModuleId || !state.currentUnitId) return;
  const done = isUnitCompleted(state.currentModuleId, state.currentUnitId);
  const topBtn = document.getElementById('btn-complete-unit');
  const footBtn = document.getElementById('btn-footer-complete');
  [topBtn, footBtn].forEach((btn) => {
    if (!btn) return;
    btn.textContent = done ? '✓ Completed' : 'Mark Complete';
    btn.classList.toggle('completed', done);
  });
}

function getFlatUnitList() {
  const flat = [];
  state.modules.forEach((mod, modIdx) => {
    (mod.units || []).forEach((u, uIdx) => {
      flat.push({
        moduleId: mod.id,
        moduleIndex: modIdx,
        moduleTitle: mod.title,
        unitId: u.id,
        unitIndex: uIdx,
        unitTitle: u.title
      });
    });
  });
  return flat;
}

function wireUnitNavigation() {
  const flat = getFlatUnitList();
  const idx = flat.findIndex(
    (item) => item.moduleId === state.currentModuleId && item.unitId === state.currentUnitId
  );

  const prevBtns = [document.getElementById('btn-prev-unit'), document.getElementById('btn-footer-prev')];
  const nextBtns = [document.getElementById('btn-next-unit'), document.getElementById('btn-footer-next')];

  if (idx > 0) {
    const prev = flat[idx - 1];
    prevBtns.forEach((b) => {
      if (!b) return;
      b.disabled = false;
      b.onclick = () => selectUnit(prev.moduleId, prev.unitId);
    });
  } else {
    prevBtns.forEach((b) => {
      if (!b) return;
      b.disabled = true;
      b.onclick = null;
    });
  }

  if (idx >= 0 && idx < flat.length - 1) {
    const next = flat[idx + 1];
    nextBtns.forEach((b) => {
      if (!b) return;
      b.disabled = false;
      b.onclick = () => selectUnit(next.moduleId, next.unitId);
    });
  } else {
    nextBtns.forEach((b) => {
      if (!b) return;
      b.disabled = true;
      b.onclick = null;
    });
  }
}

async function selectUnit(moduleId, unitId) {
  state.currentModuleId = moduleId;
  state.currentUnitId = unitId;
  state.quizSelections = {};
  renderSidebar();

  const modIndex = state.modules.findIndex((m) => m.id === moduleId);
  const mod = state.modules[modIndex];
  const unitIndex = mod ? (mod.units || []).findIndex((u) => u.id === unitId) : -1;
  const unitMeta = mod && unitIndex >= 0 ? mod.units[unitIndex] : null;

  const breadcrumb = document.getElementById('unit-breadcrumb');
  if (breadcrumb && mod && unitMeta) {
    breadcrumb.innerHTML = `${escapeHtml(mod.title.replace(/^Module\s+\d+:\s*/i, ''))} &nbsp;/&nbsp; <span>${escapeHtml(unitMeta.title)}</span>`;
  }

  const mainTitle = document.getElementById('unit-main-title');
  if (mainTitle) mainTitle.textContent = unitMeta ? unitMeta.title : '';

  const modDesc = document.getElementById('unit-module-desc');
  if (modDesc) modDesc.textContent = mod && mod.description ? mod.description : '';

  updateCompleteButtonsUI();
  wireUnitNavigation();

  try {
    const payload = await API.getUnit(moduleId, unitId);
    state.unitData = payload;

    renderInteractiveWidget(moduleId, unitId);
    renderMarkdown(payload.markdown || '');
    buildUnitTOC();

    const quizItems = (payload.unit && payload.unit.knowledge_check) || payload.quiz || [];
    renderQuiz(quizItems);

    const labFile = (payload.unit && payload.unit.lab_file) || '';
    const starterCode = payload.starter_code || '';
    const editor = document.getElementById('code-editor');
    const labLabel = document.getElementById('lab-filename-label');
    const inlineFilename = document.getElementById('inline-lab-filename');
    const tocLabName = document.getElementById('toc-lab-name');
    const inlineCard = document.getElementById('inline-lab-card');

    if (starterCode) {
      if (editor) editor.value = starterCode;
      updateEditorGutter();
      const shortName = labFile || 'lab.py';
      if (labLabel) labLabel.textContent = shortName;
      if (inlineFilename) inlineFilename.textContent = shortName;
      if (tocLabName) tocLabName.textContent = shortName;
      if (inlineCard) inlineCard.style.display = 'block';
    } else {
      if (editor) editor.value = '';
      updateEditorGutter();
      if (inlineCard) inlineCard.style.display = 'none';
    }

    const scrollPane = document.getElementById('content-scroll-pane');
    if (scrollPane) scrollPane.scrollTop = 0;
  } catch (err) {
    const mdEl = document.getElementById('unit-markdown');
    if (mdEl) {
      mdEl.innerHTML = `<p style="color:var(--status-err)">Failed to load unit: ${escapeHtml(err.message)}</p>`;
    }
  }
}

/* ==========================================================================
   Interactive Hardware & Algorithm Explorers
   ========================================================================== */
function renderInteractiveWidget(moduleId, unitId) {
  const mount = document.getElementById('interactive-widget-mount');
  if (!mount) return;

  if (moduleId.startsWith('00_')) {
    renderHardwareTopologyWidget(mount);
  } else if (moduleId.startsWith('01_')) {
    renderBitLayoutAndMxuWidget(mount);
  } else if (moduleId.startsWith('02_')) {
    renderSyncBubbleWidget(mount);
  } else if (moduleId.startsWith('03_')) {
    renderRooflineWidget(mount);
  } else if (moduleId.startsWith('04_')) {
    renderPallasTilerWidget(mount);
  } else if (moduleId.startsWith('05_')) {
    renderSpmdShardingWidget(mount);
  } else if (moduleId.startsWith('06_')) {
    renderHfSizingWidget(mount);
  } else {
    mount.innerHTML = '';
  }
}

function renderHardwareTopologyWidget(mount) {
  const specs = {
    v5e: {
      name: 'v5e',
      coresPerChip: '1 TensorCore (4 MXUs)',
      mxuDim: '128 × 128',
      bf16Tflops: 197,
      int8Tflops: 393,
      fp8Tflops: null,
      hbmCap: 16,
      hbmBw: 819,
      runtime: 'v2-alpha-tpuv5-lite'
    },
    v5p: {
      name: 'v5p',
      coresPerChip: '2 TensorCores (MegaCore)',
      mxuDim: '128 × 128',
      bf16Tflops: 459,
      int8Tflops: 918,
      fp8Tflops: null,
      hbmCap: 95,
      hbmBw: 2765,
      runtime: 'v2-alpha-tpuv5'
    },
    v6e: {
      name: 'v6e (Trillium)',
      coresPerChip: '1 TensorCore (2 MXUs)',
      mxuDim: '256 × 256',
      bf16Tflops: 918,
      int8Tflops: 1836,
      fp8Tflops: 918,
      hbmCap: 32,
      hbmBw: 1600,
      runtime: 'v2-alpha-tpuv6e'
    },
    v7: {
      name: 'v7 (Ironwood)',
      coresPerChip: '2 TensorCores',
      mxuDim: '256 × 256 (FP8/BF16)',
      bf16Tflops: 2307,
      int8Tflops: 4614,
      fp8Tflops: 4614,
      hbmCap: 192,
      hbmBw: 7370,
      runtime: 'tpu-v7-runtime'
    }
  };

  const shapes = {
    '1024x1024': [1024, 1024],
    '4096x4096': [4096, 4096],
    '8192x8192': [8192, 8192],
    '16384x16384': [16384, 16384]
  };
  const dtypes = {
    float32: { bytes: 4, label: 'float32 (4B)' },
    bfloat16: { bytes: 2, label: 'bfloat16 (2B)' },
    fp8_e4m3: { bytes: 1, label: 'fp8_e4m3fn (1B)' },
    fp8_e5m2: { bytes: 1, label: 'fp8_e5m2 (1B)' },
    int8: { bytes: 1, label: 'int8 (1B)' }
  };

  const curHw = specs[state.widgetState.hwGen] || specs.v6e;
  const [dimM, dimN] = shapes[state.widgetState.tensorShape] || [4096, 4096];
  const dtypeKey = state.widgetState.tensorDtype;
  const curDtype = dtypes[dtypeKey] || dtypes.bfloat16;

  let activeTflops = curHw.bf16Tflops;
  let computeLabel = 'Peak BF16 Compute';
  if (dtypeKey === 'int8') {
    activeTflops = curHw.int8Tflops;
    computeLabel = 'Peak INT8 Compute';
  } else if (dtypeKey === 'fp8_e4m3' || dtypeKey === 'fp8_e5m2') {
    activeTflops = curHw.fp8Tflops || curHw.bf16Tflops;
    computeLabel = curHw.fp8Tflops ? 'Peak FP8 Compute' : 'BF16 Fallback (Dequant)';
  } else if (dtypeKey === 'float32') {
    activeTflops = Math.round(curHw.bf16Tflops / 2);
    computeLabel = 'Peak FP32 Compute';
  }

  const tensorBytes = dimM * dimN * curDtype.bytes;
  const tensorMiB = (tensorBytes / (1024 * 1024)).toFixed(1);
  const matmulFlops = 2 * dimM * dimN * dimN;
  const idealUs = ((matmulFlops / (activeTflops * 1e12)) * 1e6).toFixed(1);
  const hbmTransferUs = (((tensorBytes * 3) / (curHw.hbmBw * 1e9)) * 1e6).toFixed(1);
  const pcieTransferUs = ((tensorBytes / (16 * 1e9)) * 1e6).toFixed(1);

  mount.innerHTML = `
    <div class="interactive-figure">
      <div class="fig-topbar">
        <span class="fig-badge">TPU Hardware &amp; Memory Calculator</span>
        <span class="fig-hint">${curHw.runtime}</span>
      </div>
      <div class="fig-body">
        <div style="display:flex;flex-wrap:wrap;gap:18px;margin-bottom:6px;">
          <div>
            <div class="blueprint-kicker" style="margin-bottom:6px;">TPU Generation</div>
            <div class="fig-controls" style="margin-bottom:0;">
              ${Object.keys(specs).map((k) => `
                <button type="button" class="fig-pill-btn ${state.widgetState.hwGen === k ? 'active' : ''}" data-hw="${k}">${specs[k].name}</button>
              `).join('')}
            </div>
          </div>
          <div>
            <div class="blueprint-kicker" style="margin-bottom:6px;">Matrix Shape</div>
            <div class="fig-controls" style="margin-bottom:0;">
              ${Object.keys(shapes).map((s) => `
                <button type="button" class="fig-pill-btn ${state.widgetState.tensorShape === s ? 'active' : ''}" data-shape="${s}">${s}</button>
              `).join('')}
            </div>
          </div>
          <div>
            <div class="blueprint-kicker" style="margin-bottom:6px;">Dtype</div>
            <div class="fig-controls" style="margin-bottom:0;">
              ${Object.keys(dtypes).map((d) => `
                <button type="button" class="fig-pill-btn ${state.widgetState.tensorDtype === d ? 'active' : ''}" data-dtype="${d}">${dtypes[d].label}</button>
              `).join('')}
            </div>
          </div>
        </div>

        <div class="fig-metrics-grid">
          <div class="fig-metric-box">
            <div class="fig-metric-label">MXU Array</div>
            <div class="fig-metric-val">${curHw.mxuDim}</div>
            <div class="fig-metric-sub">${curHw.coresPerChip}</div>
          </div>
          <div class="fig-metric-box">
            <div class="fig-metric-label">${computeLabel}</div>
            <div class="fig-metric-val">${activeTflops.toLocaleString()} TFLOP/s</div>
            <div class="fig-metric-sub">${dimM}² Matmul: <strong>${idealUs} µs</strong></div>
          </div>
          <div class="fig-metric-box">
            <div class="fig-metric-label">HBM Capacity &amp; BW</div>
            <div class="fig-metric-val">${curHw.hbmCap} GiB @ ${curHw.hbmBw} GB/s</div>
            <div class="fig-metric-sub">Tensor: <strong>${tensorMiB} MiB</strong> (${hbmTransferUs} µs R/W)</div>
          </div>
          <div class="fig-metric-box">
            <div class="fig-metric-label">PCIe vs HBM</div>
            <div class="fig-metric-val" style="color:var(--accent-coral);">${(pcieTransferUs / Math.max(0.1, hbmTransferUs)).toFixed(0)}× Slower</div>
            <div class="fig-metric-sub">.cpu(): ${pcieTransferUs} µs vs ${hbmTransferUs} µs</div>
          </div>
        </div>
      </div>
    </div>
  `;

  mount.querySelectorAll('[data-hw]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.hwGen = btn.dataset.hw;
      renderHardwareTopologyWidget(mount);
    };
  });
  mount.querySelectorAll('[data-shape]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.tensorShape = btn.dataset.shape;
      renderHardwareTopologyWidget(mount);
    };
  });
  mount.querySelectorAll('[data-dtype]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.tensorDtype = btn.dataset.dtype;
      renderHardwareTopologyWidget(mount);
    };
  });
}

function renderBitLayoutAndMxuWidget(mount) {
  const formats = {
    bfloat16: {
      name: 'bfloat16',
      signBits: 1,
      expBits: 8,
      mantBits: 7,
      range: '±3.40 × 10³⁸',
      maxVal: 3.4e38
    },
    float16: {
      name: 'float16',
      signBits: 1,
      expBits: 5,
      mantBits: 10,
      range: '±65,504',
      maxVal: 65504
    },
    fp8_e4m3: {
      name: 'fp8_e4m3fn',
      signBits: 1,
      expBits: 4,
      mantBits: 3,
      range: '±448',
      maxVal: 448
    },
    fp8_e5m2: {
      name: 'fp8_e5m2',
      signBits: 1,
      expBits: 5,
      mantBits: 2,
      range: '±57,344',
      maxVal: 57344
    },
    float32: {
      name: 'float32',
      signBits: 1,
      expBits: 8,
      mantBits: 23,
      range: '±3.40 × 10³⁸',
      maxVal: 3.4e38
    }
  };

  const samples = {
    '1.0': { val: 1.0, label: '1.0' },
    '0.00001': { val: 1e-5, label: '1e-5' },
    '448': { val: 448, label: '448 (E4M3 Max)' },
    '65504': { val: 65504, label: '65,504 (FP16 Max)' },
    '100000': { val: 100000, label: '100,000' }
  };

  const mxuModes = {
    default: {
      label: 'DEFAULT (1-Pass bf16)',
      passes: 1,
      relSpeed: '1.0× speed',
      desc: 'Truncates FP32 inputs to 1× bfloat16 (7-bit mantissa) and accumulates in FP32.'
    },
    high: {
      label: 'HIGH (3-Pass bf16)',
      passes: 3,
      relSpeed: '~0.33× speed',
      desc: 'Splits each FP32 operand into high + low bfloat16 parts (~14 bits effective mantissa).'
    },
    highest: {
      label: 'HIGHEST (6-Pass FP32)',
      passes: 6,
      relSpeed: '~0.16× speed',
      desc: 'Splits FP32 into 3 bfloat16 components and executes 6 MXU passes.'
    }
  };

  const curFmt = formats[state.widgetState.bitFormat] || formats.bfloat16;
  const curSample = samples[state.widgetState.bitSample] || samples['1.0'];
  const curMxu = mxuModes[state.widgetState.mxuMode] || mxuModes.default;

  const bits = computeFloatBits(curSample.val, state.widgetState.bitFormat);
  const overflows = curSample.val > curFmt.maxVal;

  mount.innerHTML = `
    <div class="interactive-figure">
      <div class="fig-topbar">
        <span class="fig-badge">Floating-Point Bit Layout &amp; MXU Precision</span>
        <span class="fig-hint">${curFmt.signBits + curFmt.expBits + curFmt.mantBits} bits</span>
      </div>
      <div class="fig-body">
        <div style="display:flex;flex-wrap:wrap;gap:18px;margin-bottom:14px;">
          <div>
            <div class="blueprint-kicker" style="margin-bottom:6px;">Dtype</div>
            <div class="fig-controls" style="margin-bottom:0;">
              ${Object.keys(formats).map((f) => `
                <button type="button" class="fig-pill-btn ${state.widgetState.bitFormat === f ? 'active' : ''}" data-fmt="${f}">${formats[f].name}</button>
              `).join('')}
            </div>
          </div>
          <div>
            <div class="blueprint-kicker" style="margin-bottom:6px;">Value</div>
            <div class="fig-controls" style="margin-bottom:0;">
              ${Object.keys(samples).map((s) => `
                <button type="button" class="fig-pill-btn ${state.widgetState.bitSample === s ? 'active' : ''}" data-sample="${s}">${samples[s].label}</button>
              `).join('')}
            </div>
          </div>
          <div>
            <div class="blueprint-kicker" style="margin-bottom:6px;">MXU Precision</div>
            <div class="fig-controls" style="margin-bottom:0;">
              ${Object.keys(mxuModes).map((m) => `
                <button type="button" class="fig-pill-btn ${state.widgetState.mxuMode === m ? 'active' : ''}" data-mxu="${m}">${m.toUpperCase()} (${mxuModes[m].passes}x)</button>
              `).join('')}
            </div>
          </div>
        </div>

        <div class="bit-row" style="background:var(--bg-paper);padding:12px 14px;border:1px solid var(--border-hairline);border-radius:var(--radius-sm);">
          <div class="bit-row-header">
            <span>${curFmt.name} = <code>${curSample.val}</code></span>
            <span style="color:${overflows ? 'var(--status-err)' : 'var(--status-ok)'};">
              ${overflows ? `Overflows (Max for ${curFmt.name} is ${curFmt.maxVal.toLocaleString()})` : 'Representable'}
            </span>
          </div>
          <div class="bit-cells">
            ${bits.map((b) => `<div class="bit-cell bit-${b.type}" title="${b.type.toUpperCase()} bit">${b.bit}</div>`).join('')}
          </div>
          <div style="display:flex;gap:16px;margin-top:8px;font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);">
            <span>■ Sign (${curFmt.signBits}b)</span>
            <span style="color:#003a9b;">■ Exponent (${curFmt.expBits}b, range ${curFmt.range})</span>
            <span style="color:#085d2b;">■ Mantissa (${curFmt.mantBits}b)</span>
          </div>
        </div>

        <div class="fig-metrics-grid">
          <div class="fig-metric-box">
            <div class="fig-metric-label">MXU Passes</div>
            <div class="fig-metric-val">${curMxu.passes}×</div>
            <div class="fig-metric-sub">${curMxu.relSpeed}</div>
          </div>
          <div class="fig-metric-box" style="grid-column: span 2;">
            <div class="fig-metric-label">${curMxu.label}</div>
            <div class="fig-metric-sub" style="margin-top:4px;font-size:12.5px;color:var(--text-ink);">${curMxu.desc}</div>
          </div>
        </div>
      </div>
    </div>
  `;

  mount.querySelectorAll('[data-fmt]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.bitFormat = btn.dataset.fmt;
      renderBitLayoutAndMxuWidget(mount);
    };
  });
  mount.querySelectorAll('[data-sample]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.bitSample = btn.dataset.sample;
      renderBitLayoutAndMxuWidget(mount);
    };
  });
  mount.querySelectorAll('[data-mxu]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.mxuMode = btn.dataset.mxu;
      renderBitLayoutAndMxuWidget(mount);
    };
  });
}

function computeFloatBits(val, fmt) {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  view.setFloat32(0, val, false);
  const uint32 = view.getUint32(0, false);
  const fullBits = uint32.toString(2).padStart(32, '0');
  const sign = fullBits[0];
  const exp32 = parseInt(fullBits.slice(1, 9), 2) - 127;

  if (fmt === 'float32') {
    return fullBits.split('').map((b, i) => ({
      bit: b,
      type: i === 0 ? 'sign' : i <= 8 ? 'exp' : 'mant'
    }));
  }
  if (fmt === 'bfloat16') {
    return fullBits.slice(0, 16).split('').map((b, i) => ({
      bit: b,
      type: i === 0 ? 'sign' : i <= 8 ? 'exp' : 'mant'
    }));
  }
  if (fmt === 'fp8_e4m3') {
    if (val > 448) {
      return '01111111'.split('').map((b, i) => ({
        bit: b,
        type: i === 0 ? 'sign' : i <= 4 ? 'exp' : 'mant'
      }));
    }
    const exp4 = Math.max(0, Math.min(15, exp32 + 7)).toString(2).padStart(4, '0');
    const mant3 = fullBits.slice(9, 12);
    return (sign + exp4 + mant3).split('').map((b, i) => ({
      bit: b,
      type: i === 0 ? 'sign' : i <= 4 ? 'exp' : 'mant'
    }));
  }
  if (fmt === 'fp8_e5m2') {
    if (val > 57344) {
      return '01111100'.split('').map((b, i) => ({
        bit: b,
        type: i === 0 ? 'sign' : i <= 5 ? 'exp' : 'mant'
      }));
    }
    const exp5 = Math.max(0, Math.min(31, exp32 + 15)).toString(2).padStart(5, '0');
    const mant2 = fullBits.slice(9, 11);
    return (sign + exp5 + mant2).split('').map((b, i) => ({
      bit: b,
      type: i === 0 ? 'sign' : i <= 5 ? 'exp' : 'mant'
    }));
  }
  if (val > 65504) {
    const infBits = '0111110000000000';
    return infBits.split('').map((b, i) => ({
      bit: b,
      type: i === 0 ? 'sign' : i <= 5 ? 'exp' : 'mant'
    }));
  }
  const exp16 = Math.max(0, Math.min(31, exp32 + 15)).toString(2).padStart(5, '0');
  const mant16 = fullBits.slice(9, 19);
  const fp16Str = sign + exp16 + mant16;
  return fp16Str.split('').map((b, i) => ({
    bit: b,
    type: i === 0 ? 'sign' : i <= 5 ? 'exp' : 'mant'
  }));
}

function renderSyncBubbleWidget(mount) {
  const isBad = state.widgetState.syncMode === 'sync';
  mount.innerHTML = `
    <div class="interactive-figure">
      <div class="fig-topbar">
        <span class="fig-badge">Host-Device Synchronization Timeline</span>
        <span class="fig-hint" style="color:${isBad ? 'var(--status-err)' : 'var(--status-ok)'};font-weight:700;">
          ${isBad ? '64% TPU Idle' : '98% TPU Utilization'}
        </span>
      </div>
      <div class="fig-body">
        <div class="fig-controls">
          <button type="button" class="fig-pill-btn ${!isBad ? 'active' : ''}" data-sync="accum">Detached Accumulator / jax.lax.scan</button>
          <button type="button" class="fig-pill-btn ${isBad ? 'active' : ''}" data-sync="sync">Per-Step loss.item()</button>
        </div>

        <div style="background:var(--bg-paper);border:1px solid var(--border-hairline);padding:14px;border-radius:var(--radius-sm);">
          <div style="display:grid;grid-template-columns:${isBad ? '1fr 1.5fr 1fr 1.5fr 1fr 1.5fr 1fr' : '1fr 1fr 1fr 1fr 0.4fr'};gap:6px;align-items:center;">
            ${isBad ? `
              <div style="background:#d1f4e0;border:1.5px solid #085d2b;padding:8px;text-align:center;font-family:var(--font-mono);font-size:11px;border-radius:4px;">Step 1<br><strong>MXU</strong></div>
              <div style="background:#fff0f0;border:1.5px dashed #c9252d;padding:8px;text-align:center;font-family:var(--font-mono);font-size:10.5px;color:#c9252d;border-radius:4px;">loss.item()<br>Stall</div>
              <div style="background:#d1f4e0;border:1.5px solid #085d2b;padding:8px;text-align:center;font-family:var(--font-mono);font-size:11px;border-radius:4px;">Step 2<br><strong>MXU</strong></div>
              <div style="background:#fff0f0;border:1.5px dashed #c9252d;padding:8px;text-align:center;font-family:var(--font-mono);font-size:10.5px;color:#c9252d;border-radius:4px;">loss.item()<br>Stall</div>
              <div style="background:#d1f4e0;border:1.5px solid #085d2b;padding:8px;text-align:center;font-family:var(--font-mono);font-size:11px;border-radius:4px;">Step 3<br><strong>MXU</strong></div>
              <div style="background:#fff0f0;border:1.5px dashed #c9252d;padding:8px;text-align:center;font-family:var(--font-mono);font-size:10.5px;color:#c9252d;border-radius:4px;">loss.item()<br>Stall</div>
              <div style="background:#d1f4e0;border:1.5px solid #085d2b;padding:8px;text-align:center;font-family:var(--font-mono);font-size:11px;border-radius:4px;">Step 4<br><strong>MXU</strong></div>
            ` : `
              <div style="background:#d1f4e0;border:1.5px solid #085d2b;padding:8px;text-align:center;font-family:var(--font-mono);font-size:11px;border-radius:4px;">Step 1<br><strong>MXU</strong></div>
              <div style="background:#d1f4e0;border:1.5px solid #085d2b;padding:8px;text-align:center;font-family:var(--font-mono);font-size:11px;border-radius:4px;">Step 2<br><strong>MXU</strong></div>
              <div style="background:#d1f4e0;border:1.5px solid #085d2b;padding:8px;text-align:center;font-family:var(--font-mono);font-size:11px;border-radius:4px;">Step 3<br><strong>MXU</strong></div>
              <div style="background:#d1f4e0;border:1.5px solid #085d2b;padding:8px;text-align:center;font-family:var(--font-mono);font-size:11px;border-radius:4px;">Step 4<br><strong>MXU</strong></div>
              <div style="background:#cfe2ff;border:1.5px solid #003a9b;padding:8px;text-align:center;font-family:var(--font-mono);font-size:10.5px;border-radius:4px;">Read<br>@ N</div>
            `}
          </div>
        </div>
      </div>
    </div>
  `;

  mount.querySelectorAll('[data-sync]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.syncMode = btn.dataset.sync;
      renderSyncBubbleWidget(mount);
    };
  });
}

function renderRooflineWidget(mount) {
  const hwMap = {
    v6e: { name: 'v6e', peakTflops: 918, bwGbs: 1600, ridge: 573.75 },
    v5p: { name: 'v5p (1 Core)', peakTflops: 229.5, bwGbs: 1382.5, ridge: 166.0 },
    v5e: { name: 'v5e', peakTflops: 197, bwGbs: 819, ridge: 240.5 }
  };
  const batches = [1, 8, 32, 128, 512];
  const curHw = hwMap[state.widgetState.rooflineHw] || hwMap.v6e;
  const B = state.widgetState.rooflineBatch || 16;

  const M = B * 16;
  const K = 4096;
  const N = 4096;
  const flops = 2 * M * K * N;
  const bytes = 2 * (M * K + K * N + M * N);
  const ai = flops / bytes;
  const achievedTflops = Math.min(curHw.peakTflops, (ai * curHw.bwGbs) / 1000);
  const mfuPct = Math.min(100, Math.round((achievedTflops / curHw.peakTflops) * 100));
  const isComputeBound = ai >= curHw.ridge;

  mount.innerHTML = `
    <div class="interactive-figure">
      <div class="fig-topbar">
        <span class="fig-badge">Roofline Calculator</span>
        <span class="fig-hint" style="color:${isComputeBound ? 'var(--status-ok)' : 'var(--accent-coral)'};font-weight:700;">
          ${isComputeBound ? 'Compute-Bound' : 'HBM Bandwidth-Bound'}
        </span>
      </div>
      <div class="fig-body">
        <div style="display:flex;flex-wrap:wrap;gap:18px;margin-bottom:8px;">
          <div>
            <div class="blueprint-kicker" style="margin-bottom:6px;">TPU</div>
            <div class="fig-controls" style="margin-bottom:0;">
              ${Object.keys(hwMap).map((k) => `
                <button type="button" class="fig-pill-btn ${state.widgetState.rooflineHw === k ? 'active' : ''}" data-rfhw="${k}">${hwMap[k].name}</button>
              `).join('')}
            </div>
          </div>
          <div>
            <div class="blueprint-kicker" style="margin-bottom:6px;">Batch Size</div>
            <div class="fig-controls" style="margin-bottom:0;">
              ${batches.map((b) => `
                <button type="button" class="fig-pill-btn ${state.widgetState.rooflineBatch === b ? 'active' : ''}" data-rfbatch="${b}">B=${b} (M=${b * 16})</button>
              `).join('')}
            </div>
          </div>
        </div>

        <div class="fig-metrics-grid">
          <div class="fig-metric-box">
            <div class="fig-metric-label">Ridge Point</div>
            <div class="fig-metric-val">${curHw.ridge.toFixed(0)} FLOP/B</div>
            <div class="fig-metric-sub">${curHw.peakTflops} TFLOP/s ÷ ${curHw.bwGbs} GB/s</div>
          </div>
          <div class="fig-metric-box">
            <div class="fig-metric-label">Arithmetic Intensity</div>
            <div class="fig-metric-val">${ai.toFixed(0)} FLOP/B</div>
            <div class="fig-metric-sub">(${M} × 4096) @ (4096 × 4096) bf16</div>
          </div>
          <div class="fig-metric-box">
            <div class="fig-metric-label">Roofline Bound</div>
            <div class="fig-metric-val">${achievedTflops.toFixed(0)} TFLOP/s (${mfuPct}%)</div>
            <div class="fig-metric-sub">${isComputeBound ? 'MXU limited' : 'HBM limited'}</div>
          </div>
        </div>
      </div>
    </div>
  `;

  mount.querySelectorAll('[data-rfhw]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.rooflineHw = btn.dataset.rfhw;
      renderRooflineWidget(mount);
    };
  });
  mount.querySelectorAll('[data-rfbatch]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.rooflineBatch = parseInt(btn.dataset.rfbatch, 10);
      renderRooflineWidget(mount);
    };
  });
}

function renderPallasTilerWidget(mount) {
  const blocks = {
    '64x64': { bm: 64, bn: 64, note: 'Under-padded for 128×128 MXU' },
    '128x128': { bm: 128, bn: 128, note: 'Aligned to 128×128 MXU' },
    '256x256': { bm: 256, bn: 256, note: 'Aligned to v6e 256×256 MXU' },
    '512x512': { bm: 512, bn: 512, note: 'Fits in 16 MiB VMEM' },
    '2048x2048': { bm: 2048, bn: 2048, note: 'Exceeds 16 MiB VMEM' }
  };

  const cur = blocks[state.widgetState.pallasBlock] || blocks['128x128'];
  const singleTileBytes = cur.bm * cur.bn * 4;
  const totalVmemKiB = Math.round((singleTileBytes * 3 * 2) / 1024);
  const vmemPct = Math.min(100, Math.round((totalVmemKiB / 16384) * 100));
  const oom = totalVmemKiB > 16384;

  mount.innerHTML = `
    <div class="interactive-figure">
      <div class="fig-topbar">
        <span class="fig-badge">Pallas BlockSpec VMEM Calculator</span>
        <span class="fig-hint" style="color:${oom ? 'var(--status-err)' : 'var(--status-ok)'};font-weight:700;">
          ${oom ? 'VMEM OOM (> 16 MiB)' : `${totalVmemKiB} KiB / 16,384 KiB`}
        </span>
      </div>
      <div class="fig-body">
        <div class="blueprint-kicker" style="margin-bottom:6px;">Block Shape (FP32, Double-Buffered)</div>
        <div class="fig-controls">
          ${Object.keys(blocks).map((b) => `
            <button type="button" class="fig-pill-btn ${state.widgetState.pallasBlock === b ? 'active' : ''}" data-pblock="${b}">(${b.replace('x', ', ')})</button>
          `).join('')}
        </div>
        <div class="fig-metrics-grid">
          <div class="fig-metric-box">
            <div class="fig-metric-label">Single Tile</div>
            <div class="fig-metric-val">${(singleTileBytes / 1024).toFixed(0)} KiB</div>
            <div class="fig-metric-sub">(${cur.bm}, ${cur.bn}) × 4B</div>
          </div>
          <div class="fig-metric-box">
            <div class="fig-metric-label">Total VMEM (2× Buffered)</div>
            <div class="fig-metric-val">${totalVmemKiB} KiB (${vmemPct}%)</div>
            <div class="fig-metric-sub">2 Inputs + 1 Output × 2</div>
          </div>
          <div class="fig-metric-box">
            <div class="fig-metric-label">MXU Alignment</div>
            <div class="fig-metric-val">${cur.bm % 128 === 0 ? 'Aligned' : 'Unaligned'}</div>
            <div class="fig-metric-sub">${cur.note}</div>
          </div>
        </div>
      </div>
    </div>
  `;

  mount.querySelectorAll('[data-pblock]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.pallasBlock = btn.dataset.pblock;
      renderPallasTilerWidget(mount);
    };
  });
}

function renderSpmdShardingWidget(mount) {
  const modes = {
    ddp: {
      name: '1D DDP',
      perChipGb: 112.0,
      commOp: 'All-Reduce Gradients',
      spec: 'Weights: P(None, None) | Batch: P("data", None)'
    },
    fsdp: {
      name: '1D FSDP / SimpleFSDP',
      perChipGb: 14.0,
      commOp: 'All-Gather Weights + Reduce-Scatter Grads',
      spec: 'Weights: P("data", None) | Batch: P("data", None)'
    },
    tp_fsdp: {
      name: '2D FSDP × Tensor Parallel (4×2)',
      perChipGb: 14.0,
      commOp: 'FSDP ("data") + All-Reduce ("model")',
      spec: 'MLP In: P("data", "model") | MLP Out: P("model", "data")'
    }
  };

  const cur = modes[state.widgetState.spmdMode] || modes.fsdp;
  const fitsV6e = cur.perChipGb <= 32;

  mount.innerHTML = `
    <div class="interactive-figure">
      <div class="fig-topbar">
        <span class="fig-badge">8-Chip SPMD Memory Calculator (7B Model)</span>
        <span class="fig-hint" style="color:${fitsV6e ? 'var(--status-ok)' : 'var(--status-err)'};font-weight:700;">
          ${fitsV6e ? 'Fits on v6e (32 GiB) & v5p (95 GiB)' : 'OOM on v6e & v5p'}
        </span>
      </div>
      <div class="fig-body">
        <div class="fig-controls">
          ${Object.keys(modes).map((m) => `
            <button type="button" class="fig-pill-btn ${state.widgetState.spmdMode === m ? 'active' : ''}" data-spmd="${m}">${modes[m].name}</button>
          `).join('')}
        </div>
        <div class="fig-metrics-grid">
          <div class="fig-metric-box">
            <div class="fig-metric-label">Per-Chip HBM</div>
            <div class="fig-metric-val">${cur.perChipGb.toFixed(1)} GB / chip</div>
            <div class="fig-metric-sub">v6e: 32 GiB | v5p: 95 GiB</div>
          </div>
          <div class="fig-metric-box" style="grid-column: span 2;">
            <div class="fig-metric-label">PartitionSpec &amp; Collective</div>
            <div class="fig-metric-val" style="font-size:13.5px;"><code>${cur.spec}</code></div>
            <div class="fig-metric-sub">${cur.commOp}</div>
          </div>
        </div>
      </div>
    </div>
  `;

  mount.querySelectorAll('[data-spmd]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.spmdMode = btn.dataset.spmd;
      renderSpmdShardingWidget(mount);
    };
  });
}

function renderHfSizingWidget(mount) {
  const models = {
    qwen06b: { name: 'Qwen3-0.6B', paramsB: 0.6, vocab: 151936 },
    llama1b: { name: 'Llama-3.2-1B', paramsB: 1.23, vocab: 128256 },
    gemma2b: { name: 'Gemma-2-2B', paramsB: 2.6, vocab: 256000 },
    llama8b: { name: 'Llama-3.1-8B', paramsB: 8.03, vocab: 128256 }
  };

  const m = models[state.widgetState.hfModel] || models.qwen06b;
  const isLora = state.widgetState.hfTuneMode === 'lora';
  const isChunked = state.widgetState.hfLossMode === 'chunked';

  const weightGb = m.paramsB * 2.0;
  const trainStateGb = isLora ? m.paramsB * 0.25 : m.paramsB * 14.0;
  const fullLogitGb = (8192 * m.vocab * 4 * 2) / 1e9;
  const chunkedLogitGb = (512 * m.vocab * 4 * 2) / 1e9;
  const logitGb = isChunked ? chunkedLogitGb : fullLogitGb;
  const actGb = m.paramsB * 1.4;

  const totalGb = weightGb + trainStateGb + logitGb + actGb;
  const fitsV6e1 = totalGb <= 30.0;
  const fitsV5p1 = totalGb <= 90.0;

  mount.innerHTML = `
    <div class="interactive-figure">
      <div class="fig-topbar">
        <span class="fig-badge">Hugging Face Fine-Tuning HBM Calculator</span>
        <span class="fig-hint" style="color:${fitsV6e1 ? 'var(--status-ok)' : fitsV5p1 ? 'var(--accent-coral)' : 'var(--status-err)'};font-weight:700;">
          ${totalGb.toFixed(1)} GiB (${fitsV6e1 ? 'Fits v6e-1' : fitsV5p1 ? 'Fits v5p' : 'Requires FSDP'})
        </span>
      </div>
      <div class="fig-body">
        <div style="display:flex;flex-wrap:wrap;gap:18px;margin-bottom:8px;">
          <div>
            <div class="blueprint-kicker" style="margin-bottom:6px;">Model</div>
            <div class="fig-controls" style="margin-bottom:0;">
              ${Object.keys(models).map((k) => `
                <button type="button" class="fig-pill-btn ${state.widgetState.hfModel === k ? 'active' : ''}" data-hfmod="${k}">${models[k].name}</button>
              `).join('')}
            </div>
          </div>
          <div>
            <div class="blueprint-kicker" style="margin-bottom:6px;">Mode</div>
            <div class="fig-controls" style="margin-bottom:0;">
              <button type="button" class="fig-pill-btn ${isLora ? 'active' : ''}" data-hftune="lora">LoRA (r=8)</button>
              <button type="button" class="fig-pill-btn ${!isLora ? 'active' : ''}" data-hftune="full">Full SFT</button>
            </div>
          </div>
          <div>
            <div class="blueprint-kicker" style="margin-bottom:6px;">Logits (V=${m.vocab.toLocaleString()})</div>
            <div class="fig-controls" style="margin-bottom:0;">
              <button type="button" class="fig-pill-btn ${isChunked ? 'active' : ''}" data-hfloss="chunked">Chunked (512)</button>
              <button type="button" class="fig-pill-btn ${!isChunked ? 'active' : ''}" data-hfloss="full">Full (8192)</button>
            </div>
          </div>
        </div>

        <div class="fig-metrics-grid">
          <div class="fig-metric-box">
            <div class="fig-metric-label">Weights + Optimizer</div>
            <div class="fig-metric-val">${(weightGb + trainStateGb).toFixed(2)} GiB</div>
            <div class="fig-metric-sub">Weights: ${weightGb.toFixed(1)} GiB | States: ${trainStateGb.toFixed(2)} GiB</div>
          </div>
          <div class="fig-metric-box">
            <div class="fig-metric-label">Logit Memory</div>
            <div class="fig-metric-val">${logitGb.toFixed(2)} GiB</div>
            <div class="fig-metric-sub">${isChunked ? `Saves ${(fullLogitGb - chunkedLogitGb).toFixed(1)} GiB` : 'Full [B*S, V] FP32'}</div>
          </div>
          <div class="fig-metric-box">
            <div class="fig-metric-label">Total HBM</div>
            <div class="fig-metric-val">${totalGb.toFixed(1)} GiB</div>
            <div class="fig-metric-sub">v6e-1: ${Math.round((totalGb / 32) * 100)}% | v5p: ${Math.round((totalGb / 95) * 100)}%</div>
          </div>
        </div>
      </div>
    </div>
  `;

  mount.querySelectorAll('[data-hfmod]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.hfModel = btn.dataset.hfmod;
      renderHfSizingWidget(mount);
    };
  });
  mount.querySelectorAll('[data-hftune]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.hfTuneMode = btn.dataset.hftune;
      renderHfSizingWidget(mount);
    };
  });
  mount.querySelectorAll('[data-hfloss]').forEach((btn) => {
    btn.onclick = () => {
      state.widgetState.hfLossMode = btn.dataset.hfloss;
      renderHfSizingWidget(mount);
    };
  });
}

/* ==========================================================================
   Rich Markdown, Code Syntax Highlighting, Diagrams & Math
   ========================================================================== */
function renderInline(text) {
  let html = escapeHtml(text);
  // Allow safe <br> line breaks inside table cells / prose
  html = html.replace(/&lt;br\s*\/?&gt;/gi, '<br>');

  const codeSpans = [];
  html = html.replace(/`([^`]+)`/g, (_, code) => {
    const idx = codeSpans.length;
    codeSpans.push(`<code>${code}</code>`);
    return `@@CODESPAN_${idx}@@`;
  });

  if (window.katex) {
    html = html.replace(/\$([^$\n]+)\$/g, (_, tex) => {
      try {
        const decoded = tex.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
        return window.katex.renderToString(decoded, { throwOnError: false, displayMode: false });
      } catch (e) {
        return `<code>${tex}</code>`;
      }
    });
  }

  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  html = html.replace(/@@CODESPAN_(\d+)@@/g, (_, idx) => codeSpans[Number(idx)]);
  return html;
}

function classifyCallout(rawText) {
  const clean = rawText.replace(/^>\s?/, '').trim();
  if (/^(\*\*|\[)?(warning|caution|anti-pattern|pitfall)/i.test(clean)) {
    return { cls: 'callout-warn' };
  }
  if (/^(\*\*|\[)?(tip|best practice|pro tip|optimization|performance|recipe)/i.test(clean)) {
    return { cls: 'callout-tip' };
  }
  return { cls: 'callout-info' };
}

function buildVisualFlowDiagramHtml(rawDiagram) {
  return `
    <div class="visual-diagram-card">
      <pre class="diagram-raw-pre"><code class="nohighlight">${escapeHtml(rawDiagram)}</code></pre>
    </div>
  `;
}

function stripCommonIndent(lines) {
  const nonEmpty = lines.filter((l) => l.trim().length > 0);
  if (nonEmpty.length === 0) return '';
  const minIndent = Math.min(...nonEmpty.map((l) => l.match(/^\s*/)[0].length));
  return lines.map((l) => l.slice(minIndent)).join('\n');
}

function renderMarkdown(md) {
  const container = document.getElementById('unit-markdown');
  if (!container) return;

  const lines = md.replace(/\r\n/g, '\n').split('\n');
  let html = '';
  let inCodeBlock = false;
  let codeLang = '';
  let codeBuffer = [];
  let inList = false;
  let listType = 'ul';
  let inTable = false;
  let tableRows = [];
  let inMathBlock = false;
  let mathBuffer = [];
  let headingCounter = 0;

  const flushList = () => {
    if (inList) {
      html += `</${listType}>`;
      inList = false;
    }
  };

  const flushTable = () => {
    if (!inTable || tableRows.length === 0) return;
    html += '<div class="table-wrapper"><table>';
    const headerCells = tableRows[0]
      .split('|')
      .map((c) => c.trim())
      .filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
    html += '<thead><tr>' + headerCells.map((c) => `<th>${renderInline(c)}</th>`).join('') + '</tr></thead><tbody>';

    for (let i = 1; i < tableRows.length; i++) {
      const row = tableRows[i];
      if (/^\|[\s-:|]+\|$/.test(row.trim())) continue;
      const cells = row
        .split('|')
        .map((c) => c.trim())
        .filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
      html += '<tr>' + cells.map((c) => `<td>${renderInline(c)}</td>`).join('') + '</tr>';
    }
    html += '</tbody></table></div>';
    inTable = false;
    tableRows = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith('$$')) {
      flushTable();
      if (!inMathBlock && trimmed !== '$$' && trimmed.endsWith('$$') && trimmed.length > 4) {
        const tex = trimmed.slice(2, -2);
        if (window.katex) {
          html += `<div class="math-display-block">${window.katex.renderToString(tex, { throwOnError: false, displayMode: true })}</div>`;
        } else {
          html += `<pre><code class="nohighlight">${escapeHtml(tex)}</code></pre>`;
        }
        continue;
      }
      if (!inMathBlock) {
        inMathBlock = true;
        mathBuffer = [];
      } else {
        inMathBlock = false;
        const tex = mathBuffer.join('\n');
        if (window.katex) {
          html += `<div class="math-display-block">${window.katex.renderToString(tex, { throwOnError: false, displayMode: true })}</div>`;
        } else {
          html += `<pre><code class="nohighlight">${escapeHtml(tex)}</code></pre>`;
        }
      }
      continue;
    }

    if (inMathBlock) {
      mathBuffer.push(line);
      continue;
    }

    // Match top-level OR indented fenced code blocks (` ``` `)
    if (trimmed.startsWith('```')) {
      flushTable();
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeLang = trimmed.slice(3).trim().toLowerCase();
        codeBuffer = [];
      } else {
        const rawCode = stripCommonIndent(codeBuffer);
        const isDiagram =
          (!codeLang || codeLang === 'text' || codeLang === 'ascii' || codeLang === 'diagram') &&
          (/[┌┐└┘├┤┬┴┼─│╭╮╯╰►▼▲]|(\+---)|(\|.*->)|(-->)/.test(rawCode));

        if (isDiagram) {
          html += buildVisualFlowDiagramHtml(rawCode);
        } else {
          const langLabel = codeLang || 'code';
          const isRunnablePython = langLabel === 'python' || langLabel === 'py';
          const encodedSnippet = encodeURIComponent(rawCode);
          html += `
            <div class="code-card">
              <div class="code-card-header">
                <div class="code-header-left">
                  <span class="code-lang-badge">${escapeHtml(langLabel)}</span>
                </div>
                <div class="code-header-actions">
                  ${isRunnablePython ? `<button class="code-action-btn" type="button" data-snippet="${encodedSnippet}">Load in Editor</button>` : ''}
                  <button class="code-action-btn" type="button" data-copy="${encodedSnippet}">Copy</button>
                </div>
              </div>
              <pre><code class="language-${escapeHtml(langLabel)}">${escapeHtml(rawCode)}</code></pre>
            </div>
          `;
        }
        inCodeBlock = false;
        codeLang = '';
      }
      continue;
    }

    if (inCodeBlock) {
      codeBuffer.push(line);
      continue;
    }

    // Ignore markdown horizontal rule dividers (--- / ***) since h2 headings already have top borders
    if (/^(\s*)(-{3,}|\*{3,}|_{3,})(\s*)$/.test(line)) {
      flushList();
      flushTable();
      continue;
    }

    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      flushList();
      inTable = true;
      tableRows.push(trimmed);
      continue;
    } else if (inTable) {
      flushTable();
    }

    if (trimmed === '') {
      // Keep list open across blank lines or indented blocks if next non-empty line is a list item or indented continuation
      let nextIdx = i + 1;
      while (nextIdx < lines.length && lines[nextIdx].trim() === '') nextIdx++;
      const nextLine = nextIdx < lines.length ? lines[nextIdx] : '';
      const continuesList =
        inList &&
        (/^\s*(\*|-|\d+\.)\s+/.test(nextLine) || /^\s{2,}\S/.test(nextLine));
      if (!continuesList) {
        flushList();
      }
      continue;
    }

    if (line.startsWith('# ')) {
      flushList();
      continue;
    } else if (line.startsWith('## ')) {
      flushList();
      headingCounter++;
      html += `<h2 id="sec-${headingCounter}" data-toc-level="2">${renderInline(line.slice(3))}</h2>`;
    } else if (line.startsWith('### ')) {
      flushList();
      headingCounter++;
      html += `<h3 id="sec-${headingCounter}" data-toc-level="3">${renderInline(line.slice(4))}</h3>`;
    } else if (line.startsWith('#### ')) {
      flushList();
      headingCounter++;
      html += `<h4 id="sec-${headingCounter}">${renderInline(line.slice(5))}</h4>`;
    } else if (trimmed.startsWith('>')) {
      flushList();
      const quoteLines = [trimmed.replace(/^>\s?/, '')];
      while (i + 1 < lines.length && lines[i + 1].trim().startsWith('>')) {
        i++;
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ''));
      }
      const joined = quoteLines.filter(Boolean).join('<br>');
      const callout = classifyCallout(joined);
      html += `
        <div class="callout-box ${callout.cls}">
          <div>${renderInline(joined)}</div>
        </div>
      `;
    } else if (/^\s*(\*|-)\s+/.test(line)) {
      if (!inList || listType !== 'ul') {
        flushList();
        html += '<ul>';
        inList = true;
        listType = 'ul';
      }
      html += `<li>${renderInline(line.replace(/^\s*(\*|-)\s+/, ''))}</li>`;
    } else if (/^\s*\d+\.\s+/.test(line)) {
      if (!inList || listType !== 'ol') {
        flushList();
        const startNum = (line.match(/^\s*(\d+)\./) || [])[1] || '1';
        html += startNum !== '1' ? `<ol start="${startNum}">` : '<ol>';
        inList = true;
        listType = 'ol';
      }
      html += `<li>${renderInline(line.replace(/^\s*\d+\.\s+/, ''))}</li>`;
    } else {
      if (inList && /^\s{2,}\S/.test(line)) {
        html += `<div style="margin:6px 0 8px 0;">${renderInline(trimmed)}</div>`;
      } else {
        flushList();
        html += `<p>${renderInline(trimmed)}</p>`;
      }
    }
  }

  flushList();
  flushTable();
  container.innerHTML = html;

  if (window.hljs) {
    container.querySelectorAll('.code-card pre code').forEach((el) => {
      window.hljs.highlightElement(el);
    });
  }

  container.querySelectorAll('[data-copy]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const raw = decodeURIComponent(btn.dataset.copy || '');
      navigator.clipboard.writeText(raw).then(() => {
        const orig = btn.textContent;
        btn.textContent = 'Copied ✓';
        setTimeout(() => { btn.textContent = orig; }, 1400);
      });
    });
  });

  container.querySelectorAll('[data-snippet]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const raw = decodeURIComponent(btn.dataset.snippet || '');
      const editor = document.getElementById('code-editor');
      if (editor) {
        editor.value = raw;
        updateEditorGutter();
      }
      setViewMode('split');
    });
  });
}

function buildUnitTOC() {
  const tocList = document.getElementById('unit-toc-list');
  const container = document.getElementById('unit-markdown');
  if (!tocList || !container) return;

  tocList.innerHTML = '';
  const headings = container.querySelectorAll('h2[id], h3[id]');
  headings.forEach((h) => {
    const a = document.createElement('a');
    a.className = 'toc-link';
    if (h.tagName === 'H3') {
      a.style.paddingLeft = '10px';
      a.style.fontSize = '11.5px';
    }
    a.href = `#${h.id}`;
    a.textContent = h.textContent;
    a.addEventListener('click', (e) => {
      e.preventDefault();
      h.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    tocList.appendChild(a);
  });
}

function renderQuiz(quizList) {
  const section = document.getElementById('knowledge-check-section');
  const listEl = document.getElementById('knowledge-check-list');
  const banner = document.getElementById('kc-result-banner');

  if (!section || !listEl) return;
  if (!quizList || quizList.length === 0) {
    section.classList.add('hidden');
    return;
  }

  section.classList.remove('hidden');
  listEl.innerHTML = '';
  if (banner) {
    banner.classList.add('hidden');
    banner.textContent = '';
  }

  quizList.forEach((q, qIdx) => {
    const card = document.createElement('div');
    card.className = 'kc-question-card';

    const qTitle = document.createElement('div');
    qTitle.className = 'kc-question-title';
    qTitle.innerHTML = `${qIdx + 1}. ${renderInline(q.question || '')}`;
    card.appendChild(qTitle);

    const optsWrap = document.createElement('div');
    const expEl = document.createElement('div');
    expEl.className = 'kc-explanation hidden';

    const correctIdx = typeof q.answer_index === 'number' ? q.answer_index : q.answer;

    (q.options || []).forEach((optText, optIdx) => {
      const label = document.createElement('label');
      label.className = 'kc-option-label';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = `kc-q-${qIdx}`;
      radio.value = String(optIdx);
      radio.addEventListener('change', () => {
        state.quizSelections[qIdx] = optIdx;
        const isCorrect = optIdx === correctIdx;
        expEl.className = `kc-explanation ${isCorrect ? 'correct' : 'incorrect'}`;
        expEl.innerHTML = `${renderInline(q.explanation || '')}`;
      });

      const span = document.createElement('span');
      span.innerHTML = renderInline(optText);

      label.appendChild(radio);
      label.appendChild(span);
      optsWrap.appendChild(label);
    });

    card.appendChild(optsWrap);
    card.appendChild(expEl);
    listEl.appendChild(card);
  });

  const checkBtn = document.getElementById('btn-check-answers');
  if (checkBtn) {
    checkBtn.onclick = () => {
      let correctCount = 0;
      quizList.forEach((q, qIdx) => {
        const correctIdx = typeof q.answer_index === 'number' ? q.answer_index : q.answer;
        if (state.quizSelections[qIdx] === correctIdx) correctCount += 1;
      });
      if (banner) {
        banner.classList.remove('hidden');
        banner.textContent = `${correctCount} / ${quizList.length} correct`;
      }
      if (correctCount === quizList.length && state.currentModuleId && state.currentUnitId) {
        toggleUnitComplete(true);
      }
    };
  }
}

async function toggleUnitComplete(forceTrue = false) {
  if (!state.currentModuleId || !state.currentUnitId) return;
  const key = unitKey(state.currentModuleId, state.currentUnitId);
  if (forceTrue || !state.completedUnits.has(key)) {
    state.completedUnits.add(key);
    try {
      await API.saveProgress({ completed_unit: key });
    } catch (e) {
      // non-fatal
    }
  } else {
    state.completedUnits.delete(key);
  }
  updateCourseProgressUI();
  updateCompleteButtonsUI();
  renderSidebar();
}

async function refreshTpuStatus() {
  const indicator = document.getElementById('tpu-status-indicator');
  const zoneEl = document.getElementById('tpu-zone-input');
  const infraLog = document.getElementById('tpu-infra-log');
  const zone = zoneEl ? zoneEl.value : 'us-east1-d';

  if (indicator) indicator.textContent = '...';

  try {
    const status = await API.getTpuStatus(zone);
    state.tpuStatus = status;

    if (status.vms && status.vms.length > 0) {
      const readyVm = status.vms.find((v) => v.state === 'READY') || status.vms[0];
      if (indicator) {
        indicator.textContent = `${readyVm.name} (${readyVm.state})`;
      }
      if (infraLog) {
        infraLog.textContent = JSON.stringify(status, null, 2);
      }
    } else {
      if (indicator) indicator.textContent = 'Local';
      if (infraLog) {
        infraLog.textContent = `Project: ${status.project_id} | Zone: ${status.zone} | No active TPU VMs.`;
      }
    }
  } catch (err) {
    if (indicator) indicator.textContent = 'Offline';
  }
}

async function runInfraAction(action) {
  const zone = document.getElementById('tpu-zone-input').value;
  const accel = document.getElementById('tpu-accel-input').value;
  const nodeId = document.getElementById('tpu-node-input').value;
  const infraLog = document.getElementById('tpu-infra-log');

  if (infraLog) {
    infraLog.textContent = `Running ${action} (${zone}, ${accel}, ${nodeId})...\n`;
  }
  try {
    const res = await API.runTpuAction(action, zone, accel, nodeId);
    if (infraLog) {
      infraLog.textContent = `[Exit ${res.returncode}]\n${res.stdout || ''}\n${res.stderr || ''}`.trim();
    }
    refreshTpuStatus();
  } catch (err) {
    if (infraLog) infraLog.textContent = `Error: ${err.message}`;
  }
}

async function runLabCode() {
  if (state.isRunningLab) return;
  state.isRunningLab = true;

  const runBtn = document.getElementById('btn-run-code');
  const inlineRunBtn = document.getElementById('btn-inline-run');
  const tocRunBtn = document.getElementById('btn-toc-run-lab');
  const consoleOut = document.getElementById('console-output');
  const inlinePreview = document.getElementById('inline-console-preview');
  const hloOut = document.getElementById('hlo-output');
  const timingBadge = document.getElementById('run-timing-badge');
  const targetSelect = document.getElementById('run-target-select');
  const chkHlo = document.getElementById('chk-capture-hlo');
  const zoneEl = document.getElementById('tpu-zone-input');
  const nodeEl = document.getElementById('tpu-node-input');
  const editor = document.getElementById('code-editor');

  const target = targetSelect ? targetSelect.value : 'local';
  const captureHlo = chkHlo ? chkHlo.checked : false;
  const zone = zoneEl ? zoneEl.value : 'us-east1-d';
  const nodeId = nodeEl ? nodeEl.value : 'tpu-lab-vm';
  const code = editor ? editor.value : '';

  [runBtn, inlineRunBtn, tocRunBtn].forEach((b) => {
    if (!b) return;
    b.disabled = true;
    b.textContent = 'Running...';
  });

  if (consoleOut) consoleOut.textContent = 'Running...\n';
  if (inlinePreview) inlinePreview.textContent = 'Running...\n';
  if (timingBadge) timingBadge.textContent = target;

  try {
    const res = await API.runCode(code, target, zone, nodeId, captureHlo);
    const combinedOut = `${res.stdout || ''}${res.stderr ? '\n' + res.stderr : ''}`.trim() || '(No output)';
    if (consoleOut) consoleOut.textContent = combinedOut;
    if (inlinePreview) inlinePreview.textContent = combinedOut;
    if (hloOut) {
      hloOut.textContent = res.hlo_dump || 'No HLO dump.';
    }
    if (timingBadge) {
      timingBadge.textContent = `exit ${res.returncode} • ${res.duration_sec}s`;
    }
  } catch (err) {
    const msg = `Error: ${err.message}`;
    if (consoleOut) consoleOut.textContent = msg;
    if (inlinePreview) inlinePreview.textContent = msg;
    if (timingBadge) timingBadge.textContent = 'error';
  } finally {
    state.isRunningLab = false;
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.textContent = 'Run';
    }
    if (inlineRunBtn) {
      inlineRunBtn.disabled = false;
      inlineRunBtn.textContent = 'Run';
    }
    if (tocRunBtn) {
      tocRunBtn.disabled = false;
      tocRunBtn.textContent = 'Run Lab';
    }
  }
}

function setupEventListeners() {
  const on = (id, evt, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener(evt, fn);
  };

  on('toggle-sidebar-btn', 'click', () => {
    const ws = document.getElementById('workspace-grid');
    if (ws) ws.classList.toggle('sidebar-hidden');
  });

  on('sidebar-tab-modules', 'click', () => setSidebarTab('modules'));
  on('sidebar-tab-recipes', 'click', () => setSidebarTab('recipes'));
  on('sidebar-search-input', 'input', (e) => {
    state.searchQuery = e.target.value || '';
    renderSidebar();
    renderRecipesSidebar();
  });

  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      runLabCode();
      return;
    }
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    if (e.key === '[') {
      const prevBtn = document.getElementById('btn-prev-unit');
      if (prevBtn && !prevBtn.disabled) prevBtn.click();
    } else if (e.key === ']') {
      const nextBtn = document.getElementById('btn-next-unit');
      if (nextBtn && !nextBtn.disabled) nextBtn.click();
    } else if (e.key === '\\') {
      const ws = document.getElementById('workspace-grid');
      if (ws) ws.classList.toggle('sidebar-hidden');
    }
  });

  on('mode-editorial-btn', 'click', () => setViewMode('editorial'));
  on('mode-split-btn', 'click', () => setViewMode('split'));
  on('btn-open-split-lab', 'click', () => setViewMode('split'));
  on('btn-close-split-lab', 'click', () => setViewMode('editorial'));
  on('btn-inline-expand', 'click', () => setViewMode('split'));

  on('toggle-tpu-drawer-btn', 'click', () => {
    const drawer = document.getElementById('tpu-drawer');
    if (drawer) drawer.classList.toggle('hidden');
  });
  on('close-tpu-drawer-btn', 'click', () => {
    const drawer = document.getElementById('tpu-drawer');
    if (drawer) drawer.classList.add('hidden');
  });

  on('btn-refresh-tpu', 'click', refreshTpuStatus);
  on('btn-bootstrap-project', 'click', () => runInfraAction('bootstrap'));
  on('btn-provision-tpu', 'click', () => runInfraAction('provision'));
  on('btn-hunt-tpu', 'click', () => runInfraAction('hunt'));
  on('btn-install-runtime', 'click', () => runInfraAction('install_runtime'));
  on('btn-delete-tpu', 'click', () => runInfraAction('delete'));

  on('btn-run-code', 'click', runLabCode);
  on('btn-inline-run', 'click', runLabCode);
  on('btn-toc-run-lab', 'click', () => {
    const card = document.getElementById('inline-lab-card');
    if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    runLabCode();
  });

  on('btn-reset-code', 'click', () => {
    if (state.unitData && state.unitData.starter_code) {
      const editor = document.getElementById('code-editor');
      if (editor) {
        editor.value = state.unitData.starter_code;
        updateEditorGutter();
      }
    }
  });

  on('btn-copy-code', 'click', () => {
    const editor = document.getElementById('code-editor');
    const copyBtn = document.getElementById('btn-copy-code');
    if (editor && copyBtn) {
      navigator.clipboard.writeText(editor.value).then(() => {
        copyBtn.textContent = 'Copied ✓';
        setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1400);
      });
    }
  });

  on('code-editor', 'input', updateEditorGutter);

  on('btn-complete-unit', 'click', () => toggleUnitComplete(false));
  on('btn-footer-complete', 'click', () => toggleUnitComplete(false));

  on('tab-console', 'click', () => {
    document.getElementById('tab-console').classList.add('active');
    document.getElementById('tab-hlo').classList.remove('active');
    document.getElementById('console-output').classList.remove('hidden');
    document.getElementById('hlo-output').classList.add('hidden');
  });

  on('tab-hlo', 'click', () => {
    document.getElementById('tab-hlo').classList.add('active');
    document.getElementById('tab-console').classList.remove('active');
    document.getElementById('hlo-output').classList.remove('hidden');
    document.getElementById('console-output').classList.add('hidden');
  });
}

async function init() {
  setupEventListeners();
  setViewMode(state.viewMode);
  refreshTpuStatus();

  try {
    const [data, recipesData] = await Promise.all([
      API.getCurriculum(),
      API.getRecipes().catch(() => ({ recipes: [] }))
    ]);
    state.modules = data.modules || [];
    state.recipes = recipesData.recipes || [];
    if (data.project_id) {
      const projBadge = document.getElementById('project-badge');
      if (projBadge) projBadge.textContent = data.project_id;
    }
    if (data.progress && Array.isArray(data.progress.completed_units)) {
      state.completedUnits = new Set(data.progress.completed_units);
    }
    updateCourseProgressUI();
    renderSidebar();
    renderRecipesSidebar();

    if (state.modules.length > 0 && state.modules[0].units && state.modules[0].units.length > 0) {
      await selectUnit(state.modules[0].id, state.modules[0].units[0].id);
    }
  } catch (err) {
    const mdEl = document.getElementById('unit-markdown');
    if (mdEl) {
      mdEl.innerHTML = `<p style="color:var(--status-err)">Failed to load curriculum: ${escapeHtml(err.message)}</p>`;
    }
  }
}

document.addEventListener('DOMContentLoaded', init);
