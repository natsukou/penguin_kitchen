"use strict";
const $ = (id) => document.getElementById(id);
const names = {
  observe: "观察与识别",
  locate_spoon: "定位量勺",
  grasp_spoon: "抓取量勺",
  acquire_food: "获取食品",
  pour_food: "倒入餐碗",
  return_spoon: "放回量勺",
  grasp_bowl: "抓取餐碗",
  place_bowl: "放置到餐垫",
  verify: "最终确认",
  complete: "已完成",
  failed: "执行失败",
  cancelled: "已停止",
};
const objects = {
  spoon: "量勺",
  food_container: "食品容器",
  bowl: "餐碗",
  mat: "餐垫",
};
const colors = {
  spoon: "#bd8030",
  food_container: "#667d95",
  bowl: "#be6e58",
  mat: "#8aa28b",
};
const faults = {
  none: "正常完整流程",
  missing_spoon: "量勺缺失",
  stale_frame: "观测过期",
  grasp_slip: "抓取未确认",
  placement_error: "餐碗放置偏差",
  depth_missing: "深度缺失",
  camera_skew: "传感器时间差",
  command_failure: "执行接口异常",
};
const statuses = {
  idle: "待启动",
  running: "运行中",
  paused: "已暂停",
  stopping: "正在停止",
  stopped: "已停止",
  complete: "已完成",
  failed: "执行失败",
  cancelled: "已停止",
};
const moduleStatus = {
  ready: "待运行",
  healthy: "最近结果正常",
  fault: "异常",
  disabled: "未启用",
};
const events = {
  hub_ready: "中枢就绪",
  run_started: "记录已建立",
  scenario: "测试场景",
  state_entered: "阶段进入",
  observation: "视觉观测",
  action_selected: "策略选择",
  action_checked: "检查通过",
  action_requested: "动作请求",
  action_dispatched: "执行回执",
  action_confirmed: "结果确认",
  failure: "异常停止",
  cancelled: "操作员停止",
  run_finished: "运行结束",
  operator_command: "操作指令",
};
const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const time = (s) =>
  s ? new Date(s).toLocaleTimeString("zh-CN", { hour12: false }) : "尚无事件";
const vec = (v) => (v ? v.map((x) => Number(x).toFixed(3)).join(" / ") : "—");
let current = null,
  token = null,
  cursor = 0,
  logRows = [],
  sceneMode = "projection",
  busy = false,
  rawVersion = "",
  selectedView = "home",
  sceneSignature = "";
$("scenario").innerHTML = Object.entries(faults)
  .map(([key, label]) => `<option value="${key}">${label}</option>`)
  .join("");

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    signal: AbortSignal.timeout(6000),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => ($("toast").hidden = true), 3500);
}
async function command(name) {
  if (busy || !token) return;
  busy = true;
  try {
    const value = await api("/api/commands", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Control-Token": token },
      body: JSON.stringify({
        command: name,
        fault: $("scenario").value,
        step_interval_s: Number($("pace").value),
      }),
    });
    render(value);
    toast(
      {
        start: "完整流程已启动",
        pause: "已暂停，等待继续",
        resume: "继续前将重新观测",
        stop: "软件停止已锁存",
        reset: "软件停止锁存已复位",
      }[name],
    );
  } catch (e) {
    toast(e.message);
  } finally {
    busy = false;
  }
}
$("start").onclick = () => command("start");
$("pause").onclick = () =>
  command(current?.status === "paused" ? "resume" : "pause");
$("stop").onclick = () => command("stop");
$("reset").onclick = () => command("reset");
$("clear-log").onclick = () => {
  logRows = [];
  renderLogs();
};
$("log-filter").onchange = renderLogs;
document
  .querySelectorAll("[data-tab]")
  .forEach((button) => (button.onclick = () => selectTab(button.dataset.tab)));
document.querySelectorAll("[data-scene]").forEach(
  (button) =>
    (button.onclick = () => {
      sceneMode = button.dataset.scene;
      document
        .querySelectorAll("[data-scene]")
        .forEach((b) => b.classList.toggle("selected", b === button));
      renderScene(current);
    }),
);
function selectTab(name) {
  selectedView = name;
  document.body.classList.toggle("home-page", name === "home");
  $("operations-heading").hidden = name === "home";
  document
    .querySelectorAll(".view")
    .forEach((v) => (v.hidden = v.id !== `view-${name}`));
  document
    .querySelectorAll("[data-tab]")
    .forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $("breadcrumb").textContent = {
    home: "今日餐桌",
    overview: "任务运行",
    modules: "模块拓扑",
    validation: "测试报告",
    history: "运行记录",
  }[name];
  if (name === "validation") {
    loadValidation();
    loadVisionEvidence();
    loadRelevanceEvidence();
  }
  if (name === "history") loadHistory();
  if (current) render(current);
}
function render(data) {
  current = data;
  if (data.control_token) token = data.control_token;
  if (selectedView === "home") {
    if (!$("live-observation").hidden) renderScene(data);
    return;
  }
  if (selectedView === "history" || selectedView === "validation") return;
  const index = data.sequence.indexOf(data.state),
    finished = data.status === "complete";
  $("run-status").textContent = statuses[data.status] || data.status;
  $("run-status").className = `status ${data.status}`;
  $("clock").textContent = time(data.server_utc);
  $("metric-state").textContent = names[data.state] || "等待任务";
  $("stage-count").textContent =
    `${index < 0 ? "—" : String(index + 1).padStart(2, "0")} / 09`;
  $("metric-actions").textContent = data.confirmed_actions;
  $("metric-time").textContent =
    data.elapsed_wall_s == null ? "—" : `${data.elapsed_wall_s.toFixed(1)} s`;
  $("start").disabled = data.active || data.stop_latched;
  $("stop").disabled = false;
  $("pause").disabled = !data.active || data.stop_latched;
  $("pause").textContent = data.status === "paused" ? "继续" : "暂停";
  $("reset").disabled = data.active || !data.stop_latched;
  $("scenario").disabled = data.active;
  $("pace").disabled = data.active;
  $("steps").innerHTML = data.sequence
    .map((state, i) => {
      const done = finished || i < index,
        active = i === index && !finished;
      const error = active && ["failed", "cancelled"].includes(data.status);
      return `<li class="${done ? "done" : active ? "current" : ""} ${error ? "error" : ""}"><span class="step-index">${done ? "✓" : String(i + 1).padStart(2, "0")}</span><span>${esc(names[state])}</span></li>`;
    })
    .join("");
  $("alert").hidden = !data.stop_latched;
  if (data.stop_latched)
    $("alert").textContent =
      `软件停止已锁存。${data.summary?.error || "操作员已发出停止请求。"} ${data.active ? "等待任务结束后可复位。" : "检查记录并复位后，才能重新启动。"}`;
  const obs = data.observation,
    fb = obs?.feedback;
  $("held").textContent = fb ? objects[fb.held_object] || "无握持" : "未读取";
  $("spoon-loaded").textContent = fb
    ? fb.spoon_loaded
      ? "已装食"
      : "未装食"
    : "—";
  $("bowl-loaded").textContent = fb
    ? fb.bowl_loaded
      ? "已装食"
      : "未装食"
    : "—";
  $("skew").textContent = obs
    ? `${(obs.sensor_skew_ns / 1e6).toFixed(1)} ms`
    : "—";
  $("latch").textContent = data.stop_latched ? "已锁存" : "未触发";
  $("target-name").textContent = data.last_action
    ? names[data.last_action.name] || data.last_action.name
    : "等待动作";
  $("target-position").textContent = vec(data.last_action?.target_m);
  $("detections").innerHTML = obs
    ? Object.values(obs.detections)
        .map(
          (d) =>
            `<div class="detection-row"><span><i class="detection-dot ${d.object_id === "food_container" ? "container" : esc(d.object_id)}"></i>${esc(objects[d.object_id] || d.object_id)}</span><code>${esc(vec(d.position_m))}</code><span class="verified">已观测</span></div>`,
        )
        .join("")
    : '<p class="muted">等待有效定位数据</p>';
  $("module-list").innerHTML = data.modules
    .map(
      (m) =>
        `<article class="module-item"><h3>${esc(m.name)}<span class="${esc(m.status)}">${esc(moduleStatus[m.status])}</span></h3><p>${esc(m.description)}</p><code>${esc(m.detail)}</code><time>${esc(time(m.last_event_utc))}</time></article>`,
    )
    .join("");
  renderScene(data);
}
function renderScene(data) {
  const obs = data?.observation;
  const signature = JSON.stringify([sceneMode, obs, data?.frame_available, data?.last_action]);
  if (signature === sceneSignature) return;
  sceneSignature = signature;
  $("scene-empty").hidden =
    sceneMode === "raw" ? Boolean(data?.frame_available) : Boolean(obs);
  $("scene").hidden = sceneMode !== "projection";
  $("raw-frame").hidden = sceneMode !== "raw" || !data?.frame_available;
  $("scene-source").textContent = data?.frame_available
    ? "合成 RGB-D / 最近观测"
    : "等待第一帧";
  if (data?.frame_available && sceneMode === "raw") {
    const version = `${data.run_id}-${obs?.frame_id ?? obs?.timestamp_ns ?? cursor}`;
    if (version !== rawVersion) {
      $("raw-frame").src = `/api/frame?v=${encodeURIComponent(version)}`;
      rawVersion = version;
    }
  }
  if (!obs) return;
  const xy = (position) => [
    75 + ((position[0] - 0.2) / 0.6) * 490,
    340 - ((position[1] + 0.4) / 0.8) * 280,
  ];
  let svg = `<defs><pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M 28 0 L 0 0 0 28" fill="none" stroke="#dce5d4" stroke-width=".6"/></pattern></defs><rect width="640" height="400" fill="#eaf0e5"/><rect x="35" y="25" width="570" height="345" rx="10" fill="url(#grid)" stroke="#d3dfc9"/><text x="52" y="49" font-size="8" fill="#95a78b" letter-spacing="1">WORKSPACE / SYNTHETIC OBSERVATION</text><path d="M52 344v-27m0 27h27" stroke="#9cae92" stroke-width="1.5"/><text x="82" y="347" font-size="9" fill="#8da27f">X</text><text x="49" y="311" font-size="9" fill="#8da27f">Y</text>`;
  for (const id of ["mat", "food_container", "spoon", "bowl"]) {
    const d = obs.detections[id];
    if (!d) continue;
    const [x, y] = xy(d.position_m),
      c = colors[id],
      held = obs.feedback.held_object === id;
    svg += `<g transform="translate(${x},${y})">`;
    if (id === "mat")
      svg += `<rect x="-63" y="-42" width="126" height="84" rx="12" fill="#d4dfcc" stroke="#8ca581" stroke-width="1.5" stroke-dasharray="4 4"/>`;
    if (id === "food_container")
      svg += `<rect x="-32" y="-32" width="64" height="64" rx="8" fill="#d3dce0" stroke="${c}" stroke-width="2"/><rect x="-21" y="-23" width="42" height="46" rx="4" fill="#e6ebeb"/><circle cx="-8" cy="-3" r="5" fill="#a9b8bf"/><circle cx="7" cy="5" r="5" fill="#a9b8bf"/><circle cx="5" cy="-11" r="5" fill="#a9b8bf"/>`;
    if (id === "spoon")
      svg += `<g transform="rotate(-30)"><rect x="-4" y="-7" width="8" height="47" rx="4" fill="#e0c69f" stroke="${c}" stroke-width="1.5"/><ellipse cy="-16" rx="14" ry="20" fill="#f0e1c8" stroke="${c}" stroke-width="2"/>${obs.feedback.spoon_loaded ? `<ellipse cy="-17" rx="9" ry="14" fill="#bf975e"/>` : ""}</g>`;
    if (id === "bowl")
      svg += `<ellipse cy="5" rx="35" ry="27" fill="#d9b7a7" stroke="${c}" stroke-width="2"/><ellipse cy="-2" rx="35" ry="23" fill="#f4e5d9" stroke="${c}" stroke-width="2"/>${obs.feedback.bowl_loaded ? `<ellipse cy="-2" rx="26" ry="16" fill="#bd9670"/><circle cx="-10" cy="-6" r="3" fill="#e2c49e"/><circle cx="8" cy="4" r="3" fill="#e2c49e"/>` : ""}`;
    if (held)
      svg += `<circle r="45" fill="none" stroke="#bd4e2e" stroke-width="1.5" stroke-dasharray="4 4"/>`;
    const labelY = id === "mat" ? 58 : id === "spoon" ? 57 : 48;
    svg += `<text y="${labelY}" text-anchor="middle" font-size="11" fill="#526b4c">${esc(objects[id])}${held ? " · 握持中" : ""}</text></g>`;
  }
  if (data.last_action) {
    const [x, y] = xy(data.last_action.target_m);
    svg += `<path d="M${x - 9} ${y - 50}h18m-9-9v18" stroke="#bd4e2e" stroke-width="1.5"/><text x="${x + 15}" y="${y - 47}" font-size="8" fill="#bd4e2e">TARGET</text>`;
  }
  $("scene").innerHTML = svg;
}
function describe(row) {
  const p = row.payload || {};
  if (row.event === "state_entered") return names[p.state] || p.state;
  if (row.event === "observation")
    return `${Object.keys(p.detections).length} 个目标 · ${p.source}`;
  if (
    [
      "action_requested",
      "action_selected",
      "action_checked",
      "action_dispatched",
      "action_confirmed",
    ].includes(row.event)
  )
    return `${names[p.name] || p.name} · ${vec(p.target_m)}`;
  if (row.event === "failure") return `${p.module || ""}: ${p.error}`;
  if (row.event === "cancelled") return p.reason;
  if (row.event === "scenario") return faults[p.fault] || p.fault;
  if (row.event === "run_finished")
    return `${statuses[p.state]} / ${p.successful_commands ?? 0} 个动作已执行`;
  if (row.event === "operator_command")
    return (
      {
        start: "启动",
        pause: "暂停",
        resume: "继续",
        stop: "软件停止",
        reset: "复位",
      }[p.command] || p.command
    );
  if (row.event === "run_started")
    return `运行 ${row.run_id.slice(0, 12)} · 源码与配置哈希已保存`;
  return "simulation_only";
}
function renderLogs() {
  const important = $("log-filter").value === "important";
  const rows = logRows.filter(
    (r) =>
      !important ||
      [
        "action_requested",
        "action_confirmed",
        "failure",
        "cancelled",
        "run_finished",
        "operator_command",
      ].includes(r.event),
  );
  const element = $("events"),
    atBottom =
      element.scrollHeight - element.scrollTop - element.clientHeight < 40;
  element.innerHTML = rows
    .slice(-150)
    .map(
      (row) =>
        `<div class="event-row ${["failure", "cancelled"].includes(row.event) ? "error" : ""}"><time>${esc(time(row.utc))}</time><span class="event-type">${esc(events[row.event] || row.event)}</span><span class="event-message">${esc(describe(row))}</span></div>`,
    )
    .join("");
  $("event-count").textContent =
    `${logRows.length} 个事件 / 内存显示最近 300 条`;
  if (atBottom) element.scrollTop = element.scrollHeight;
}
async function loadValidation() {
  try {
    const r = await api("/api/validation");
    if (!r.checks) {
      $("validation-content").textContent =
        "尚无持久测试报告，请运行 scripts/validate.py。";
      return;
    }
    $("validation-content").innerHTML =
      `<div class="report-banner"><div><b>${r.all_passed ? "软件验证全部通过" : "存在未通过项"}</b><p>${esc(r.checks.tests.output.trim().split("\n").at(-1))} · ${r.scenarios.filter((s) => s.passed).length} / ${r.scenarios.length} 个场景符合预期</p></div><span>${esc(new Date(r.generated_at_utc).toLocaleString("zh-CN"))}</span></div><div class="checks">${Object.entries(
        r.checks,
      )
        .map(([name, c]) => `<span>${c.passed ? "✓" : "×"} ${esc(name)}</span>`)
        .join(
          "",
        )}</div>${r.scenarios.map((s) => `<div class="scenario-row"><strong>${esc(faults[s.scenario] || s.scenario)}</strong><span>${s.passed ? "符合预期" : "未通过"}</span><code>${esc(s.summary.error || "完整流程完成 / 实机验收未测试")}</code></div>`).join("")}<p class="scope-note">这些结果覆盖软件链路与故障处理，不证明抓取稳定性、食品计量或真实餐具识别效果。</p>`;
  } catch (e) {
    toast(e.message);
  }
}
async function loadVisionEvidence() {
  const panel = $("vision-evidence");
  try {
    const result = await api("/api/vision-sample");
    if (!result.detections) return;
    const labels = {
      bowl: "碗",
      spoon: "勺",
      knife: "刀",
      "dining table": "餐桌",
    };
    panel.innerHTML = `<div class="panel-head"><h2>公开照片视觉测试</h2><span>YOLO11n / GPU</span></div>
      <div class="vision-evidence-body"><img src="/api/vision-sample.jpg" alt="碗和勺的 YOLO11n 检测标注">
      <div><p>模型输出的类别与置信度</p>${result.detections.map((d) => `<div class="vision-score"><span>${esc(labels[d.label] || d.label)}</span><b>${d.confidence.toFixed(3)}</b></div>`).join("")}
      <p>这张样例中，勺的区域也出现了一个刀具候选框。静态照片测试与合成任务独立运行。</p>
      <a href="/api/vision-sample" target="_blank" rel="noreferrer">查看推理结果 JSON ↗</a></div></div>`;
    panel.hidden = false;
  } catch {
    panel.hidden = true;
  }
}
async function loadRelevanceEvidence() {
  const panel = $("relevance-evidence");
  const roles = {
    spoon: "勺",
    bowl: "餐碗",
    food_container: "食品容器",
    mat: "餐垫",
  };
  const labels = {
    relevant: "相关",
    irrelevant: "无关",
    insufficient: "证据不足",
  };
  const outcome = (row) =>
    row.outcome === "candidate" ? "可进入本地复核" : "暂停，缺少可见证据";
  try {
    const [fixtures, live] = await Promise.all([
      api("/api/relevance-test"),
      api("/api/bailian-vision"),
    ]);
    let body = `<div class="panel-head"><h2>百炼视觉与任务相关性</h2><span>VISION / RELEVANCE</span></div>`;
    if (fixtures.cases) {
      body += `<div class="panel-head"><strong>离线样例 ${esc(fixtures.passed)} / ${esc(fixtures.total)}</strong><span>预设证据 · 本地规则</span></div>`;
      body += fixtures.cases
        .map(
          (row) =>
            `<div class="scenario-row"><strong>${esc(names[row.phase])}</strong><span>${row.passed ? "符合预期" : "未通过"}</span><code>${esc(outcome(row))} · ${esc(row.name)}</code></div>`,
        )
        .join("");
    }
    const liveStatus =
      live.status === "completed" ? "云端已返回" : "等待云端实测";
    body += `<div class="panel-head"><strong>${liveStatus}</strong><span>${esc(live.model || "qwen3-vl-plus")}</span></div>`;
    if (live.status === "completed") {
      body += live.decisions
        .map(
          (row) =>
            `<div class="scenario-row"><strong>${esc(names[row.phase])}</strong><span>${esc(roles[row.target_role])}</span><code>${esc(outcome(row))}<br>${row.objects.map((item) => `${esc(roles[item.role])} ${esc(labels[item.relevance])}`).join(" · ")}</code></div>`,
        )
        .join("");
      body += `<div class="panel-head"><a href="/api/bailian-vision" target="_blank" rel="noreferrer">查看视觉证据与决策 JSON ↗</a><span>任务级复核</span></div>`;
    } else if (live.error) {
      body += `<div class="panel-head"><span>${esc(live.error)}</span></div>`;
    }
    panel.innerHTML = body;
    panel.hidden = false;
  } catch {
    panel.hidden = true;
  }
}
async function loadHistory() {
  try {
    const runs = await api("/api/runs");
    $("history-list").innerHTML = runs.length
      ? runs
          .map(
            ({ summary: s, artifact_directory: path }) =>
              `<article class="history-row"><div><b>${esc(s.run_id?.slice(0, 16) || "运行记录")}</b><small>合成场景 / simulation_only</small></div><span class="status ${esc(s.state)}">${esc(statuses[s.state])}</span><span>${s.confirmed_actions == null ? `${s.successful_commands ?? 0} 已执行` : `${s.confirmed_actions} / 6 已确认`}</span><code>${esc(s.error || path)}</code></article>`,
          )
          .join("")
      : `<p class="empty-history">暂无中枢运行记录。启动一个完整流程后，记录会自动保存在本机。</p>`;
  } catch (e) {
    toast(e.message);
  }
}
async function poll() {
  try {
    const data = await api("/api/status");
    render(data);
    if (selectedView === "overview" || selectedView === "modules") {
      const stream = await api(`/api/events?after=${cursor}`);
      if (stream.truncated) logRows = [];
      if (stream.events.length) {
        cursor = stream.cursor;
        logRows = [...logRows, ...stream.events].slice(-300);
        renderLogs();
      }
    }
    $("connection").textContent = "服务在线";
    $("connection").className = "connection online";
  } catch (e) {
    $("connection").textContent = "服务断开";
    $("connection").className = "connection offline";
    ["start", "pause", "stop", "reset"].forEach(
      (id) => ($(id).disabled = true),
    );
  }
  const interval = document.hidden ? 5000 : selectedView === "overview" || (selectedView === "home" && !$("live-observation").hidden) ? 750 : 2000;
  setTimeout(poll, interval);
}
poll();

// The recipe pantry is an explicit frontend demo, independent of robot observations.
const recipeCatalog = [
  {id:"mushroom-tofu",name:"蘑菇豆腐蒸菜",ingredients:["蘑菇","豆腐","小葱"],groups:["mushroom","tofu"],kind:"mushroom",time:18,method:"清蒸",tag:"鲜味叠加 · 清爽不寡淡",steps:["豆腐切块、蘑菇切片，放入浅盘。","加少量清水，水开后入锅蒸约 12–15 分钟。","出锅撒上葱花，再按所选模式调味。"]},
  {id:"lemon-chicken",name:"柠檬香菜鸡胸肉蒸菜",ingredients:["鸡胸肉","柠檬","香菜"],groups:["chicken"],kind:"chicken",time:25,method:"清蒸",tag:"柠檬清香 · 一口鲜嫩",steps:["鸡胸肉切成厚度均匀的小块，加入柠檬汁拌匀。","放入耐热盘，水开后蒸约 15–20 分钟，按厚度调整，确保肉中心完全熟透。","出锅加入香菜，淋上少量柠檬汁调味。"]},
  {id:"vegetable-salad",name:"蔬菜沙拉",ingredients:["生菜","小番茄","黄瓜"],groups:["vegetable"],kind:"salad",time:10,method:"凉拌",tag:"缤纷蔬菜 · 新鲜一整碗",steps:["生菜、黄瓜和小番茄洗净，沥干水分。","生菜撕小片，黄瓜切片，小番茄对半切。","加入所选模式的调味汁，食用前轻轻拌匀。"]},
  {id:"mushroom-vegetables",name:"蘑菇时蔬暖沙拉",ingredients:["蘑菇","生菜","小番茄"],groups:["mushroom","vegetable"],kind:"salad",time:15,method:"焯拌",tag:"温热菌菇 · 搭配脆嫩时蔬",steps:["蘑菇切片，入沸水煮熟后沥干。","生菜和番茄洗净切好，与温热蘑菇混合。","按模式加入调味汁，轻轻拌匀。"]},
  {id:"mushroom-chicken",name:"香菇鸡胸肉蒸菜",ingredients:["蘑菇","鸡胸肉","姜丝"],groups:["mushroom","chicken"],kind:"chicken",time:25,method:"清蒸",tag:"菌菇提鲜 · 一盘刚刚好",steps:["鸡胸肉切成均匀小块，蘑菇切片，放入姜丝。","铺在浅盘中，水开后蒸约 15–20 分钟，确保鸡肉中心完全熟透。","出锅轻拌，根据所选模式调味。"]},
  {id:"tofu-vegetables",name:"时蔬豆腐蒸菜",ingredients:["豆腐","西兰花","胡萝卜"],groups:["tofu","vegetable"],kind:"mushroom",time:20,method:"清蒸",tag:"软嫩豆腐 · 一盘蔬食色彩",steps:["豆腐切块，西兰花掰成小朵，胡萝卜切薄片并洗净。","铺入浅盘，水开后蒸约 12–15 分钟至食材熟透。","沥去多余蒸汽水，根据模式加入调味汁。"]},
  {id:"tofu-salad",name:"黄瓜豆腐沙拉",ingredients:["豆腐","黄瓜","小番茄"],groups:["tofu","vegetable"],kind:"salad",time:12,method:"凉拌",tag:"柔软与爽脆 · 简单又合拍",steps:["豆腐切块，放入沸水中煮熟后沥干放凉。","黄瓜、番茄洗净切好，与豆腐混合。","加入所选模式的调味汁，轻拌后食用。"]},
  {id:"chicken-salad",name:"鸡胸肉蔬菜沙拉",ingredients:["鸡胸肉","生菜","小番茄"],groups:["chicken","vegetable"],kind:"salad",time:25,method:"蒸拌",tag:"一碗丰盛 · 荤素好搭配",steps:["鸡胸肉切块后蒸约 15–20 分钟，确保中心完全熟透。","洗净生菜和小番茄，沥干切好。","熟鸡肉稍放凉后撕小块，与蔬菜和调味汁拌匀。"]},  {"id": "tomato-tofu", "name": "番茄豆腐汤", "ingredients": ["番茄", "豆腐", "小葱"], "groups": ["tofu", "vegetable"], "kind": "mushroom", "time": 20, "method": "炖煮", "tag": "番茄酸甜 · 豆腐软嫩", "steps": ["番茄洗净切块，豆腐切小块，小葱切碎。", "锅中加少量油，炒软番茄后倒入清水煮开。", "加入豆腐，小火煮约 8–10 分钟，按所选模式调味，撒葱花。"]},
  {"id": "broccoli-mushroom", "name": "蒜香西兰花蘑菇", "ingredients": ["西兰花", "蘑菇", "大蒜"], "groups": ["mushroom", "vegetable"], "kind": "salad", "time": 15, "method": "快炒", "tag": "蒜香提鲜 · 脆嫩有层次", "steps": ["西兰花洗净掰小朵，蘑菇切片，大蒜切末。", "西兰花先焯至断生，沥干；锅中加少量油，炒香蒜末。", "放入蘑菇炒熟，再加入西兰花翻炒，按所选模式调味。"]},
  {"id": "cucumber-chicken", "name": "黄瓜拌鸡丝", "ingredients": ["鸡胸肉", "黄瓜", "白芝麻"], "groups": ["chicken", "vegetable"], "kind": "chicken", "time": 25, "method": "煮拌", "tag": "清爽鸡丝 · 一口脆嫩", "steps": ["鸡胸肉切成厚度均匀的块，煮至中心完全熟透。", "鸡肉稍放凉后撕成丝；黄瓜洗净切细丝。", "混合鸡丝与黄瓜，撒白芝麻，根据模式用柠檬汁或少量香醋调味。"]},
  {"id": "spinach-egg", "name": "菠菜炒鸡蛋", "ingredients": ["菠菜", "鸡蛋", "大蒜"], "groups": ["vegetable"], "kind": "salad", "time": 12, "method": "快炒", "tag": "鲜嫩蛋香 · 家常好滋味", "steps": ["菠菜洗净后焯水、沥干切段；鸡蛋打散，大蒜切末。", "不粘锅中放少量油，将蛋液炒至完全凝固后盛出。", "炒香蒜末，放入菠菜和鸡蛋翻匀，按所选模式调味。"]},

];
const kitchenModes = {
  fat:{label:"减脂",description:"少一点油，多一点轻盈。",note:"使用柠檬汁、黑胡椒提味；避免额外淋油和浓稠沙拉酱。",order:["mushroom-tofu","lemon-chicken","vegetable-salad","cucumber-chicken","tofu-salad","mushroom-vegetables","chicken-salad","tofu-vegetables","tomato-tofu","broccoli-mushroom","mushroom-chicken","spinach-egg"]},
  salt:{label:"减盐",description:"借一点天然鲜味，让调味轻一些。",note:"不额外加盐或酱油，用柠檬、姜丝、香菜与菌菇提鲜。",order:["tofu-vegetables","broccoli-mushroom","mushroom-vegetables","lemon-chicken","mushroom-tofu","tofu-salad","tomato-tofu","vegetable-salad","cucumber-chicken","mushroom-chicken","spinach-egg","chicken-salad"]},
  daily:{label:"日常",description:"熟悉的食材，换一点新滋味。",note:"少量盐调味，可按口味加入香醋和少量橄榄油。",order:["tomato-tofu","spinach-egg","chicken-salad","mushroom-chicken","tofu-vegetables","broccoli-mushroom","tofu-salad","cucumber-chicken","mushroom-vegetables","lemon-chicken","mushroom-tofu","vegetable-salad"]},
};
let kitchenMode = "fat", kitchenIngredient = null, kitchenBatch = 0;
const ingredientLabels = {mushroom:"蘑菇",tofu:"豆腐",chicken:"鸡胸肉",vegetable:"蔬菜"};
const foodPhotos = {"mushroom-tofu": {"src": "/images/mushroom.e5b1442f88a9.webp", "source": "https://www.dienmayxanh.com/vao-bep/tim-kiem/dau-hu-hap-nam", "credit": "DienmayXanh · 蘑菇豆腐蒸菜实拍"}, "lemon-chicken": {"src": "/images/chicken.f99b233bc22a.webp", "source": "https://news.infoseek.co.jp/article/iemone_338529/", "credit": "iemone / Infoseek · 柠檬香菜蒸鸡胸实拍"}, "vegetable-salad": {"src": "/images/salad.61077b37f62c.webp", "source": "https://unsplash.com/photos/LgOsoMm38Cc", "credit": "Rashpal Singh / Unsplash · 蔬菜沙拉实拍"}, "mushroom-vegetables": {"src": "/images/mushroom-vegetables.b0447eff220a.webp", "source": "https://recipe.rakuten.co.jp/recipe/1700007979/", "credit": "Rakuten Recipe · 温热菌菇沙拉"}, "mushroom-chicken": {"src": "/images/mushroom-chicken.c2186a911eda.webp", "source": "https://cookpad.com/tw/食譜/16907892", "credit": "Cookpad / 安妮食堂 · 香菇蒸鸡"}, "tofu-vegetables": {"src": "/images/tofu-vegetables.27a7291c362f.webp", "source": "https://m.10000recipe.com/recipe/7045296", "credit": "10000recipe · 时蔬豆腐蒸菜"}, "tofu-salad": {"src": "/images/tofu-salad.13a137c7a88d.webp", "source": "https://recipes.timesofindia.com/recipes/tofu-cucumber-salad/rs60702182.cms", "credit": "Times of India · 黄瓜豆腐沙拉"}, "chicken-salad": {"src": "/images/chicken-salad.ece0ad95ebd3.webp", "source": "https://kitchen.sayidaty.net/node/12084", "credit": "Sayidaty Kitchen · 鸡胸肉蔬菜沙拉"}, "tomato-tofu": {"src": "/images/tomato-tofu.77943a82b7e0.webp", "source": "https://www.madewithlau.com/recipes/tomato-tofu-soup", "credit": "Made With Lau · 番茄豆腐汤"}, "broccoli-mushroom": {"src": "/images/broccoli-mushroom.1ac23d5e1fa3.webp", "source": "https://www.foodtalkdaily.com/recipes/cuisine/asian/broccoli-and-mushroom-stir-fry-quick-and-healthy-44684593", "credit": "Foodtalk · 西兰花炒蘑菇"}, "cucumber-chicken": {"src": "/images/cucumber-chicken.992d7d2ea3a9.webp", "source": "https://recipe.rakuten.co.jp/recipe/1470012399/", "credit": "Rakuten Recipe · 黄瓜拌鸡丝"}, "spinach-egg": {"src": "/images/spinach-egg.0f3ae81ac8f5.webp", "source": "https://www.foodfaithfitness.com/spinach-scrambled-eggs/", "credit": "Food Faith Fitness · 菠菜炒鸡蛋"}};
function plateArt(kind, name = "菜品实拍参考") {
  return `<img class="food-photo" src="${foodPhotos[kind].src}" alt="${esc(name)}实拍参考" decoding="async" width="320" height="240" />`;
}

function kitchenPool() {
  const mode = kitchenModes[kitchenMode];
  const rank = id => {const i=mode.order.indexOf(id);return i < 0 ? mode.order.length : i;};
  return recipeCatalog.filter(r => !kitchenIngredient || r.groups.includes(kitchenIngredient)).sort((a,b) => rank(a.id)-rank(b.id));
}
function kitchenRecipes() {
  const pool = kitchenPool(), start = (kitchenBatch * 3) % pool.length;
  return Array.from({length:Math.min(3,pool.length)},(_,i)=>pool[(start+i)%pool.length]);
}

function renderKitchen() {
  const mode = kitchenModes[kitchenMode], recipes = kitchenRecipes();
  $("more-recipes").disabled = kitchenPool().length <= 3;
  $("more-recipes").title = `从 ${kitchenPool().length} 道菜中换一组推荐`;
  $("mode-description").textContent = kitchenIngredient ? `${ingredientLabels[kitchenIngredient]}的三种好吃法 · ${mode.label}模式` : mode.description;
  document.querySelectorAll("[data-recipe-mode]").forEach(b => {
    const selected = b.dataset.recipeMode === kitchenMode;
    b.setAttribute("aria-selected", String(selected)); b.tabIndex = selected ? 0 : -1;
  });
  document.querySelectorAll("[data-ingredient]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.ingredient === kitchenIngredient)));
  $("all-ingredients").classList.toggle("active", !kitchenIngredient);
  $("recipe-results").setAttribute("aria-labelledby", `mode-${kitchenMode}`);
  $("recipe-list").innerHTML = recipes.map((r,i) => `<button class="recipe-card" data-open-recipe="${r.id}" aria-label="查看${esc(r.name)}做法"><span class="dish-art ${r.kind}">${plateArt(r.id,r.name)}<small>0${i+1}</small></span><span class="dish-copy"><span class="recipe-label">${mode.label}推荐 <i>·</i> ${r.method}</span><strong>${r.name}</strong><span class="dish-tagline">${r.tag}</span><span class="recipe-ingredients">${r.ingredients.join('<i> / </i>')}</span><span class="recipe-meta"><span>◷ ${r.time} 分钟</span><span>简单易做</span><b>查看做法 ↗</b></span></span></button>`).join("");
  $("recipe-announcement").textContent = `已切换为${mode.label}模式，推荐${recipes.map(r=>r.name).join("、")}`;
  document.querySelectorAll("[data-open-recipe]").forEach(b => b.onclick = () => openRecipe(b.dataset.openRecipe));
}
function openRecipe(id) {
  const r = recipeCatalog.find(r=>r.id===id), mode = kitchenModes[kitchenMode];
  $("recipe-detail").innerHTML = `<div class="detail-art ${r.kind}">${plateArt(r.id,r.name)}</div><p class="kicker">${mode.label}模式 / ${r.method} / ${r.time} 分钟</p><h2 id="recipe-dialog-title">${r.name}</h2><p class="detail-ingredients">食材：${r.ingredients.join("、")}</p><div class="mode-cooking-note">${mode.note}</div><ol>${r.steps.map(s=>`<li>${s}</li>`).join("")}</ol><p class="photo-credit">同类菜品实拍参考 · <a href="${foodPhotos[r.id].source}" target="_blank" rel="noreferrer">${esc(foodPhotos[r.id].credit)} ↗</a></p><p class="recipe-detail-note">演示菜谱 · 时间为参考，请以食材实际熟透状态为准。</p>`;
  $("recipe-dialog").showModal();
}
document.querySelectorAll("[data-recipe-mode]").forEach(b => {
  b.onclick = () => {kitchenMode=b.dataset.recipeMode; kitchenBatch=0; renderKitchen();};
  b.onkeydown = e => {
    const buttons = [...document.querySelectorAll("[data-recipe-mode]")];
    let i = buttons.indexOf(b);
    if(e.key === "ArrowRight") i=(i+1)%buttons.length;
    else if(e.key === "ArrowLeft") i=(i+buttons.length-1)%buttons.length;
    else if(e.key === "Home") i=0;
    else if(e.key === "End") i=buttons.length-1;
    else return;
    e.preventDefault(); buttons[i].click(); buttons[i].focus();
  };
});
document.querySelectorAll("[data-ingredient]").forEach(b => b.onclick = () => {kitchenIngredient=b.dataset.ingredient; kitchenBatch=0; renderKitchen();});
$("all-ingredients").onclick = () => {kitchenIngredient=null; kitchenBatch=0; renderKitchen();};
$("more-recipes").onclick = () => {kitchenBatch++; renderKitchen();};
$("close-recipe").onclick = () => $("recipe-dialog").close();
$("recipe-dialog").onclick = e => {if(e.target === $("recipe-dialog")) {const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}};
$("live-observation").append(document.querySelector(".scene-panel"));
document.querySelectorAll("[data-observation]").forEach(b => b.onclick = () => {
  const live=b.dataset.observation === "live";
  $("ingredient-projection").hidden=live; $("live-observation").hidden=!live;
  document.querySelector(".demo-label").textContent=live ? "模拟器观测" : "演示场景";
  if (live) renderScene(current);
  document.querySelectorAll("[data-observation]").forEach(other=>{other.classList.toggle("selected",other===b);other.setAttribute("aria-pressed",String(other===b));});
});
renderKitchen();


const ingredientPhotos = {"raw-mushroom": {"src": "/images/raw-mushroom.25881edee45e.webp", "source": "https://unsplash.com/photos/RpyEAQS1I5g"}, "raw-tofu": {"src": "/images/raw-tofu.ad4e6520250f.webp", "source": "https://food.daily.co.kr/read3132202425"}, "raw-chicken": {"src": "/images/raw-chicken.b79c65be5b0c.webp", "source": "https://unsplash.com/photos/HsEnMY-ZYO8"}, "raw-vegetable": {"src": "/images/raw-vegetable.77eb9a323e12.webp", "source": "https://unsplash.com/photos/lzyaeNQ9wU4"}};
const portraitProjection = document.createElement("div");
portraitProjection.className = "portrait-projection";
portraitProjection.setAttribute("aria-label", "原始食材状态投影，演示数据");
portraitProjection.innerHTML = [
  ["raw-mushroom", "01", "蘑菇"],
  ["raw-tofu", "02", "豆腐"],
  ["raw-chicken", "03", "鸡胸肉"],
  ["raw-vegetable", "04", "蔬菜"],
].map(([kind, number, label]) => `<div class="projection-photo"><img class="food-photo" src="${ingredientPhotos[kind].src}" alt="${label}原始食材实拍" width="160" height="120" decoding="async" /><span><i>${number}</i>${label}</span><b>⌖</b></div>`).join("");
$("ingredient-projection").prepend(portraitProjection);
