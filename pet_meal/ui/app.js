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
  rawVersion = "";
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
  document
    .querySelectorAll(".view")
    .forEach((v) => (v.hidden = v.id !== `view-${name}`));
  document
    .querySelectorAll("[data-tab]")
    .forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $("breadcrumb").textContent = {
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
}
function render(data) {
  current = data;
  if (data.control_token) token = data.control_token;
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
  $("scene-empty").hidden =
    sceneMode === "raw" ? Boolean(data?.frame_available) : Boolean(obs);
  $("scene").hidden = sceneMode !== "projection";
  $("raw-frame").hidden = sceneMode !== "raw" || !data?.frame_available;
  $("scene-source").textContent = data?.frame_available
    ? "合成 RGB-D / 最近观测"
    : "等待第一帧";
  if (data?.frame_available && sceneMode === "raw") {
    const version = `${data.run_id}-${cursor}`;
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
    const stream = await api(`/api/events?after=${cursor}`);
    if (stream.truncated) logRows = [];
    if (stream.events.length) {
      cursor = stream.cursor;
      logRows = [...logRows, ...stream.events].slice(-300);
      renderLogs();
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
  setTimeout(poll, 450);
}
poll();
