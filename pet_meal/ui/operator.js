const panel = document.querySelector("#operator");
const home = panel.parentElement;
const $ = (id) => panel.querySelector("#" + id);
const form = $("plan-form");
const fields = form.elements;
const transcript = $("transcript");
const modes = { g1_plan_only: "G1 规划预览", g1_simulation: "G1 水瓶模拟", simulation_only: "模拟任务", g1_hardware: "G1 实机分段" };
const states = {
  idle: "待机", running: "运行中", paused: "已暂停", stopping: "正在停止",
  stopped: "已停止", cancelled: "已取消", held: "待复核", preview: "预览已生成",
  complete: "已完成", failed: "失败", unconfirmed: "实机结果待核对",
};
const reasons = {
  target_not_detected: "本次采集未找到目标",
  multiple_instances: "找到多个候选，请选实例 ID",
  matched_pixels_missing: "缺少跨相机匹配点",
  geometry_rejected: "三维定位未通过",
  gripper_unknown: "夹爪持物状态未知",
  gripper_reported_occupied: "夹爪已有物体",
  scene_geometry_unavailable: "场景几何或碰撞模型未就绪",
  sdk_planning_not_run: "已生成候选姿态，SDK 轨迹尚未校验",
  simulated_task_verified: "模拟阶段均通过",
  simulated_stage_unconfirmed: "模拟阶段验证失败，流程已暂停",
  operator_stop: "操作者已停止软件任务",
};
const stages = {
  pregrasp: "预抓取", insert: "接近", close: "闭合", test_lift: "试提",
  raise_for_shake: "抬升", shake: "摇动", lower: "下降", release: "放开", retreat: "退出",
};

let snapshot;
let token;
let pending = false;
let revision = 0;
let recognition;
let floating;
let confirmedTranscript = "";

function speechType() {
  const view = panel.ownerDocument.defaultView;
  return view?.SpeechRecognition || view?.webkitSpeechRecognition;
}

function updateControls() {
  const ready = Boolean(token && !pending);
  $("interpret").disabled = !ready || Boolean(recognition) || !transcript.value.trim();
  $("mic").disabled = !speechType() || pending;
  const text = transcript.value.trim();
  $("plan").disabled = !ready || snapshot?.active || snapshot?.stop_latched ||
    Boolean(recognition || (text && text !== confirmedTranscript));
  $("pause").disabled = !ready || !snapshot?.active || snapshot?.stop_latched || snapshot.mode === "g1_hardware";
  $("stop").disabled = !ready || !snapshot?.active || snapshot?.stop_latched;
  $("reset").disabled = !ready || snapshot?.active || !snapshot?.stop_latched;
  fields.repeat_count.disabled = fields.skill.value !== "shake_then_place";
}

function render(data) {
  snapshot = data;
  $("connection").textContent = "本机中枢在线";
  $("mode").textContent = modes[data.mode] || "模式未知";
  $("pause").textContent = data.status === "paused" ? "继续" : "暂停";
  const result = data.summary;
  const parts = [states[data.status] || data.status];
  if (data.stop_latched) parts.push("停止已锁存");
  if (result?.reason) parts.push(reasons[result.reason] || result.reason);
  if (result?.motion_commands_sent !== undefined) parts.push(`运动派发 ${result.motion_commands_sent} 条`);
  if (result?.confirmed_stages !== undefined) parts.push(`${result.confirmed_stages} 段实机验收`);
  const status = parts.join(" · ");
  if ($("run-status").textContent !== status) $("run-status").textContent = status;
  const candidates = result?.candidates || [];
  const stageResults = result?.simulated_stage_results;
  $("stage-summary").textContent = stageResults
    ? `${result.simulated_verified_stages} 个模拟阶段通过 · 摇动 ${result.shake_cycles} 次 · ${stageResults.map(({ stage_id }) => stages[stage_id] || stage_id).join(" → ")}`
    : candidates.length
    ? `${candidates.length} 组候选 · ${candidates[0].stages.map(({ stage_id }) => stages[stage_id] || stage_id).join(" → ")}`
    : result?.detail || result?.error || "";
  $("receipt-link").hidden = !result;
  updateControls();
}

async function api(path, body) {
  const options = { signal: AbortSignal.timeout(6000) };
  if (body !== undefined) {
    options.method = "POST";
    options.headers = { "Content-Type": "application/json", "X-Control-Token": token };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function poll() {
  const current = revision;
  try {
    if (pending) return;
    const data = await api("/api/status");
    if (current !== revision) return;
    token = data.control_token;
    render(data);
  } catch {
    if (current !== revision) return;
    token = null;
    $("connection").textContent = "本机中枢离线";
    $("run-status").textContent = "等待重连 · 上次回执仅供参考";
    updateControls();
  } finally {
    setTimeout(poll, 900);
  }
}

async function send(path, body, onSuccess, errorTarget = "request-status") {
  if (pending || !token) return;
  pending = true;
  revision += 1;
  $("request-status").textContent = "";
  updateControls();
  try {
    onSuccess(await api(path, body));
  } catch (error) {
    $(errorTarget).textContent = error.message;
  } finally {
    pending = false;
    updateControls();
  }
}

$("voice-form").onsubmit = (event) => {
  event.preventDefault();
  if (recognition) return;
  const text = transcript.value.trim();
  send("/api/tabletop/interpret", { text }, ({ task }) => {
    if (transcript.value.trim() !== text) {
      $("mic-status").textContent = "转写已编辑，请重新解析。";
      return;
    }
    fields.skill.value = task.skill;
    fields.selector.value = task.object_id ? "object_id" : "category";
    fields.target.value = task.object_id || task.category;
    fields.repeat_count.value = task.repeat_count ?? 1;
    // An earlier visual choice may name a different target than this voice request.
    fields.vision_choice.value = "";
    confirmedTranscript = text;
    $("mic-status").textContent = "任务已填入规划表单。核对动作和目标后，再生成规划。";
  }, "mic-status");
};

form.onsubmit = (event) => {
  event.preventDefault();
  if ($("plan").disabled) return;
  const input = new FormData(form);
  const task = { skill: input.get("skill"), [input.get("selector")]: input.get("target").trim() };
  const count = input.get("repeat_count");
  if (count) task.repeat_count = Number(count);
  const payload = {
    task,
    capture: input.get("capture").trim(),
    observation: input.get("observation").trim(),
  };
  const choice = input.get("vision_choice").trim();
  if (choice) payload.vision_choice = choice;
  send("/api/tabletop/plan", payload, render);
};

for (const name of ["pause", "stop", "reset"]) {
  $(name).onclick = () => {
    const command = name === "pause" && snapshot?.status === "paused" ? "resume" : name;
    send("/api/commands", { command }, render);
  };
}

function finishMic() {
  recognition = null;
  $("mic").textContent = "开始听写";
  updateControls();
}

function abortMic() {
  const previous = recognition;
  if (!previous) return;
  finishMic();
  previous.abort();
  $("mic-status").textContent = "听写已结束。";
}

$("mic").onclick = () => {
  if (recognition) {
    recognition.stop();
    return;
  }
  const SpeechRecognition = speechType();
  if (!SpeechRecognition) return;
  const session = new SpeechRecognition();
  session.lang = "zh-CN";
  session.continuous = false;
  session.interimResults = false;
  session.onresult = (event) => {
    if (recognition !== session) return;
    transcript.value = event.results[event.resultIndex][0].transcript.trim();
    $("mic-status").textContent = "听写完成，请核对转写文本。";
  };
  session.onerror = (event) => {
    if (recognition !== session) return;
    const errors = {
      "not-allowed": "麦克风权限未获准",
      "audio-capture": "找不到可用的麦克风",
      network: "浏览器听写服务无法连接",
      "no-speech": "未听到语音",
    };
    $("mic-status").textContent = (errors[event.error] || "听写失败") + "，可直接输入文本。";
  };
  session.onend = () => {
    if (recognition !== session) return;
    if ($("mic-status").textContent === "正在听写…") $("mic-status").textContent = "听写已结束。";
    finishMic();
  };
  try {
    recognition = session;
    session.start();
    $("mic").textContent = "结束听写";
    $("mic-status").textContent = "正在听写…";
    updateControls();
  } catch {
    finishMic();
    $("mic-status").textContent = "无法启动听写，可直接输入文本。";
  }
};

$("float").onclick = async () => {
  if (floating) {
    floating.close();
    return;
  }
  $("float").disabled = true;
  try {
    floating = await window.documentPictureInPicture.requestWindow({ width: 470, height: 690 });
    abortMic();
    window.dispatchEvent(new Event("operator-window-change"));
    floating.document.documentElement.lang = "zh-CN";
    floating.document.title = document.title;
    const style = floating.document.createElement("link");
    style.rel = "stylesheet";
    style.href = new URL("/operator.css", location.origin).href;
    floating.document.head.append(style);
    floating.document.body.append(panel);
    $("float").textContent = "返回主页面";
    $("window-status").textContent = speechType() ? "" : "浮窗中可输入文本。听写请返回主页面。";
    floating.addEventListener("pagehide", () => {
      abortMic();
      window.dispatchEvent(new Event("operator-window-change"));
      home.append(panel);
      floating = null;
      $("float").textContent = "置顶浮窗 ↗";
      $("window-status").textContent = "";
      updateControls();
    }, { once: true });
  } catch {
    $("window-status").textContent = "无法打开置顶窗口，可以继续在此页操作。";
  } finally {
    $("float").disabled = false;
    updateControls();
  }
};

if (!window.documentPictureInPicture) {
  $("float").disabled = true;
  $("window-status").textContent = "当前浏览器未提供置顶浮窗，可以在此页操作。";
}
if (!speechType()) $("mic-status").textContent = "当前浏览器未提供听写，可以直接输入文本。";
const params = new URLSearchParams(location.search);
for (const name of ["capture", "observation", "vision_choice"]) {
  fields[name].value = params.get(name) || "";
}
fields.skill.onchange = updateControls;
transcript.oninput = () => {
  $("mic-status").textContent = "文本已编辑。点击「解析到任务」更新表单。";
  updateControls();
};
window.addEventListener("pagehide", abortMic);
updateControls();
poll();
