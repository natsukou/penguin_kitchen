{
  const file = $("show-file");
  const offset = $("show-offset");
  const status = $("show-status");
  let manifest;
  let clip;
  let active;
  let loading = false;
  let preset = false;

  function updateShowControls() {
    $("show-play").disabled = !clip || Boolean(active);
    $("show-export").disabled = !clip || Boolean(active);
    $("show-stop").disabled = !active;
    $("show-load").disabled = loading || Boolean(active);
    file.disabled = loading || Boolean(active);
    offset.disabled = loading || Boolean(active);
  }

  async function prepare() {
    if (loading || !manifest) return;
    clip = null;
    let input = file.files[0];
    if (!input && !preset) {
      status.textContent = "等待音源";
      updateShowControls();
      return;
    }
    loading = true;
    status.textContent = "正在准备音频";
    updateShowControls();
    try {
      if (preset) {
        const response = await fetch("/api/show/audio", { signal: AbortSignal.timeout(6000) });
        if (!response.ok) throw new Error("演出音源暂不可用。");
        input = new File([await response.blob()], "好运来 · 祖海（央视）.wav");
      }
      const start = offset.valueAsNumber;
      const rate = 16000;
      const duration = manifest.duration_s;
      if (input.size > 24 * 1024 * 1024) throw new Error("请选择不超过 24 MiB 的音频。");
      if (!Number.isFinite(start) || start < 0) throw new Error("起点须为非负秒数。");
      const context = new OfflineAudioContext(1, duration * rate, rate);
      const audio = await context.decodeAudioData(await input.arrayBuffer());
      if (audio.duration - start < duration) throw new Error("从选定起点开始需要至少 10 秒音频。");
      const source = context.createBufferSource();
      source.buffer = audio;
      source.connect(context.destination);
      source.start(0, start, duration);
      clip = await context.startRendering();
      renderTime(0);
      status.textContent = `${input.name} · ${start.toFixed(1)}–${(start + duration).toFixed(1)} 秒已就绪`;
    } catch (error) {
      status.textContent = error.message || "音频解码失败。";
    } finally {
      loading = false;
      updateShowControls();
    }
  }

  function renderTime(seconds) {
    const time = Math.min(seconds, manifest.duration_s);
    const frames = manifest.head_yaw_rad;
    const endIndex = frames.findIndex(([at], i) => i > 0 && time <= at);
    const [start, initial] = frames[endIndex - 1];
    const [end, target] = frames[endIndex];
    const phase = (time - start) / (end - start);
    const yaw = initial + (target - initial) * (1 - Math.cos(Math.PI * phase)) / 2;
    $("show-head").setAttribute("transform", `translate(${yaw * 130} 0)`);
    $("show-progress").value = time;
    $("show-clock").value = `${time.toFixed(1)} / ${manifest.duration_s} 秒`;
  }

  function endShow(completed) {
    const run = active;
    if (!run) return;
    active = null;
    if (run.started && !completed) run.source.stop();
    run.context.close().catch(() => {});
    renderTime(completed ? manifest.duration_s : 0);
    status.textContent = completed ? "10 秒预演完成 · 未向 G1 派发" : "预演已停止";
    updateShowControls();
  }

  $("show-play").onclick = async () => {
    if (!clip || active) return;
    const context = new AudioContext();
    const source = context.createBufferSource();
    const run = { context, source, started: false };
    active = run;
    updateShowControls();
    try {
      await context.resume();
      if (active !== run) return;
      source.buffer = clip;
      source.connect(context.destination);
      source.onended = () => {
        if (active === run) endShow(true);
      };
      const began = context.currentTime;
      source.start(began);
      run.started = true;
      status.textContent = "正在预演 · 电脑播放";
      function tick() {
        if (active !== run) return;
        renderTime(context.currentTime - began);
        panel.ownerDocument.defaultView.requestAnimationFrame(tick);
      }
      tick();
    } catch {
      if (active === run) {
        endShow(false);
        status.textContent = "无法启动浏览器音频。";
      }
    }
  };
  $("show-stop").onclick = () => endShow(false);
  window.addEventListener("pagehide", () => endShow(false));
  window.addEventListener("operator-window-change", () => endShow(false));
  file.onchange = () => {
    preset = false;
    prepare();
  };
  offset.onchange = prepare;
  $("show-load").onclick = () => {
    preset = true;
    file.value = "";
    offset.value = "0";
    prepare();
  };

  $("show-export").onclick = () => {
    if (!clip || active) return;
    const samples = clip.getChannelData(0);
    const bytes = new Uint8Array(44 + samples.length * 2);
    const view = new DataView(bytes.buffer);
    const encoder = new TextEncoder();
    bytes.set(encoder.encode("RIFF"), 0);
    view.setUint32(4, bytes.length - 8, true);
    bytes.set(encoder.encode("WAVEfmt "), 8);
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, clip.sampleRate, true);
    view.setUint32(28, clip.sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    bytes.set(encoder.encode("data"), 36);
    view.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, Math.round(sample * 32767), true);
    }
    const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    const link = panel.ownerDocument.createElement("a");
    link.href = url;
    link.download = "g1-show-10s.wav";
    panel.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  api("/api/show").then((data) => {
    manifest = data;
    $("show-load").hidden = !manifest.audio_available;
    $("show-progress").max = manifest.duration_s;
    prepare();
  }).catch(() => {
    status.textContent = "演出时间线暂不可用。";
  });
}
