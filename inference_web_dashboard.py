#!/usr/bin/env python3
"""Local-only web dashboard for real Franka inference.

The HTTP server is intentionally bound to 127.0.0.1 by default.  Operators
reach it through an SSH local-port forward.  A per-process token protects the
control endpoints from other users/processes on the remote host.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import secrets
import threading
import time
from collections import deque
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np
from aiohttp import web


def _array_or_none(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    return array.copy()


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _latest_row(value: Optional[np.ndarray]) -> Optional[list]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    if array.ndim == 1:
        row = array
    else:
        row = array[-1]
    return np.asarray(row, dtype=np.float64).reshape(-1).tolist()


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3:
        raise ValueError(f"frame must have 3 dimensions, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if np.issubdtype(image.dtype, np.floating):
        finite = image[np.isfinite(image)]
        max_value = float(np.max(finite)) if finite.size else 1.0
        if max_value <= 1.5:
            image = image * 255.0
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0, 255).astype(np.uint8)


class InferenceWebDashboard:
    """Threaded aiohttp dashboard plus a thread-safe inference control state."""

    VALID_PHASES = {
        "booting", "paused", "running", "resetting", "error", "stopping", "stopped"
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        heartbeat_timeout: float = 5.0,
        history_seconds: float = 10.0,
        jpeg_quality: int = 85,
        token: Optional[str] = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.history_seconds = float(history_seconds)
        self.jpeg_quality = int(jpeg_quality)
        self.token = token or secrets.token_urlsafe(24)

        self._lock = threading.RLock()
        self._phase = "booting"
        self._reason = "正在加载模型和硬件"
        self._running_requested = False
        self._reset_requested = False
        self._stop_requested = False
        self._client_seen = False
        self._last_heartbeat_mono = 0.0
        self._updated_wall = time.time()
        self._frames: Dict[str, np.ndarray] = {}
        self._frame_versions: Dict[str, int] = {}
        self._payload: Dict[str, Any] = {
            "raw_model_chunk": None,
            "processed_arm_chunk": None,
            "submitted_arm_chunk": None,
            "processed_hand_chunk": None,
            "submitted_hand_chunk": None,
            "observed_arm": None,
            "observed_hand": None,
            "metadata": {},
        }
        self._history = deque()

        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._startup_error: Optional[BaseException] = None

    @property
    def url(self) -> str:
        browser_host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{browser_host}:{self.port}/?token={self.token}"

    def start(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._thread_main,
            name="inference-web-dashboard",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout):
            raise TimeoutError(f"dashboard did not start within {timeout:.1f}s")
        if self._startup_error is not None:
            raise RuntimeError(f"dashboard startup failed: {self._startup_error}")

    def close(self, timeout: float = 5.0) -> None:
        loop = self._loop
        event = self._shutdown_event
        if loop is not None and event is not None and loop.is_running():
            loop.call_soon_threadsafe(event.set)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        runner = None
        watchdog_task = None
        try:
            app = web.Application()
            app.router.add_get("/", self._handle_index)
            app.router.add_get("/api/state", self._handle_state)
            app.router.add_get("/api/frame/{name}", self._handle_frame)
            app.router.add_post("/api/heartbeat", self._handle_heartbeat)
            app.router.add_post("/api/control/{command}", self._handle_control)
            runner = web.AppRunner(app, access_log=None)
            loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, self.host, self.port)
            loop.run_until_complete(site.start())
            self._shutdown_event = asyncio.Event()
            watchdog_task = loop.create_task(self._watchdog())
            self._ready_event.set()
            loop.run_until_complete(self._shutdown_event.wait())
        except BaseException as exc:
            self._startup_error = exc
            self._ready_event.set()
        finally:
            if watchdog_task is not None:
                watchdog_task.cancel()
                try:
                    loop.run_until_complete(watchdog_task)
                except (asyncio.CancelledError, RuntimeError):
                    pass
            if runner is not None:
                loop.run_until_complete(runner.cleanup())
            loop.close()

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(0.2)
            with self._lock:
                if not self._running_requested or not self._client_seen:
                    continue
                age = time.monotonic() - self._last_heartbeat_mono
                if age <= self.heartbeat_timeout:
                    continue
                self._running_requested = False
                self._phase = "paused"
                self._reason = f"网页心跳超过 {self.heartbeat_timeout:.1f} 秒，已自动暂停"
                self._updated_wall = time.time()

    def _authorized(self, request: web.Request) -> bool:
        supplied = request.headers.get("X-Dashboard-Token") or request.query.get("token", "")
        return bool(supplied) and hmac.compare_digest(str(supplied), self.token)

    def _require_auth(self, request: web.Request) -> None:
        if not self._authorized(request):
            raise web.HTTPForbidden(text="invalid dashboard token")

    async def _handle_index(self, request: web.Request) -> web.Response:
        self._require_auth(request)
        return web.Response(text=_DASHBOARD_HTML, content_type="text/html")

    async def _handle_state(self, request: web.Request) -> web.Response:
        self._require_auth(request)
        return web.json_response(self.snapshot())

    async def _handle_frame(self, request: web.Request) -> web.Response:
        self._require_auth(request)
        name = request.match_info["name"]
        with self._lock:
            frame = self._frames.get(name)
        if frame is None:
            raise web.HTTPNotFound(text=f"no frame for {name}")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise web.HTTPInternalServerError(text="JPEG encoding failed")
        return web.Response(
            body=encoded.tobytes(),
            content_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    async def _handle_heartbeat(self, request: web.Request) -> web.Response:
        self._require_auth(request)
        with self._lock:
            self._client_seen = True
            self._last_heartbeat_mono = time.monotonic()
        return web.json_response({"ok": True})

    async def _handle_control(self, request: web.Request) -> web.Response:
        self._require_auth(request)
        command = request.match_info["command"]
        if command == "start":
            result = self.request_start()
        elif command == "pause":
            result = self.request_pause("用户点击暂停")
        elif command == "reset":
            result = self.request_reset()
        elif command == "stop":
            result = self.request_stop()
        else:
            raise web.HTTPNotFound(text=f"unknown command: {command}")
        status = 200 if result[0] else 409
        return web.json_response({"ok": result[0], "message": result[1]}, status=status)

    def request_start(self) -> tuple[bool, str]:
        with self._lock:
            if self._phase in {"booting", "resetting", "error", "stopping", "stopped"}:
                return False, f"当前状态 {self._phase} 不允许开始"
            self._client_seen = True
            self._last_heartbeat_mono = time.monotonic()
            self._reset_requested = False
            self._running_requested = True
            self._phase = "running"
            self._reason = "正在推理并执行"
            self._updated_wall = time.time()
            return True, "已开始"

    def request_pause(self, reason: str = "已暂停") -> tuple[bool, str]:
        with self._lock:
            if self._phase in {"stopping", "stopped"}:
                return False, f"当前状态 {self._phase} 不允许暂停"
            self._running_requested = False
            if self._phase != "error":
                self._phase = "paused"
                self._reason = str(reason)
            self._updated_wall = time.time()
            return True, str(reason)

    def request_reset(self) -> tuple[bool, str]:
        with self._lock:
            if self._phase in {"booting", "stopping", "stopped"}:
                return False, f"当前状态 {self._phase} 不允许复位"
            self._running_requested = False
            self._reset_requested = True
            self._phase = "resetting"
            self._reason = "等待推理主线程执行复位"
            self._updated_wall = time.time()
            return True, "已请求复位"

    def request_stop(self) -> tuple[bool, str]:
        with self._lock:
            self._running_requested = False
            self._stop_requested = True
            self._phase = "stopping"
            self._reason = "正在安全结束推理"
            self._updated_wall = time.time()
            return True, "已请求结束"

    def can_execute(self) -> bool:
        with self._lock:
            if self._phase != "running" or not self._running_requested:
                return False
            if self._reset_requested or self._stop_requested:
                return False
            if not self._client_seen:
                return False
            if time.monotonic() - self._last_heartbeat_mono > self.heartbeat_timeout:
                return False
            return True

    def reset_pending(self) -> bool:
        with self._lock:
            return self._reset_requested

    def consume_reset_request(self) -> bool:
        with self._lock:
            if not self._reset_requested:
                return False
            self._reset_requested = False
            return True

    def stop_pending(self) -> bool:
        with self._lock:
            return self._stop_requested

    def mark_ready(self, reason: str = "已暂停，等待开始") -> None:
        with self._lock:
            if self._stop_requested:
                return
            self._running_requested = False
            self._phase = "paused"
            self._reason = str(reason)
            self._updated_wall = time.time()

    def mark_running(self, reason: str = "正在推理并执行") -> None:
        with self._lock:
            if self._stop_requested or self._reset_requested:
                return
            self._phase = "running"
            self._reason = str(reason)
            self._updated_wall = time.time()

    def mark_resetting(self, reason: str = "正在复位机械臂和灵巧手") -> None:
        with self._lock:
            self._running_requested = False
            self._phase = "resetting"
            self._reason = str(reason)
            self._updated_wall = time.time()

    def mark_error(self, reason: str) -> None:
        with self._lock:
            self._running_requested = False
            self._phase = "error"
            self._reason = str(reason)
            self._updated_wall = time.time()

    def mark_stopped(self, reason: str = "推理已结束") -> None:
        with self._lock:
            self._running_requested = False
            self._phase = "stopped"
            self._reason = str(reason)
            self._updated_wall = time.time()

    def update(
        self,
        *,
        frames: Optional[Mapping[str, np.ndarray]] = None,
        raw_model_chunk: Any = None,
        processed_arm_chunk: Any = None,
        submitted_arm_chunk: Any = None,
        processed_hand_chunk: Any = None,
        submitted_hand_chunk: Any = None,
        observed_arm: Any = None,
        observed_hand: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            if frames:
                for name, frame in frames.items():
                    try:
                        image = _frame_to_uint8(frame)
                    except Exception:
                        continue
                    self._frames[str(name)] = image.copy()
                    self._frame_versions[str(name)] = self._frame_versions.get(str(name), 0) + 1

            updates = {
                "raw_model_chunk": raw_model_chunk,
                "processed_arm_chunk": processed_arm_chunk,
                "submitted_arm_chunk": submitted_arm_chunk,
                "processed_hand_chunk": processed_hand_chunk,
                "submitted_hand_chunk": submitted_hand_chunk,
                "observed_arm": observed_arm,
                "observed_hand": observed_hand,
            }
            for key, value in updates.items():
                if value is not None:
                    self._payload[key] = _array_or_none(value)
            if metadata is not None:
                self._payload["metadata"] = dict(metadata)

            history_item = {
                "time": now,
                "model": _latest_row(self._payload["raw_model_chunk"]),
                "arm_command": _latest_row(self._payload["submitted_arm_chunk"]),
                "hand_command": _latest_row(self._payload["submitted_hand_chunk"]),
                "arm_observed": _latest_row(_array_or_none(observed_arm)),
                "hand_observed": _latest_row(_array_or_none(observed_hand)),
            }
            if any(history_item[key] is not None for key in history_item if key != "time"):
                self._history.append(history_item)
            cutoff = now - self.history_seconds
            while self._history and self._history[0]["time"] < cutoff:
                self._history.popleft()
            self._updated_wall = now

    def snapshot(self) -> dict:
        with self._lock:
            heartbeat_age = None
            if self._client_seen:
                heartbeat_age = max(0.0, time.monotonic() - self._last_heartbeat_mono)
            payload = {
                "phase": self._phase,
                "reason": self._reason,
                "running_requested": self._running_requested,
                "reset_requested": self._reset_requested,
                "stop_requested": self._stop_requested,
                "heartbeat_timeout": self.heartbeat_timeout,
                "heartbeat_age": heartbeat_age,
                "updated_at": self._updated_wall,
                "frame_names": sorted(self._frames),
                "frame_versions": dict(self._frame_versions),
                "history": list(self._history),
            }
            payload.update({key: _jsonable(value) for key, value in self._payload.items()})
            return payload


_DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Franka 实时推理面板</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --card:#151c30; --line:#2b3855; --text:#e8edf7; --muted:#97a4bb; --green:#29c67d; --orange:#ffb648; --red:#ff5f68; --blue:#54a8ff; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--text); }
    header { position:sticky; top:0; z-index:4; display:flex; gap:16px; align-items:center; justify-content:space-between; padding:14px 18px; background:rgba(11,16,32,.95); border-bottom:1px solid var(--line); backdrop-filter:blur(10px); }
    h1 { margin:0; font-size:19px; }
    .status { display:flex; align-items:center; gap:9px; color:var(--muted); font-size:13px; }
    .dot { width:10px; height:10px; border-radius:50%; background:var(--orange); box-shadow:0 0 12px currentColor; }
    .controls { display:flex; flex-wrap:wrap; gap:9px; }
    button { border:1px solid var(--line); border-radius:8px; color:var(--text); background:#202a43; padding:9px 15px; cursor:pointer; font-weight:650; }
    button:hover { filter:brightness(1.18); }
    button:disabled { opacity:.42; cursor:not-allowed; }
    button.start { background:#0c6c45; } button.pause { background:#805814; } button.reset { background:#245a91; } button.stop { background:#8f2730; }
    main { padding:16px; display:grid; gap:14px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:14px; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; min-width:0; }
    .card h2 { margin:0 0 11px; font-size:15px; }
    .camera-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,520px)); justify-content:center; gap:12px; }
    .camera { display:flex; width:100%; flex-direction:column; align-items:center; gap:6px; }
    .camera img { display:block; width:min(520px,46vh,100%); height:auto; object-fit:contain; background:#05070d; border-radius:8px; image-rendering:auto; }
    .camera label, .muted { color:var(--muted); font-size:12px; }
    pre { margin:0; max-height:290px; overflow:auto; font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; color:#cbd7ec; white-space:pre-wrap; overflow-wrap:anywhere; }
    canvas { width:100%; height:260px; border-radius:8px; background:#0c1221; }
    select { color:var(--text); background:#202a43; border:1px solid var(--line); border-radius:7px; padding:7px; }
    .chart-tools { display:flex; gap:8px; margin-bottom:9px; flex-wrap:wrap; align-items:center; }
    .legend { display:flex; gap:14px; color:var(--muted); font-size:12px; margin-top:7px; }
    .legend span::before { content:""; display:inline-block; width:15px; height:3px; margin-right:5px; vertical-align:middle; background:var(--blue); }
    .legend span:nth-child(2)::before { background:var(--green); }
    .message { color:var(--muted); max-width:720px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    @media (max-width:700px) { header { align-items:flex-start; flex-direction:column; } .message { max-width:95vw; } .camera-grid { grid-template-columns:minmax(0,1fr); } .camera img { width:min(100%,42vh); } }
  </style>
</head>
<body>
<header>
  <div>
    <h1>Franka + O6 实时推理面板</h1>
    <div class="status"><span class="dot" id="dot"></span><b id="phase">连接中</b><span class="message" id="reason"></span><span id="heartbeat"></span></div>
  </div>
  <div class="controls">
    <button class="start" id="start">开始推理</button>
    <button class="pause" id="pause">暂停</button>
    <button class="reset" id="reset">复位</button>
    <button class="stop" id="stop">结束</button>
  </div>
</header>
<main>
  <section class="card"><h2>模型实际输入图像</h2><div class="camera-grid" id="cameras"><span class="muted">等待图像…</span></div></section>
  <section class="card">
    <h2>最近 10 秒动作与观测曲线</h2>
    <div class="chart-tools"><label>分组 <select id="group"><option value="arm_pos">机械臂位置</option><option value="arm_rot">机械臂旋转</option><option value="hand">灵巧手</option><option value="model">模型原始输出</option></select></label><label>维度 <select id="dimension"></select></label></div>
    <canvas id="chart" width="1100" height="300"></canvas>
    <div class="legend"><span>实际发送/模型输出</span><span>机器人观测</span></div>
  </section>
  <section class="grid">
    <article class="card"><h2>模型原始 action chunk</h2><pre id="raw">暂无</pre></article>
    <article class="card"><h2>后处理后的机械臂 action</h2><pre id="processed-arm">暂无</pre></article>
    <article class="card"><h2>实际提交给 Franka 的 action</h2><pre id="submitted-arm">暂无</pre></article>
    <article class="card"><h2>后处理 / 实际发送的灵巧手 action</h2><pre id="hand">暂无</pre></article>
    <article class="card"><h2>当前机器人观测状态</h2><pre id="observed">暂无</pre></article>
    <article class="card"><h2>运行元数据</h2><pre id="metadata">暂无</pre></article>
  </section>
</main>
<script>
const token = new URLSearchParams(location.search).get('token') || '';
let latest = null, knownCameras = '';
const phaseNames = {booting:'启动中',paused:'已暂停',running:'运行中',resetting:'复位中',error:'错误',stopping:'结束中',stopped:'已结束'};
const colors = {running:'#29c67d',paused:'#ffb648',error:'#ff5f68',resetting:'#54a8ff',booting:'#97a4bb',stopping:'#ff5f68',stopped:'#97a4bb'};
async function api(path, options={}) {
  options.headers = Object.assign({}, options.headers || {}, {'X-Dashboard-Token':token});
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
async function command(name) { try { await api('/api/control/'+name,{method:'POST'}); } catch(e) { alert(e.message); } }
document.querySelector('#start').onclick=()=>command('start');
document.querySelector('#pause').onclick=()=>command('pause');
document.querySelector('#reset').onclick=()=>{ if(confirm('确认复位机械臂和灵巧手？复位完成后保持暂停。')) command('reset'); };
document.querySelector('#stop').onclick=()=>{ if(confirm('确认安全结束本次推理并关闭服务？')) command('stop'); };
setInterval(()=>api('/api/heartbeat',{method:'POST'}).catch(()=>{}),1000);
function format(value) { return value == null ? '暂无' : JSON.stringify(value,null,2); }
function updateCameras(state) {
  const key=(state.frame_names||[]).join('|');
  if (key!==knownCameras) {
    knownCameras=key; const root=document.querySelector('#cameras'); root.innerHTML='';
    for (const name of state.frame_names||[]) { const box=document.createElement('div'); box.className='camera'; box.innerHTML=`<img id="cam-${name}" alt="${name}"><label>${name}（模型实际输入）</label>`; root.appendChild(box); }
    if (!key) root.innerHTML='<span class="muted">等待图像…</span>';
  }
  for (const name of state.frame_names||[]) { const img=document.getElementById('cam-'+name); const v=(state.frame_versions||{})[name]||0; if(img && img.dataset.v!=String(v)){img.dataset.v=String(v);img.src=`/api/frame/${encodeURIComponent(name)}?token=${encodeURIComponent(token)}&v=${v}`;} }
}
function dimensionsFor(state,group){
  if(group==='arm_pos'||group==='arm_rot') return 3;
  if(group==='hand') return Math.max((state.submitted_hand_chunk?.at(-1)||[]).length,(state.observed_hand||[]).length,1);
  return Math.max((state.raw_model_chunk?.at(-1)||[]).length,1);
}
function syncDimensions(){ const sel=document.querySelector('#dimension'), n=dimensionsFor(latest||{},document.querySelector('#group').value), old=Number(sel.value||0); sel.innerHTML=''; for(let i=0;i<n;i++){const o=document.createElement('option');o.value=i;o.textContent='dim '+i;sel.appendChild(o);} sel.value=Math.min(old,n-1); }
document.querySelector('#group').onchange=()=>{syncDimensions();drawChart();}; document.querySelector('#dimension').onchange=drawChart;
function drawChart(){
  const canvas=document.querySelector('#chart'), ctx=canvas.getContext('2d'), history=latest?.history||[], group=document.querySelector('#group').value, dim=Number(document.querySelector('#dimension').value||0); ctx.clearRect(0,0,canvas.width,canvas.height); ctx.strokeStyle='#27324a'; ctx.lineWidth=1; for(let i=0;i<=5;i++){let y=20+i*(canvas.height-40)/5;ctx.beginPath();ctx.moveTo(45,y);ctx.lineTo(canvas.width-15,y);ctx.stroke();}
  if(history.length<2){ctx.fillStyle='#97a4bb';ctx.fillText('等待历史数据…',55,45);return;}
  const first=history[0].time,last=history.at(-1).time||first+1; let a=[],b=[];
  for(const h of history){let av=null,bv=null;if(group==='arm_pos'){av=h.arm_command?.[dim];bv=h.arm_observed?.[dim];}else if(group==='arm_rot'){av=h.arm_command?.[dim+3];bv=h.arm_observed?.[dim+3];}else if(group==='hand'){av=h.hand_command?.[dim];bv=h.hand_observed?.[dim];}else{av=h.model?.[dim];}if(Number.isFinite(av))a.push([h.time,av]);if(Number.isFinite(bv))b.push([h.time,bv]);}
  const all=a.concat(b).map(x=>x[1]);if(!all.length){ctx.fillStyle='#97a4bb';ctx.fillText('该维度暂无数据',55,45);return;}let min=Math.min(...all),max=Math.max(...all);if(Math.abs(max-min)<1e-9){min-=1;max+=1;}const x=t=>45+(t-first)/Math.max(last-first,.001)*(canvas.width-65),y=v=>canvas.height-20-(v-min)/(max-min)*(canvas.height-40);
  function line(points,color){if(!points.length)return;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(x(p[0]),y(p[1])):ctx.moveTo(x(p[0]),y(p[1])));ctx.stroke();}line(a,'#54a8ff');line(b,'#29c67d');ctx.fillStyle='#97a4bb';ctx.fillText(max.toFixed(4),4,24);ctx.fillText(min.toFixed(4),4,canvas.height-20);
}
function render(state){
  latest=state; document.querySelector('#phase').textContent=phaseNames[state.phase]||state.phase; document.querySelector('#reason').textContent=state.reason||''; document.querySelector('#dot').style.background=colors[state.phase]||'#97a4bb'; document.querySelector('#heartbeat').textContent=state.heartbeat_age==null?'':`心跳 ${state.heartbeat_age.toFixed(1)}s`;
  document.querySelector('#start').disabled=!['paused'].includes(state.phase); document.querySelector('#pause').disabled=state.phase!=='running'; document.querySelector('#reset').disabled=['booting','resetting','stopping','stopped'].includes(state.phase); document.querySelector('#stop').disabled=['stopping','stopped'].includes(state.phase);
  document.querySelector('#raw').textContent=format(state.raw_model_chunk); document.querySelector('#processed-arm').textContent=format(state.processed_arm_chunk); document.querySelector('#submitted-arm').textContent=format(state.submitted_arm_chunk); document.querySelector('#hand').textContent=format({processed:state.processed_hand_chunk,submitted:state.submitted_hand_chunk}); document.querySelector('#observed').textContent=format({arm:state.observed_arm,hand:state.observed_hand}); document.querySelector('#metadata').textContent=format(state.metadata);
  updateCameras(state); syncDimensions(); drawChart();
}
async function poll(){try{render(await api('/api/state'));}catch(e){document.querySelector('#reason').textContent='页面与远端服务失去连接';document.querySelector('#dot').style.background='#ff5f68';}}
setInterval(poll,250); poll();
</script>
</body></html>"""


if __name__ == "__main__":
    dashboard = InferenceWebDashboard()
    dashboard.start()
    dashboard.mark_ready("独立面板测试模式")
    print(dashboard.url, flush=True)
    try:
        while not dashboard.stop_pending():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.mark_stopped()
        dashboard.close()
