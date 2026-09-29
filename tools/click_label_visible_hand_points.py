#!/usr/bin/env python3
"""Browser-based click annotation for visible hand landmarks.

The tool is intentionally small and dependency-free. It serves selected dataset
images over a local HTTP server and writes an annotation JSON after each edit.
It supports both the old 6-point fingertip profile and a full 21-point
MANO/MediaPipe-order profile.

Typical usage on a remote server:

ssh -L 8899:127.0.0.1:8899 server-03

cd /home/zjc/Desktop/human2dex

/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python \
  tools/click_label_visible_hand_points.py \
  --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/wrist_2D/images \
  --output n/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/wrist_2D/annotations/visible_hand_points_v1.json \
  --host 127.0.0.1 \
  --port 8899

Then open the printed URL through SSH port forwarding.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import re
import socket
import sys
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FINGERTIP6_LABELS = (
    "palm_center",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)

# A single functional target used to fit the two human grasp-pocket offsets.
# It is intentionally a separate profile: it must not be confused with the
# anatomical palm_center annotation used by camera-extrinsic fitting.
GRASP_POCKET_LABELS = (
    "grasp_pocket",
)

MANO21_LABELS = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

DEFAULT_LABELS = FINGERTIP6_LABELS

HAND_BONES_LABELS = (
    ("wrist", "thumb_cmc"),
    ("thumb_cmc", "thumb_mcp"),
    ("thumb_mcp", "thumb_ip"),
    ("thumb_ip", "thumb_tip"),
    ("wrist", "index_mcp"),
    ("index_mcp", "index_pip"),
    ("index_pip", "index_dip"),
    ("index_dip", "index_tip"),
    ("wrist", "middle_mcp"),
    ("middle_mcp", "middle_pip"),
    ("middle_pip", "middle_dip"),
    ("middle_dip", "middle_tip"),
    ("wrist", "ring_mcp"),
    ("ring_mcp", "ring_pip"),
    ("ring_pip", "ring_dip"),
    ("ring_dip", "ring_tip"),
    ("wrist", "pinky_mcp"),
    ("pinky_mcp", "pinky_pip"),
    ("pinky_pip", "pinky_dip"),
    ("pinky_dip", "pinky_tip"),
)

LABEL_COLORS = {
    "palm_center": "#ff3c50",
    "wrist": "#f5f5f5",
    "thumb_cmc": "#ffb450",
    "thumb_mcp": "#ffb450",
    "thumb_ip": "#ffb450",
    "thumb_tip": "#ffb450",
    "index_mcp": "#78ff50",
    "index_pip": "#78ff50",
    "index_dip": "#78ff50",
    "index_tip": "#78ff50",
    "middle_mcp": "#50d2ff",
    "middle_pip": "#50d2ff",
    "middle_dip": "#50d2ff",
    "middle_tip": "#50d2ff",
    "ring_mcp": "#b478ff",
    "ring_pip": "#b478ff",
    "ring_dip": "#b478ff",
    "ring_tip": "#b478ff",
    "pinky_mcp": "#ff78b4",
    "pinky_pip": "#ff78b4",
    "pinky_dip": "#ff78b4",
    "pinky_tip": "#ff78b4",
    "grasp_pocket": "#ffe050",
}


def labels_for_profile(profile: str) -> list[str]:
    if profile == "mano21":
        return list(MANO21_LABELS)
    if profile == "fingertips6":
        return list(FINGERTIP6_LABELS)
    if profile == "grasp_pocket":
        return list(GRASP_POCKET_LABELS)
    raise ValueError(f"unknown label profile: {profile}")


def skeleton_edges_for_labels(labels: list[str]) -> list[tuple[str, str]]:
    label_set = set(labels)
    return [(a, b) for a, b in HAND_BONES_LABELS if a in label_set and b in label_set]


def natural_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def read_image_list(path: Path) -> list[Path]:
    images: list[Path] = []
    base = path.parent
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        img = Path(line)
        if not img.is_absolute():
            img = base / img
        images.append(img.resolve())
    return images


def list_images_in_dir(path: Path) -> list[Path]:
    return sorted(
        [
            item.resolve()
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_key,
    )


def discover_images(
    input_path: Path,
    *,
    episodes: list[str],
    max_episodes: int | None,
) -> list[tuple[str, Path]]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        return [(image.parent.name, image) for image in read_image_list(input_path)]

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    episode_dirs: list[Path] = []
    if (input_path / "images").is_dir():
        episode_dirs = [input_path]
    elif list_images_in_dir(input_path):
        return [(input_path.name, image) for image in list_images_in_dir(input_path)]
    else:
        requested = set(episodes)
        for child in sorted(input_path.iterdir(), key=natural_key):
            if not child.is_dir():
                continue
            if requested and child.name not in requested:
                continue
            if (child / "images").is_dir():
                episode_dirs.append(child)
        if requested:
            found = {path.name for path in episode_dirs}
            missing = sorted(requested - found)
            if missing:
                raise FileNotFoundError(f"episodes not found: {', '.join(missing)}")
        if max_episodes is not None:
            episode_dirs = episode_dirs[: int(max_episodes)]

    frames: list[tuple[str, Path]] = []
    for episode_dir in episode_dirs:
        for image in list_images_in_dir(episode_dir / "images"):
            frames.append((episode_dir.name, image))
    return frames


def apply_sampling(
    frames: list[tuple[str, Path]],
    *,
    stride: int,
    start: int,
    limit: int | None,
    shuffle: bool,
    seed: int,
) -> list[tuple[str, Path]]:
    stride = max(1, int(stride))
    start = max(0, int(start))
    sampled = frames[start::stride]
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(sampled)
    if limit is not None:
        sampled = sampled[: int(limit)]
    return sampled


def merge_existing_annotations(
    frames: list[tuple[str, Path]],
    *,
    input_path: Path,
    output_path: Path,
    labels: list[str],
) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    old_by_image: dict[str, dict[str, Any]] = {}
    created_at = now
    if output_path.exists():
        try:
            old = json.loads(output_path.read_text(encoding="utf-8"))
            created_at = str(old.get("created_at", now))
            for record in old.get("frames", []):
                image = record.get("image")
                if isinstance(image, str):
                    old_by_image[image] = record
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"existing output is not valid JSON: {output_path}") from exc

    records: list[dict[str, Any]] = []
    for idx, (episode, image_path) in enumerate(frames):
        image_str = str(image_path)
        previous = old_by_image.get(image_str, {})
        points = previous.get("points") if isinstance(previous.get("points"), dict) else {}
        cleaned_points = {
            label: points[label]
            for label in labels
            if isinstance(points.get(label), dict)
        }
        records.append(
            {
                "frame_id": idx,
                "episode": episode,
                "image": image_str,
                "image_rel": (
                    str(image_path.relative_to(input_path.expanduser().resolve()))
                    if input_path.expanduser().resolve().is_dir()
                    and image_path.is_relative_to(input_path.expanduser().resolve())
                    else image_path.name
                ),
                "points": cleaned_points,
                "notes": str(previous.get("notes", "")),
            }
        )

    return {
        "version": 1,
        "task": "visible_hand_points",
        "created_at": created_at,
        "updated_at": now,
        "input": str(input_path.expanduser().resolve()),
        "output": str(output_path.expanduser().resolve()),
        "labels": labels,
        "label_colors": {label: LABEL_COLORS.get(label, "#ffffff") for label in labels},
        "skeleton_edges": skeleton_edges_for_labels(labels),
        "point_format": {
            "visible": {"x": "pixel x in original served image", "y": "pixel y in original served image", "visible": True},
            "occluded": {"x": None, "y": None, "visible": False},
        },
        "frames": records,
    }


def first_incomplete_frame(annotation: dict[str, Any]) -> int:
    labels = list(annotation["labels"])
    for idx, frame in enumerate(annotation["frames"]):
        points = frame.get("points", {})
        if not all(label in points for label in labels):
            return idx
    return 0


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Visible hand point annotator</title>
  <style>
    :root { color-scheme: dark; }
    body {
      margin: 0;
      background: #111;
      color: #eee;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #top {
      position: sticky;
      top: 0;
      z-index: 3;
      background: #1b1b1b;
      border-bottom: 1px solid #333;
      padding: 10px 12px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
    }
    .muted { color: #aaa; }
    button {
      border: 1px solid #444;
      background: #272727;
      color: #eee;
      border-radius: 7px;
      padding: 7px 10px;
      cursor: pointer;
      margin: 2px;
    }
    button:hover { background: #333; }
    button.active { border-color: #fff; box-shadow: 0 0 0 2px #777 inset; }
    button.done { background: #17321f; }
    button.occluded { background: #3a2820; }
    #main {
      display: grid;
      grid-template-columns: minmax(760px, 1fr) 430px;
      gap: 14px;
      padding: 14px;
    }
    #canvasWrap {
      overflow: auto;
      background: #050505;
      border: 1px solid #333;
      min-height: calc(100vh - 105px);
      text-align: center;
    }
    canvas {
      width: min(960px, 100%);
      height: auto;
      image-rendering: auto;
      cursor: crosshair;
    }
    #side {
      border: 1px solid #333;
      background: #181818;
      border-radius: 8px;
      padding: 12px;
      max-height: calc(100vh - 105px);
      overflow: auto;
    }
    #guideWrap {
      background: #0b0b0b;
      border: 1px solid #333;
      border-radius: 8px;
      padding: 8px;
      margin: 8px 0 12px;
    }
    #handGuide {
      width: 100%;
      height: auto;
      display: block;
    }
    .guide-edge {
      stroke-width: 3;
      stroke-linecap: round;
      opacity: 0.82;
    }
    .guide-point {
      cursor: pointer;
      stroke: #111;
      stroke-width: 2;
    }
    .guide-point.done { stroke: #d8ffd8; stroke-width: 3; }
    .guide-point.occluded { opacity: 0.35; }
    .guide-point.active {
      stroke: #fff;
      stroke-width: 5;
      filter: drop-shadow(0 0 5px #fff);
    }
    .guide-index {
      font-size: 11px;
      fill: #ddd;
      text-anchor: middle;
      dominant-baseline: central;
      pointer-events: none;
    }
    #currentPoint {
      font-size: 15px;
      color: #fff;
      margin: 4px 0 8px;
    }
    @media (max-width: 1180px) {
      #main { grid-template-columns: 1fr; }
      #side { max-height: none; }
      canvas { width: min(900px, 100%); }
    }
    #labels button {
      width: 100%;
      text-align: left;
      display: block;
      margin: 5px 0;
    }
    pre {
      white-space: pre-wrap;
      background: #0c0c0c;
      border: 1px solid #333;
      padding: 8px;
      border-radius: 6px;
      max-height: 180px;
      overflow: auto;
    }
    input[type=number] {
      width: 80px;
      background: #111;
      color: #eee;
      border: 1px solid #444;
      border-radius: 5px;
      padding: 5px;
    }
  </style>
</head>
<body>
  <div id="top">
    <div>
      <span id="progress"></span>
      <span class="muted" id="imageName"></span>
      <div class="muted" id="status">loading...</div>
    </div>
    <div>
      <button id="prevBtn">← 上一帧/P</button>
      <button id="nextBtn">下一帧/N →</button>
      <button id="saveBtn">保存/S</button>
    </div>
  </div>
  <div id="main">
    <div id="canvasWrap"><canvas id="canvas"></canvas></div>
    <div id="side">
      <div>
        跳转帧:
        <input id="jumpInput" type="number" min="1" value="1" />
        <button id="jumpBtn">跳转</button>
      </div>
      <h3>当前标注点</h3>
      <div id="currentPoint"></div>
      <div id="guideWrap">
        <svg id="handGuide" viewBox="0 0 360 300" role="img" aria-label="21-point hand guide"></svg>
      </div>
      <div id="labels"></div>
      <div>
        <button id="occludeBtn">当前点不可见/X</button>
        <button id="clearBtn">清除当前点/Backspace</button>
        <button id="clearFrameBtn">清空本帧</button>
      </div>
      <h3>快捷键</h3>
      <pre>1-9/0: 选择前10个点
[/] 或 A/D: 上一个/下一个点
鼠标左键: 标当前点，并自动跳到下一个未完成点
X: 当前点不可见
Backspace/Delete: 清除当前点
N/P: 下一帧/上一帧
S: 保存
G: 跳到第一帧未完成</pre>
      <h3>JSON 输出</h3>
      <pre id="outputPath"></pre>
    </div>
  </div>
<script>
let state = null;
let idx = 0;
let currentLabel = null;
let image = new Image();
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const labelButtons = {};
const GUIDE_POS = {
  wrist: [178, 262],
  thumb_cmc: [132, 222], thumb_mcp: [96, 188], thumb_ip: [70, 150], thumb_tip: [48, 110],
  index_mcp: [155, 190], index_pip: [145, 140], index_dip: [138, 96], index_tip: [132, 55],
  middle_mcp: [185, 185], middle_pip: [188, 128], middle_dip: [190, 78], middle_tip: [192, 35],
  ring_mcp: [215, 190], ring_pip: [230, 142], ring_dip: [242, 100], ring_tip: [252, 62],
  pinky_mcp: [242, 205], pinky_pip: [272, 168], pinky_dip: [292, 132], pinky_tip: [312, 98],
  palm_center: [180, 218],
};

async function api(path, options={}) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function frame() { return state.frames[idx]; }

function pointState(label) {
  const p = frame().points[label];
  if (!p) return "missing";
  return p.visible ? "done" : "occluded";
}

function setStatus(text) {
  document.getElementById("status").textContent = text;
}

function chooseNextMissing() {
  for (const label of state.labels) {
    if (!frame().points[label]) return label;
  }
  return currentLabel || state.labels[0];
}

function selectLabel(label) {
  currentLabel = label;
  renderButtons();
  renderGuide();
  draw();
}

function stepLabel(delta) {
  const currentIndex = Math.max(0, state.labels.indexOf(currentLabel));
  const nextIndex = (currentIndex + delta + state.labels.length) % state.labels.length;
  selectLabel(state.labels[nextIndex]);
}

function renderButtons() {
  const labelsDiv = document.getElementById("labels");
  labelsDiv.innerHTML = "";
  document.getElementById("currentPoint").textContent =
    `当前点：${state.labels.indexOf(currentLabel) + 1}. ${currentLabel}`;
  state.labels.forEach((label, i) => {
    const btn = document.createElement("button");
    const p = frame().points[label];
    const status = pointState(label);
    btn.classList.toggle("active", label === currentLabel);
    btn.classList.toggle("done", status === "done");
    btn.classList.toggle("occluded", status === "occluded");
    btn.style.borderLeft = `8px solid ${state.label_colors[label] || "#fff"}`;
    btn.textContent = `${i + 1}. ${label}` + (
      p ? (p.visible ? `  (${p.x.toFixed(1)}, ${p.y.toFixed(1)})` : "  [不可见]") : "  [未标]"
    );
    btn.onclick = () => selectLabel(label);
    labelsDiv.appendChild(btn);
    labelButtons[label] = btn;
  });
}

function renderGuide() {
  const svg = document.getElementById("handGuide");
  svg.innerHTML = "";
  const ns = "http://www.w3.org/2000/svg";
  const edges = state.skeleton_edges || [];
  for (const edge of edges) {
    const a = GUIDE_POS[edge[0]];
    const b = GUIDE_POS[edge[1]];
    if (!a || !b) continue;
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", a[0]); line.setAttribute("y1", a[1]);
    line.setAttribute("x2", b[0]); line.setAttribute("y2", b[1]);
    line.setAttribute("stroke", state.label_colors[edge[1]] || "#777");
    line.setAttribute("class", "guide-edge");
    svg.appendChild(line);
  }
  state.labels.forEach((label, i) => {
    const pos = GUIDE_POS[label];
    if (!pos) return;
    const status = pointState(label);
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", pos[0]);
    circle.setAttribute("cy", pos[1]);
    circle.setAttribute("r", label === currentLabel ? 9 : 7);
    circle.setAttribute("fill", state.label_colors[label] || "#fff");
    circle.setAttribute("class", `guide-point ${status} ${label === currentLabel ? "active" : ""}`);
    circle.addEventListener("click", () => selectLabel(label));
    svg.appendChild(circle);
    const text = document.createElementNS(ns, "text");
    text.setAttribute("x", pos[0]);
    text.setAttribute("y", pos[1]);
    text.setAttribute("class", "guide-index");
    text.textContent = String(i + 1);
    svg.appendChild(text);
  });
}

function updateHeader() {
  const complete = state.frames.filter(f => state.labels.every(l => f.points[l])).length;
  document.getElementById("progress").textContent =
    `Frame ${idx + 1}/${state.frames.length} | 完成 ${complete}/${state.frames.length}`;
  document.getElementById("imageName").textContent =
    ` | ${frame().episode} / ${frame().image_rel}`;
  document.getElementById("jumpInput").value = idx + 1;
}

function draw() {
  if (!state || !image.complete || image.naturalWidth === 0) return;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  ctx.drawImage(image, 0, 0);

  const edges = state.skeleton_edges || [];
  for (const edge of edges) {
    const a = frame().points[edge[0]];
    const b = frame().points[edge[1]];
    if (a && a.visible && b && b.visible) {
      ctx.strokeStyle = state.label_colors[edge[1]] || "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }

  for (const label of state.labels) {
    const p = frame().points[label];
    if (!p || !p.visible) continue;
    const color = state.label_colors[label] || "#fff";
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, label === currentLabel ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

async function saveFrame() {
  await api("/api/save_frame", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({frame_id: frame().frame_id, points: frame().points, notes: frame().notes || ""})
  });
  setStatus("saved " + new Date().toLocaleTimeString());
}

async function loadFrame(newIdx) {
  idx = Math.max(0, Math.min(state.frames.length - 1, newIdx));
  currentLabel = chooseNextMissing();
  updateHeader();
  renderButtons();
  renderGuide();
  setStatus("loading image...");
  image = new Image();
  image.onload = () => { draw(); setStatus("ready"); };
  image.onerror = () => setStatus("image load failed");
  image.src = `/image/${idx}?t=${Date.now()}`;
}

async function setPointVisible(x, y) {
  frame().points[currentLabel] = {x, y, visible: true};
  const missingAfter = state.labels.find(l => !frame().points[l]);
  if (missingAfter) currentLabel = missingAfter;
  renderButtons();
  renderGuide();
  draw();
  await saveFrame();
}

async function markOccluded() {
  frame().points[currentLabel] = {x: null, y: null, visible: false};
  const missingAfter = state.labels.find(l => !frame().points[l]);
  if (missingAfter) currentLabel = missingAfter;
  renderButtons();
  draw();
  await saveFrame();
}

async function clearCurrent() {
  delete frame().points[currentLabel];
  renderButtons();
  renderGuide();
  draw();
  await saveFrame();
}

async function clearFrame() {
  if (!confirm("清空当前帧所有点？")) return;
  frame().points = {};
  currentLabel = state.labels[0];
  renderButtons();
  renderGuide();
  draw();
  await saveFrame();
}

function goFirstIncomplete() {
  const target = state.frames.findIndex(f => !state.labels.every(l => f.points[l]));
  if (target >= 0) loadFrame(target);
}

canvas.addEventListener("click", async (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * canvas.width / rect.width;
  const y = (event.clientY - rect.top) * canvas.height / rect.height;
  await setPointVisible(x, y);
});

document.getElementById("prevBtn").onclick = () => loadFrame(idx - 1);
document.getElementById("nextBtn").onclick = () => loadFrame(idx + 1);
document.getElementById("saveBtn").onclick = () => saveFrame();
document.getElementById("occludeBtn").onclick = () => markOccluded();
document.getElementById("clearBtn").onclick = () => clearCurrent();
document.getElementById("clearFrameBtn").onclick = () => clearFrame();
document.getElementById("jumpBtn").onclick = () => loadFrame(Number(document.getElementById("jumpInput").value) - 1);

document.addEventListener("keydown", async (event) => {
  if (event.target.tagName === "INPUT") return;
  if (/^[1-9]$/.test(event.key) || event.key === "0") {
    const shortcutIndex = event.key === "0" ? 9 : Number(event.key) - 1;
    const label = state.labels[shortcutIndex];
    if (label) selectLabel(label);
  } else if (event.key === "]" || event.key === "d" || event.key === "D") {
    stepLabel(1);
  } else if (event.key === "[" || event.key === "a" || event.key === "A") {
    stepLabel(-1);
  } else if (event.key === "n" || event.key === "N" || event.key === "ArrowRight") {
    loadFrame(idx + 1);
  } else if (event.key === "p" || event.key === "P" || event.key === "ArrowLeft") {
    loadFrame(idx - 1);
  } else if (event.key === "x" || event.key === "X") {
    await markOccluded();
  } else if (event.key === "Backspace" || event.key === "Delete") {
    event.preventDefault();
    await clearCurrent();
  } else if (event.key === "s" || event.key === "S") {
    await saveFrame();
  } else if (event.key === "g" || event.key === "G") {
    goFirstIncomplete();
  }
});

async function init() {
  state = await api("/api/state");
  idx = state.start_index || 0;
  document.getElementById("outputPath").textContent = state.output;
  await loadFrame(idx);
}
init().catch(err => setStatus("ERROR: " + err.message));
</script>
</body>
</html>
"""


class AnnotationApp:
    def __init__(self, annotation: dict[str, Any], output_path: Path):
        self.annotation = annotation
        self.output_path = output_path.expanduser().resolve()
        self.image_by_index = {
            idx: Path(frame["image"])
            for idx, frame in enumerate(self.annotation["frames"])
        }

    def save(self) -> None:
        self.annotation["updated_at"] = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(self.output_path, self.annotation)

    def state(self) -> dict[str, Any]:
        return {
            "version": self.annotation["version"],
            "task": self.annotation["task"],
            "output": str(self.output_path),
            "labels": self.annotation["labels"],
            "label_colors": self.annotation["label_colors"],
            "skeleton_edges": self.annotation.get("skeleton_edges", skeleton_edges_for_labels(list(self.annotation["labels"]))),
            "frames": self.annotation["frames"],
            "start_index": first_incomplete_frame(self.annotation),
        }

    def update_frame(self, frame_id: int, points: dict[str, Any], notes: str = "") -> None:
        labels = set(self.annotation["labels"])
        if frame_id < 0 or frame_id >= len(self.annotation["frames"]):
            raise IndexError(frame_id)
        cleaned: dict[str, Any] = {}
        for label, point in points.items():
            if label not in labels or not isinstance(point, dict):
                continue
            visible = bool(point.get("visible", False))
            if visible:
                x = float(point["x"])
                y = float(point["y"])
                if not (0 <= x < 100000 and 0 <= y < 100000):
                    continue
                cleaned[label] = {"x": round(x, 3), "y": round(y, 3), "visible": True}
            else:
                cleaned[label] = {"x": None, "y": None, "visible": False}
        frame = self.annotation["frames"][frame_id]
        frame["points"] = cleaned
        frame["notes"] = str(notes)
        self.save()


def make_handler(app: AnnotationApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VisibleHandPointAnnotator/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

        def send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def send_error_json(self, message: str, status: HTTPStatus) -> None:
            self.send_json({"error": message}, status=status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/" or path == "/index.html":
                self.send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                self.send_json(app.state())
                return
            if path.startswith("/image/"):
                try:
                    idx = int(path.rsplit("/", 1)[-1])
                    image_path = app.image_by_index[idx]
                    data = image_path.read_bytes()
                except Exception as exc:
                    self.send_error_json(str(exc), HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
                self.send_bytes(data, content_type)
                return
            self.send_error_json(f"unknown path: {path}", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/save_frame":
                self.send_error_json(f"unknown path: {parsed.path}", HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                app.update_frame(
                    int(payload["frame_id"]),
                    payload.get("points", {}),
                    payload.get("notes", ""),
                )
            except Exception as exc:
                self.send_error_json(str(exc), HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "updated_at": app.annotation["updated_at"]})

    return Handler


def local_ip_hint() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        try:
            sock.close()
        except Exception:
            pass


def parse_labels(value: str) -> list[str]:
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("labels cannot be empty")
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Dataset root, episode dir, image dir, or text file with image paths.")
    parser.add_argument("--output", type=Path, required=True, help="Annotation JSON path.")
    parser.add_argument("--episode", action="append", default=[], help="Episode name to include. Repeat for multiple episodes.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Use only the first N discovered episodes.")
    parser.add_argument("--start", type=int, default=0, help="Start image offset before stride sampling.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Maximum number of frames after sampling.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle sampled frames.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-profile", choices=("fingertips6", "mano21", "grasp_pocket"), default="fingertips6", help="Preset label set: fingertips6, mano21, or one functional grasp_pocket click.")
    parser.add_argument("--labels", type=parse_labels, default=None, help="Comma-separated labels. Overrides --label-profile.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host. Use 127.0.0.1 with SSH port forwarding.")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--open-browser", action="store_true", help="Try to open a browser on the current machine.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected frames without starting server.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frames = discover_images(args.input, episodes=args.episode, max_episodes=args.max_episodes)
    frames = apply_sampling(
        frames,
        stride=args.stride,
        start=args.start,
        limit=args.limit_frames,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not frames:
        raise SystemExit(f"no images selected from: {args.input}")

    print(f"selected frames: {len(frames)}")
    print(f"first image: {frames[0][1]}")
    print(f"output: {args.output.expanduser().resolve()}")
    if args.dry_run:
        for episode, image in frames[:20]:
            print(f"{episode}\t{image}")
        if len(frames) > 20:
            print(f"... {len(frames) - 20} more")
        return 0

    labels = list(args.labels) if args.labels is not None else labels_for_profile(str(args.label_profile))
    annotation = merge_existing_annotations(
        frames,
        input_path=args.input,
        output_path=args.output,
        labels=labels,
    )
    app = AnnotationApp(annotation=annotation, output_path=args.output)
    app.save()

    handler = make_handler(app)
    httpd = ThreadingHTTPServer((args.host, int(args.port)), handler)
    url = f"http://{args.host}:{args.port}/"
    print("server started")
    print(f"local URL: {url}")
    if args.host in {"0.0.0.0", "::"}:
        print(f"LAN URL hint: http://{local_ip_hint()}:{args.port}/")
    print("SSH tunnel example from your laptop:")
    print(f"  ssh -L {args.port}:127.0.0.1:{args.port} server-03")
    print(f"then open: http://127.0.0.1:{args.port}/")
    if args.open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutdown")
    finally:
        app.save()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
