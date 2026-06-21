#!/usr/bin/env python

"""Local web UI for SO101 data collection, training, and rollout.

Run from the LeRobot repo:

    python scripts/robot_training_ui.py

Then open http://127.0.0.1:8787.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_PATH = REPO_ROOT / "robot_training_ui_presets.json"
ARCHIVE_DIR = REPO_ROOT / "archive"
DEFAULT_PRESETS = {
    "robot_type": "so101_follower",
    "robot_port": "/dev/tty.usbmodem5B7B0145651",
    "robot_id": "my_follower_arm",
    "teleop_type": "so101_leader",
    "teleop_port": "/dev/tty.usbmodem5B7B0098471",
    "teleop_id": "my_leader_arm",
    "camera_name": "overhead",
    "camera_index": "0",
    "camera_width": "640",
    "camera_height": "480",
    "camera_fps": "30",
    "task": "Pick up the object and place it in the bin",
    "max_relative_target": "15",
    "encoder_threads": "2",
    "num_episodes": "50",
    "dataset_namespace": "ethan",
    "dataset_new_name": "so101_pickup",
    "dataset_repo_id": "ethan/so101_overhead_test",
    "dataset_root": "data/so101_overhead_test",
    "policy_output_dir": "outputs/train/act_so101_overhead_ui",
    "policy_job_name": "act_so101_overhead_ui",
    "train_steps": "30000",
    "batch_size": "8",
    "log_freq": "50",
    "save_freq": "5000",
    "device": "mps",
    "policy_path": "outputs/train/act_so101_overhead_ui/checkpoints/last/pretrained_model",
    "rollout_duration": "20",
}


class ProcessManager:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.kind: str | None = None
        self.command: list[str] = []
        self.started_at: float | None = None
        self.logs: deque[str] = deque(maxlen=1200)
        self.lock = threading.Lock()

    def start(self, kind: str, command: list[str], pre_logs: list[str] | None = None) -> None:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                raise RuntimeError(f"{self.kind} is already running")

            command = resolve_command(command)
            env = command_env(kind)
            self.logs.clear()
            self.kind = kind
            self.command = command
            self.started_at = time.time()
            for line in pre_logs or []:
                self.logs.append(line)
            self.logs.append("$ " + " ".join(command))
            self.proc = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self.lock:
                self.logs.append(line.rstrip())
        code = self.proc.wait()
        with self.lock:
            self.logs.append(f"[process exited with code {code}]")

    def stop(self) -> None:
        with self.lock:
            proc = self.proc
            if not proc or proc.poll() is not None:
                return
            self.logs.append("[stopping process]")
            proc.send_signal(signal.SIGINT)

    def status(self) -> dict:
        with self.lock:
            running = bool(self.proc and self.proc.poll() is None)
            return {
                "running": running,
                "kind": self.kind,
                "command": self.command,
                "started_at": self.started_at,
                "logs": list(self.logs),
            }


MANAGER = ProcessManager()


def command_env(kind: str | None = None) -> dict:
    env = os.environ.copy()
    python_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")
    if kind == "record":
        env["LEROBOT_RECORD_CONTROLS_NO_OPEN"] = "1"
    return env


def resolve_command(command: list[str]) -> list[str]:
    exe = command[0]
    if "/" in exe:
        return command

    env = command_env()
    resolved = shutil.which(exe, path=env["PATH"])
    if resolved:
        return [resolved, *command[1:]]

    python_bin_candidate = Path(sys.executable).resolve().parent / exe
    if python_bin_candidate.exists():
        return [str(python_bin_candidate), *command[1:]]

    raise FileNotFoundError(
        f"Could not find {exe}. Launch this UI from the lerobot conda env: "
        "conda activate lerobot && python scripts/robot_training_ui.py"
    )


def load_presets() -> dict:
    if PRESETS_PATH.exists():
        data = json.loads(PRESETS_PATH.read_text())
        return {**DEFAULT_PRESETS, **data}
    return dict(DEFAULT_PRESETS)


def save_presets(data: dict) -> dict:
    merged = {**load_presets(), **data}
    PRESETS_PATH.write_text(json.dumps(merged, indent=2) + "\n")
    return merged


def resolved_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def archive_dataset(root: str) -> dict:
    ds_root = resolved_path(root)
    info_path = ds_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{ds_root} does not look like a LeRobot dataset")

    target_dir = ARCHIVE_DIR / "datasets"
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = target_dir / f"{ds_root.name}_{timestamp}"
    shutil.move(str(ds_root), str(target))
    return {"archived_to": display_path(target)}


def rename_dataset(root: str, new_name: str) -> dict:
    ds_root = resolved_path(root)
    info_path = ds_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{ds_root} does not look like a LeRobot dataset")

    clean_name = clean_folder_name(new_name)
    target = ds_root.with_name(clean_name)
    if target.exists():
        raise FileExistsError(f"{display_path(target)} already exists")

    try:
        info = json.loads(info_path.read_text())
    except Exception:
        info = {}
    old_repo_id = info.get("repo_id") or infer_repo_id(ds_root)
    namespace = old_repo_id.rsplit("/", 1)[0] if "/" in old_repo_id else None
    new_repo_id = f"{namespace}/{clean_name}" if namespace else clean_name

    shutil.move(str(ds_root), str(target))

    new_info_path = target / "meta" / "info.json"
    if new_info_path.exists():
        try:
            info = json.loads(new_info_path.read_text())
            info["repo_id"] = new_repo_id
            new_info_path.write_text(json.dumps(info, indent=4) + "\n")
        except Exception:
            pass

    return {"root": display_path(target), "repo_id": new_repo_id}


def clean_folder_name(name: str) -> str:
    cleaned = "".join(ch for ch in name.strip() if ch.isalnum() or ch in "._-")
    if not cleaned:
        raise ValueError("New name cannot be empty")
    return cleaned


def rename_policy_run(run_root: str, new_name: str) -> dict:
    source = resolved_path(run_root)
    try:
        source.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("Can only rename policy runs inside the LeRobot repo") from exc

    if not (source / "checkpoints").exists():
        raise FileNotFoundError(f"{source} does not look like a training run")

    target = source.with_name(clean_folder_name(new_name))
    if target.exists():
        raise FileExistsError(f"{display_path(target)} already exists")
    source.rename(target)
    return {"renamed_to": display_path(target)}


def archive_policy_run(run_root: str) -> dict:
    source = resolved_path(run_root)
    try:
        source.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("Can only archive policy runs inside the LeRobot repo") from exc

    if not (source / "checkpoints").exists():
        raise FileNotFoundError(f"{source} does not look like a training run")

    target_dir = ARCHIVE_DIR / "policies"
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = target_dir / f"{source.name}_{timestamp}"
    shutil.move(str(source), str(target))
    return {"archived_to": display_path(target)}


def remove_keys_recursive(value, keys_to_remove: set[str]) -> bool:
    changed = False
    if isinstance(value, dict):
        for key in list(value):
            if key in keys_to_remove:
                value.pop(key)
                changed = True
            else:
                changed = remove_keys_recursive(value[key], keys_to_remove) or changed
    elif isinstance(value, list):
        for item in value:
            changed = remove_keys_recursive(item, keys_to_remove) or changed
    return changed


def sanitize_policy_for_rollout(policy_path: str) -> list[str]:
    """Fix selected checkpoint configs that older LeRobot versions cannot parse."""
    model_dir = resolved_path(policy_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Policy path does not exist: {model_dir}")

    changed_files: list[str] = []
    for filename in ["config.json", "train_config.json"]:
        fpath = model_dir / filename
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text())
        if remove_keys_recursive(data, {"pretrained_revision"}):
            fpath.write_text(json.dumps(data, indent=4) + "\n")
            changed_files.append(display_path(fpath))

    if not changed_files:
        return []
    return [
        "[auto-fix] Removed unsupported pretrained_revision from selected policy checkpoint:",
        *[f"[auto-fix]   {path}" for path in changed_files],
    ]


def dataset_candidates() -> list[dict]:
    roots = [
        REPO_ROOT / "data",
        Path.home() / ".cache" / "huggingface" / "lerobot",
    ]
    out: list[dict] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for info_path in root.rglob("meta/info.json"):
            ds_root = info_path.parents[1]
            if ds_root in seen:
                continue
            seen.add(ds_root)
            try:
                info = json.loads(info_path.read_text())
            except Exception:
                info = {}
            root_label = display_path(ds_root)
            repo_id = info.get("repo_id") or infer_repo_id(ds_root)
            out.append(
                {
                    "repo_id": repo_id,
                    "root": root_label,
                    "episodes": info.get("total_episodes", "?"),
                    "frames": info.get("total_frames", "?"),
                    "modified": ds_root.stat().st_mtime,
                }
            )
    out.sort(key=lambda item: item["modified"], reverse=True)
    return out


def infer_repo_id(path: Path) -> str:
    parts = path.parts
    if "lerobot" in parts:
        idx = len(parts) - 2
        if idx >= 0:
            return f"{parts[-2]}/{parts[-1]}"
    return path.name


def policy_candidates() -> list[dict]:
    out: list[dict] = []
    for cfg in sorted((REPO_ROOT / "outputs" / "train").glob("*/checkpoints/*/pretrained_model/config.json")):
        model_dir = cfg.parent
        run_dir = model_dir.parents[2]
        out.append(
            {
                "path": display_path(model_dir),
                "run_root": display_path(run_dir),
                "run_name": run_dir.name,
                "checkpoint": model_dir.parent.name,
                "modified": model_dir.stat().st_mtime,
            }
        )
    for cfg in sorted((REPO_ROOT / "policies").glob("*/checkpoints/*/pretrained_model/config.json")):
        model_dir = cfg.parent
        run_dir = model_dir.parents[2]
        out.append(
            {
                "path": display_path(model_dir),
                "run_root": display_path(run_dir),
                "run_name": run_dir.name,
                "checkpoint": model_dir.parent.name,
                "modified": model_dir.stat().st_mtime,
            }
        )
    out.sort(key=lambda item: item["modified"], reverse=True)
    return out


def camera_arg(p: dict) -> str:
    name = p["camera_name"]
    return (
        "{ "
        f"{name}: {{type: opencv, index_or_path: {p['camera_index']}, "
        f"width: {p['camera_width']}, height: {p['camera_height']}, fps: {p['camera_fps']}}}"
        "}"
    )


def bool_arg(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "true" if str(value).lower() in {"true", "1", "yes", "on"} else "false"


def cmd_record(p: dict) -> list[str]:
    cmd = [
        "lerobot-record",
        f"--robot.type={p['robot_type']}",
        f"--robot.port={p['robot_port']}",
        f"--robot.id={p['robot_id']}",
        f"--robot.max_relative_target={p['max_relative_target']}",
        f"--robot.cameras={camera_arg(p)}",
        f"--teleop.type={p['teleop_type']}",
        f"--teleop.port={p['teleop_port']}",
        f"--teleop.id={p['teleop_id']}",
        f"--dataset.repo_id={p['dataset_repo_id']}",
        f"--dataset.num_episodes={p['num_episodes']}",
        "--dataset.episode_time_s=0",
        "--dataset.reset_time_s=0",
        f"--dataset.single_task={p['task']}",
        "--dataset.push_to_hub=false",
        "--dataset.streaming_encoding=true",
        f"--dataset.encoder_threads={p['encoder_threads']}",
        f"--display_data={bool_arg(p.get('display_data', 'true'))}",
    ]
    if p.get("dataset_root"):
        cmd.append(f"--dataset.root={p['dataset_root']}")
    if bool_arg(p.get("resume", "false")) == "true":
        cmd.append("--resume=true")
    return cmd


def cmd_train(p: dict) -> list[str]:
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={p['dataset_repo_id']}",
        f"--policy.type={p.get('policy_type', 'act')}",
        f"--policy.device={p['device']}",
        f"--output_dir={p['policy_output_dir']}",
        f"--job_name={p['policy_job_name']}",
        f"--steps={p['train_steps']}",
        f"--batch_size={p['batch_size']}",
        f"--log_freq={p['log_freq']}",
        f"--save_freq={p['save_freq']}",
        f"--dataset.image_transforms.enable={bool_arg(p.get('image_transforms', 'true'))}",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
    ]
    if p.get("dataset_root"):
        cmd.insert(2, f"--dataset.root={p['dataset_root']}")
    return cmd


def cmd_rollout(p: dict) -> list[str]:
    return [
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={p['policy_path']}",
        f"--robot.type={p['robot_type']}",
        f"--robot.port={p['robot_port']}",
        f"--robot.id={p['robot_id']}",
        f"--robot.max_relative_target={p.get('rollout_max_relative_target', p['max_relative_target'])}",
        f"--robot.cameras={camera_arg(p)}",
        f"--task={p['task']}",
        f"--duration={p['rollout_duration']}",
        f"--fps={p['camera_fps']}",
        f"--display_data={bool_arg(p.get('display_data', 'true'))}",
    ]


def cmd_teleoperate(p: dict) -> list[str]:
    return [
        "lerobot-teleoperate",
        f"--robot.type={p['robot_type']}",
        f"--robot.port={p['robot_port']}",
        f"--robot.id={p['robot_id']}",
        f"--robot.max_relative_target={p['max_relative_target']}",
        f"--teleop.type={p['teleop_type']}",
        f"--teleop.port={p['teleop_port']}",
        f"--teleop.id={p['teleop_id']}",
    ]


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SO101 Robot Training UI</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f6f7f9; color: #1f2937; }
    header { padding: 16px 22px; background: #111827; color: white; display: flex; align-items: center; justify-content: space-between; }
    main { padding: 18px; display: grid; grid-template-columns: var(--sidebar-width, 360px) 8px 1fr; gap: 12px; }
    main.sidebar-hidden { grid-template-columns: 0 8px 1fr; }
    main.sidebar-hidden #sidebar { display: none; }
    #sidebar { min-width: 260px; overflow: hidden; }
    #resizer { cursor: col-resize; border-radius: 6px; background: #d8dee8; min-height: 480px; }
    #resizer:hover { background: #9ca3af; }
    section { background: white; border: 1px solid #d7dce2; border-radius: 8px; padding: 14px; margin-bottom: 14px; }
    h2 { margin: 0 0 12px; font-size: 18px; }
    label { display: block; font-size: 12px; font-weight: 700; color: #4b5563; margin: 10px 0 4px; }
    .help { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; margin-left: 5px; border-radius: 50%; background: #dbeafe; color: #1d4ed8; font-size: 11px; font-weight: 800; cursor: help; vertical-align: middle; }
    input, select, textarea { width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #cfd5dd; border-radius: 6px; font-size: 14px; }
    textarea { height: 58px; }
    button { border: 0; border-radius: 6px; padding: 10px 12px; font-weight: 700; cursor: pointer; margin: 8px 6px 0 0; }
    .primary { background: #2563eb; color: white; }
    .train { background: #059669; color: white; }
    .danger { background: #dc2626; color: white; }
    .muted { background: #e5e7eb; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .tabs button { background: transparent; color: white; border: 1px solid #4b5563; }
    .tabs button.active { background: #2563eb; border-color: #2563eb; }
    .panel { display: none; }
    .panel.active { display: block; }
    pre { background: #0b1020; color: #d1e7ff; border-radius: 8px; padding: 12px; height: 360px; overflow: auto; white-space: pre-wrap; }
    .list { max-height: 260px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 6px; }
    .item { padding: 8px; border-bottom: 1px solid #edf0f3; cursor: pointer; }
    .item:hover { background: #f3f4f6; }
    .small { font-size: 12px; color: #6b7280; }
    .list-action { float: right; padding: 5px 7px; margin: 0 0 0 6px; font-size: 12px; }
    .status-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .sidebar-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .policy-run { padding: 8px; border-bottom: 1px solid #edf0f3; }
    .policy-run-head { cursor: pointer; }
    .checkpoint-row { padding: 7px 8px; margin-top: 6px; border-radius: 6px; cursor: pointer; background: #f8fafc; }
    .checkpoint-row:hover { background: #e5e7eb; }
    .checkpoint-row.selected { outline: 2px solid #2563eb; background: #eff6ff; }
    #sidebarReveal { display: none; position: fixed; left: 8px; bottom: 10px; z-index: 5; }
    main.sidebar-hidden + #sidebarReveal { display: block; }
    .modal-backdrop { display: none; position: fixed; inset: 0; background: rgba(17, 24, 39, 0.58); z-index: 10; align-items: center; justify-content: center; padding: 24px; }
    .modal-backdrop.open { display: flex; }
    .modal { width: min(960px, 96vw); height: min(720px, 88vh); background: white; border-radius: 8px; box-shadow: 0 24px 80px rgba(0,0,0,.32); display: grid; grid-template-rows: auto 1fr; overflow: hidden; }
    .modal-header { padding: 12px 14px; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .modal-header h2 { margin: 0; }
    iframe { width: 100%; height: 100%; border: 0; background: white; }
  </style>
</head>
<body>
  <header>
    <div><strong>SO101 Robot Training UI</strong></div>
    <div class="tabs">
      <button data-tab="collect" class="active">Collect</button>
      <button data-tab="teleoperate">Teleoperate</button>
      <button data-tab="train">Train</button>
      <button data-tab="rollout">Rollout</button>
    </div>
  </header>
  <main id="mainLayout">
    <div id="sidebar">
      <section>
        <div class="sidebar-head">
          <h2>Saved Parameters</h2>
          <button class="muted" onclick="hideSidebar()">Hide</button>
        </div>
        <button class="muted" onclick="savePreset()">Save Current Parameters</button>
        <button class="muted" onclick="loadData()">Reload Lists</button>
      </section>
      <section>
        <h2>Existing Datasets</h2>
        <div id="datasets" class="list"></div>
      </section>
      <section>
        <h2>Policies</h2>
        <div id="policies" class="list"></div>
      </section>
    </div>
    <div id="resizer" title="Drag to resize sidebar"></div>
    <div>
      <section id="collect" class="panel active">
        <h2>Collect Data</h2>
        <div class="row">
          <div><label>Mode</label><select id="resume" onchange="updateCollectMode()"><option value="false">Create new dataset</option><option value="true">Add to existing dataset</option></select></div>
          <div><label>Episodes to collect</label><input id="num_episodes"></div>
        </div>
        <div id="newDatasetFields">
          <div class="row">
            <div><label>Dataset name</label><input id="dataset_new_name" placeholder="so101_pickup"></div>
            <div><label>Namespace</label><input id="dataset_namespace" placeholder="ethan"></div>
          </div>
          <div class="small">This will save as <span id="newDatasetPreview"></span></div>
        </div>
        <div id="existingDatasetFields">
          <div class="row">
            <div><label>Dataset repo id</label><input id="dataset_repo_id"></div>
            <div><label>Dataset root</label><input id="dataset_root"></div>
          </div>
          <div class="small">Pick a dataset from the left list to fill these fields.</div>
        </div>
        <label>Task</label><textarea id="task"></textarea>
        <div class="row">
          <div><label>Robot max relative target</label><input id="max_relative_target"></div>
          <div><label>Encoder threads</label><input id="encoder_threads"></div>
        </div>
        <h2>Robot / Camera</h2>
        <div class="row">
          <div><label>Follower port</label><input id="robot_port"></div>
          <div><label>Leader port</label><input id="teleop_port"></div>
        </div>
        <div class="row">
          <div><label>Camera index</label><input id="camera_index"></div>
          <div><label>Display data</label><select id="display_data"><option>true</option><option>false</option></select></div>
        </div>
        <button class="primary" onclick="startJob('record')">Start Collecting</button>
        <button class="danger" onclick="stopJob()">Stop</button>
      </section>

      <section id="teleoperate" class="panel">
        <h2>Teleoperate</h2>
        <div class="row">
          <div><label>Follower port</label><input id="teleop_robot_port"></div>
          <div><label>Leader port</label><input id="teleop_leader_port"></div>
        </div>
        <div class="row">
          <div><label>Max relative target</label><input id="teleop_max_relative_target"></div>
          <div><label>Status</label><input value="Manual leader control" disabled></div>
        </div>
        <button class="primary" onclick="startJob('teleoperate')">Start Teleoperate</button>
        <button class="danger" onclick="stopJob()">Stop</button>
      </section>

      <section id="train" class="panel">
        <h2>Train Policy</h2>
        <div class="row">
          <div><label>Dataset repo id</label><input id="train_dataset_repo_id"></div>
          <div><label>Dataset root</label><input id="train_dataset_root"></div>
        </div>
        <div class="row">
          <div><label>Output dir</label><input id="policy_output_dir"></div>
          <div><label>Job name</label><input id="policy_job_name"></div>
        </div>
        <div class="row">
          <div><label>Steps</label><input id="train_steps"></div>
          <div><label>Batch size</label><input id="batch_size"></div>
        </div>
        <div class="row">
          <div><label>Device</label><select id="device"><option>mps</option><option>cuda</option><option>cpu</option></select></div>
          <div><label>Image transforms</label><select id="image_transforms"><option>true</option><option>false</option></select></div>
        </div>
        <div class="row">
          <div><label>Log freq</label><input id="log_freq"></div>
          <div><label>Save freq</label><input id="save_freq"></div>
        </div>
        <button class="train" onclick="startJob('train')">Start Training</button>
        <button class="danger" onclick="stopJob()">Stop</button>
      </section>

      <section id="rollout" class="panel">
        <h2>Run Policy</h2>
        <label>Policy path</label><input id="policy_path">
        <div class="row">
          <div><label>Duration seconds</label><input id="rollout_duration"></div>
          <div><label>Max relative target</label><input id="rollout_max_relative_target"></div>
        </div>
        <button class="primary" onclick="startJob('rollout')">Start Rollout</button>
        <button class="danger" onclick="stopJob()">Stop</button>
      </section>

      <section>
        <div class="status-head">
          <h2>Status</h2>
          <button class="muted" onclick="copyLogs()">Copy Output</button>
        </div>
        <div id="status" class="small"></div>
        <pre id="logs"></pre>
      </section>
    </div>
  </main>
  <button id="sidebarReveal" class="muted" onclick="showSidebar()">Show Lists</button>
  <div id="recordControlsModal" class="modal-backdrop">
    <div class="modal">
      <div class="modal-header">
        <div>
          <h2>Recording Controls</h2>
          <div class="small">Use Space inside this panel to finish task/reset. The live feed is embedded here.</div>
        </div>
        <div>
          <button class="muted" onclick="reloadRecordControls()">Reload</button>
          <button class="muted" onclick="hideRecordControls()">Hide</button>
        </div>
      </div>
      <iframe id="recordControlsFrame" title="Recording controls"></iframe>
    </div>
  </div>
<script>
const ids = ["robot_port","teleop_port","teleop_robot_port","teleop_leader_port","teleop_max_relative_target","camera_index","display_data","dataset_namespace","dataset_new_name","dataset_repo_id","dataset_root","resume","num_episodes","task","max_relative_target","encoder_threads","policy_output_dir","policy_job_name","train_steps","batch_size","device","image_transforms","log_freq","save_freq","policy_path","rollout_duration","rollout_max_relative_target"];
const mapTrain = {train_dataset_repo_id:"dataset_repo_id", train_dataset_root:"dataset_root"};
const expandedPolicyRuns = new Set(JSON.parse(localStorage.getItem("expandedPolicyRuns") || "[]"));
let selectedPolicyPath = "";
let currentPolicies = [];
const fieldDescriptions = {
  dataset_repo_id: "Dataset identifier stored in LeRobot metadata. Keep the username/name style, for example ethan/so101_pickup.",
  dataset_root: "Local folder where this dataset is saved. Use data/... for easy access and transfer.",
  resume: "Create a fresh dataset or append new episodes to the selected existing dataset.",
  dataset_new_name: "Short local name for a new dataset. Use letters, numbers, dashes, or underscores.",
  dataset_namespace: "Repo namespace stored in metadata. Usually your name, for example ethan.",
  num_episodes: "How many new demonstration episodes to record before stopping.",
  task: "Language task attached to every episode and passed to policies during rollout.",
  max_relative_target: "Maximum joint movement per control step. Higher is faster, lower is safer and smoother.",
  encoder_threads: "Number of video encoding threads. Higher can save faster but may make the Mac sluggish.",
  robot_port: "USB serial port for the follower arm controller.",
  teleop_port: "USB serial port for the leader arm controller.",
  camera_index: "OpenCV camera index. Your overhead USB camera has usually been camera 0.",
  display_data: "Whether to show Rerun visualization while recording or rolling out. Useful, but can slow things down.",
  teleop_robot_port: "Follower arm USB serial port used for manual leader/follower control.",
  teleop_leader_port: "Leader arm USB serial port used for manual leader/follower control.",
  teleop_max_relative_target: "Manual teleop speed/safety limit. Start low if the arm is acting unexpectedly.",
  train_dataset_repo_id: "Dataset identifier to train from. Pick a dataset from the left list to fill this.",
  train_dataset_root: "Local dataset folder to train from. This should contain meta, data, and videos folders.",
  policy_output_dir: "Folder where training checkpoints and logs will be written.",
  policy_job_name: "Human-readable name used in training logs.",
  train_steps: "Number of optimizer updates. More steps can improve fitting but takes longer and can overfit.",
  batch_size: "Examples per training step. Larger batches can be faster but require more memory.",
  device: "Compute backend for training. Use mps on Apple Silicon, cuda on NVIDIA, cpu only as a fallback.",
  image_transforms: "Applies visual augmentation during training to improve robustness to lighting and small camera changes.",
  log_freq: "How often training prints loss and speed metrics, in steps.",
  save_freq: "How often training saves checkpoints, in steps.",
  policy_path: "Specific pretrained_model checkpoint folder to run during rollout.",
  rollout_duration: "How many seconds the policy runs before stopping automatically.",
  rollout_max_relative_target: "Rollout speed/safety limit. Lower this when testing a new or questionable policy."
};

document.querySelectorAll(".tabs button").forEach(btn => btn.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  btn.classList.add("active");
  document.getElementById(btn.dataset.tab).classList.add("active");
});

function applyTooltips() {
  for (const [id, description] of Object.entries(fieldDescriptions)) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.title = description;
    const parentLabel = el.parentElement?.querySelector("label");
    const previousLabel = el.previousElementSibling?.tagName === "LABEL" ? el.previousElementSibling : null;
    const label = document.querySelector(`label[for="${id}"]`) || previousLabel || parentLabel;
    if (label) {
      label.title = description;
      if (!label.querySelector(".help")) {
        const help = document.createElement("span");
        help.className = "help";
        help.textContent = "?";
        help.title = description;
        label.appendChild(help);
      }
    }
  }
}

function cleanName(value) {
  const cleaned = String(value || "").trim().replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
  return cleaned || `so101_dataset_${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "")}`;
}

function updateCollectMode() {
  const isAppend = document.getElementById("resume").value === "true";
  document.getElementById("newDatasetFields").style.display = isAppend ? "none" : "block";
  document.getElementById("existingDatasetFields").style.display = isAppend ? "block" : "none";
  const name = cleanName(document.getElementById("dataset_new_name").value);
  const namespace = cleanName(document.getElementById("dataset_namespace").value || "ethan");
  document.getElementById("newDatasetPreview").textContent = `${namespace}/${name} -> data/${name}`;
}

document.addEventListener("input", (event) => {
  if (event.target && ["dataset_new_name", "dataset_namespace"].includes(event.target.id)) updateCollectMode();
});

function values(kind=null) {
  const out = {};
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) out[id] = el.value;
  }
  if (kind === "train") {
    out.dataset_repo_id = document.getElementById("train_dataset_repo_id").value || out.dataset_repo_id;
    out.dataset_root = document.getElementById("train_dataset_root").value || out.dataset_root;
  }
  if (kind === "record" && out.resume !== "true") {
    const name = cleanName(out.dataset_new_name);
    const namespace = cleanName(out.dataset_namespace || "ethan");
    out.dataset_new_name = name;
    out.dataset_namespace = namespace;
    out.dataset_repo_id = `${namespace}/${name}`;
    out.dataset_root = `data/${name}`;
  }
  if (kind === "teleoperate") {
    out.robot_port = document.getElementById("teleop_robot_port").value || out.robot_port;
    out.teleop_port = document.getElementById("teleop_leader_port").value || out.teleop_port;
    out.max_relative_target = document.getElementById("teleop_max_relative_target").value || out.max_relative_target;
  }
  return out;
}

function setValues(p) {
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el && p[id] !== undefined) el.value = p[id];
  }
  document.getElementById("train_dataset_repo_id").value = p.dataset_repo_id || "";
  document.getElementById("train_dataset_root").value = p.dataset_root || "";
  document.getElementById("teleop_robot_port").value = p.robot_port || "";
  document.getElementById("teleop_leader_port").value = p.teleop_port || "";
  document.getElementById("teleop_max_relative_target").value = p.max_relative_target || "10";
  if (!document.getElementById("rollout_max_relative_target").value) {
    document.getElementById("rollout_max_relative_target").value = p.max_relative_target || "5";
  }
  selectedPolicyPath = document.getElementById("policy_path").value || "";
  updateCollectMode();
}

async function api(path, body=null) {
  const opts = body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {};
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function loadData() {
  const data = await api("/api/state");
  setValues(data.presets);
  renderDatasets(data.datasets);
  renderPolicies(data.policies);
  renderStatus(data.status);
}

function renderDatasets(items) {
  const el = document.getElementById("datasets");
  el.innerHTML = items.map(d => {
    const payload = JSON.stringify(d).replaceAll("'", "&#39;");
    return `<div class="item" onclick='pickDataset(${payload})'>
      <button class="danger list-action" onclick='archiveDataset(event, ${payload})'>Archive</button>
      <button class="muted list-action" onclick='renameDataset(event, ${payload})'>Rename</button>
      <b>${d.repo_id}</b><br><span class="small">${d.root} | ${d.episodes} eps | ${d.frames} frames</span>
    </div>`;
  }).join("");
}

function renderPolicies(items) {
  currentPolicies = items;
  const el = document.getElementById("policies");
  const groups = {};
  for (const p of items) {
    if (!groups[p.run_root]) groups[p.run_root] = [];
    groups[p.run_root].push(p);
  }
  el.innerHTML = Object.entries(groups).map(([runRoot, policies]) => {
    const first = policies[0];
    const runPayload = JSON.stringify({run_root: runRoot, run_name: first.run_name}).replaceAll("'", "&#39;");
    const expanded = expandedPolicyRuns.has(runRoot);
    const rows = policies.map(p => {
      const payload = JSON.stringify(p).replaceAll("'", "&#39;");
      const selected = p.path === selectedPolicyPath ? " selected" : "";
      return `<div class="checkpoint-row${selected}" onclick='pickPolicy(event, ${payload})'>
        <b>${p.checkpoint}</b><br><span class="small">${p.path}</span>
      </div>`;
    }).join("");
    return `<div class="policy-run">
      <button class="muted list-action" onclick='renamePolicyRun(event, ${runPayload})'>Rename</button>
      <button class="danger list-action" onclick='archivePolicyRun(event, ${runPayload})'>Archive</button>
      <div class="policy-run-head" onclick='togglePolicyRun(${JSON.stringify(runRoot)})'>
        <b>${expanded ? "▾" : "▸"} ${first.run_name}</b><br>
        <span class="small">${runRoot} | ${policies.length} checkpoints</span>
      </div>
      <div style="${expanded ? "" : "display:none"}">${rows}</div>
    </div>`;
  }).join("");
}

function pickDataset(d) {
  document.getElementById("dataset_repo_id").value = d.repo_id;
  document.getElementById("dataset_root").value = d.root;
  document.getElementById("train_dataset_repo_id").value = d.repo_id;
  document.getElementById("train_dataset_root").value = d.root;
  document.getElementById("resume").value = "true";
  updateCollectMode();
}

function pickPolicy(event, p) {
  if (event) event.stopPropagation();
  selectedPolicyPath = p.path;
  document.getElementById("policy_path").value = p.path;
  renderPolicies(currentPolicies);
}

function togglePolicyRun(runRoot) {
  if (expandedPolicyRuns.has(runRoot)) expandedPolicyRuns.delete(runRoot);
  else expandedPolicyRuns.add(runRoot);
  localStorage.setItem("expandedPolicyRuns", JSON.stringify([...expandedPolicyRuns]));
  renderPolicies(currentPolicies);
}

async function archiveDataset(event, d) {
  event.stopPropagation();
  if (!confirm(`Archive dataset?\n\n${d.root}\n\nIt will be moved to ~/lerobot/archive/datasets and removed from this list.`)) return;
  try {
    await api("/api/archive_dataset", {root: d.root});
    await loadData();
  } catch (e) {
    alert(e.message);
  }
}

async function renameDataset(event, d) {
  event.stopPropagation();
  const currentName = d.root.split("/").filter(Boolean).pop() || d.repo_id.split("/").pop();
  const name = prompt("New dataset folder name:", currentName);
  if (!name || name === currentName) return;
  try {
    const result = await api("/api/rename_dataset", {root: d.root, new_name: name});
    if (document.getElementById("dataset_root").value === d.root) {
      document.getElementById("dataset_root").value = result.root;
      document.getElementById("dataset_repo_id").value = result.repo_id;
    }
    if (document.getElementById("train_dataset_root").value === d.root) {
      document.getElementById("train_dataset_root").value = result.root;
      document.getElementById("train_dataset_repo_id").value = result.repo_id;
    }
    await loadData();
  } catch (e) {
    alert(e.message);
  }
}

async function renamePolicyRun(event, p) {
  event.stopPropagation();
  const name = prompt("New policy run folder name:", p.run_name);
  if (!name || name === p.run_name) return;
  try {
    await api("/api/rename_policy_run", {run_root: p.run_root, new_name: name});
    await loadData();
  } catch (e) {
    alert(e.message);
  }
}

async function archivePolicyRun(event, p) {
  event.stopPropagation();
  if (!confirm(`Archive policy run?\n\n${p.run_root}\n\nIt will be moved to ~/lerobot/archive/policies and removed from this list.`)) return;
  try {
    await api("/api/archive_policy_run", {run_root: p.run_root});
    if (document.getElementById("policy_path").value.startsWith(p.run_root + "/")) {
      document.getElementById("policy_path").value = "";
      selectedPolicyPath = "";
    }
    expandedPolicyRuns.delete(p.run_root);
    localStorage.setItem("expandedPolicyRuns", JSON.stringify([...expandedPolicyRuns]));
    await loadData();
  } catch (e) {
    alert(e.message);
  }
}

async function savePreset() {
  await api("/api/presets", values());
  await loadData();
}

async function startJob(kind) {
  try {
    await api("/api/start/" + kind, values(kind));
    if (kind === "record") showRecordControls();
  } catch (e) {
    alert(e.message);
  }
  poll();
}

async function stopJob() {
  await api("/api/stop", {});
  poll();
}

async function copyLogs() {
  const text = document.getElementById("logs").textContent;
  await navigator.clipboard.writeText(text);
}

function renderStatus(s) {
  document.getElementById("status").textContent = s.running ? `Running ${s.kind}` : "Idle";
  document.getElementById("logs").textContent = (s.logs || []).join("\n");
  document.getElementById("logs").scrollTop = document.getElementById("logs").scrollHeight;
  const controlUrl = findRecordControlUrl(s.logs || []);
  if (controlUrl) showRecordControls(controlUrl);
  if (!s.running || s.kind !== "record") hideRecordControls(false);
}

async function poll() {
  const data = await api("/api/status");
  renderStatus(data);
}

function findRecordControlUrl(logs) {
  for (let i = logs.length - 1; i >= 0; i--) {
    const match = String(logs[i]).match(/LeRobot browser recording controls: (http:\/\/127\.0\.0\.1:\d+)/);
    if (match) return match[1];
  }
  return null;
}

function showRecordControls(url=null) {
  const modal = document.getElementById("recordControlsModal");
  const frame = document.getElementById("recordControlsFrame");
  if (url) {
    if (frame.dataset.controlUrl !== url) {
      frame.dataset.controlUrl = url;
      frame.removeAttribute("srcdoc");
      frame.src = url;
    }
  } else if (!frame.dataset.controlUrl) {
    frame.src = "about:blank";
    frame.srcdoc = "<body style='font-family:-apple-system,BlinkMacSystemFont,sans-serif;padding:24px'><h2>Starting recording controls...</h2><p>The controls and live feed will appear here when the recorder is ready.</p></body>";
  }
  modal.classList.add("open");
}

function hideRecordControls(clear=true) {
  const modal = document.getElementById("recordControlsModal");
  modal.classList.remove("open");
  if (clear) {
    const frame = document.getElementById("recordControlsFrame");
    frame.dataset.controlUrl = "";
    frame.removeAttribute("srcdoc");
    frame.src = "about:blank";
  }
}

function reloadRecordControls() {
  const frame = document.getElementById("recordControlsFrame");
  const src = frame.src || "http://127.0.0.1:8765";
  frame.src = src;
}

function hideSidebar() {
  document.getElementById("mainLayout").classList.add("sidebar-hidden");
  localStorage.setItem("sidebarHidden", "true");
}

function showSidebar() {
  document.getElementById("mainLayout").classList.remove("sidebar-hidden");
  localStorage.setItem("sidebarHidden", "false");
}

function initSidebar() {
  const savedWidth = localStorage.getItem("sidebarWidth");
  if (savedWidth) document.documentElement.style.setProperty("--sidebar-width", savedWidth + "px");
  if (localStorage.getItem("sidebarHidden") === "true") hideSidebar();
  const resizer = document.getElementById("resizer");
  let dragging = false;
  resizer.addEventListener("mousedown", () => {
    dragging = true;
    document.body.style.userSelect = "none";
  });
  window.addEventListener("mousemove", (event) => {
    if (!dragging) return;
    const width = Math.max(260, Math.min(720, event.clientX - 18));
    document.documentElement.style.setProperty("--sidebar-width", width + "px");
    localStorage.setItem("sidebarWidth", String(width));
    showSidebar();
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    document.body.style.userSelect = "";
  });
}

initSidebar();
applyTooltips();
loadData();
setInterval(poll, 1200);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(HTML)
        elif parsed.path == "/api/state":
            self.send_json(
                {
                    "presets": load_presets(),
                    "datasets": dataset_candidates(),
                    "policies": policy_candidates(),
                    "status": MANAGER.status(),
                }
            )
        elif parsed.path == "/api/status":
            self.send_json(MANAGER.status())
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            parsed = urlparse(self.path)
            if parsed.path == "/api/presets":
                self.send_json(save_presets(body))
            elif parsed.path == "/api/start/record":
                presets = save_presets(body)
                MANAGER.start("record", cmd_record(presets))
                self.send_json({"ok": True})
            elif parsed.path == "/api/start/train":
                presets = save_presets(body)
                MANAGER.start("train", cmd_train(presets))
                self.send_json({"ok": True})
            elif parsed.path == "/api/start/rollout":
                presets = save_presets(body)
                pre_logs = sanitize_policy_for_rollout(presets["policy_path"])
                MANAGER.start("rollout", cmd_rollout(presets), pre_logs=pre_logs)
                self.send_json({"ok": True})
            elif parsed.path == "/api/start/teleoperate":
                presets = save_presets(body)
                MANAGER.start("teleoperate", cmd_teleoperate(presets))
                self.send_json({"ok": True})
            elif parsed.path == "/api/stop":
                MANAGER.stop()
                self.send_json({"ok": True})
            elif parsed.path == "/api/archive_dataset":
                self.send_json(archive_dataset(body["root"]))
            elif parsed.path == "/api/rename_dataset":
                self.send_json(rename_dataset(body["root"], body["new_name"]))
            elif parsed.path == "/api/rename_policy_run":
                self.send_json(rename_policy_run(body["run_root"], body["new_name"]))
            elif parsed.path == "/api/archive_policy_run":
                self.send_json(archive_policy_run(body["run_root"]))
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def log_message(self, fmt: str, *args) -> None:
        return

    def send_html(self, html: str) -> None:
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Robot training UI: {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        MANAGER.stop()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
