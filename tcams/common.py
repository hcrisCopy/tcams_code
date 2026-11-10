"""Strict artifact handling, fixed-seed execution, and independent GPU workers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone

import numpy as np
import torch
import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sample_filename(sample_id):
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', sample_id).strip('._')
    if not name or len(name) > 180:
        raise ValueError('Sample ID cannot form a short, readable artifact name.')
    return name + '.json'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def load_tensor(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if 'extends' in cfg:
        parent = config(cfg.pop('extends'))
        cfg = {**parent, **cfg}
    if cfg['seed'] != 42:
        raise ValueError('This project fixes seed=42.')
    for key in ('data_root', 'run_dir', 'official_repo', 'source', 'target', 'judge'):
        if Path(cfg[key]).is_absolute():
            raise ValueError(f'{key} must be relative to tcams_code.')
    root = Path(cfg['data_root']).resolve()
    output = Path(cfg['run_dir']).resolve()
    if root not in output.parents or output == root:
        raise ValueError('run_dir must be a child of data_root.')
    return cfg


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f'Nonfinite values: {name}')


def bind_run(cfg, clean=False):
    root = Path(cfg['run_dir'])
    marker = root / 'run_config.json'
    if clean and root.exists():
        if not marker.is_file():
            raise ValueError('Refusing to clean a directory without a TCAMS run marker.')
        old = read_json(marker)
        if old.get('run_dir') != cfg['run_dir']:
            raise ValueError('Run marker path mismatch.')
        resolved = root.resolve()
        if Path(cfg['data_root']).resolve() not in resolved.parents:
            raise ValueError('Cleanup escaped data_root.')
        shutil.rmtree(resolved)
    if marker.exists() and read_json(marker) != cfg:
        raise ValueError('Configuration changed; use another run_dir or explicit --clean.')
    save_json(marker, cfg)
    return root


def bind_stage(root, stage, signature):
    path = Path(root) / 'manifests' / f'{stage}.json'
    if path.exists() and read_json(path) != signature:
        raise ValueError(f'{stage}: inputs changed; use a new run_dir or --clean.')
    save_json(path, signature)


def model_identity(path):
    path = Path(path)
    files = sorted(list(path.glob('*.safetensors')) + list(path.glob('*.bin')))
    if not files:
        raise FileNotFoundError(f'No downloaded model weights in {path}')
    # Hash weights once during inspect; subsequent stages verify size/mtime signatures.
    names = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'generation_config.json']
    return {'path': str(path), 'metadata': {n: file_hash(path / n) for n in names if (path / n).exists()},
            'weights': {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in files}}


def official_modules(cfg):
    repo = Path(cfg['official_repo'])
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    if head != cfg['official_commit'] or dirty:
        raise RuntimeError('Official repository must be the clean, pinned revision.')
    sys.path.insert(0, str(repo.resolve()))
    from core import hot_transport_chunk, generate_hot_residual
    return hot_transport_chunk, generate_hot_residual


def launch_workers(args, devices):
    """One independent process per GPU; no distributed collectives or rank barriers."""
    workers = []
    try:
        for rank, device in enumerate(devices):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            cmd = [sys.executable, 'run.py', *args, '--worker-rank', str(rank), '--world-size', str(len(devices))]
            workers.append(subprocess.Popen(cmd, env=env))
        while any(p.poll() is None for p in workers):
            failures = [p.returncode for p in workers if p.poll() not in (None, 0)]
            if failures:
                raise RuntimeError(f'GPU worker failed: {failures}')
            time.sleep(0.5)
        if any(p.returncode != 0 for p in workers):
            raise RuntimeError('GPU worker failed; no aggregation was performed.')
    finally:
        for p in workers:
            if p.poll() is None:
                p.terminate()
        for p in workers:
            p.wait()


class Resources:
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.start = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.peak_ram = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def sample(self):
        self.peak_ram = max(self.peak_ram, self.process.memory_info().rss)

    def _monitor(self):
        while not self._stop.wait(0.25):
            self.sample()

    def report(self):
        self._stop.set()
        self._thread.join()
        self.sample()
        return {'started_at_utc': self.started_at, 'elapsed_seconds': time.monotonic() - self.start,
                'sampled_peak_cpu_ram_gb': self.peak_ram / 2**30,
                'peak_allocated_gb': torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0,
                'peak_reserved_gb': torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
