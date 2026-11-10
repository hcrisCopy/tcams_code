"""Download only this pilot's selected assets, with immutable revision records.

This is the only network-enabled project entry point. It executes hf download
for Hub assets and git for the small publisher repositories. Reruns reuse the
resolved revision lock and the official download caches.
"""
import argparse
import os
from pathlib import Path
import subprocess

from huggingface_hub import HfApi

from tcams.common import config, read_json, save_json


def git_checkout(url, destination, revision):
    destination = Path(destination)
    if not destination.exists():
        subprocess.run(['git', 'clone', url, str(destination)], check=True)
    remote = subprocess.check_output(['git', '-C', str(destination), 'remote', 'get-url', 'origin'], text=True).strip()
    if remote.removesuffix('.git') != url.removesuffix('.git'):
        raise ValueError(f'Unexpected upstream repository: {destination}')
    if subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain'], text=True).strip():
        raise ValueError(f'Refusing to modify a dirty repository: {destination}')
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', revision], check=True)
    return subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/pilot_thai.yaml')
    args = p.parse_args()
    cfg = config(args.config)
    if os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/') != 'https://huggingface.co':
        raise ValueError('Mirrors are not permitted; unset HF_ENDPOINT.')
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    raw = Path(cfg['data_root']) / 'datasets'
    lock_path = Path(cfg['data_root']) / 'asset_revisions.json'
    lock = read_json(lock_path) if lock_path.exists() else {}
    api = HfApi(endpoint='https://huggingface.co')
    pinned = {'DAMO-NLP-SG/MultiJail': '93d65c778925d55973b9ff30c7bccc8545494eb6',
              'facebook/belebele': '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2',
              'typhoon-ai/ThaiSafetyBench': '4330735f39fbe3cbdc17aa437e378811621fd760',
              'Qwen/Qwen3Guard-Gen-0.6B': 'fada3b2f655b89601929198343c94cd2f64d93cc'}
    assets = [
        ('model', cfg['source_model_id'], cfg['source']),
        ('model', cfg['target_model_id'], cfg['target']),
        ('model', cfg['judge_model_id'], cfg['judge']),
        ('dataset', 'DAMO-NLP-SG/MultiJail', str(raw / 'DAMO-NLP-SG/MultiJail')),
        ('dataset', 'facebook/belebele', str(raw / 'facebook/belebele')),
        ('dataset', 'typhoon-ai/ThaiSafetyBench', str(raw / 'typhoon-ai/ThaiSafetyBench')),
        ('dataset', 'ChavyvAkvar/fineweb-2-1M-Sample-Thai', str(raw / 'ChavyvAkvar/fineweb-2-1M-Sample-Thai')),
    ]
    # Model paths must preserve publisher/repository names, as requested.
    for kind, repo, destination in assets:
        if kind == 'model' and Path(destination).as_posix() != '../' + repo:
            raise ValueError('Model path differs from the explicit asset list; update the selected repo ID first.')
        key = kind + ':' + repo
        if key not in lock:
            info = api.repo_info(repo_id=repo, repo_type=kind, revision=pinned.get(repo))
            card = info.card_data.to_dict() if info.card_data is not None else {}
            lock[key] = {'revision': info.sha, 'license': card.get('license', 'not declared'), 'repo': repo}
            save_json(lock_path, lock)
        cmd = ['hf', 'download', repo, '--repo-type', kind, '--revision', lock[key]['revision'], '--local-dir', destination]
        if repo == 'facebook/belebele':
            cmd.extend(['--include', 'data/tha_Thai.jsonl', 'README.md'])
        print(' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    repos = [
        ('https://github.com/chenhangcuisg-code/Cross-Architecture-Merging-for-Large-Language-Models.git',
         cfg['official_repo'], cfg['official_commit']),
        ('https://github.com/google-research-datasets/MGSM-Rev2.git',
         str(raw / 'MGSM-Rev2'), '5f091edeae66ebc605ac4f98dca07a6c345f0060'),
        ('https://github.com/paul-rottger/xstest.git', str(raw / 'XSTest'),
         'd7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d'),
    ]
    for url, destination, revision in repos:
        key = 'git:' + url
        if key in lock:
            revision = lock[key]['revision']
        elif revision is None:
            revision = subprocess.check_output(['git', 'ls-remote', url, 'HEAD'], text=True).split()[0]
        sha = git_checkout(url, destination, revision)
        lock[key] = {'revision': sha, 'url': url}
        save_json(lock_path, lock)
    print(f'Assets ready; resolved revisions: {lock_path}')


if __name__ == '__main__':
    main()
