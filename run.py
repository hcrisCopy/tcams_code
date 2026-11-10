"""All commands are launched from tcams_code; computation is offline by default."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tcams.common import (Resources, bind_run, bind_stage, config, file_hash,
                          launch_workers, model_identity, read_json, save_json, seed_all)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['inspect', 'prepare', 'activations', 'transport', 'fuse',
                                    'delta', 'probe-plan', 'evaluate', 'judge', 'select', 'summarize', 'audit'])
    p.add_argument('--config', default='configs/pilot_thai.yaml')
    p.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    p.add_argument('--model', choices=['source', 'target', 'fused'], default='fused')
    p.add_argument('--split', choices=['probe', 'test'], default='test')
    p.add_argument('--limit', type=int, default=-1, help='Per dataset/language/task; -1 means all.')
    p.add_argument('--plan', default=None)
    p.add_argument('--conditions', default='source,target,fused')
    p.add_argument('--extra-thai-benign', default=None)
    p.add_argument('--clean', action='store_true', help='Delete this marked run only, including all downstream artifacts.')
    p.add_argument('--worker-rank', type=int, default=-1, help=argparse.SUPPRESS)
    p.add_argument('--world-size', type=int, default=1, help=argparse.SUPPRESS)
    return p


def inspect(cfg):
    import importlib.metadata
    import platform
    import shutil
    import subprocess
    import torch
    from transformers import AutoConfig
    from tcams.common import official_modules
    official_modules(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Inspect is intended for the remote GPU server.')
    ids = {}
    for role in ('source', 'target', 'judge'):
        identity = model_identity(cfg[role])
        identity['weight_sha256'] = {p.name: file_hash(p) for p in sorted(Path(cfg[role]).glob('*.safetensors'))}
        ids[role] = identity
    a = AutoConfig.from_pretrained(cfg['target'], local_files_only=True)
    b = AutoConfig.from_pretrained(cfg['source'], local_files_only=True)
    if a.model_type != 'llama' or b.model_type != 'llama':
        raise ValueError('This adapter has only been audited for Llama.')
    def widths(c):
        hd = getattr(c, 'head_dim', c.hidden_size // c.num_attention_heads)
        return [c.num_attention_heads * hd, c.num_key_value_heads * hd,
                c.num_key_value_heads * hd, c.hidden_size]
    qa, qb = widths(a), widths(b)
    qbytes = a.num_hidden_layers * b.num_hidden_layers * 4 * (
        sum(x * y for x, y in zip(qa, qb, strict=True)) + 4 * a.hidden_size * b.hidden_size)
    result = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'packages': {p: importlib.metadata.version(p) for p in ['transformers', 'accelerate', 'datasets']},
              'gpus': [{'name': torch.cuda.get_device_name(i),
                        'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 2**30}
                       for i in range(torch.cuda.device_count())],
              'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
              'disk_free_gb': shutil.disk_usage(cfg['data_root']).free / 2**30,
              'models': ids, 'official_commit': cfg['official_commit'],
              'estimated_dense_Q_gb_one_copy': qbytes / 2**30,
              'resource_note': 'Q pair cache + assembled bundles need about twice Q storage, plus models and activations. CPU assembly loads one module at a time.'}
    save_json(Path(cfg['run_dir']) / 'environment.json', result)
    print(f'Q storage estimate: {qbytes / 2**30:.2f} GiB per copy; model/config fingerprints saved.')


def bind_inputs(cfg, args):
    root = Path(cfg['run_dir'])
    code = {str(p): file_hash(p) for p in sorted(Path('tcams').glob('*.py'))}
    code['run.py'] = file_hash('run.py')
    stage = args.stage
    signature = {'code': code, 'config': cfg}
    if stage not in ('inspect', 'prepare'):
        data_signature = {
            'manifest': read_json(root / 'data/manifest.json'),
            'files': {name: file_hash(root / 'data' / name) for name in ('align.jsonl', 'eval.jsonl')}}
        base_models = {role: model_identity(cfg[role]) for role in ('source', 'target', 'judge')}
        bind_stage(root, 'pipeline_inputs', {**signature, 'data': data_signature, 'models': base_models})
        if stage in ('evaluate', 'judge'):
            from tcams.evaluation import evaluation_spec
            rows, conditions = evaluation_spec(cfg, args)
            model = cfg[args.model] if args.model != 'fused' else root / 'fused'
            for condition in conditions:
                item = {**signature, 'data': data_signature,
                        'model': model_identity(model), 'condition': condition}
                if stage == 'judge':
                    item['judge'] = model_identity(cfg['judge'])
                bind_stage(root, f'{stage}_{args.split}_{condition["name"]}', item)
            return
        signature['data'] = data_signature
        signature['models'] = base_models
    bind_stage(root, stage, signature)


def execute_worker(cfg, args):
    from tcams.ot_adapter import activation_worker, fuse, transport_worker
    from tcams.evaluation import evaluate_worker, judge_worker
    if args.stage == 'activations':
        activation_worker(cfg, args.worker_rank, args.world_size)
    elif args.stage == 'transport':
        transport_worker(cfg, args.worker_rank, args.world_size)
    elif args.stage == 'fuse':
        fuse(cfg)
    elif args.stage == 'evaluate':
        evaluate_worker(cfg, args)
    elif args.stage == 'judge':
        judge_worker(cfg, args)
    else:
        raise ValueError('Unsupported GPU worker stage.')


def main():
    args = parser().parse_args()
    cfg = config(args.config)
    seed_all()
    resources = Resources()
    root = Path(cfg['run_dir'])
    if args.worker_rank >= 0:
        try:
            execute_worker(cfg, args)
        finally:
            save_json(root / 'resources' / f'{args.stage}_{args.model}_{args.split}_rank{args.worker_rank}.json', resources.report())
        return
    if args.clean and args.stage != 'prepare':
        raise ValueError('--clean is only accepted by prepare; it resets the entire marked run.')
    bind_run(cfg, clean=args.clean)
    bind_inputs(cfg, args)
    if args.stage in ('activations', 'transport', 'fuse', 'evaluate', 'judge'):
        devices = args.devices.split(',')
        if not devices or len(devices) != len(set(devices)) or any(not x.isdigit() for x in devices):
            raise ValueError('--devices must be unique integer CUDA device indices.')
        if args.stage == 'fuse':
            devices = devices[:1]
            print('Fusion uses the upstream single-GPU function; activation/OT/evaluation stages shard across all selected GPUs.')
        argv = [x for x in sys.argv[1:] if x != '--clean']
        launch_workers(argv, devices)
        from tcams.ot_adapter import finish_activations, finish_transport
        if args.stage == 'activations':
            finish_activations(cfg)
        elif args.stage == 'transport':
            finish_transport(cfg)
    elif args.stage == 'inspect':
        inspect(cfg)
    elif args.stage == 'prepare':
        from tcams.data import prepare
        prepare(cfg, args.extra_thai_benign)
    elif args.stage == 'delta':
        from tcams.delta import extract_groups
        extract_groups(cfg)
    elif args.stage == 'probe-plan':
        from tcams.delta import probe_conditions
        probe_conditions(cfg)
    elif args.stage == 'select':
        from tcams.analysis import select_plan
        select_plan(cfg, args.limit)
    elif args.stage in ('summarize', 'audit'):
        from tcams.analysis import export_audit, summarize
        names = args.conditions.split(',')
        if args.plan:
            plan = read_json(args.plan)
            if plan['split'] != args.split:
                raise ValueError('Summary plan split mismatch.')
            names = [c['name'] for c in plan['conditions']]
        if args.stage == 'summarize':
            summarize(cfg, args.split, names, args.limit)
        else:
            export_audit(cfg, args.split, names, args.limit)
    save_json(root / 'resources' / f'{args.stage}_parent.json', resources.report())
    print(f'Completed {args.stage}; artifacts: {root}')


if __name__ == '__main__':
    main()
