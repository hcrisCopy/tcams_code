"""Disjoint layer x projection interventions using the realized checkpoint delta."""
from contextlib import contextmanager
from pathlib import Path

import torch
from tqdm import tqdm

from .common import finite, load_tensor, read_json, save_json, save_tensor
from .models import PROJECTIONS, load_model


def extract_groups(cfg):
    root = Path(cfg['run_dir'])
    target, _ = load_model(cfg['target'], cfg, device='cpu', llama_only=True)
    fused, _ = load_model(root / 'fused', cfg, device='cpu', llama_only=True)
    base = dict(target.named_parameters())
    merged = dict(fused.named_parameters())
    if base.keys() != merged.keys():
        raise ValueError('Target/fused parameter names differ.')
    strength = read_json(root / 'activation_strength.json')
    groups, expected = [], {}
    for layer in range(target.config.num_hidden_layers):
        for kind, attr in PROJECTIONS.items():
            name = f'model.layers.{layer}.self_attn.{attr}.weight'
            expected[name] = (layer, kind)
    unchanged = 0
    for name in tqdm(base, desc='Validate and reconstruct delta'):
        a, b = base[name].detach(), merged[name].detach()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f'Parameter structure changed: {name}')
        finite(b, name)
        if name not in expected:
            if not torch.equal(a, b):
                raise ValueError(f'Unexpected changed parameter: {name}')
            unchanged += a.numel()
            continue
        layer, kind = expected[name]
        axis = 1 if kind == 'O' else 0
        ids = load_tensor(root / 'top_neurons' / f'top_neurons_{kind}_layer_{layer}.pt')
        before, after = a.index_select(axis, ids), b.index_select(axis, ids)
        delta = after.float() - before.float()
        reconstructed = a.clone()
        reconstructed.index_copy_(axis, ids, (before.float() + delta).to(a.dtype))
        if not torch.equal(reconstructed, b):
            raise ValueError(f'Exact delta reconstruction failed: {name}')
        gid = f'layer_{layer:03d}_{kind}'
        save_tensor(root / 'delta' / f'{gid}.pt',
                    {'target': before, 'fused': after, 'delta': delta, 'indices': ids, 'axis': axis, 'name': name})
        groups.append({'id': gid, 'name': name, 'layer': layer, 'kind': kind,
                       'axis': axis, 'parameter_budget': delta.numel(),
                       'changed_parameters': int(torch.count_nonzero(delta)),
                       'delta_norm': float(delta.norm()), 'activation': strength[gid],
                       'distance_from_target_shrink_only': float((after.float() -
                           ((1 - cfg['alpha']) * before).float()).norm()),
                       'ot_post': f'transport/hot_{kind}.pt', 'ot_pre': f'transport/hot_{kind}_pre.pt'})
    if cfg['alpha'] > 0 and not any(g['changed_parameters'] for g in groups):
        raise ValueError('Positive-alpha fusion changed no parameters.')
    save_json(root / 'delta/groups.json', groups)
    save_json(root / 'plans/shrink_control.json', {'split': 'test', 'conditions': [
        {'name': 'fused', 'groups': []},
        {'name': 'shrink_only', 'groups': [g['id'] for g in groups], 'mode': 'shrink', 'fraction': cfg['alpha']}],
        'purpose': 'Check whether the realized source contribution differs from shrinking selected target slices.'})
    save_json(root / 'delta/verification.json', {'exact_reconstruction': True,
                                               'unchanged_parameter_count': unchanged,
                                               'groups': len(groups), 'overlap': False})
    # Test in-memory patch and bit-exact restoration on the actual fused checkpoint.
    for g in groups:
        param = dict(fused.named_parameters())[g['name']]
        original = param.detach().clone()
        with rollback_groups(fused, root / 'delta', [g['id']]):
            stored = load_tensor(root / 'delta' / f'{g["id"]}.pt')
            if not torch.equal(param.index_select(stored['axis'], stored['indices']), stored['target']):
                raise ValueError('Rollback did not restore target slices.')
        if not torch.equal(original, param):
            raise ValueError('Patch polluted the fused checkpoint.')


@contextmanager
def rollback_groups(model, directory, group_ids, mode='rollback', fraction=0.0):
    """Restore exact target slices, then exact fused slices; avoid subtraction drift."""
    if len(group_ids) != len(set(group_ids)):
        raise ValueError('Overlapping/repeated group IDs.')
    params = dict(model.named_parameters())
    backups = []
    try:
        with torch.no_grad():
            for gid in group_ids:
                obj = load_tensor(Path(directory) / f'{gid}.pt')
                parameter = params[obj['name']]
                ids = obj['indices'].to(parameter.device)
                current = parameter.index_select(obj['axis'], ids)
                expected = obj['fused'].to(parameter.device, parameter.dtype)
                if not torch.equal(current, expected):
                    raise ValueError('Patch base is not the recorded fused checkpoint.')
                backups.append((parameter, obj['axis'], ids, current.clone()))
                if mode == 'rollback':
                    replacement = obj['target']
                elif mode == 'scale_delta':
                    replacement = (obj['target'].float() + fraction * obj['delta']).to(obj['target'].dtype)
                elif mode == 'shrink':
                    replacement = (1 - fraction) * obj['target']
                else:
                    raise ValueError(f'Unknown intervention mode: {mode}')
                parameter.index_copy_(obj['axis'], ids, replacement.to(parameter.device, parameter.dtype))
        yield
    finally:
        with torch.no_grad():
            for parameter, axis, ids, value in reversed(backups):
                parameter.index_copy_(axis, ids, value)
                if not torch.equal(parameter.index_select(axis, ids), value):
                    raise RuntimeError('Patch restoration failed.')


def probe_conditions(cfg):
    root = Path(cfg['run_dir'])
    groups = read_json(root / 'delta/groups.json')
    # Stratified deterministic coverage avoids selecting candidates on test outcomes.
    by_kind = {k: sorted([g for g in groups if g['kind'] == k], key=lambda g: g['delta_norm'], reverse=True)
               for k in PROJECTIONS}
    chosen = []
    for index in range(max(map(len, by_kind.values()))):
        for kind in PROJECTIONS:
            if index < len(by_kind[kind]) and len(chosen) < cfg['candidate_count']:
                chosen.append(by_kind[kind][index])
    conditions = [{'name': 'fused', 'groups': []}]
    conditions.extend({'name': 'remove_' + g['id'], 'groups': [g['id']]} for g in chosen)
    save_json(root / 'plans/probe.json', {'split': 'probe', 'conditions': conditions,
                                        'candidate_policy': 'round-robin projection kind, descending delta norm'})
    return conditions
