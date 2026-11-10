"""Use upstream correlation, Sinkhorn and fusion functions without training."""
from pathlib import Path
import hashlib

import torch
from tqdm import tqdm
from transformers import AutoConfig

from .common import finite, load_tensor, official_modules, read_jsonl, save_json, save_tensor
from .models import KINDS, PROJECTIONS, activation_batches, assemble_activations, load_model


def activation_worker(cfg, rank, world):
    root = Path(cfg['run_dir'])
    rows = read_jsonl(root / 'data/align.jsonl')
    for role in ('target', 'source'):
        model, tokenizer = load_model(cfg[role], cfg, llama_only=True)
        activation_batches(model, tokenizer, rows, root / 'activations' / role, cfg, rank, world)
        del model, tokenizer
        torch.cuda.empty_cache()


def finish_activations(cfg):
    root = Path(cfg['run_dir'])
    rows = read_jsonl(root / 'data/align.jsonl')
    for role in ('target', 'source'):
        layers = AutoConfig.from_pretrained(cfg[role], local_files_only=True).num_hidden_layers
        assemble_activations(root / 'activations' / role, rows, cfg, layers)
        if role == 'target':
            stats = {}
            for layer in range(layers):
                for kind in PROJECTIONS:
                    # Preserve upstream O-post ranking, even though fusion writes O columns.
                    acts = load_tensor(root / 'activations/target' / f'layer_{layer:03d}_{kind}.pt')
                    strength = acts.abs().mean(0)
                    if cfg['top_k'] > strength.numel():
                        raise ValueError('top_k exceeds the projection dimension.')
                    ids = strength.topk(cfg['top_k'], sorted=False).indices.unique()
                    save_tensor(root / 'top_neurons' / f'top_neurons_{kind}_layer_{layer}.pt', ids)
                    stats[f'layer_{layer:03d}_{kind}'] = float(strength[ids].mean())
            save_json(root / 'activation_strength.json', stats)


def marginal_report(plan, tolerance, label):
    finite(plan, label)
    if plan.ndim != 2 or (plan < 0).any():
        raise ValueError(f'Invalid transport plan: {label}')
    row_error = float((plan.sum(1) - 1 / plan.shape[0]).abs().sum())
    col_error = float((plan.sum(0) - 1 / plan.shape[1]).abs().sum())
    if max(row_error, col_error) > tolerance:
        raise ValueError(f'{label}: marginal error {row_error}, {col_error} exceeds {tolerance}')
    return {'row_l1': row_error, 'column_l1': col_error, 'mass': float(plan.sum())}


def normalized(acts):
    finite(acts, 'activations')
    if acts.shape[0] < 2:
        raise ValueError('Correlation requires at least two paired stimuli.')
    sigma = acts.std(0) + 1e-6
    return (acts - acts.mean(0)) / sigma, sigma


def transport_worker(cfg, rank, world):
    upstream, _ = official_modules(cfg)
    root = Path(cfg['run_dir'])
    L = AutoConfig.from_pretrained(cfg['target'], local_files_only=True).num_hidden_layers
    M = AutoConfig.from_pretrained(cfg['source'], local_files_only=True).num_hidden_layers
    jobs = [(kind, i, j) for kind in KINDS for i in range(L) for j in range(M)]
    for kind, i, j in tqdm(jobs[rank::world], desc='OT layer pairs', disable=rank != 0):
        path = root / 'ot_pairs' / kind / f'target_{i:03d}_source_{j:03d}.pt'
        if path.exists():
            obj = load_tensor(path)
            finite(obj['Q_raw'], str(path))
            continue
        x = load_tensor(root / 'activations/target' / f'layer_{i:03d}_{kind}.pt')
        y = load_tensor(root / 'activations/source' / f'layer_{j:03d}_{kind}.pt')
        if x.shape[0] != y.shape[0]:
            raise ValueError('OT stimuli must be paired; refusing upstream min(T) truncation.')
        xn, sx = normalized(x)
        yn, sy = normalized(y)
        cost = upstream.corr_distance_matrix(xn.cuda(), yn.cuda())
        finite(cost, 'correlation cost')
        a = torch.full((cost.shape[0],), 1 / cost.shape[0], device=cost.device)
        b = torch.full((cost.shape[1],), 1 / cost.shape[1], device=cost.device)
        q = upstream.sinkhorn_uniform_streaming(
            cost, a, b, reg=cfg['reg_inner'], n_iters=cfg['sinkhorn_iters'],
            block_cols=cfg['chunk_cols'], dtype=getattr(torch, cfg['ot_storage_dtype']))
        diag = marginal_report(q, cfg['marginal_l1_tolerance'], str(path))
        c = float((cost * q).sum())
        raw = q.cpu() * sx[:, None] / sy[None, :]
        finite(raw, 'raw-space Q')
        save_tensor(path, {'Q_raw': raw, 'cost': c, 'marginals_before_sigma': diag,
                           'target_layer': i, 'source_layer': j, 'kind': kind})


def finish_transport(cfg):
    upstream, _ = official_modules(cfg)
    root = Path(cfg['run_dir'])
    L = AutoConfig.from_pretrained(cfg['target'], local_files_only=True).num_hidden_layers
    M = AutoConfig.from_pretrained(cfg['source'], local_files_only=True).num_hidden_layers
    diagnostics = {}
    for kind in tqdm(KINDS, desc='Assemble full-depth transport'):
        qgrid, costs, diags = [], torch.empty(L, M), []
        for i in range(L):
            row = []
            for j in range(M):
                obj = load_tensor(root / 'ot_pairs' / kind / f'target_{i:03d}_source_{j:03d}.pt')
                row.append(obj['Q_raw'])
                costs[i, j] = obj['cost']
                diags.append(obj['marginals_before_sigma'])
            qgrid.append(row)
        finite(costs, 'layer costs')
        # Disable epsilon escalation; invalid plans fail the checks below.
        p = upstream.compute_P_stable(costs, reg_outer=cfg['reg_outer'],
                                      max_iters=cfg['outer_iters'], max_eps_grow=0)
        diag = marginal_report(p, cfg['marginal_l1_tolerance'], f'P/{kind}')
        save_tensor(root / 'transport' / f'hot_{kind}.pt', {
            'P': p, 'Q_list': qgrid, 'C_layer': costs, 'L': L, 'M': M,
            'kept_layer_idx': list(range(L)), 'kept_layer_idx_B': list(range(M)),
            'notes': 'A=target, B=source; full depth; Q_raw=sigma_A*Q/sigma_B'})
        diagnostics[kind] = {'P': diag, 'Q_before_sigma': diags, 'L': L, 'M': M}
    save_json(root / 'ot_diagnostics.json', diagnostics)


def fuse(cfg):
    _, upstream = official_modules(cfg)
    root = Path(cfg['run_dir'])
    output = root / 'fused'
    if (output / 'complete.json').exists():
        return
    target, tokenizer = load_model(cfg['target'], cfg, llama_only=True)
    source, _ = load_model(cfg['source'], cfg, llama_only=True)
    L = target.config.num_hidden_layers
    for layer in range(L):
        for kind, attr in PROJECTIONS.items():
            indices = load_tensor(root / 'top_neurons' / f'top_neurons_{kind}_layer_{layer}.pt')
            weight = getattr(target.model.layers[layer].self_attn, attr).weight
            dim = weight.shape[1 if kind == 'O' else 0]
            if indices.numel() != cfg['top_k'] or indices.min() < 0 or indices.max() >= dim:
                raise ValueError('Invalid neuron selection; dense fallback is forbidden.')
    def checksum(model):
        result = hashlib.sha256()
        for name, parameter in model.named_parameters():
            result.update(name.encode())
            result.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return result.hexdigest()

    def apply(alpha):
        return upstream.fuse_attention_only_from_hot_dir(
            target, source, hot_dir=str(root / 'transport'), alpha=alpha,
            use_language_model_only=True, verbose=False, attn_device='cuda:0',
            attn_autocast=False, use_hot_as_residual=False,
            hot_neuron_select_dir=str(root / 'top_neurons'))
    from .models import messages, generate
    probes = [messages('What is 2 + 3?'), messages('สองบวกสามเท่ากับเท่าไร')]
    text = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in probes]
    enc = tokenizer(text, padding=True, return_tensors='pt', add_special_tokens=False).to(target.device)
    with torch.no_grad():
        before_hash = checksum(target)
        before_logits = target(**enc).logits[:, -1].float().cpu()
        before_response = generate(target, tokenizer, probes, cfg, 32)
        zero_report = apply(0.0)
        zero_logits = target(**enc).logits[:, -1].float().cpu()
        if before_hash != checksum(target) or not torch.equal(before_logits, zero_logits):
            raise ValueError('Actual upstream alpha=0 changed target parameters or logits.')
        if zero_report.get('hot_neuron_select_error') or any(
                c['pairs_skipped'] for c in zero_report['components'].values()):
            raise ValueError('Alpha-zero test did not traverse all expected layer pairs.')
        report = apply(cfg['alpha'])
        after_logits = target(**enc).logits[:, -1].float().cpu()
        finite(after_logits, 'fused probe logits')
        after_response = generate(target, tokenizer, probes, cfg, 32)
    save_json(root / 'e0_inference.json', {
        'alpha_zero_parameters_exact': True, 'alpha_zero_logits_exact': True,
        'target_responses': before_response, 'fused_responses': after_response,
        'probe_logit_max_change': float((after_logits - before_logits).abs().max()),
        'optimizer_steps': 0})
    if report.get('hot_neuron_select_error'):
        raise RuntimeError(report['hot_neuron_select_error'])
    for kind, comp in report['components'].items():
        if comp['pairs_skipped'] or not comp['neuron_select_enabled']:
            raise ValueError(f'{kind}: incomplete fusion: {comp}')
    for name, parameter in target.named_parameters():
        finite(parameter, name)
    target.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    save_json(root / 'fusion_report.json', report)
    save_json(output / 'complete.json', {'alpha': cfg['alpha'], 'training': False,
                                       'target_shape_preserved': True})
