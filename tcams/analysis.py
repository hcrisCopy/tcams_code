"""Paired estimates, held-out plans, and restrained feasibility reporting."""
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np

from .common import digest, read_json, read_jsonl, save_json, write_jsonl
from .data import subset
from .evaluation import collect_scores


def paired_ci(base, changed, field, repeats=2000):
    """Cluster bootstrap the same question/passage in both conditions."""
    a = {r['id']: r for r in base}
    b = {r['id']: r for r in changed}
    if a.keys() != b.keys() or not a:
        raise ValueError('Paired metrics require the same nonempty set of IDs.')
    groups = defaultdict(list)
    for key in a:
        groups[a[key]['split_group']].append(float(a[key][field]) - float(b[key][field]))
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(42)
    boots = np.empty(repeats)
    for i in range(repeats):
        selected = rng.integers(0, len(sums), len(sums))
        boots[i] = sums[selected].sum() / sizes[selected].sum()
    low, high = np.quantile(boots, [.025, .975])
    return {'estimate': float(sums.sum() / sizes.sum()), 'ci_low': float(low), 'ci_high': float(high),
            'n': int(sizes.sum()), 'clusters': len(sums)}


def effect(base, changed, repeats, split):
    out = {}
    specs = {'risk': ('safety', 'unsafe'), 'risk_including_controversial': ('safety', 'unsafe_or_controversial'),
             'math': ('math', 'correct'), 'reading': ('reading', 'correct'),
             'overrefusal': ('overrefusal', 'refusal')}
    for lang in ('en', 'th'):
        for name, (task, field) in specs.items():
            a = [r for r in base if r['language'] == lang and r['task'] == task
                 and (task != 'safety' or r['dataset'] == 'multijail')]
            b = [r for r in changed if r['language'] == lang and r['task'] == task
                 and (task != 'safety' or r['dataset'] == 'multijail')]
            if a:
                out[f'{lang}_{name}'] = paired_ci(a, b, field, repeats)
    # Predeclared equal weighting prevents the larger reading dataset dominating utility.
    if split == 'probe':
        out['th_utility_loss'] = out['th_math']['estimate']
        out['utility_definition'] = 'Thai MGSM only; Belebele is reserved for test'
    elif split == 'test':
        out['th_utility_loss'] = (out['th_math']['estimate'] + out['th_reading']['estimate']) / 2
        out['utility_definition'] = 'Equal-weight Thai MGSM and Belebele'
    else:
        raise ValueError('Unknown split for the utility definition.')
    return out


def select_plan(cfg, limit):
    root = Path(cfg['run_dir'])
    plan = read_json(root / 'plans/probe.json')
    rows = subset(read_jsonl(root / 'data/eval.jsonl'), 'probe', limit)
    baseline = collect_scores(cfg, 'probe', 'fused', rows)
    evidence = []
    for condition in plan['conditions']:
        if condition['name'] == 'fused':
            continue
        scores = collect_scores(cfg, 'probe', condition['name'], rows)
        evidence.append({'group': condition['groups'][0], **effect(baseline, scores, cfg['bootstrap_repeats'], 'probe')})
    save_json(root / 'probe_effects.json', evidence)
    eligible = [r for r in evidence if r['th_risk']['estimate'] > 0
                and r['th_risk_including_controversial']['estimate'] > 0
                and r['th_utility_loss'] <= cfg['utility_loss_tolerance']]
    eligible.sort(key=lambda r: (-r['th_risk']['estimate'], r['th_utility_loss'], r['group']))
    if len(eligible) < cfg['joint_count']:
        save_json(root / 'selection_status.json', {'status': 'insufficient_probe_candidates',
                                                   'eligible_count': len(eligible), 'required': cfg['joint_count']})
        raise ValueError('Probe did not find enough candidate groups. Do not tune on test; inspect probe_effects.json.')
    chosen = [r['group'] for r in eligible[:cfg['joint_count']]]
    all_groups = read_json(root / 'delta/groups.json')
    by_id = {g['id']: g for g in all_groups}
    budget = Counter(by_id[g]['parameter_budget'] for g in chosen)
    buckets = {size: [g for g in all_groups if g['parameter_budget'] == size] for size in budget}
    conditions = [{'name': 'fused', 'groups': []}, {'name': 'selected_joint', 'groups': chosen}]
    conditions.extend({'name': 'selected_single_' + g, 'groups': [g]} for g in chosen)
    for fraction in (.5, .75):
        conditions.append({'name': f'global_delta_{int(fraction * 100)}',
                           'groups': [g['id'] for g in all_groups],
                           'mode': 'scale_delta', 'fraction': fraction})
    # Keep a positive safety contributor as a preregistered mechanistic counterpoint.
    positive = min(evidence, key=lambda r: r['th_risk']['estimate'])
    if positive['th_risk']['estimate'] < 0:
        conditions.append({'name': 'remove_safety_helpful', 'groups': [positive['group']]})
    for metric in ('activation', 'delta_norm'):
        ids = []
        for size, count in budget.items():
            ranked = sorted(buckets[size], key=lambda g: (-g[metric], g['id']))
            ids.extend(g['id'] for g in ranked[:count])
        conditions.append({'name': 'control_' + metric, 'groups': ids})
    rng = np.random.default_rng(42)
    for i in range(cfg['random_controls']):
        ids = []
        for size, count in budget.items():
            options = sorted(g['id'] for g in buckets[size])
            ids.extend(rng.choice(options, count, replace=False).tolist())
        conditions.append({'name': f'control_random_{i:02d}', 'groups': ids})
    result = {'split': 'test', 'conditions': conditions, 'probe_limit_per_cell': limit,
              'selection': 'exploratory point estimates, frozen before test',
              'matched_budget': {str(k): v for k, v in budget.items()}, 'probe_evidence_hash': digest(evidence),
              'comparison_limit': 'intervention parameter/count matched; selection compute not matched',
              'random_policy': f'{cfg["random_controls"]} draws from one RNG seeded 42; duplicates retained'}
    destination = root / 'plans/test.json'
    if destination.exists() and read_json(destination) != result:
        raise ValueError('Test plan already frozen with different selections.')
    save_json(destination, result)
    # Evaluate the same joint/control sets on probe to measure joint interactions separately.
    save_json(root / 'plans/joint_probe.json', {**result, 'split': 'probe'})


def write_csv(path, rows):
    if not rows:
        raise ValueError('No summary rows.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(cfg, split, conditions, limit):
    root = Path(cfg['run_dir'])
    rows = subset(read_jsonl(root / 'data/eval.jsonl'), split, limit)
    metrics, score_sets, category_metrics = [], {}, []
    for name in conditions:
        scores = collect_scores(cfg, split, name, rows)
        score_sets[name] = scores
        cells = defaultdict(list)
        for r in scores:
            cells[(r['dataset'], r['language'], r['task'])].append(r)
        for (dataset, language, task), cell in cells.items():
            field = {'safety': 'unsafe', 'overrefusal': 'refusal', 'math': 'correct', 'reading': 'correct'}[task]
            zero = [{**r, field: 0} for r in cell]
            ci = paired_ci(cell, zero, field, cfg['bootstrap_repeats'])
            metrics.append({'condition': name, 'split': split, 'dataset': dataset, 'language': language,
                            'metric': field, **ci,
                            'empty_response_rate': np.mean([r['empty_response'] for r in cell]),
                            'parse_failure_rate': np.mean([r.get('parse_failure', 0) for r in cell]),
                            'controversial_rate': np.mean([r.get('controversial', 0) for r in cell]),
                            'generation_limit_rate': np.mean([r['hit_generation_limit'] for r in cell])})
            if task == 'safety':
                for category in sorted({str(r['category']) for r in cell}):
                    items = [r for r in cell if str(r['category']) == category]
                    category_metrics.append({'condition': name, 'dataset': dataset, 'language': language,
                                             'category': category, 'n': len(items),
                                             'unsafe': np.mean([r['unsafe'] for r in items]),
                                             'controversial': np.mean([r['controversial'] for r in items])})
    out = root / 'reports' / split
    write_csv(out / 'metrics.csv', metrics)
    if {'source', 'target', 'fused'}.issubset(score_sets):
        write_csv(out / 'baseline_metrics.csv', metrics)
    if category_metrics:
        write_csv(out / 'risk_categories.csv', category_metrics)
    if 'target' in score_sets and 'fused' in score_sets:
        paired = effect(score_sets['fused'], score_sets['target'], cfg['bootstrap_repeats'], split)
        paired['th_utility_gain'] = paired.pop('th_utility_loss')
        save_json(out / 'fused_minus_target.json', {
            'direction': 'fused minus target; risk negative is better, utility positive is better', **paired})
    lines = ['# Pilot results', '', '这是远程运行生成的结果，不自动判定论文假设成立。', '',
             '| Condition | Dataset | Language | Metric | Estimate | 95% CI | N |',
             '|---|---|---|---|---:|---|---:|']
    for r in metrics:
        lines.append(f'| {r["condition"]} | {r["dataset"]} | {r["language"]} | {r["metric"]} | '
                     f'{r["estimate"]:.4f} | [{r["ci_low"]:.4f}, {r["ci_high"]:.4f}] | {r["n"]} |')
    effects = []
    if 'fused' in score_sets:
        for name, scores in score_sets.items():
            if name in ('source', 'target', 'fused'):
                continue
            e = effect(score_sets['fused'], scores, cfg['bootstrap_repeats'], split)
            effects.append({'condition': name, 'utility_loss': e['th_utility_loss'],
                            'risk_reduction': e['th_risk']['estimate'],
                            'risk_ci_low': e['th_risk']['ci_low'], 'risk_ci_high': e['th_risk']['ci_high'],
                            'math_loss_upper_ci': e['th_math']['ci_high'],
                            'reading_loss_upper_ci': e['th_reading']['ci_high'] if split == 'test' else None})
        if effects:
            write_csv(out / 'ablation.csv', effects)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            for r in effects:
                ax.scatter(r['utility_loss'], r['risk_reduction'],
                           label=r['condition'] if not r['condition'].startswith('control_random') else None,
                           color='gray' if r['condition'].startswith('control_random') else None)
            ax.axhline(0, color='black', lw=.7)
            ax.axvline(0, color='black', lw=.7)
            ax.set(xlabel='Thai utility loss after rollback (lower is better)',
                   ylabel='Thai MultiJail unsafe-rate reduction (higher is better)')
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out / 'risk_utility.png', dpi=200)
            plt.close(fig)
    lines.extend(['', '## 解释边界', '',
                  f'- 当前模型：{cfg["source_model_id"]} → {cfg["target_model_id"]}；只合并 attention Q/K/V/O，不训练。',
                  '- MGSM 使用 zero-shot 显式 final-answer 解析；Belebele 是阅读理解，不能称为 MMLU 知识得分。',
                  '- Unsafe、Controversial、空回答、解析失败分别记录；拒答只表示 guard 判定，未校准成人类标签。',
                  '- CI 是同题/同 passage 配对 bootstrap；单组比较未做多重比较校正，属于探索分析。',
                  '- 合并后的安全分数不能替代泰语理解和过度拒答检查；缺少人工 TH 良性集时此证据缺口仍在。',
                  '- Go/No-Go 需同时核验独立 test 风险改善、两项能力非劣、过拒答和语言理解，不能只看散点象限。',
                  '- 等参数量删除不等于等评测预算；当前也没有实现 SWE/SubSafe-Merge 强方法对照。',
                  '- 本轮干预的是实际融合增量；没有识别单条 OT 神经元路径，也没有验证来源信息的增量价值。'])
    report_text = '\n'.join(lines) + '\n'
    (out / 'pilot_report.md').write_text(report_text, encoding='utf-8')
    if {'source', 'target', 'fused'}.issubset(score_sets):
        (out / 'baseline_report.md').write_text(report_text, encoding='utf-8')


def export_audit(cfg, split, conditions, limit):
    root = Path(cfg['run_dir'])
    rows = subset(read_jsonl(root / 'data/eval.jsonl'), split, limit)
    selected = sorted(rows, key=lambda r: digest([42, r['id']]))[:min(50, len(rows))]
    audit = []
    for condition in conditions:
        for r in collect_scores(cfg, split, condition, selected):
            audit.append({'blind_id': digest([condition, r['id']])[:20], 'language': r['language'],
                          'prompt': r['prompt'], 'response': r['response'],
                          'understands_request': None, 'appropriate_language': None,
                          'unsafe_assistance': None, 'unnecessary_refusal': None})
    write_jsonl(root / 'private_audit' / f'{split}.jsonl', sorted(audit, key=lambda r: r['blind_id']))
