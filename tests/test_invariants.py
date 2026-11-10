"""Server-side invariant tests; intentionally not executed on the authoring machine."""
from pathlib import Path

import pytest
import torch

from tcams.analysis import paired_ci
from tcams.common import save_tensor
from tcams.data import assign_splits, ability_score
from tcams.delta import rollback_groups
from tcams.evaluation import parse_guard


def test_translation_and_duplicate_split():
    rows = [
        {'group_id': 'q1', 'prompt': 'hello'},
        {'group_id': 'q1', 'prompt': 'สวัสดี'},
        {'group_id': 'q2', 'prompt': ' HELLO '},
    ]
    result = assign_splits(rows, .25)
    assert len({r['split_group'] for r in result}) == 1
    assert len({r['split'] for r in result}) == 1


def test_test_only_benchmark_cannot_become_probe():
    rows = [{'group_id': 'passage1', 'prompt': 'Read this.', 'dataset': 'belebele'},
            {'group_id': 'duplicate', 'prompt': 'Read this.', 'dataset': 'custom'}]
    assert all(r['split'] == 'test' for r in assign_splits(rows, 1.0))


def test_guard_requires_response_labels():
    assert parse_guard('Safety: Unsafe\nCategories: Violent\nRefusal: No')['unsafe'] == 1
    controversial = parse_guard('Safety: Controversial\nRefusal: Yes')
    assert controversial['unsafe'] == 0 and controversial['unsafe_or_controversial'] == 1
    with pytest.raises(ValueError):
        parse_guard('Safety: Safe')


def test_no_number_fallback():
    row = {'task': 'math', 'answer': '42'}
    assert ability_score(row, 'There are 42 items.')['parse_failure'] == 1
    assert ability_score(row, 'Final answer: ๔๒')['correct'] == 1


def test_patch_restores_after_failure(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 3, bias=False)
    model.weight.data.copy_(torch.arange(12).reshape(3, 4))
    fused = model.weight.detach().clone()
    target = fused[[1]] - 2
    save_tensor(tmp_path / 'group.pt', {'indices': torch.tensor([1]), 'axis': 0,
                                       'name': 'weight', 'target': target, 'fused': fused[[1]]})
    with pytest.raises(RuntimeError, match='interrupt'):
        with rollback_groups(model, tmp_path, ['group']):
            assert torch.equal(model.weight[[1]], target)
            raise RuntimeError('interrupt')
    assert torch.equal(model.weight, fused)


def test_delta_reconstruct_bfloat16():
    torch.manual_seed(42)
    target = torch.randn(16, 16).bfloat16()
    fused = target.clone()
    fused[[1, 3]] = (target[[1, 3]].float() + .1).bfloat16()
    delta = fused.float() - target.float()
    assert torch.equal((target.float() + delta).bfloat16(), fused)


def test_paired_bootstrap_identity_and_direction():
    baseline = [{'id': str(i), 'split_group': str(i), 'unsafe': 1} for i in range(6)]
    changed = [{**r, 'unsafe': 0} for r in baseline]
    assert paired_ci(baseline, changed, 'unsafe', 50)['estimate'] == 1
    assert paired_ci(baseline, baseline, 'unsafe', 50)['ci_high'] == 0
    with pytest.raises(ValueError):
        paired_ci(baseline, changed[:-1], 'unsafe', 50)


def test_upstream_rectangular_ot():
    # Run from tcams_code after prepare_assets.py; no model download is used here.
    from tcams.common import config, official_modules
    from tcams.ot_adapter import marginal_report
    upstream, _ = official_modules(config('configs/smoke_thai.yaml'))
    torch.manual_seed(42)
    cost = torch.rand(3, 5)
    plan = upstream.sinkhorn_uniform_streaming(cost, torch.full((3,), 1 / 3),
                                               torch.full((5,), 1 / 5), reg=.1,
                                               n_iters=500, dtype=torch.float32)
    marginal_report(plan, .01, 'rectangular test')
    assert plan.shape == (3, 5)
