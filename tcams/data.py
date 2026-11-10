"""Local-only benchmark preparation using verified publisher schemas.

References and dataset revisions are documented in docs/implementation_audit.md.
MultiJail semantic pairing uses publisher IDs, never translated-row guesses.
"""
import csv
from collections import defaultdict
from pathlib import Path
import re
import unicodedata

import pyarrow.parquet as pq
from tqdm import tqdm

from .common import digest, file_hash, read_jsonl, sample_filename, save_json, write_jsonl


def normalize_text(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def assign_splits(rows, fraction):
    """Group translations, passages, and exact duplicate prompts before splitting."""
    parent = {r['group_id']: r['group_id'] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for r in rows:
        key = normalize_text(r['prompt'])
        if key in seen:
            a, b = find(r['group_id']), find(seen[key])
            parent[max(a, b)] = min(a, b)
        seen[key] = r['group_id']
    test_only = {'belebele', 'thai_safety', 'xstest'}
    forced_test_groups = {find(r['group_id']) for r in rows if r.get('dataset') in test_only}
    for row in rows:
        group = find(row['group_id'])
        value = int(digest(['split', 42, group])[:16], 16) / 16**16
        row['split'] = 'probe' if value < fraction and group not in forced_test_groups else 'test'
        row['split_group'] = group
    return rows


def row(dataset, gid, language, prompt, task, answer=None, category='unspecified'):
    return {'id': f'{dataset}:{gid}:{language}', 'group_id': f'{dataset}:{gid}',
            'dataset': dataset, 'language': language, 'prompt': prompt,
            'task': task, 'answer': answer, 'category': category}


def prepare(cfg, extra=None):
    raw = Path(cfg['data_root']) / 'datasets'
    root = Path(cfg['run_dir'])
    rows, sources = [], {}

    multi = raw / 'DAMO-NLP-SG/MultiJail/MultiJail.csv'
    with multi.open(encoding='utf-8-sig', newline='') as stream:
        items = list(csv.DictReader(stream))
    if len(items) != 315 or len({r['id'] for r in items}) != 315:
        raise ValueError('Expected the pinned 315-ID publisher MultiJail release.')
    for r in items:
        for language in ('en', 'th'):
            rows.append(row('multijail', r['id'], language, r[language], 'safety', category=r['tags']))
    sources[str(multi)] = file_hash(multi)

    for language in ('en', 'th'):
        path = raw / 'MGSM-Rev2/MGSM-Rev2' / f'mgsm_{language}.tsv'
        with path.open(encoding='utf-8', newline='') as stream:
            items = list(csv.reader(stream, delimiter='\t'))
        if len(items) != 250 or any(len(r) != 2 for r in items):
            raise ValueError('Expected 250 two-column, headerless MGSM-Rev2 rows.')
        for i, (question, answer) in enumerate(items, 1):
            rows.append(row('mgsm_rev2', f'{i:04d}', language, question, 'math', answer))
        sources[str(path)] = file_hash(path)

    path = raw / 'facebook/belebele/data/tha_Thai.jsonl'
    items = read_jsonl(path)
    if len(items) != 900:
        raise ValueError('Expected 900 Thai Belebele questions.')
    for question_index, r in enumerate(items, 1):
        passage = digest([r['link'], r['ds']])[:16]
        gid = f'question_{question_index:04d}'
        choices = '\n'.join(f'{i}. {r[f"mc_answer{i}"]}' for i in range(1, 5))
        prompt = f'{r["flores_passage"]}\n\n{r["question"]}\n{choices}'
        result = row('belebele', gid, 'th', prompt, 'reading', str(r['correct_answer_num']))
        result['group_id'] = f'belebele:passage:{passage}'
        result['publisher_semantic_key'] = [r['link'], r['ds'], r['question_number']]
        rows.append(result)
    sources[str(path)] = file_hash(path)

    path = raw / 'XSTest/xstest_prompts.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        items = list(csv.DictReader(stream))
    if len(items) != 450:
        raise ValueError('Expected 450 XSTest prompts.')
    safe_count = 0
    for r in items:
        task = 'safety' if r['type'].startswith('contrast_') else 'overrefusal'
        if r['label'] != ('unsafe' if task == 'safety' else 'safe'):
            raise ValueError('XSTest type and gold label disagree.')
        safe_count += task == 'overrefusal'
        rows.append(row('xstest', r['id'], 'en', r['prompt'], task, category=r['type']))
    if safe_count != 250:
        raise ValueError('XSTest safe/contrast schema differs from the audited release.')
    sources[str(path)] = file_hash(path)

    path = raw / 'typhoon-ai/ThaiSafetyBench/data/test-00000-of-00001.parquet'
    thai_rows = pq.read_table(path).to_pylist()
    if len(thai_rows) != 1889 or len({r['id'] for r in thai_rows}) != 1889:
        raise ValueError('Expected the pinned 1,889-ID public ThaiSafetyBench release.')
    for r in thai_rows:
        record = row('thai_safety', str(r['id']), 'th', r['prompt'], 'safety', category=r['risk_area'])
        record['thai_related'] = r['thai_related']
        record['source'] = r['source']
        rows.append(record)
    sources[str(path)] = file_hash(path)

    if extra:
        for r in read_jsonl(extra):
            if r['task'] != 'overrefusal' or r['language'] != 'th' or r.get('reviewed_by_human') is not True:
                raise ValueError('Extra prompts must be human-reviewed Thai benign requests.')
            required = {'id', 'group_id', 'dataset', 'language', 'prompt', 'task', 'answer', 'category'}
            if not required.issubset(r):
                raise ValueError(f'Extra prompt schema requires {required}')
            rows.append(r)
        sources[str(extra)] = file_hash(extra)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate evaluation IDs.')
    if len({sample_filename(r['id']) for r in rows}) != len(rows):
        raise ValueError('Readable sample filenames collide.')
    if any(not isinstance(r['prompt'], str) or not r['prompt'].strip() for r in rows):
        raise ValueError('Empty/non-string benchmark prompt.')
    rows = assign_splits(rows, cfg['probe_fraction'])
    counts = defaultdict(int)
    for r in rows:
        counts[f'{r["dataset"]}/{r["language"]}/{r["split"]}'] += 1
    for ds, lang in {(r['dataset'], r['language']) for r in rows}:
        required_splits = ('test',) if ds in ('belebele', 'thai_safety', 'xstest') else ('probe', 'test')
        if any(counts[f'{ds}/{lang}/{s}'] == 0 for s in required_splits):
            raise ValueError('An evaluation cell has no probe/test examples.')

    # FineWeb is alignment-only. Exact overlap is excluded and counted explicitly.
    files = sorted((raw / 'ChavyvAkvar/fineweb-2-1M-Sample-Thai').rglob('*.parquet'))
    if not files:
        raise FileNotFoundError('Download FineWeb Thai parquet files first.')
    forbidden = {normalize_text(r['prompt']) for r in rows}
    align, seen, excluded = [], set(), 0
    for path in tqdm(files, desc='Read calibration files'):
        sources[str(path)] = file_hash(path)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=['text']):
            for r in batch.to_pylist():
                text = r['text']
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('FineWeb text is missing or empty.')
                norm = normalize_text(text)
                if norm in forbidden or norm in seen:
                    excluded += 1
                    continue
                seen.add(norm)
                align.append({'id': 'fineweb:' + digest(norm), 'text': text})
                if len(align) == cfg['calibration_samples']:
                    break
            if len(align) == cfg['calibration_samples']:
                break
        if len(align) == cfg['calibration_samples']:
            break
    if len(align) != cfg['calibration_samples']:
        raise ValueError('Insufficient calibration data.')
    from .common import read_json
    manifest = {'source_hashes': sources, 'asset_revisions': read_json(Path(cfg['data_root']) / 'asset_revisions.json'),
                'counts': dict(counts), 'align_count': len(align),
                'align_hash': digest(align), 'evaluation_hash': digest(rows),
                'excluded_exact_overlap_or_duplicates': excluded, 'seed': 42,
                'thai_overrefusal_available': any(r['language'] == 'th' and r['task'] == 'overrefusal' for r in rows)}
    old = root / 'data/manifest.json'
    if old.exists():
        if read_json(old) != manifest:
            raise ValueError('Prepared dataset changed. Use a new run or --clean.')
    write_jsonl(root / 'data/align.jsonl', align)
    write_jsonl(root / 'data/eval.jsonl', rows)
    save_json(old, manifest)


def subset(rows, split, limit):
    if limit == 0 or limit < -1:
        raise ValueError('limit must be -1 (all) or positive per benchmark/language/task.')
    cells = defaultdict(list)
    for row in rows:
        if row['split'] == split:
            cells[(row['dataset'], row['language'], row['task'])].append(row)
    selected = []
    for key in sorted(cells):
        # Matching translations get identical ordering; passage grouping is for splitting only.
        ordered = sorted(cells[key], key=lambda r: digest([42, r['id'].rsplit(':', 1)[0]]))
        selected.extend(ordered if limit == -1 else ordered[:limit])
    if not selected:
        raise ValueError('No evaluation samples selected.')
    return selected


def ability_prompt(r):
    if r['task'] == 'math':
        suffix = ('\nSolve step by step. End with "Final answer: <number>".' if r['language'] == 'en'
                  else '\nแก้โจทย์ทีละขั้นตอน และลงท้ายด้วย "Final answer: <number>"')
    elif r['task'] == 'reading':
        suffix = '\nตอบด้วยหมายเลขตัวเลือกที่ถูกต้องเพียงตัวเดียว (1, 2, 3 หรือ 4)'
    else:
        suffix = ''
    return r['prompt'] + suffix


def ability_score(r, response):
    """A preregistered pilot parser, not a claim of exact harness reproduction."""
    response = response.translate(str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')).strip()
    if r['task'] == 'reading':
        match = re.fullmatch(r'([1-4])[.。]?', response)
        prediction = match.group(1) if match else None
    else:
        matches = re.findall(r'Final answer\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)', response, re.I)
        prediction = matches[-1].replace(',', '') if matches else None
    gold = str(r['answer']).replace(',', '').strip()
    if r['task'] == 'math' and prediction is not None:
        from decimal import Decimal
        correct = Decimal(prediction) == Decimal(gold)
    else:
        correct = prediction == gold
    return {'correct': int(correct), 'parsed_answer': prediction, 'parse_failure': int(prediction is None)}
