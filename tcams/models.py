"""Llama hooks adapted from the pinned T&M activs_llama3_modules.py.

Differences: native RoPE configuration, strict errors, attention-only hooks,
atomic batch checkpoints, and explicit sample identities.
"""
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import finite, load_tensor, save_tensor

PROJECTIONS = {'Q': 'q_proj', 'K': 'k_proj', 'V': 'v_proj', 'O': 'o_proj'}
KINDS = [k + suffix for k in PROJECTIONS for suffix in ('', '_pre')]


def load_model(path, cfg, device='cuda:0', llama_only=False):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError(f'Missing native chat template: {path}')
    if tokenizer.eos_token_id is None:
        raise ValueError('An EOS token is required.')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        torch_dtype=getattr(torch, cfg['dtype']), attn_implementation='eager').to(device)
    model.eval().requires_grad_(False)
    if llama_only and model.config.model_type != 'llama':
        raise ValueError('The audited OT adapter currently supports text-only Llama models.')
    return model, tokenizer


def messages(prompt):
    return [{'role': 'user', 'content': prompt}]


def activation_batches(model, tokenizer, rows, output, cfg, rank, world):
    output = Path(output)
    blocks = list(model.model.layers)
    buffers, current = {}, {}
    handles = []

    def hook(kind, layer):
        def collect(module, inputs, result):
            mask = current['mask'].unsqueeze(-1)
            for suffix, value in (('_pre', inputs[0]), ('', result)):
                if not isinstance(value, torch.Tensor) or value.ndim != 3:
                    raise ValueError('Unexpected projection activation shape.')
                # Use the upstream masked mean, preserving its model-dtype arithmetic.
                m = mask.to(value.dtype)
                pooled = (value * m).sum(1) / m.sum(1)
                finite(pooled, f'{layer}/{kind}{suffix}')
                buffers[f'{layer}/{kind}{suffix}'] = pooled.detach().cpu()
        return collect

    for layer, block in enumerate(blocks):
        for kind, name in PROJECTIONS.items():
            projection = getattr(block.self_attn, name)
            if projection.bias is not None:
                raise ValueError('Bias-bearing attention requires a separate audited adapter.')
            handles.append(projection.register_forward_hook(hook(kind, layer)))
    batch_size = cfg['batch_size']
    indices = list(range(0, len(rows), batch_size))[rank::world]
    try:
        for start in tqdm(indices, desc='Calibration batches', disable=rank != 0):
            chunk = rows[start:start + batch_size]
            path = output / f'batch_{start:06d}.pt'
            ids = [r['id'] for r in chunk]
            if path.exists():
                if load_tensor(path)['ids'] != ids:
                    raise ValueError('Activation cache identity mismatch.')
                continue
            prompts = [tokenizer.apply_chat_template(messages(r['text']), tokenize=False,
                                                     add_generation_prompt=True) for r in chunk]
            enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True,
                            max_length=cfg['calibration_max_tokens'], add_special_tokens=False).to(model.device)
            current['mask'] = enc['attention_mask']
            buffers.clear()
            with torch.inference_mode():
                model(**enc, use_cache=False)
            if len(buffers) != len(blocks) * len(KINDS):
                raise ValueError('Not every projection hook fired.')
            save_tensor(path, {'ids': ids, 'acts': dict(buffers)})
    finally:
        for handle in handles:
            handle.remove()


def assemble_activations(output, rows, cfg, layers):
    chunks = []
    for start in range(0, len(rows), cfg['batch_size']):
        obj = load_tensor(Path(output) / f'batch_{start:06d}.pt')
        if obj['ids'] != [r['id'] for r in rows[start:start + cfg['batch_size']]]:
            raise ValueError('Activation sample order mismatch.')
        chunks.append(obj['acts'])
    for layer in range(layers):
        for kind in KINDS:
            path = Path(output) / f'layer_{layer:03d}_{kind}.pt'
            value = torch.cat([chunk[f'{layer}/{kind}'] for chunk in chunks]).float()
            if value.shape[0] != len(rows):
                raise ValueError('Activation cardinality mismatch.')
            finite(value, str(path))
            save_tensor(path, value)


def generate(model, tokenizer, prompts, cfg, max_new_tokens=None, with_metadata=False):
    text = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tokenizer(text, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
    if enc.input_ids.shape[1] > cfg['max_input_tokens']:
        raise ValueError('Input exceeds max_input_tokens; refusing silent truncation.')
    with torch.inference_mode():
        result = model.generate(**enc, do_sample=False, num_beams=1,
                                max_new_tokens=max_new_tokens or cfg['max_new_tokens'],
                                pad_token_id=tokenizer.pad_token_id)
    continuation = result[:, enc.input_ids.shape[1]:]
    texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
    if not with_metadata:
        return texts
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [tokenizer.eos_token_id])
    result = []
    for text, tokens in zip(texts, continuation.tolist(), strict=True):
        terminal = next((i for i, token in enumerate(tokens) if token in eos), None)
        result.append({'response': text, 'generated_tokens': terminal + 1 if terminal is not None else len(tokens),
                       'hit_generation_limit': int(terminal is None and len(tokens) == (max_new_tokens or cfg['max_new_tokens']))})
    return result
