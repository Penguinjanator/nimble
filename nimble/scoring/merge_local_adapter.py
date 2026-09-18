"""Merge a local adapter with its pinned, cached base for Mac/Linux inference."""
import argparse
import hashlib
import json
import shutil
import time
from importlib.metadata import version
from pathlib import Path

from nimble.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.adapter/'schema_config.json').read_text())
    prompt_hash = hashlib.sha256((PROJECT_ROOT/'nimble/scoring/parallel_schema.py').read_bytes()).hexdigest()
    assert contract['prompt_code_sha256'] == prompt_hash
    base_path = PROJECT_ROOT/'.cache/huggingface/hub'/('models--'+contract['model'].replace('/', '--'))/'snapshots'/contract['revision']
    if not base_path.is_dir():
        raise FileNotFoundError(f'Pinned base must already be cached: {base_path}')
    adapter_hash = hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
    ready = args.output/'READY.json'
    if ready.exists():
        saved = json.loads(ready.read_text())
        assert saved['adapter_sha256'] == adapter_hash and saved['base_revision'] == contract['revision']
        print('Reusing verified merge:', args.output)
        return
    if args.output.exists():
        raise FileExistsError('Output exists without READY.json; use a fresh directory')
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(8)
    started = time.perf_counter()
    print('Loading cached base on CPU', flush=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(base_path, local_files_only=True,
                dtype=torch.bfloat16, device_map='cpu', attn_implementation='sdpa', trust_remote_code=False)
    print('Merging adapter in BF16', flush=True)
    merged = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).merge_and_unload(safe_merge=True)
    print('Saving merged checkpoint', flush=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size='4GB')
    AutoTokenizer.from_pretrained(args.adapter, local_files_only=True).save_pretrained(args.output)
    shutil.copyfile(args.adapter/'schema_config.json', args.output/'schema_config.json')
    hashes = {}
    for path in sorted(args.output.glob('*.safetensors')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    manifest = {'base_model': contract['model'], 'base_revision': contract['revision'],
                'adapter_sha256': adapter_hash, 'prompt_sha256': prompt_hash,
                'training_examples': contract['data_audit']['training_rows'],
                'heldout_examples': contract['data_audit']['validation_rows'],
                'precision': 'bfloat16', 'quantized': False, 'merge': 'PEFT safe_merge on CPU',
                'merge_seconds': time.perf_counter()-started, 'weight_sha256': hashes,
                'versions': {p: version(p) for p in ('torch', 'peft', 'transformers')}}
    ready.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'weight_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
