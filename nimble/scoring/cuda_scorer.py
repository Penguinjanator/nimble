"""CUDA counterpart of the MLX independent candidate scorer.

Shares prompt construction and projects only candidate output rows in FP32.
Checkpoint weights remain BF16; no generated text or probability fitting.
"""

import hashlib
import json
import math
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

from nimble.scoring.parallel_schema import choice_key, prepare_prompts


def candidate_projection(hidden, weight, token_ids):
    indices = torch.tensor(token_ids, device=weight.device)
    return hidden.float() @ weight.index_select(0, indices).float().T


class CudaCandidateScorer:
    def __init__(self, model_path, model_id, revision, max_input_tokens=4096, temperature=1.0):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This runner requires a CUDA GPU with BF16 support")
        if not isinstance(max_input_tokens, int) or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        # Avoid TF32 rounding in the small FP32 candidate projection.
        torch.backends.cuda.matmul.allow_tf32 = False
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if config.model_type not in {"qwen3_5", "gemma3_text"}:
            raise ValueError(f"Unsupported architecture: {config.model_type}")
        cls = Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else AutoModelForCausalLM
        # Torch 2.8's optimized CUDA SDPA produced incorrect Gemma outputs once
        # prompts crossed its 512-token sliding window. Eager agrees with the
        # CPU and math-only SDPA checks; retain that validated path for Gemma.
        attention = "eager" if config.model_type == "gemma3_text" else "sdpa"
        self.model, loading = cls.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation=attention, output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Checkpoint did not load exactly: {loading}")
        self.model = self.model.eval().to("cuda")
        self.backbone = self.model.model
        self.head_weight = self.model.get_output_embeddings().weight
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.system_role = config.model_type != "gemma3_text"
        self.model_id, self.revision = model_id, revision
        text_config = getattr(config, "text_config", config)
        self.max_input_tokens = min(max_input_tokens, text_config.max_position_embeddings - 1)
        self.temperature = temperature
        self.device = torch.device("cuda")

    def prepare(self, context, schema):
        return prepare_prompts(self.tokenizer, context, schema, self.max_input_tokens,
                               system_role=self.system_role)

    @torch.inference_mode()
    def score(self, context, schema, mode="independent"):
        if mode != "independent":
            raise ValueError("CUDA runner currently supports independent full-prompt prefill only")
        prepared = self.prepare(context, schema)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        fields, output = {}, {}
        for name, choices, ids, candidates in zip(prepared.names, prepared.choices,
                                                   prepared.full_ids, prepared.candidate_ids):
            tokens = torch.tensor([ids], device=self.device)
            hidden = self.backbone(input_ids=tokens, use_cache=False).last_hidden_state[:, -1, :]
            logits = candidate_projection(hidden, self.head_weight, candidates)[0]
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite candidate logits")
            probabilities = torch.softmax(logits / self.temperature, dim=-1)
            best = logits.argmax().item()
            keys = [choice_key(value) for value in choices]
            output[name] = choices[best]
            fields[name] = {
                "value": choices[best], "scores": dict(zip(keys, probabilities.tolist())),
                "logits": dict(zip(keys, logits.tolist())), "candidate_token_ids": candidates,
                "code_to_choice": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", choices)),
                "prompt_token_count": len(ids),
                "prompt_token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            }
            del hidden, logits, probabilities
        torch.cuda.synchronize()
        return {"model": self.model_id, "revision": self.revision, "backend": "cuda",
                "temperature": self.temperature, "temperature_fitted": False,
                "context": context, "output": output, "fields": fields,
                "metrics": {"mode": mode, "fields": len(fields),
                            "total_seconds": time.perf_counter() - started,
                            "cuda_peak_active_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "full_vocabulary_projection": False}}
