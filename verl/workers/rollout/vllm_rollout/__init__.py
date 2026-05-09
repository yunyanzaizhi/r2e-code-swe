# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


vllm_package_name = "vllm"
vllm_package_version = get_version(vllm_package_name)
if vllm_package_version is None:
    raise PackageNotFoundError("To use vllm rollout, please ensure the 'vllm' package is properly installed. See https://verl.readthedocs.io/en/latest/start/install.html for more details")

###
# package_version = get_version(package_name)
# [SUPPORT AMD:]
# Do not call any torch.cuda* API here, or ray actor creation import class will fail.
if "ROCM_PATH" in os.environ:
    import re

    match = re.match(r"(\d+\.\d+\.?\d*)", vllm_package_version)
    if match:
        vllm_package_version = match.group(1)
    else:
        raise ValueError(f"Warning: Could not parse version format: {vllm_package_version}")
###

if Version(vllm_package_version) <= Version("0.6.3"):
    vllm_mode = "customized"
    from .fire_vllm_rollout import FIREvLLMRollout  # noqa: F401
    from .vllm_rollout import vLLMRollout  # noqa: F401
else:
    vllm_mode = "spmd"
    from .vllm_rollout_spmd import vLLMAsyncRollout, vLLMRollout  # noqa: F401

# ============================================================
# V100 compatibility patches:
# 1. Avoid triton prefix_prefill kernel (LLVM crash on sm_70)
# 2. Disable flash_attn cross-entropy (uses bf16 triton ops)
# 3. Replace flash_attention_forward with sdpa for actor model
# ============================================================

def _apply_v100_patches():
    """Apply monkey-patches for V100 (sm_70) GPU compatibility.
    
    These patches are safe to apply on any GPU:
    - On pre-Ampere GPUs, they avoid triton kernel crashes
    - On Ampere+ GPUs, FlashAttention-2 is used instead of XFormers,
      so the XFormers patch is never triggered
    
    NOTE: We must NOT call torch.cuda.* APIs at import time, as this
    breaks Ray actor creation. GPU capability checks are deferred to
    runtime inside the patched functions.
    """

    # Patch 1: Replace triton prefix_prefill kernel with xformers fallback
    # The triton kernel context_attention_fwd crashes with
    # "LLVM ERROR: Failed to compute parent layout for slice layout" on sm_70.
    # We monkey-patch XFormersImpl.forward to always use the xformers
    # memory-efficient attention path (C++ backend) for prefill, instead of
    # the triton prefix_prefill kernel path.
    try:
        from vllm.attention.backends.xformers import XFormersImpl
        _original_xformers_forward = XFormersImpl.forward

        def _patched_xformers_forward(self, layer, query, key, value,
                                      kv_cache, attn_metadata, output=None):
            import torch
            # Force block_tables to empty for prefill metadata so that
            # the code takes the _run_memory_efficient_xformers_forward path
            # instead of the triton PagedAttention.forward_prefix path.
            # This is safe when prefix caching is disabled because:
            # - KV cache is already written before this check
            # - During prefill without prefix caching, new KV == all KV
            prefill_meta = getattr(attn_metadata, 'prefill_metadata', None)
            saved_block_tables = None
            if prefill_meta is not None and hasattr(prefill_meta, 'block_tables'):
                bt = prefill_meta.block_tables
                if bt is not None and bt.numel() > 0:
                    saved_block_tables = bt
                    prefill_meta.block_tables = torch.empty(
                        0, dtype=torch.int32, device=query.device)
            try:
                return _original_xformers_forward(
                    self, layer, query, key, value, kv_cache,
                    attn_metadata, output)
            finally:
                if saved_block_tables is not None:
                    prefill_meta.block_tables = saved_block_tables

        XFormersImpl.forward = _patched_xformers_forward
    except Exception as e:
        print(f"[V100 Patch] Warning: Failed to patch XFormersImpl: {e}")

    # Patch 2: Replace xformers memory_efficient_attention_forward with PyTorch SDPA
    # The xformers cutlass kernel crashes intermittently on V100 with
    # "CUDA error: an illegal memory access was encountered".
    # We replace the core xformers call with torch.nn.functional.scaled_dot_product_attention
    # which is stable on V100.
    try:
        from vllm.attention.backends.xformers import XFormersImpl
        import torch.nn.functional as _F_sdpa

        _original_run_xformers = XFormersImpl._run_memory_efficient_xformers_forward

        def _sdpa_run_memory_efficient_forward(self, query, key, value,
                                                attn_metadata, attn_type=None):
            """Replace xformers cutlass with PyTorch SDPA for V100 compatibility."""
            import torch
            from vllm.attention.backends.xformers import AttentionType

            if attn_type is None:
                attn_type = AttentionType.DECODER

            original_query = query

            # Handle GQA: expand key/value to match query heads
            if self.num_kv_heads != self.num_heads:
                query = query.view(query.shape[0], self.num_kv_heads,
                                   self.num_queries_per_kv, query.shape[-1])
                key = key[:, :, None, :].expand(key.shape[0], self.num_kv_heads,
                                                self.num_queries_per_kv, key.shape[-1])
                value = value[:, :, None, :].expand(value.shape[0], self.num_kv_heads,
                                                    self.num_queries_per_kv, value.shape[-1])

            # Process each sequence independently using SDPA
            assert attn_metadata.seq_lens is not None
            seq_lens = attn_metadata.seq_lens
            output = torch.empty_like(original_query)
            is_causal = (attn_type == AttentionType.DECODER)

            start = 0
            for seq_len in seq_lens:
                end = start + seq_len
                if seq_len == 0:
                    start = end
                    continue

                q_seq = query[start:end]  # (seq_len, num_kv_heads, [G,] head_size)
                k_seq = key[start:end]
                v_seq = value[start:end]

                if q_seq.dim() == 4:
                    # GQA: (seq_len, num_kv_heads, G, head_size)
                    # Merge kv_heads and G: (seq_len, num_heads, head_size)
                    s, h, g, d = q_seq.shape
                    q_seq = q_seq.reshape(s, h * g, d)
                    k_seq = k_seq.reshape(s, h * g, d)
                    v_seq = v_seq.reshape(s, h * g, d)

                # SDPA expects (batch, heads, seq_len, head_dim)
                q_sdpa = q_seq.permute(1, 0, 2).unsqueeze(0)  # (1, heads, seq, dim)
                k_sdpa = k_seq.permute(1, 0, 2).unsqueeze(0)
                v_sdpa = v_seq.permute(1, 0, 2).unsqueeze(0)

                out = _F_sdpa.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa,
                    is_causal=is_causal,
                    scale=self.scale,
                    dropout_p=0.0)

                # (1, heads, seq, dim) -> (seq, heads, dim)
                out = out.squeeze(0).permute(1, 0, 2)
                output[start:end] = out
                start = end

            return output

        XFormersImpl._run_memory_efficient_xformers_forward = _sdpa_run_memory_efficient_forward
        print("[V100 Patch] Replaced xformers cutlass with PyTorch SDPA")
    except Exception as e:
        print(f"[V100 Patch] Warning: Failed to patch xformers with SDPA: {e}")

    # Patch 3: Disable flash_attn cross-entropy loss (uses bf16 triton ops)
    try:
        import verl.utils.torch_functional as vtf
        vtf.FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False
    except Exception:
        pass

    # Patch 4: Also patch context_attention_fwd directly as a safety net.
    # Even if the XFormersImpl patch above works, this ensures any direct
    # calls to context_attention_fwd also use a safe fallback.
    try:
        import vllm.attention.ops.prefix_prefill as _prefix_prefill_mod
        import vllm.attention.ops.paged_attn as _paged_attn_mod
        _orig_context_attn = _prefix_prefill_mod.context_attention_fwd

        def _safe_context_attention_fwd(q, k, v, o, kv_cache_dtype, k_cache,
                                        v_cache, b_loc, b_start_loc,
                                        b_seq_len, max_seq_len, max_input_len,
                                        k_scale, v_scale, alibi_slopes=None,
                                        sliding_window=None, sm_scale=None,
                                        skip_decode=False):
            """PyTorch SDPA fallback for triton prefix_prefill on V100."""
            import torch
            import torch.nn.functional as F
            num_tokens, num_heads, head_dim = q.shape
            num_kv_heads = k.shape[1]
            batch_size = b_seq_len.shape[0]
            groups = num_heads // num_kv_heads

            if sm_scale is None:
                sm_scale = 1.0 / (head_dim ** 0.5)

            for i in range(batch_size):
                start = b_start_loc[i].item()
                if i + 1 < b_start_loc.shape[0]:
                    end = b_start_loc[i + 1].item()
                else:
                    end = num_tokens
                seq_len = end - start
                if seq_len == 0:
                    continue

                q_i = q[start:end].permute(1, 0, 2).unsqueeze(0)
                k_i = k[start:end].permute(1, 0, 2).unsqueeze(0)
                v_i = v[start:end].permute(1, 0, 2).unsqueeze(0)

                if groups > 1:
                    k_i = k_i.repeat_interleave(groups, dim=1)
                    v_i = v_i.repeat_interleave(groups, dim=1)

                out_i = F.scaled_dot_product_attention(
                    q_i, k_i, v_i, is_causal=True, scale=sm_scale)
                o[start:end] = out_i.squeeze(0).permute(1, 0, 2)

        _prefix_prefill_mod.context_attention_fwd = _safe_context_attention_fwd
        # Also patch the reference in paged_attn module
        _paged_attn_mod.context_attention_fwd = _safe_context_attention_fwd
    except Exception as e:
        print(f"[V100 Patch] Warning: Failed to patch context_attention_fwd: {e}")


_apply_v100_patches()
