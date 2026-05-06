[![Tips](https://img.shields.io/badge/%E2%98%95_Tips-Support_the_work-ff5e5b?style=flat)](#support-the-work)

## Quick Links

| Resource | Link |
|---|---|
| **Model Weights + Full Documentation** | [AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4 on HuggingFace](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) |
| **DFlash vLLM Container (DGX Spark)** | [`ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`](https://github.com/users/AEON-7/packages/container/package/aeon-gemma-4-26b-a4b-dflash) |
| **DFlash Drafter** | [z-lab/gemma-4-26B-A4B-it-DFlash](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) |

## Quick Start

```bash
# 1. Pull the DGX Spark / GB10 DFlash v2 image.
docker pull ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2

# 2. Download the target model and DFlash drafter.
mkdir -p models
huggingface-cli download AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4 \
  --local-dir ./models/gemma4
huggingface-cli download z-lab/gemma-4-26B-A4B-it-DFlash \
  --local-dir ./models/gemma4-dflash

# 3. Serve with native Blackwell FP4 kernels + DFlash k=15.
docker run --gpus all --ipc host --network host \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_MATMUL_PRECISION=high \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -v "$PWD/models/gemma4:/models/gemma4:ro" \
  -v "$PWD/models/gemma4-dflash:/models/gemma4-dflash:ro" \
  ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2 \
  vllm serve /models/gemma4 \
    --served-model-name gemma4-aeon-uncensored gemma4-fast gemma4-deep \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.76 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --speculative-config '{"method":"dflash","model":"/models/gemma4-dflash","num_speculative_tokens":15,"attention_backend":"flash_attn"}'
```

This profile is tuned for interactive and low-to-medium concurrency agent traffic on DGX Spark. `:latest` tracks the same v2 image. For longer per-request contexts, raise `--max-model-len` and reduce `--max-num-seqs` to match the KV cache budget.

## Model Specs

| Property | Value |
|---|---|
| Architecture | Gemma 4 Mixture of Experts |
| Total / Active Parameters | 26B / ~4B per token (top-8 of 128 experts) |
| Layers | 30 (25 sliding-window + 5 full-attention) |
| Max Context | 262,144 tokens |
| Quantization | NVFP4 (compressed-tensors) |
| Model Size on Disk | 15.3 GB |
| VRAM Loaded | 16.25 GB |
| Vision | 27-layer ViT (BF16) |
| Tool Calling | Native Gemma 4 format |

## Performance (DGX Spark GB10)

Benchmarked with [`ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`](https://github.com/users/AEON-7/packages/container/package/aeon-gemma-4-26b-a4b-dflash) on NVIDIA DGX Spark (GB10, SM 12.1, 128 GB unified memory). The server used the official vLLM 0.20.1 base with the AEON DFlash overlay baked into a single container, native FlashInfer CUTLASS NVFP4 GEMM, VLLM CUTLASS MoE, CUDA graphs, `--gpu-memory-utilization 0.76`, `--max-num-batched-tokens 32768`, `--max-num-seqs 256`, and DFlash `num_speculative_tokens=15`.

Interactive sweep: these are the most relevant numbers for chat, coding, tool use, and small agent teams.

| Category | c=1 tok/s | c=1 TTFT p50 | c=1 TPOT p50 | c=4 agg tok/s | c=8 agg tok/s | c=16 agg tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Coding | 81.60 | 94 ms | 11.83 ms | 223.92 | 481.38 | 740.22 |
| Math | 56.19 | 880 ms | 13.28 ms | 248.34 | 421.30 | 614.34 |
| Reasoning | 63.09 | 251 ms | 14.62 ms | 215.63 | 352.00 | 533.43 |
| Prose | 39.81 | 258 ms | 23.83 ms | 152.79 | 247.07 | 405.97 |
| Natural language | 44.52 | 238 ms | 21.10 ms | 183.90 | 321.85 | 491.19 |
| Extraction / JSON | 65.92 | 811 ms | 5.46 ms | 411.85 | 743.66 | 1,299.40 |

High-concurrency sweep: 6 natural prompt categories x 8 concurrency levels (`1, 4, 8, 16, 32, 64, 128, 256`) = 48 benchmark points, **0 request errors**.

| Category | c=1 tok/s | Peak aggregate tok/s | c=256 aggregate tok/s | c=256 TTFT p50 |
|---|---:|---:|---:|---:|
| Coding | 81.60 | 1,142.76 @ c=64 | 1,076.05 | 3,965 ms |
| Math | 56.19 | 992.82 @ c=64 | 947.76 | 1,651 ms |
| Reasoning | 63.09 | 874.56 @ c=256 | 874.56 | 782 ms |
| Prose | 39.81 | 591.29 @ c=64 | 541.10 | 1,232 ms |
| Natural language | 44.52 | 653.83 @ c=128 | 647.37 | 1,144 ms |
| Extraction / JSON | 65.92 | 2,069.83 @ c=128 | 2,066.46 | 917 ms |

DFlash v2 is strongest for interactive decode and short agent/tool-call bursts. At very high concurrency, the drafter adds scheduling pressure and per-request latency rises; c=256 is best read as a saturation probe, not the recommended production target. For raw many-request aggregate throughput without speculation, compare against the stock vLLM baseline below.

## Stock Community vLLM Baseline (No DFlash)

Benchmarked with the official community image `vllm/vllm-openai:latest` pulled on 2026-05-06 (`vLLM 0.20.1`, PyTorch `2.11.0+cu130`, transformers `5.7.0`, image digest `sha256:9eff9734a30b6713a8566217d36f8277630fd2d31cec7f0a0292835901a23aa4`). This run used the same model weights, 32K context, `--max-num-batched-tokens 32768`, and `--max-num-seqs 256`, but no DFlash drafter and no AEON container env overrides. Upstream vLLM now boots this model on GB10 with FlashInfer CUTLASS NVFP4 linear kernels and VLLM CUTLASS MoE.

Full sweep: 6 natural prompt categories x 8 concurrency levels (`1, 4, 8, 16, 32, 64, 128, 256`) = 48 benchmark points, **0 request errors**.

| Category | c=1 tok/s | c=1 TTFT p50 | Peak aggregate tok/s | c=256 aggregate tok/s | c=256 TTFT p50 |
|---|---:|---:|---:|---:|---:|
| Coding | 49.12 | 130.7 ms | 3,356.61 @ c=256 | 3,356.61 | 542 ms |
| Math | 48.79 | 134.0 ms | 3,006.60 @ c=256 | 3,006.60 | 1,078 ms |
| Reasoning | 48.90 | 113.8 ms | 3,241.42 @ c=256 | 3,241.42 | 274 ms |
| Prose | 48.86 | 115.9 ms | 3,222.85 @ c=256 | 3,222.85 | 662 ms |
| Natural language | 49.38 | 72.4 ms | 3,418.94 @ c=256 | 3,418.94 | 650 ms |
| Extraction / JSON | 47.34 | 120.6 ms | 3,674.70 @ c=256 | 3,674.70 | 385 ms |

Use the stock community path when raw many-request aggregate throughput matters more than speculative single-stream speed. Use the DFlash image when you want the lower interactive TPOT and the integrated Gemma 4 DFlash serving recipe.

---

## Why This Is Hard: Gemma 4 on DGX Spark

Running Gemma 4 NVFP4 on a DGX Spark used to require a source-built stack. As of the 2026-05-06 community `vllm/vllm-openai:latest` image, upstream vLLM can boot this model on GB10, and AEON's v2 image packages the optimized DFlash path as a single pull-and-run container. Every layer of the stack, from the silicon to the serving framework to the model weights themselves, has had compatibility gaps worth understanding.

### The DGX Spark Problem

The NVIDIA DGX Spark ships with a **GB10 Grace Blackwell** chip: SM 12.1 on ARM64 (aarch64). This is bleeding-edge silicon that much of the ML ecosystem is still catching up to:

- **Python wheels remain risky on SM 12.1.** Official PyPI releases have historically targeted SM 8.0/8.9/9.0 (Ampere/Ada/Hopper). Installing `pip install vllm` can give you CUDA kernels compiled for the wrong GPU; use a tested Docker image or build from source.
- **No pre-built FlashInfer wheels** for SM 12.1. FlashInfer provides the fused MoE dispatch kernels that make expert routing fast. Without it compiled for your architecture, MoE models can't use the optimized CUTLASS/Triton backends.
- **ARM64 architecture** means many x86-only prebuilt binaries don't run at all. Even when packages claim CUDA support, the host-side code is often x86-compiled.
- **273 GB/s memory bandwidth**: fast for a desktop-class device, but a fraction of what data center GPUs offer (H100: 3.35 TB/s, A100: 2 TB/s). This makes model architecture choice critical: dense models that need to read all parameters every token are bandwidth-starved here.

The practical result: current stock vLLM can serve this model, but high-confidence production recipes still need to pin image versions, model format, attention backend, KV dtype, and concurrency settings instead of assuming any vLLM tag will behave the same way.

### The Gemma 4 Problem

Gemma 4 is not just a new model. It is architecturally unusual in ways that break assumptions in existing tooling:

**1. Requires transformers v5+ (nothing else does yet)**

Gemma 4 was the first major model to require the `transformers` v5 major version bump. Older stock vLLM images shipped with v4.x and failed to parse the Gemma 4 config. Current community images may include transformers v5, but pin the version because v4/v5 API differences can still break model loading.

**2. Heterogeneous attention head dimensions**

Most models have uniform head dimensions across all layers. Gemma 4 has `head_dim=256` for sliding-window layers and `global_head_dim=512` for full-attention layers. This breaks attention backends that assume a single head dimension. vLLM forces the `TRITON_ATTN` backend specifically for Gemma 4 to handle this — other backends (FlashAttention, FlashInfer attention) produce numerical divergence or crash.

**3. Hybrid sliding-window + full-attention layers**

Of the 30 layers, 25 use a sliding window of 1024 tokens and 5 use full global attention. The sliding-window layers use regular MoE (128 experts, top-8), while the full-attention layers use dense MLPs. This means the model has two completely different layer types with different weight shapes, different compute patterns, and different KV cache requirements — all interleaved.

**4. Massive MoE expert count**

128 experts per layer with top-8 routing. That's 128 x 25 = 3,200 expert weight matrices in the MoE layers alone, each with 4 NVFP4 tensors (weight_packed, weight_scale, weight_global_scale, input_global_scale). The total tensor count in this model is **47,648**. Loading and routing these correctly requires FusedMoE kernels that can handle the stacked expert format, and the compressed-tensors naming convention doesn't match what vLLM expects (see below).

### The NVFP4 Quantization Problem

NVFP4 (4-bit NormalFloat) quantization is how we get a 26B-parameter model into 15.3 GB. But there are two completely different NVFP4 formats in the ecosystem, and they are not compatible:

**ModelOpt NVFP4** (NVIDIA's TensorRT-LLM toolchain): Stores weights as `weight`, `weight_scale_inverse`, `input_scale`. This is what NVIDIA's own tools produce and what most vLLM NVFP4 code paths expect.

**Compressed-tensors NVFP4** (llmcompressor/vLLM community): Stores weights as `weight_packed`, `weight_scale`, `weight_global_scale`, `input_global_scale`. Different tensor names, different scale conventions, different packing format.

This model uses compressed-tensors format (quantized with llmcompressor on an H200). vLLM's Gemma 4 weight loader has hard-coded assumptions about tensor naming that don't match. Specifically:

- **Expert path mismatch**: Compressed-tensors names MoE experts as `layers.X.experts.{id}.{proj}.weight_packed`. vLLM's FusedMoE expects `layers.X.moe.experts.{id}.{proj}.weight_packed` — note the `.moe.` segment. Without patching, every single expert tensor fails to load with a KeyError.
- **Suffix format mismatch**: The weight loader constructs names like `w2_weight.weight_packed` when it should be `w2_weight_packed`. The `_weight.` needs to be collapsed to `_`.
- **Dimension assertion failure**: The original code asserts `dim == 2` for weight tensors, but NVFP4 packed tensors have different dimensionality due to the 4-bit packing.

The included `gemma4_patched.py` fixes all three issues with targeted patches to the weight loading pipeline.

### The Accidental Quantization Problem

When quantizing with llmcompressor, you specify ignore patterns for layers that should stay in BF16 (full precision). The original quantization used patterns like `re:.*visual.*` and `re:.*gate.*` to skip vision and routing layers. But Gemma 4's naming conventions didn't match:

| Layer | Expected Pattern | Actual Name in Gemma 4 | Result |
|---|---|---|---|
| Vision tower | `re:.*visual.*` | `model.vision_tower.*` | **Quantized** (wrong) |
| Vision embedding | `re:.*visual.*` | `model.embed_vision.*` | **Quantized** (wrong) |
| MoE routers | `re:.*gate.*` | `model.*.router.proj.*` | **Quantized** (wrong) |

Quantizing these layers breaks the model:
- **Vision tower** in NVFP4 crashes because vLLM allocates standard `Linear` layers (expects `.weight` tensor, gets `weight_packed`/`weight_scale`/etc.)
- **MoE routers** in NVFP4 corrupts expert routing — the router decides which experts to activate for each token, and 4-bit precision on routing logits causes degenerate expert selection
- **Vision embedding projection** bridges the ViT output to the language model — quantization here cascades errors through every subsequent layer

We fixed this by extracting the original BF16 weights from the base model ([TrevorJS/gemma-4-26B-A4B-it-uncensored](https://huggingface.co/TrevorJS/gemma-4-26B-A4B-it-uncensored)) and replacing the incorrectly quantized tensors in the safetensors file:
- **760 NVFP4 tensors removed** from the vision tower, replaced with **190 original BF16 weights** (355 total vision tensors including biases and layernorms)
- **120 NVFP4 tensors removed** from router.proj layers, replaced with **30 BF16 weights**
- **4 NVFP4 tensors removed** from embed_vision, replaced with **1 BF16 weight**

### The Token Leakage Problem

Gemma 4 uses internal control tokens for multi-channel generation (thinking, tool calls, output). These tokens have specific IDs in the vocabulary:

| Token ID | Token | Purpose |
|---:|---|---|
| 100 | `<\|channel>` | Start internal channel (e.g., thinking) |
| 101 | `<channel\|>` | End internal channel |
| 98 | `<\|think\|>` | Enter thinking mode |
| 48 | `<\|tool_call>` | Start tool call |
| 49 | `<tool_call\|>` | End tool call |

Without proper EOS configuration, the model can enter its "thinking" channel mid-generation, and those internal tokens stream through as plaintext in the API response. Worse, it can get stuck in a repetition loop — endlessly generating `<|channel>thought<channel|>call:process{...}` as visible text. This manifests as the model appearing to "spam" garbage in the chat.

The fix is adding tokens 98, 100, and 101 to the `eos_token_id` list in `generation_config.json`, so vLLM terminates generation cleanly before any internal channel tokens leak into the output.

### What's In The Container (The Special Sauce)

The `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2` container starts from the official community vLLM 0.20.1 runtime and bakes in the AEON Gemma 4 DFlash overlay. Users pull one image; no local patching or source build is required.

| Component | What It Is | Why It Matters |
|---|---|---|
| **Official vLLM 0.20.1 base** | Upstream `vllm/vllm-openai` runtime | Keeps the strong low-concurrency behavior of the community image while adding the DFlash path. |
| **AEON DFlash overlay** | Python patchset baked into site-packages at build time | Adds `method="dflash"` support, Gemma 4 drafter wiring, and backend isolation so users do not apply patches manually. |
| **PyTorch 2.11.0 + CUDA 13 runtime** | Framework + CUDA runtime from the official image | Provides current SM 12.1 support for GB10. |
| **transformers 5.7.0+** | Model config/tokenizer loading | Gemma 4 support requires transformers v5+. |
| **DFlash drafter** | `z-lab/gemma-4-26B-A4B-it-DFlash`, k=15 | Speculative decoding for the Gemma 4 26B A4B target model. |
| **Native FP4 CUTLASS kernels** | FlashInfer CUTLASS for linear layers, VLLM CUTLASS for MoE | Do not force Marlin on this image; the native FP4 path is faster on GB10. |
| **TRITON_ATTN backend** | Attention computation | Handles Gemma 4's heterogeneous head dimensions (256/512) without numerical divergence. Other backends assume uniform head_dim. |
| **FlashAttention drafter backend** | DFlash draft attention | Keeps non-causal DFlash attention on a backend that supports it while the Gemma target model stays on Triton attention. |
| **torch.compile + CUDA graphs** | Graph capture and kernel fusion | Captures decode graphs for the configured batch sizes, reducing Python overhead on the decode hot path. |

### Why MoE Makes This Possible

The fundamental constraint on DGX Spark is memory bandwidth: **273 GB/s**. During autoregressive decode, the GPU must read the model weights for every single token generated. This is what determines tok/s:

```
tok/s = memory_bandwidth / bytes_read_per_token
```

For a **dense 27B model at NVFP4** (~13.5 GB weights):
```
273 GB/s / 13.5 GB = ~20 tok/s (theoretical max, before KV cache and overhead)
```

For this **MoE model** (top-8 of 128 experts, ~2.8 GB active per token):
```
273 GB/s / 2.8 GB = ~97 tok/s (theoretical max)
```

We achieve **~40-82 tok/s single-stream** depending on prompt shape, with enough headroom to pass 1,000 aggregate tok/s on coding and more than 2,000 aggregate tok/s on extraction/JSON workloads. The gap from the theoretical limit comes from KV cache reads, attention computation, router overhead, drafter verification, and memory access patterns. But the key insight is that MoE turns a bandwidth-impossible problem (dense 27B) into a bandwidth-comfortable one.

| Model Type | Params Read/Token | Max tok/s on GB10 | Practical tok/s |
|---|---|---|---|
| Dense 27B BF16 | ~54 GB | 5 | Not viable |
| Dense 27B NVFP4 | ~13.5 GB | 20 | ~15 |
| **MoE 26B top-8/128 NVFP4 + DFlash** | **~2.8 GB + drafter** | **97** | **40-82 c=1, 1K+ aggregate** |

This is why architecture choice matters more than raw parameter count on bandwidth-limited hardware. A 26B MoE model at NVFP4 is faster than a dense 7B at BF16 on the same hardware.

## Container Image Details

### DFlash v2 Image

**`ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`**

`latest` currently points to the same v2 image.

| Component | Version |
|---|---|
| vLLM | 0.20.1 official base + AEON DFlash overlay |
| PyTorch | 2.11.0+cu130 |
| transformers | 5.7.0+ |
| AEON overlay revision | `06e292d0ce7e0ddc4f84bd200c3bdf55c7875eb7` |
| DFlash drafter | z-lab/gemma-4-26B-A4B-it-DFlash |
| Target GPU | NVIDIA GB10 (DGX Spark, SM 12.1) |

The v2 image is the recommended default for real interactive use cases. Previous tags such as `v0.1.0` and `pr41703-20260506` remain available for historical comparison and high-concurrency experiments.

### Stock Community Baseline Image

**`vllm/vllm-openai:latest@sha256:9eff9734a30b6713a8566217d36f8277630fd2d31cec7f0a0292835901a23aa4`**

| Component | Version |
|---|---|
| vLLM | 0.20.1 |
| PyTorch | 2.11.0+cu130 |
| transformers | 5.7.0 |
| Speculative decoding | None |

This image is useful as a current upstream reference point. It is not the AEON DFlash package and does not include the Gemma 4 DFlash drafter path.

## All Fixes Included

This model required several post-quantization fixes to work correctly with vLLM. **All fixes are baked into the HuggingFace release** — no additional debugging needed:

- De-quantized 760 vision tower tensors (27 ViT layers), 120 router tensors (30 MoE layers), and 4 embedding projection tensors — all restored from original BF16 weights
- Patched vLLM weight loader for compressed-tensors NVFP4 MoE format (`gemma4_patched.py` — 3 targeted patches to `_weight_iterator` and `load_weights`)
- Added `audio_config` and `num_experts_per_tok` to `config.json` (vLLM config parser requirements)
- Created `preprocessor_config.json` and `processor_config.json` for multimodal support
- Configured EOS token IDs [1, 106, 50, 98, 100, 101] to prevent thinking/channel token leakage

Full technical details: [HuggingFace Model Card](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4)

## Related Models

| Model | Type | Size | tok/s (DGX Spark) | Links |
|---|---|---|---|---|
| **This model (Gemma 4 26B MoE + DFlash v2)** | MoE NVFP4 | 15.3 GB | 81.6 c=1 coding / 1,143 aggregate coding / 2,070 aggregate extraction | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) |
| **Gemma 4 31B DECKARD AWQ_FULL** | Dense NVFP4 | 20.5 GB | ~12-14 | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4) \| [GitHub](https://github.com/AEON-7/Gemma-4-31B-DECKARD-HERETIC-Uncensored-NVFP4) |
| **Gemma 4 31B DECKARD SVDQuant** | Dense NVFP4 | 20.9 GB | ~10-13 | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4-SVDQuant) |
| **Qwen3.5-27B Uncensored** | Dense NVFP4 | ~15 GB | ~15-18 | [HuggingFace](https://huggingface.co/AEON-7/Qwen3.5-27B-Uncensored-NVFP4) |

> **MoE vs Dense**: The MoE model is 3-4x faster than dense models because it only reads ~4B parameters per token (top-8 of 128 experts) vs 27-31B for dense models. Choose MoE for speed and concurrency, dense for maximum quality.

## Disclaimer, Liability Waiver, and Assumption of Risk

**THIS IS AN UNCENSORED MODEL.** By downloading, accessing, or using this model, the associated container image ([`ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash`](https://github.com/users/AEON-7/packages/container/package/aeon-gemma-4-26b-a4b-dflash)), or any derivative works thereof, you expressly acknowledge and agree to the following:

### Assumption of Risk

Uncensored language models present materially elevated risks compared to safety-aligned models, including but not limited to: generation of harmful, misleading, illegal, or objectionable content; susceptibility to adversarial misuse; potential for facilitating activities that violate applicable laws or regulations; and amplified risk in automated or agentic pipelines where outputs may be executed without human review.

These tools are powerful and serve a multitude of legitimate and essential purposes — including security research, red-teaming, content analysis, creative work, and applications where safety filters interfere with valid use cases. However, the absence of safety guardrails demands a correspondingly higher standard of care from the operator. **You must implement your own safeguards, content filtering, access controls, and monitoring appropriate to your use case and jurisdiction.**

### Limitation of Liability

The authors, contributors, and distributors of this model and container image ("Providers") are not responsible or liable, directly or indirectly, for any actions taken, content generated, damages incurred, or legal consequences arising from the use or misuse of these materials. This includes, without limitation:

- Any harmful, illegal, unethical, or objectionable outputs produced by the model
- Any decisions made or actions taken based on model outputs
- Any damages — direct, indirect, incidental, consequential, special, or exemplary — arising from the use of the model or container, regardless of whether the Providers were advised of the possibility of such damages
- Any violation of local, state, national, or international laws or regulations by the user

### User Responsibility

**You, the user, assume full and sole responsibility and liability** for:

- All outputs generated by the model under your operation
- Ensuring your use complies with all applicable laws, regulations, and ethical standards in your jurisdiction
- Implementing appropriate access controls, content filtering, and human oversight
- Any consequences of deploying this model in production, automated, or public-facing systems
- Evaluating whether an uncensored model is appropriate for your specific use case

### Acceptance

**By downloading or using any component of this release — including the model weights, container image, configuration files, or patched code — you indicate your acceptance of these terms and your assumption of all associated risks and liabilities.** If you do not agree to these terms, do not download or use these materials.

## License

This model inherits the [Gemma license](https://ai.google.dev/gemma/terms) from Google.

## Support the work

If this release has been useful, tips are deeply appreciated. They go directly toward more compute, more models, and more open releases.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <strong>Bitcoin (BTC)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/btc.png" alt="BTC QR" width="200"/><br/>
      <sub><code>bc1q09xmzn00q4z3c5raene0f3pzn9d9pvawfm0py4</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>Ethereum (ETH)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/eth.png" alt="ETH QR" width="200"/><br/>
      <sub><code>0x1512667F6D61454ad531d2E45C0a5d1fd82D0500</code></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Solana (SOL)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/sol.png" alt="SOL QR" width="200"/><br/>
      <sub><code>DgQsjHdAnT5PNLQTNpJdpLS3tYGpVcsHQCkpoiAKsw8t</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>Monero (XMR)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/xmr.png" alt="XMR QR" width="200"/><br/>
      <sub><code>836XrSKw4R76vNi3QPJ5Fa9ugcyvE2cWmKSPv3AhpTNNKvqP8v5ba9JRL4Vh7UnFNjDz3E2GXZDVVenu3rkZaNdUFhjAvgd</code></sub>
    </td>
  </tr>
</table>

> **Ethereum L2s (Base, Arbitrum, Optimism, Polygon, etc.) and EVM-compatible tokens** can be sent to the same Ethereum address.
