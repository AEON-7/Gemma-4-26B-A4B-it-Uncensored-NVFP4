# Gemma 4 26B-A4B-it Uncensored NVFP4

NVFP4-quantized [Gemma 4 26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (abliterated/uncensored) optimized for NVIDIA DGX Spark and Blackwell GPUs. Quantized from [TrevorJS/gemma-4-26B-A4B-it-uncensored](https://huggingface.co/TrevorJS/gemma-4-26B-A4B-it-uncensored) using [llmcompressor](https://github.com/vllm-project/llmcompressor).

## Quick Links

| Resource | Link |
|---|---|
| **Model Weights + Full Documentation** | [AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4 on HuggingFace](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) |
| **Pre-built vLLM Container (DGX Spark)** | [ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4](https://github.com/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4) |

## Quick Start

```bash
# 1. Pull the pre-built vLLM container (DGX Spark / SM 12.1)
docker pull ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest

# 2. Download the model
huggingface-cli download AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4 --local-dir ./model

# 3. Run
docker run --gpus all --ipc host -p 8000:8000 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -v ./model:/models/Gemma-4-26B-A4B-it-Uncensored-NVFP4 \
  -v ./model/gemma4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py \
  ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest \
  vllm serve /models/Gemma-4-26B-A4B-it-Uncensored-NVFP4 \
    --served-model-name Gemma-4-26B-A4B-it-Uncensored-NVFP4 \
    --max-model-len 262000 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --load-format safetensors \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4
```

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

Benchmarked with [`ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest`](https://github.com/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4) on NVIDIA DGX Spark (GB10, SM 12.1, 128 GB unified memory). **Zero failures** across all concurrency levels.

| Concurrent | Aggregate tok/s | Per-Request tok/s | Avg Latency (200 tok) |
|---:|---:|---:|---:|
| 1 | 50 | 50 | 4.0s |
| 2 | 89 | 45 | 4.5s |
| 4 | 144 | 36 | 5.6s |
| 8 | 159 | 20 | 10.1s |
| 16 | 368 | 23 | 8.7s |
| 32 | 599 | 19 | 10.7s |
| 64 | 951 | 15 | 13.5s |
| 128 | 1,430 | 11 | 17.9s |

---

## Why This Is Hard: Gemma 4 on DGX Spark

Running Gemma 4 NVFP4 on a DGX Spark is not a download-and-run situation. There is no pre-built path that works out of the box. Every layer of the stack — from the silicon to the serving framework to the model weights themselves — has a compatibility gap that had to be bridged. This section explains what those gaps are and how this release solves each one.

### The DGX Spark Problem

The NVIDIA DGX Spark ships with a **GB10 Grace Blackwell** chip — SM 12.1 on ARM64 (aarch64). This is bleeding-edge silicon that most of the ML ecosystem hasn't caught up to yet:

- **No pre-built vLLM wheels** exist for SM 12.1. The official PyPI releases target SM 8.0/8.9/9.0 (Ampere/Ada/Hopper). Installing `pip install vllm` gives you CUDA kernels compiled for the wrong GPU — they either crash with ABI mismatches or silently fall back to unoptimized paths.
- **No pre-built FlashInfer wheels** for SM 12.1. FlashInfer provides the fused MoE dispatch kernels that make expert routing fast. Without it compiled for your architecture, MoE models can't use the optimized CUTLASS/Triton backends.
- **ARM64 architecture** means many x86-only prebuilt binaries don't run at all. Even when packages claim CUDA support, the host-side code is often x86-compiled.
- **273 GB/s memory bandwidth** — fast for a desktop-class device, but a fraction of what data center GPUs offer (H100: 3.35 TB/s, A100: 2 TB/s). This makes model architecture choice critical: dense models that need to read all parameters every token are bandwidth-starved here.

The result: you can't just `docker pull` a stock vLLM image and serve Gemma 4. Everything must be compiled from source, targeting SM 12.1 specifically.

### The Gemma 4 Problem

Gemma 4 is not just a new model — it's architecturally unusual in ways that break assumptions in existing tooling:

**1. Requires transformers v5+ (nothing else does yet)**

Gemma 4 was the first major model to require the `transformers` v5 major version bump. Stock vLLM images ship with v4.x. Even if you have vLLM compiled for your GPU, it will fail to parse the Gemma 4 config without upgrading transformers — and upgrading transformers in a pre-built vLLM image risks breaking other dependencies due to API changes between v4 and v5.

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

The `ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4` container is a from-source build of the entire inference stack, targeting the GB10 specifically:

| Component | What It Is | Why It Matters |
|---|---|---|
| **vLLM 0.19.1rc1** | Inference engine, compiled with `TORCH_CUDA_ARCH_LIST=12.1a` | All CUDA kernels (attention, MoE, quantization, sampling) emit native SM 12.1 machine code. No JIT recompilation, no fallback to PTX emulation. |
| **FlashInfer 0.6.7** | Fused MoE dispatch kernels, compiled with `FLASHINFER_CUDA_ARCH_LIST=12.1a` | The `VLLM_CUTLASS` MoE backend uses FlashInfer's fused expert kernels. Without SM 12.1 compilation, MoE dispatch falls back to slow unfused paths. |
| **PyTorch 2.12.0 + CUDA 13.0** | Framework + CUDA runtime | CUDA 13.0 is required for full SM 12.1 support. Older CUDA versions either don't recognize GB10 or fall back to compatibility mode. |
| **transformers 5.5.0** | Model config/tokenizer loading | Gemma 4's architecture registration only exists in transformers v5+. The `--tf5` build flag handles the v4→v5 migration. |
| **Marlin W4A16 kernel** | NVFP4 weight decompression for dense GEMM | GB10 has no native FP4 tensor cores (those are only on data center Blackwell like B200). Marlin decompresses FP4→FP16 on-the-fly during GEMM, optimized for the memory-bandwidth-bound decode regime. |
| **TRITON_ATTN backend** | Attention computation | Handles Gemma 4's heterogeneous head dimensions (256/512) without numerical divergence. Other backends assume uniform head_dim. |
| **torch.compile + CUDA graphs** | Graph capture and kernel fusion | Captures the full decode graph as a CUDA graph for each batch size 1-256. Eliminates Python overhead and CPU-GPU synchronization on the decode hot path. ~2.5s one-time compilation cost at startup. |

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

We achieve **50 tok/s in practice** (51% efficiency) — the gap comes from KV cache reads, attention computation, router overhead, and memory access patterns. But the key insight is that MoE turns a bandwidth-impossible problem (dense 27B) into a bandwidth-comfortable one, with enough headroom to scale to 128 concurrent requests at 1,430 aggregate tok/s.

| Model Type | Params Read/Token | Max tok/s on GB10 | Practical tok/s |
|---|---|---|---|
| Dense 27B BF16 | ~54 GB | 5 | Not viable |
| Dense 27B NVFP4 | ~13.5 GB | 20 | ~15 |
| **MoE 26B top-8/128 NVFP4** | **~2.8 GB** | **97** | **50** |

This is why architecture choice matters more than raw parameter count on bandwidth-limited hardware. A 26B MoE model at NVFP4 is faster than a dense 7B at BF16 on the same hardware.

## Container Image Details

**`ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4:latest`**

| Component | Version |
|---|---|
| vLLM | 0.19.1rc1 (compiled for SM 12.1) |
| PyTorch | 2.12.0 + CUDA 13.0 |
| transformers | 5.5.0 |
| FlashInfer | 0.6.7 |
| Target GPU | NVIDIA GB10 (DGX Spark, SM 12.1) |

Built from [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) with `--tf5` flag. For other GPU architectures, see the [build-from-source instructions](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4#building-from-source) on the HuggingFace model page.

## All Fixes Included

This model required several post-quantization fixes to work correctly with vLLM. **All fixes are baked into the HuggingFace release** — no additional debugging needed:

- De-quantized 760 vision tower tensors (27 ViT layers), 120 router tensors (30 MoE layers), and 4 embedding projection tensors — all restored from original BF16 weights
- Patched vLLM weight loader for compressed-tensors NVFP4 MoE format (`gemma4_patched.py` — 3 targeted patches to `_weight_iterator` and `load_weights`)
- Added `audio_config` and `num_experts_per_tok` to `config.json` (vLLM config parser requirements)
- Created `preprocessor_config.json` and `processor_config.json` for multimodal support
- Configured EOS token IDs [1, 106, 50, 98, 100, 101] to prevent thinking/channel token leakage

Full technical details: [HuggingFace Model Card](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4)

## Disclaimer, Liability Waiver, and Assumption of Risk

**THIS IS AN UNCENSORED MODEL.** By downloading, accessing, or using this model, the associated container image ([`ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4`](https://github.com/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4)), or any derivative works thereof, you expressly acknowledge and agree to the following:

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
