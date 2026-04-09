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

### Why MoE Fits DGX Spark

The GB10 has 273 GB/s memory bandwidth. A dense 27B NVFP4 model would need ~675 GB/s at 50 tok/s — impossible. This MoE model activates only ~4B params/token, requiring just ~140 GB/s (51% of bandwidth), leaving headroom for concurrent requests.

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

## What's Included

This model required several post-quantization fixes to work correctly with vLLM. All fixes are included in the HuggingFace release — no additional debugging needed:

- De-quantized vision tower, embedding projection, and MoE router layers (incorrectly quantized to NVFP4, restored to BF16)
- Patched vLLM weight loader for compressed-tensors NVFP4 MoE format (`gemma4_patched.py`)
- Added missing processor configs for multimodal support
- Configured EOS tokens to prevent internal thinking/channel token leakage

Full technical details: [HuggingFace Model Card](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4)

## License

This model inherits the [Gemma license](https://ai.google.dev/gemma/terms) from Google.
