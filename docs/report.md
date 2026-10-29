**Qwen2.5-7B-Instruct — FP8 Quantization & Cost Analysis**

> Cell letters in this report (A–J, G2, G3) refer to sections of the original working notebook. In the cleaned notebooks: quantization, tokenizer patch, perplexity (E) and profiling (J) are in `01_quantize_and_evaluate.ipynb`; serving, load tests (F/G/G3), offline batch (G2) and cost (I) are in `02_serve_and_benchmark.ipynb`.

*Project report · Google Colab (L4 GPU) · llmcompressor + vLLM*

Prepared from the working notebooks (now `notebooks/01_quantize_and_evaluate.ipynb` and `notebooks/02_serve_and_benchmark.ipynb`) — first written August 25, 2026; revised October 15, 2026 with a re-validation session, a long-prompt benchmark and a BF16 offline baseline

1\. Project Overview

The project set out to quantize Qwen2.5-7B-Instruct from its native BF16 weights down to FP8, then answer three practical questions about whether that quantization was worth doing: does it stay accurate, does it serve faster under concurrent load, and does it actually lower the per-token cost of running the model. The work runs entirely inside Google Colab against a single L4 GPU (24 GB card, ~22 GB usable), with the notebook designed as a budget-conscious workflow: install and import checks run first, and the GPU-heavy cells (quantization, serving, load testing) run only once the environment is known to work.

This mirrors, and extends, the methodology used in the earlier [Phi-2 quantization project](https://github.com/namrathasanjay/phi2-fp8-quantization-vllm) (GPTQ-based W4A8/W4A16/FP8 comparison on an H100). The Qwen project narrows the scope to a single scheme — FP8 dynamic quantization — but goes further operationally: it adds a live vLLM OpenAI-compatible server, an async concurrent load tester with latency percentiles, an offline batch benchmark, two different traffic shapes (short and long prompts), and a cost-per-million-tokens model.

The October 2026 revision matters for how the results should be read. The original benchmark used very short prompts, which flatters FP8. A second benchmark with production-length prompts (≈250–2,000 tokens) showed that FP8’s advantage depends strongly on the workload: about 30% cheaper for short-prompt chat, but only about 10% cheaper for long-prompt traffic under load. Section 5.6 explains why.

Model & Environment

| **Item**            | **Detail**                                                                                                                |
|---------------------|---------------------------------------------------------------------------------------------------------------------------|
| Model               | Qwen/Qwen2.5-7B-Instruct                                                                                                  |
| Hardware            | NVIDIA L4 (24 GB, ~22.03 GiB usable; ~300 GB/s memory bandwidth; 72 W power limit)                                        |
| Platform            | Google Colab notebook                                                                                                     |
| Quantization scheme | FP8_DYNAMIC: weights FP8 with static per-channel scales; activations FP8 with per-token scales computed at inference time |
| Serving engine      | vLLM 0.7.3 (OpenAI-compatible API server + offline batch mode), Python 3.12                                               |
| Storage             | Google Drive (quantized checkpoint persisted across sessions)                                                             |

Results at a Glance

| **Measurement**                     | **BF16 baseline**        | **FP8**                 | **FP8 vs BF16**   |
|-------------------------------------|--------------------------|-------------------------|-------------------|
| Checkpoint size                     | ~15 GB                   | 8.14 GB                 | ~46% smaller      |
| Perplexity (WikiText-2, n=20)       | 6.469                    | 6.536                   | +1.0% (worse)     |
| Decode time per token, 1 user       | 57.4–57.9 ms             | 36.9–37.0 ms            | 36% faster        |
| Short prompts, 50 users: throughput | 434.9 tok/s              | 605.1 tok/s             | +39%, 28% cheaper |
| Long prompts, 50 users: throughput  | 193.7 tok/s              | 213.0 tok/s             | +10%, 9% cheaper  |
| Long prompts, 50 users: median TTFT | 4.20 s                   | 1.65 s                  | 61% faster        |
| Offline batch, 240 short prompts    | 2,019 tok/s              | 2,298 tok/s             | +14%, 12% cheaper |
| KV cache memory available           | 3.02 GiB (~56.5K tokens) | 8.76 GiB (~164K tokens) | ~2.9× more room   |

2\. Idea Behind the Project

Running a 7B-parameter model in BF16 costs roughly twice the memory of an 8-bit representation, which directly caps how many concurrent users a single GPU can serve and how much each token costs to generate. FP8 is attractive on FP8-capable hardware (L4 and newer) for two reasons. First, it halves the bytes of weights that must be read from memory for every generated token. Second, FP8 tensor cores can in principle double matrix-multiply throughput compared with BF16. The project tested which of these actually shows up in practice on an L4, under realistic concurrent load rather than a single-prompt benchmark.

The answer turned out to be mostly the first. Decoding is memory-bound and got the full benefit (36% faster per token). Prompt processing (prefill) is compute-bound and saw little or no speed-up in this setup, which is why the gains shrink on long-prompt workloads (Section 5.6).

A second idea running through the notebook is the choice of quantization method. FP8_DYNAMIC computes activation scales on the fly at inference time, so it needs no calibration dataset — a deliberate contrast to the GPTQ approach used in the Phi-2 project, which required 256 calibration samples before it could quantize a single layer. That trade (calibration-free FP8 vs calibration-heavy GPTQ INT4) is itself part of what this project set out to validate: is the simpler FP8 path “good enough” for a 7B instruct model on FP8-native hardware.

3\. Requirements

To reproduce this work end to end, you need:

- A Colab GPU runtime with FP8 tensor-core support (L4 or H100; A100 has no FP8 tensor cores). This notebook targets L4.

- Google Drive mounted and writable, since the quantized checkpoint (~8 GB) is saved there and reloaded across separate Colab sessions.

- Two separate environments, run in two separate Colab sessions, because the quantization and serving stacks pin incompatible dependency versions (Section 5.1).

- **A Python 3.12 (or older) runtime for the serving session.** vLLM 0.7.3 and its pinned dependencies do not install on Python 3.13 (Section 7.1).

- Packages: llmcompressor + transformers + datasets for quantization; vllm==0.7.3 + transformers \<5.0 for serving; aiohttp + nest_asyncio for the concurrent load tester; torch.profiler and nvidia-smi for profiling.

4\. What the Project Was Trying to Achieve

Five concrete deliverables, one per stage of the notebook:

- A working FP8 checkpoint of Qwen2.5-7B-Instruct that loads and generates coherent text under vLLM.

- A quantitative accuracy check (perplexity on WikiText-2) comparing the FP8 checkpoint against the BF16 original (Section 5.9).

- A realistic serving benchmark: TTFT, inter-token latency (ITL), percentiles and throughput at 1, 10 and 50 concurrent users, on two traffic shapes — short chat prompts (Section 5.5) and production-length prompts (Section 5.6).

- An offline batch benchmark giving the engine’s throughput ceiling with no server overhead (Section 5.7).

- A cost model translating measured throughput into dollars per million tokens, to quantify whether FP8 is worth its accuracy cost (Section 5.8).

5\. Methodology & Results

5.1 — Two isolated environments, one shared GPU

The notebook installs and runs in two separate Colab sessions, because llmcompressor and vLLM 0.7.3 pin incompatible torch, transformers and compressed-tensors versions. Installing both into one runtime makes whichever installs second break the other — the same failure mode documented in the Phi-2 project.

| **Session** | **Purpose**                                           | **torch**   | **transformers**                  | **Other**               |
|-------------|-------------------------------------------------------|-------------|-----------------------------------|-------------------------|
| A1/B1       | Quantization, perplexity (Cell E), profiling (Cell J) | 2.11.0      | 5.x (5.14.1 at quantization time) | llmcompressor 0.13.0    |
| A2/B2       | vLLM serving and benchmarking (D2, F/G/H, G2, G3)     | 2.5.1+cu124 | 4.57.6 (pinned \<5.0)             | vLLM 0.7.3, Python 3.12 |

The rule that came out of this: anything that loads the FP8 checkpoint through plain transformers runs in the quantization session; anything that uses vLLM runs in the serving session. Running the perplexity cell in the vLLM session fails (Section 5.9), and running vLLM on the quantization session’s Python 3.13 runtime fails to install (Section 7.1).

5.2 — Quantization run

The model was loaded in BF16, then quantized in a single oneshot() call using QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=\["lm_head"\]). Every linear projection (attention Q/K/V/output, MLP up/gate/down) is quantized to FP8 except the final vocabulary projection, which stays in BF16 because errors there directly distort next-token probabilities. Weights get one scale per output channel (verified on disk: weight_scale shape (3584, 1) for down_proj); activations get one scale per token, computed at runtime. Because no calibration set is needed, the quantization pass itself took about 4 seconds of compute; model loading dominated. The checkpoint’s weights occupy 8.14 GB versus ~15 GB for BF16 — close to the theoretical 2× reduction, with lm_head and embeddings still in BF16.

5.3 — A cross-version tokenizer bug, found and patched

After quantization, the saved tokenizer_config.json stores extra_special_tokens as a list of 13 tokens — a transformers-5.x-only field that the older transformers in the serving session cannot parse. Cell D1b patches it to an empty dict. The special tokens remain registered through special_tokens_map.json and added_tokens_decoder, so tokenization is unaffected; only an optional v5 convenience field is disabled. Lesson: a checkpoint written by one transformers major version can carry config fields the consuming version does not expect.

5.4 — Correctness sanity check

Before any benchmarking, the FP8 checkpoint was loaded in vLLM and asked to “Explain what FP8 quantization does in one sentence.” It returned a coherent, on-topic answer. The check catches catastrophic quantization failure (garbage tokens, empty output) cheaply, before spending GPU time on full evaluation. It was repeated on October 12 after the config repair in Section 7.2, with the same answer and the same decode speed (26.2 vs 26.8 tok/s).

5.5 — Online load test 1: short prompts (chat-style)

A custom async client (aiohttp + asyncio, streaming responses) measures time to first token (TTFT) separately from total completion time, so inter-token latency (ITL) can be derived per request. It ran against a live vLLM OpenAI-compatible server at 1, 10 and 50 concurrent requests — once serving the FP8 checkpoint, once serving BF16 Qwen/Qwen2.5-7B-Instruct.

**Workload:** 4 distinct prompts of ~10–15 tokens each, max_tokens=128, and c×5 requests per level drawn from a 60-prompt list (so 5, 50 and 60 requests). Prefix caching was off, so repeated prompts were not served from cache.

| **Concurrency** | **TTFT p50** | **Total p50** | **ITL (avg)** | **System TPS** | **Per-user TPS** |
|-----------------|--------------|---------------|---------------|----------------|------------------|
| 1 (FP8)         | 0.081 s      | 4.82 s        | 37.0 ms       | 26.8 tok/s     | 27.0 tok/s       |
| 1 (BF16)        | 0.125 s      | 7.53 s        | 57.9 ms       | 17.1 tok/s     | 17.3 tok/s       |
| 10 (FP8)        | 0.143 s      | 5.30 s        | 40.3 ms       | 235.4 tok/s    | 24.8 tok/s       |
| 10 (BF16)       | 0.206 s      | 8.01 s        | 61.0 ms       | 161.0 tok/s    | 16.4 tok/s       |
| 50 (FP8)        | 0.690 s      | 7.22 s        | 49.5 ms       | 605.1 tok/s    | 20.3 tok/s       |
| 50 (BF16)       | 0.660 s      | 9.77 s        | 70.2 ms       | 434.9 tok/s    | 14.3 tok/s       |

On this workload FP8 wins clearly. Total request time drops 26–36%, ITL drops 29–36%, and system throughput is 39–57% higher, with the largest gain at low concurrency.

**TTFT at 50 users is a tie, not a regression.** The original report treated FP8’s 0.690 s vs 0.660 s as a +4.5% anomaly. With only 60 requests at concurrency 50 — essentially one burst of 50 followed by 10 more, which is why p50, p95 and p99 are nearly identical — a 30 ms difference is within run-to-run noise. It should not be reported as a finding without repeated runs and more requests.

**Why decode improved by 36% and not 50%.** Decode is memory-bound: generating each token requires reading every weight once. At the L4’s ~300 GB/s, reading ~15 GB of BF16 weights takes at least ~51 ms, and the measured ITL was 57.9 ms (~88% of the bandwidth limit). FP8’s ~8.1 GB takes at least ~27 ms; measured ITL was 37.0 ms. The remaining time is work FP8 does not shrink: reading the KV cache, attention, BF16 lm_head and embeddings, activation quantization, and kernel launch overhead.

A separate check (Cell G3) sent single requests of different prompt lengths. These were run against the BF16 server (the ~58 ms ITL matches BF16, not FP8; the original report labelled this as FP8 in error):

| **Prompt length** | **TTFT** | **Total time** | **ITL** |
|-------------------|----------|----------------|---------|
| Short (~20 tok)   | 0.075 s  | 7.50 s         | 58.0 ms |
| Medium (~200 tok) | 0.133 s  | 7.57 s         | 58.1 ms |
| Long (~500 tok)   | 0.175 s  | 7.61 s         | 58.1 ms |

TTFT grows with prompt length (prefill cost scales with input size) while ITL is flat: once generation starts, each token costs the same regardless of how long the prompt was. This motivated the long-prompt benchmark below.

5.6 — Online load test 2: production-length prompts

The short-prompt test is a fair A/B comparison but not representative of production traffic, where prompts carry system instructions, chat history or retrieved documents. The same load test was re-run on October 12 with a new prompt list: a shared technical passage repeated to give prompts of roughly 250, 1,000 and 2,000 tokens (≈1,100 on average), followed by one of four questions; 260 prompts with a fixed random seed so both models saw identical inputs; same max_tokens=128. At concurrency 50 that is 250 requests, about 270K prompt tokens against ~27K output tokens — a prefill-heavy workload. Both models averaged ~110 output tokens per request at concurrency 50, so throughput figures are comparable.

| **Concurrency** | **TTFT p50** | **TTFT p99** | **ITL (avg)** | **System TPS** | **Req/s** |
|-----------------|--------------|--------------|---------------|----------------|-----------|
| 1 (FP8)         | 0.371 s      | 0.692 s      | 36.9 ms       | 22.9 tok/s     | 0.44      |
| 1 (BF16)        | 0.351 s      | 0.620 s      | 57.4 ms       | 15.9 tok/s     | 0.28      |
| 10 (FP8)        | 0.716 s      | 3.448 s      | 95.5 ms       | 86.9 tok/s     | 1.53      |
| 10 (BF16)       | 0.644 s      | 3.023 s      | 111.1 ms      | 77.0 tok/s     | 1.38      |
| 50 (FP8)        | 1.647 s      | 19.94 s      | 231.8 ms      | 213.0 tok/s    | 1.94      |
| 50 (BF16)       | 4.204 s      | 21.12 s      | 218.5 ms      | 193.7 tok/s    | 1.76      |

**What changed compared with short prompts (BF16 shown):** TTFT at one user rose 2.8× (0.125 → 0.351 s), ITL at 50 users rose 3.1× (70.2 → 218.5 ms), system throughput at 50 users fell 55% (434.9 → 193.7 tok/s), and the TTFT tail exploded to 21 s at p99. Request throughput barely grew from 10 to 50 users (1.38 → 1.76 req/s), a sign the server was saturated.

**FP8 vs BF16 on this workload:**

- **Decode still wins at low load.** At one user, ITL is 36% faster (36.9 vs 57.4 ms) — identical to the short-prompt result. Halving the bytes read per token pays off regardless of prompt length.

- **Prompt processing did not get faster.** TTFT at 1 and 10 users is slightly worse for FP8 (+6% and +11%). Prefill of long prompts is compute-bound, not memory-bound, and in this setup FP8 did not deliver its theoretical 2× compute advantage. Likely contributors: attention still runs in BF16 and grows with prompt length; per-token activation quantization adds work; and the L4’s 72 W power limit caps sustained compute. These are hypotheses — separating them would need a vLLM-side profile.

- **Most of the work is the part FP8 does not speed up.** About 90% of tokens processed were prompt tokens. FP8’s decode advantage applies to roughly a tenth of the work.

- **The extra KV-cache room helps admission, not throughput.** FP8’s median TTFT at 50 users is 2.5× better (1.65 vs 4.20 s): more requests fit in memory and start immediately instead of queuing. But admitting more requests means each decode step serves more users and more long prompts interrupt generation, so ITL rose slightly (231.8 vs 218.5 ms). GPU compute is saturated either way, so throughput rose only 10% and the p99 TTFT tail is about the same.

- **Long prompts stall everyone else.** Chunked prefill was off (vLLM 0.7.3 V0 default), so while the server processes a new 2,000-token prompt, every other user’s token stream pauses. This is the main reason ITL tripled under load for both models.

**Why the KV cache matters here.** The KV cache stores each token’s attention keys and values so they are not recomputed for every new token. For Qwen2.5-7B it costs 28 layers × 2 (K and V) × 4 KV heads × 128 dims × 2 bytes ≈ 56 KB per token. vLLM gets 0.85 × 22 GiB ≈ 18.7 GiB; whatever the weights and activations do not use goes to KV cache. Measured from the vLLM startup logs: FP8 left 8.76 GiB (10,256 blocks × 16 tokens ≈ 164K tokens; room for 40 full 4,096-token requests), while BF16, with 14.25 GiB of weights, left 3.02 GiB (3,534 blocks ≈ 56.5K tokens; room for 13.8). Fifty requests averaging ~1,100 prompt tokens plus output need ~60K tokens, so BF16 had to queue or evict requests while FP8 could admit them all. A BF16 offline run over 1,040 of these long prompts confirmed it directly: vLLM logged 51 preemptions (“not enough KV cache space”), meaning requests were evicted mid-generation and recomputed later. Note the KV cache itself was BF16 in both runs (kv_cache_dtype=auto); FP8 freed room by shrinking the weights, not the cache.

All long-prompt results are single runs per model and should be treated as directional.

5.7 — Offline batch throughput (Cell G2)

G2 loads the model directly in the notebook (no HTTP server) and submits all prompts at once, so the scheduler can batch nearly everything together. It originally failed with an out-of-memory error (cause and fix in Section 7.3). Run in fresh sessions on October 12 (FP8) and October 15 (BF16) with the same 240 short prompts:

| **Measurement**               | **BF16**         | **FP8**          | **FP8 vs BF16** |
|-------------------------------|------------------|------------------|-----------------|
| Output tokens / time          | 30,691 / 15.20 s | 30,612 / 13.32 s | —               |
| Throughput                    | 2,019 tok/s      | 2,298 tok/s      | +13.8%          |
| Cost at \$0.70/hr             | \$0.096 per 1M   | \$0.085 per 1M   | 12.1% cheaper   |
| Weights / KV cache (vLLM log) | 14.25 / 3.02 GiB | 8.13 / 8.76 GiB  | 2.9× KV room    |

**FP8’s lead is smallest at the largest batch.** Putting the short-prompt results side by side shows a clear trend: FP8’s throughput advantage falls as more requests share each step.

| **Requests in flight** | **BF16 tok/s** | **FP8 tok/s** | **FP8 advantage** |
|------------------------|----------------|---------------|-------------------|
| 1 (online)             | 17.1           | 26.8          | +57%              |
| 10 (online)            | 161.0          | 235.4         | +46%              |
| 50 (online)            | 434.9          | 605.1         | +39%              |
| Up to 240 (offline)    | 2,019          | 2,298         | +14%              |

This follows from how decode works. Each decode step reads all the weights once and shares that read across every sequence in the batch. With one user, weight reading dominates, so halving the weight bytes gives a large gain. With ~240 sequences per step, the weight read is amortized over many tokens, and time goes to the matrix multiplies themselves, attention and KV-cache reads — work that FP8 weights shrink little or not at all on this setup. Memory headroom did not matter here: 240 short sequences (~34K tokens) fit in BF16’s 56.5K-token KV cache. I had expected FP8’s lead to grow in batch mode; it shrank, which is consistent with the long-prompt finding that FP8 gave little compute speed-up on the L4.

A separate BF16 offline run on 1,040 long prompts (the long-prompt list ×4, run by accident before the short-prompt run) reached only 214 output tok/s — about the same as the online 50-user figure (194 tok/s) — with ~2,066 input tok/s and 51 KV-cache preemptions. Long-prompt workloads are limited by prompt processing whether or not a server is involved. FP8 has not been run on that workload offline.

These offline figures are ceilings for batch jobs, not serving numbers: no network or streaming overhead and all requests present from the start.

5.8 — Cost model

Measured system throughput was converted to dollars per million output tokens at an assumed L4 on-demand rate of \$0.70/hr (still to be re-verified; cloud pricing changes):

| **Workload / concurrency** | **FP8 \$/1M** | **BF16 \$/1M** | **Savings** | **Basis**  |
|----------------------------|---------------|----------------|-------------|------------|
| Short prompts, c=1         | \$7.26        | \$11.37        | 36.2%       | Online     |
| Short prompts, c=10        | \$0.83        | \$1.21         | 31.6%       | Online     |
| Short prompts, c=50        | \$0.32        | \$0.45         | 28.1%       | Online     |
| Long prompts, c=1          | \$8.49        | \$12.23        | 30.6%       | Online     |
| Long prompts, c=10         | \$2.24        | \$2.53         | 11.4%       | Online     |
| Long prompts, c=50         | \$0.91        | \$1.00         | 9.1%        | Online     |
| Short-prompt batch (240)   | \$0.085       | \$0.096        | 12.1%       | Offline G2 |

The headline is no longer a single number. **FP8 cuts serving cost by ~28–36% for short-prompt traffic at low-to-moderate load, ~12% for large offline batches, and only ~9–11% for long-prompt traffic under load.** Long prompts also make serving 2–3× more expensive per output token for both models, because most GPU time goes to processing input that is never billed as output.

5.9 — Perplexity / accuracy validation

The WikiText-2 perplexity comparison (Cell E) initially failed twice. First, loading the FP8 checkpoint with AutoModelForCausalLM threw AttributeError: 'ModelCompressor' object has no attribute 'compress_model'. The cell had been run in the vLLM session, where vLLM 0.7.3 pins an older compressed-tensors library that predates compress_model(), which newer transformers calls when loading compressed checkpoints. Re-running Cell E in the quantization session fixed it. Second, load_dataset("wikitext", ...) failed with HfUriError because huggingface_hub ≥1.16 rejects the legacy bare "wikitext" id; switching to "Salesforce/wikitext" fixed it.

| **Configuration** | **Perplexity (WikiText-2, n=20 windows of 1,024 tokens)** |
|-------------------|-----------------------------------------------------------|
| BF16 baseline     | 6.469                                                     |
| FP8 (FP8_DYNAMIC) | 6.536 (identical when re-run on October 12)               |
| Delta             | +0.066 (≈1.0% relative increase)                          |

A 1% perplexity increase is a small, healthy result. Its limits: 20 windows is a small sample, perplexity is a general language-modelling metric rather than a task metric, and there is no per-layer sensitivity analysis. A production rollout would add task-specific evaluations (for example coding, summarization or instruction following) on the real workload.

6\. Profiling Results (Cell J) & Caveats

Cell J runs in the quantization session and produces two outputs: an op-level torch.profiler breakdown of one forward pass through the FP8 checkpoint, and a GPU utilization/power sample. Neither measures what it first appears to measure.

6.1 — Op-level breakdown: this profiled BF16 compute, not FP8 kernels

Top rows by CUDA time (August run; 524.5 ms CPU / 258.0 ms CUDA total):

| **Op**                            | **Self CUDA %** | **CUDA total** | **Avg / call** | **\# calls** |
|-----------------------------------|-----------------|----------------|----------------|--------------|
| aten::mul                         | 39.6%           | 102.2 ms       | 157.8 µs       | 648          |
| elementwise_kernel                | 39.5%           | 102.0 ms       | 181.9 µs       | 561          |
| aten::to / \_to_copy / copy\_     | 34.2%           | 88.1 ms        | 24–32 µs       | 2,723–3,568  |
| unrolled_elementwise_kernel       | 33.6%           | 86.7 ms        | 110.6 µs       | 784          |
| aten::linear                      | —               | 62.7 ms        | 318.1 µs       | 197          |
| aten::mm / cutlass bf16 wmma GEMM | 20.9%           | 58.3 ms        | 511.6 µs       | 113–114      |

The tell is the GEMM kernel name, cutlass_80_wmma_tensorop_bf16\_…, a BF16 tensor-core kernel. The log lines before the profiler confirm it: “Applying quantization config → Compressing model → Loading weights → Decompressing model.” Plain transformers has no FP8 execution kernels, so it decompresses the FP8 weights back to BF16 before the forward pass runs.

Roughly three-quarters of CUDA time is that decompression, not inference: about 35% is dtype conversion and copying (aten::to / \_to_copy / copy\_), and about 39% is aten::mul, which is most likely the dequantization multiply (FP8 weight × per-channel scale). Its 648 calls allocate about 12.2 GB of GPU memory, which is weight-sized; a forward pass over a 6-token prompt produces activations measured in MB, not GB. Actual matrix multiplies account for only about a fifth of CUDA time. A re-run on October 12 reproduced these figures within about 1% (261.5 ms vs 258.0 ms total CUDA time).

Takeaway: this profile shows the cost of a naive load path, which is exactly why FP8 checkpoints are served through vLLM. It is not a measurement of FP8 inference performance; the vLLM benchmarks in Sections 5.5–5.7 are.

6.2 — GPU utilization/power sample isn’t representative

*Results: 100.0% utilization / 276 MB peak memory / 30 W (August) and 100.0% / 278 MB / 34 W (October).*

The sampling window brackets a commented-out workload line (# asyncio.run(run_load_test(50))), and the model was deleted before the sampler starts, so it measured an idle GPU. 278 MB is far below the ~8.1 GB the model occupies, which confirms nothing was loaded. To get a meaningful reading, run real work inside the window — ideally the load test against a live vLLM server, from the serving session.

7\. Re-validation Session (October 12, 2026): Issues Found and Fixed

7.1 — vLLM 0.7.3 would not install on a fresh runtime

pip install vllm==0.7.3 failed during a dependency’s setup.py egg_info step. The fresh Colab runtime was Python 3.13; vLLM 0.7.3 and its pinned dependencies predate 3.13, so a dependency without a 3.13 wheel fell back to a source build that failed. Moving the serving session to a Python 3.12 runtime fixed it. The quantization session is unaffected.

7.2 — Checkpoint config lost its quantization_config

Cell D2 crashed with KeyError: 'layers.0.mlp.down_proj.weight_scale'. The key clue was quantization=None in vLLM’s engine-config log line: config.json no longer contained a quantization_config block, so vLLM built plain BF16 layers with no slot for the checkpoint’s scale tensors.

The fix was to rebuild the block from the evidence on disk, not from memory. Inspecting the safetensors file confirmed FP8 (float8_e4m3fn) weights, one scale per output channel (shape (3584, 1)), no static input scales, and lm_head in BF16 — exactly the FP8_DYNAMIC layout. The standard compressed-tensors description (float weights, channel strategy, dynamic per-token activations, lm_head ignored) was written back, with the broken config saved as config.json.bak_noquant.

The repair was verified three independent ways: vLLM decode speed matched the original run (26.2 vs 26.8 tok/s), perplexity matched to three decimals (6.536), and the Cell J profile matched within 1%. What removed the block is unknown — nothing in the notebook rewrites config.json after quantization (Cell D1b only edits tokenizer_config.json) — so a known-good copy (config.json.good) is now kept alongside the checkpoint.

7.3 — Root cause of the original G2 out-of-memory error

The original G2 run failed because another process still held 17.97 GiB. Cell H’s server_proc.terminate() returned exit code 1: vLLM’s API server launches a separate engine process that holds the GPU memory, and terminate() stops only the parent. The fix is to start the server with subprocess.Popen(..., start_new_session=True) and stop the whole process group with os.killpg(os.getpgid(pid), SIGTERM), then confirm memory is free with nvidia-smi before starting another engine.

8\. Key Technical Deductions

- **FP8 helps decode far more than prefill on an L4.** Decode is memory-bound, and halving the weight bytes gave a consistent 36% faster ITL. Long-prompt prefill is compute-bound and saw no gain in this setup.

- **The benefit of quantization depends on the workload.** 28–36% cheaper for short-prompt chat; ~9–11% cheaper for long-prompt traffic under load. A single headline number would have been misleading.

- **Measured decode speed is close to the memory-bandwidth limit.** BF16 ran at ~88% of the theoretical minimum time set by 300 GB/s; FP8 at ~73%. The gap to a full 2× is the KV-cache reads, attention and other work that FP8 weights do not shrink.

- **FP8’s advantage shrinks as the batch grows.** Short-prompt throughput gain: +57% at 1 user, +46% at 10, +39% at 50, +14% for a 240-request offline batch. Larger batches amortize weight reads, so the memory-bandwidth saving matters less.

- **Smaller weights mean more KV-cache room, which improves admission rather than throughput.** FP8 had ~2.9× more KV memory (8.76 vs 3.02 GiB, measured); BF16 logged 51 preemptions on long prompts. With long prompts at 50 users this cut median TTFT by 61%, but throughput rose only 10% because compute was the bottleneck.

- **Without chunked prefill, long prompts stall other users.** ITL tripled under long-prompt load for both models.

- **FP8_DYNAMIC vs GPTQ:** no calibration data and seconds to quantize, at the cost of requiring FP8-capable hardware. Quality cost was ~1% perplexity.

- **Checkpoint portability is asymmetric.** vLLM reads compressed-tensors FP8 natively; plain transformers can only load it with a compatible compressed-tensors version, and then decompresses it to BF16.

- **Config files are part of the model.** Losing quantization_config silently turns an FP8 checkpoint into an unloadable BF16 one. Verify the engine log says quantization=compressed-tensors, and keep a known-good copy.

- **Verify measurements against expectations.** The profiler’s BF16 kernel name, the sampler’s 278 MB reading and the BF16-matching ITL in G3 each revealed that a cell was measuring something other than what it claimed.

- **Small samples hide noise.** A 30 ms TTFT difference across 60 requests is not a finding; p99 of 60 requests is just the slowest one.

9\. Limitations

- Each load-test configuration was run once; results are directional, not statistically tested.

- Synthetic prompts: the long-prompt test repeats one passage and does not include real conversation data or mixed output lengths.

- Closed-loop load (a fixed number of users in flight) rather than open-loop arrivals at a target request rate; no goodput measurement against a latency SLO.

- No vLLM server metrics (KV-cache utilization, preemptions, queue depth) were captured.

- Quality was checked with one metric (perplexity, 20 windows); no task-specific evaluation or per-layer sensitivity analysis.

- Cost figures assume \$0.70/hr for the L4 and count output tokens only.

- One GPU type (L4). On an H100, with more compute and memory bandwidth, the balance between decode and prefill gains would differ.

10\. Tools & Stack Summary

| **Tool / Library**                            | **Role in this project**                                                                  |
|-----------------------------------------------|-------------------------------------------------------------------------------------------|
| llmcompressor (oneshot, QuantizationModifier) | Applies FP8_DYNAMIC quantization to the BF16 model                                        |
| compressed-tensors                            | Checkpoint format and (de)compression library; version mismatches caused the Cell E error |
| transformers                                  | Model/tokenizer loading; different pinned versions in the two sessions                    |
| vLLM 0.7.3                                    | Serving engine: OpenAI-compatible API server and offline batch generation                 |
| datasets (Salesforce/wikitext)                | Source text for the perplexity comparison                                                 |
| aiohttp + asyncio                             | Custom async load tester with streaming TTFT/ITL measurement                              |
| nest_asyncio                                  | Allows asyncio.run() inside the Colab event loop                                          |
| safetensors                                   | Inspecting on-disk tensors to rebuild and verify quantization_config                      |
| torch.profiler                                | Op-level profile — captured BF16 decompression overhead, not FP8 kernels                  |
| nvidia-smi                                    | GPU memory checks between engines; utilization/power sampling                             |
| Google Drive                                  | Persists the FP8 checkpoint across Colab sessions                                         |

11\. Recommendations / Next Steps

- **Enable chunked prefill** (--enable-chunked-prefill) and re-run the long-prompt test. It should reduce the ITL blow-up under load for both models.

- **Try an FP8 KV cache** (--kv-cache-dtype fp8) to roughly double KV capacity again, with a quality check.

- **Run G2 for FP8 on the 1,040 long prompts** to compare against BF16’s 214 tok/s and 51 preemptions; it should show whether FP8’s extra KV room removes the preemptions.

- **Benchmark like production:** vLLM’s benchmarks/benchmark_serving.py with real conversations (ShareGPT), random arrivals (--request-rate) and goodput against a TTFT/ITL SLO; repeat each configuration several times.

- **Capture server metrics** from vLLM’s /metrics endpoint (KV-cache usage, preemptions, queue time) during load tests.

- **Profile FP8 inside vLLM** to explain why prefill did not speed up (attention share, activation quantization cost, power throttling).

- **Add task-level quality evaluation** and a per-layer sensitivity analysis, as in the Phi-2 project.

- **Re-verify the L4 hourly rate** before using cost figures in a decision-facing document.

