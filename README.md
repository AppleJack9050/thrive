# THRIVE: Recursive Multi-Agent LLM Throughput Optimization

**THRIVE** (**TH**Roughput via **R**ecursive, **I**terative, **V**erified **E**xperts) is a
training-free, recursive multi-agent framework for optimizing the throughput of LLM systems.

> 📌 **Accepted as a poster at [The Fourth UK AI Conference 2026](http://uk-ai.org/ukai2026/)**
> (Nottingham, UK, 29–30 September 2026).

**Sicheng Zhao<sup>1,2</sup>, Soumyabrata Dev<sup>1,2</sup>**<br>
<sup>1</sup> School of Computer Science and Statistics, Trinity College Dublin, Dublin, Ireland<br>
<sup>2</sup> The ADAPT Research Centre, Dublin, Ireland

---

## Overview

An LLM system's throughput is set by its slowest stage, which can be GPU kernels, data
movement, fusion, or communication. Optimizing one stage on its own achieves little. A local
speedup on a stage that takes a small share of the time barely moves the system, and the
bottleneck moves somewhere else. Yet most optimizers target kernels only.

THRIVE profiles the whole system and sends the dominant bottleneck to a specialist agent that
is checked by a verifier. It then keeps or reverts each change based on a global throughput
metric. Weights are never updated. The system improves through search plus a dual-level memory
that carries over between runs.

## The THRIVE framework

THRIVE runs a closed, system-level loop:

1. **Profile** the system's throughput.
2. **Attribute** runtime across stages and rank bottlenecks by time share. A stage can only
   improve the system as much as its time share allows, so THRIVE *attributes before it
   optimizes*.
3. **Dispatch** the bottleneck with the largest time share to a specialist agent. The
   specialists cover kernels, data, memory, compilation, and communication.
4. **Verify**, then **keep or revert** the change using the thresholds `r_t = a_t = 0.3`,
   and re-profile.

It is recursive in two ways. The orchestrator runs the loop again each time the bottleneck
moves, and each specialist runs its own refinement loop. A dual-level memory (long-term
strategy cards and a best-kernel library, plus a short-term refinement trajectory) grounds
both levels.

```
Algorithm: THRIVE recursive throughput optimization
Require: system S, metric T, specialists E, thresholds r_t, a_t, budget N

best ← S;  T_best ← Profile(S)
for i = 1 .. N:
    j* ← argmax_j Headroom(Attribute(S))          # rank by time share
    S' ← Select(E, j*).Optimize(S)                # specialist's inner loop
    T' ← Profile(S');  update memory with (j*, T')
    if T'/T_best > 1 + r_t  or  T' − T_best > a_t:
        S, best ← S';  T_best ← T'                # accept; else revert
return best
```

## Results

All results are on an **NVIDIA RTX 5090** (Blackwell, sm_120). The paper names the model
**Qwen3.5-1.7B**.

### Kernel quality

The kernel specialist writes, verifies, and benchmarks Triton kernels on its own. All outputs
match the reference numerically.

| Kernel | vs eager PyTorch | vs production (CUDA graphs) | Production baseline |
|---|---|---|---|
| RMSNorm + residual | **4.93×** | **1.3–1.4×** | vLLM CUDA |
| SwiGLU | **1.62×** | 1.00–1.01× | vLLM CUDA |
| GQA attention | **~12×** | 0.97× (tie) | cuDNN FlashAttention |

*vs eager:* geomean speedup over eager PyTorch. *vs production:* latency compared with
hand-written production kernels under CUDA graphs (>1 means THRIVE's kernel is faster).

### Recursion without training

Memory turns search into reuse. Starting from empty memory, a cold run needs **two phases** to
reach 4.93×. A warm run retrieves that kernel from memory and reaches **4.77× in one phase**,
in **one-third the time and at half the cost**.

### End-to-end

Plugged into vLLM's CUDA graphs, the kernels raise full-model throughput by **+3.3% (prefill)**
and **+7.8% (decode)** at batch size 256. GEMMs dominate the runtime and use the same cuBLAS
kernels in both systems, which caps the gain. This matches THRIVE's attribute-before-optimize
principle, which single-stage optimizers do not follow.

**Limitations:** the results cover a single model on a single GPU, and they depend on stable
timing measurements.

See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for the full measurements, the vLLM
head-to-head, and the decode-step breakdown.

---

## This repository

This repository contains the **kernel specialist**, which is the inner loop validated in the
paper. It is built on the **Claude Agent SDK** and writes and optimizes **Triton** kernels for
small-LLM inference ops. Each kernel is checked for correctness and benchmarked for
**per-kernel speedup** against a PyTorch reference. The repository also includes the vLLM
integration and the throughput harnesses behind the end-to-end numbers. The code config targets
**Qwen3-1.7B** dimensions (in `rsi/config.py`; changing them takes one edit).

> The Python package and CLI are still called `rsi`, from the project's original name
> (*recursive self-improvement*).

Improvement comes from *search* plus *memory accumulated across runs*, never from weight
updates. This follows [KernelMem](https://github.com/0satan0/KernelMem), whose ablation shows
that memory is the main factor in kernel-agent performance.

### How it works

```
Orchestrator (Opus, Python-driven loop)            rsi/orchestrator.py
  └─ per op: seed → (repair) → optimize × N         deterministic phases + budget
       └─ kernel subagents (Sonnet/Haiku)           rsi/agents.py
            ├─ kernel-generator   correct seed
            ├─ kernel-optimizer   profile→memory→one change  (may Task→analyst)
            ├─ kernel-repairer    fix compile/verify failures
            └─ profiler-analyst   bottleneck diagnosis
       backed by in-process MCP tools               rsi/tools.py
         get_op_spec / compile / verify / benchmark / profile / read_memory / record_strategy
       + dual-level memory                          rsi/memory/store.py
         long-term: strategy cards + best-kernel library (persists across runs)
         short-term: per-task refinement trajectory
```

The agents never use the shell or filesystem directly. They pass Triton **source** to the MCP
tools, which compile it (import + JIT launch), verify it (`torch.allclose`), and benchmark it
(`triton.testing.do_bench`) in-process. Every attempt is saved under `rsi/kernels/<op>/`. The
best kernels and the lessons learned go into long-term memory, so the **next** run starts from
them.

### Install

```bash
git clone https://github.com/AppleJack9050/thrive.git
cd thrive
bash scripts/setup_env.sh        # rsi + PyTorch/Triton for Blackwell sm_120
python3 scripts/smoke_test.py    # deterministic: torch+Triton+harness+memory, no LLM
```

### Run

```bash
rsi ops                                  # list target ops
rsi optimize --op rmsnorm_residual --rounds 5
rsi optimize --op all --passes 2         # full sweep + one recursive re-attack pass
rsi optimize --op all --autonomous       # single Opus orchestrator delegating via Task
rsi leaderboard                          # best kernel + speedup per op
```

Run it **twice**. The second run starts from long-term memory and reaches higher speedups in
fewer phases. This is how the training-free recursion claim is measured
(`python3 scripts/rsi_demo.py --op rmsnorm_residual` runs the cold/warm comparison). For the
vLLM head-to-head and end-to-end throughput commands, see
[EXPERIMENT_REPORT.md §7](EXPERIMENT_REPORT.md#7-reproduce).

### Target ops (Qwen3-1.7B)

`rmsnorm_residual` · `swiglu_act` · `rope` · `softmax` · `gqa_decode_attention` (stretch).
These decode, MLP, and norm ops are memory-bandwidth-bound. Fusing PyTorch's separate kernel
launches into one Triton kernel is where the gains come from.

### Config knobs

Every setting can be overridden with an environment variable (see `rsi/config.py`):

- Model dims: `RSI_HIDDEN`, …
- Agent models: `RSI_MODEL_*`
- Budgets: `RSI_PER_RUN_USD`, `RSI_ROUNDS`, `RSI_EFFORT`
- Target speedup: `RSI_TARGET_SPEEDUP`
- Peak bandwidth for roofline scoring: `RSI_PEAK_BW_GBPS`

---

## Citation

If you use THRIVE, please cite:

```bibtex
@inproceedings{zhao2026thrive,
  title     = {{THRIVE}: Recursive Multi-Agent {LLM} Throughput Optimization},
  author    = {Zhao, Sicheng and Dev, Soumyabrata},
  booktitle = {The Fourth UK AI Conference},
  year      = {2026},
  note      = {Poster}
}
```
