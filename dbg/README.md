# run_model.py — run a model through vLLM and get a trace

## What you need

**One file: `run_model.py`.** It imports only stdlib + `vllm`, so copy it anywhere.

| requirement | why | needed for |
|---|---|---|
| a working vLLM + CUDA GPU | it's a vLLM script | everything |
| HF access (`HF_TOKEN` for gated repos) | weight download | everything |
| `~/fbsource` checkout | `arvr/scripts/perfetto/share_trace.py` | `--share` only |

Without `~/fbsource`, drop `--share` — you still get the trace file in
`--profile DIR` and can open it in Perfetto by hand.

A stock `pip install vllm` is *easier* than this workspace's setup. This env
builds vLLM from source against a source-built torch 2.15 (vLLM pins 2.13.0),
which cost ~15 hand-installed dependency fixes because the build script uses
`--no-deps`. None of that is needed to run the script.

## Getting a trace

    python run_model.py --profile /tmp/p --share

Gives a torch profiler trace of CPU + CUDA activity and prints a Perfetto URL.
`--eager` skips compilation, which is only needed for models whose compiled
path is broken.

## Reading the trace

vLLM annotates every engine step:

    execute_context_N(tokens)_generation_N(tokens)

`context` is prefill, `generation` is decode. Two gotchas: annotations appear
**twice** (two tracks), so halve counts before comparing against kernel launches;
and decode runs as CUDA-graph replays, so decode `aten::mm` ops do **not** appear
even with `record_shapes` — only prefill's do.

## Models

See the docstring at the top of `run_model.py` for the verified list: which
models work, which need `--eager`, the per-model TP and kv-cache-dtype
constraints, and the `--warmup >= 2` requirement for MoE (the Triton
`fused_moe_kernel` JIT otherwise lands inside the measured run and cost one
model 35s instead of 0.34s).

## Notes on the numbers

`--batch` is concurrency and `--prefill` is per-request prompt length, so the
GEMM's `M` is `batch * prefill` in prefill and `batch` in decode. Defaults are
small for smoke tests; `M=2` decode is deep in the bandwidth-bound regime and
is not representative of serving. Reported time is end-to-end for one measured
batch — read the trace for per-phase attribution.
