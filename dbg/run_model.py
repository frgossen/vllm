#!/usr/bin/env python
"""Run a model through vLLM and capture traces, shaped for GEMM work.

Reports end-to-end wall clock for one measured batch. For per-phase attribution
read the trace instead: vLLM annotates every engine step as
`execute_context_N(tokens)_generation_N(tokens)` -- context is prefill,
generation is decode.

    tlp python dbg/run_model.py                      # torch trace + tlparse
    python dbg/run_model.py --profile /tmp/p --share # CUDA kernels + Perfetto

MODELS -- all run on this box (8x H100 96GB = 768GB HBM) unless noted. Timings
are a 512x2 prefill + 8 decode smoke test, not benchmarks.

  # WORKS
  --model Qwen/Qwen3.5-9B                        # 32L h4096 dense       0.106s
  --model meta-models/Muse-Glimmer-30B           # 52L h6656 dense       0.335s
  --model google/gemma-4-26B-A4B-it              # 30L h2816 128 exp     0.081s
  --model openai/gpt-oss-120b                    # 36L h2880 128 exp     0.145s
  --model MiniMaxAI/MiniMax-M2.5   --tp 4        # 62L h3072 256 exp     0.188s
  --model deepseek-ai/DeepSeek-V4-Flash --tp 8 --kv-cache-dtype fp8      0.336s

  # WORKS WITH A CAVEAT
  --model google/gemma-4-31B-it --eager          # 60L h5376 dense       0.365s
      Compiled path dies in capture_model with an illegal memory access;
      CUDA_LAUNCH_BLOCKING=1 does not mask it. --eager also means no torch trace.
  --model openai/gpt-oss-20b                     # 24L h2880  32 exp     0.062s
      Runs, but emits garbage (replacement chars). Fine for kernel shapes, not
      for correctness.

  # DOES NOT WORK HERE
  --model zai-org/GLM-5.2-FP8 --tp 8
      756GB. Memory is NOT the problem -- 8x96GB = 768GB holds it and it loads.
      The sparse-MLA attention path faults. Four configurations tried:
        default                  -> FLASH_ATTN_MLA_SPARSE, illegal memory access
                                    in hopper/flash_fwd_launch_template.h:199
        VLLM_ATTENTION_BACKEND=TRITON_ATTN   -> ignored, same backend, same fault
        VLLM_ATTENTION_BACKEND=FLASHINFER    -> ignored, same backend, same fault
        --kv-cache-dtype fp8     -> FLASHMLA_SPARSE instead, then
                                    CUBLAS_STATUS_EXECUTION_FAILED
      GlmMoeDsaForCausalLM pins its own backend, so VLLM_ATTENTION_BACKEND has
      no effect. Smaller INT4/NVFP4 checkpoints will not help: they change
      memory, not the attention path (and NVFP4 needs sm100).
      Suspect the torch mismatch -- vLLM pins torch==2.13.0, this env is a
      source-built 2.15 with vllm-flash-attn compiled against it. Third
      illegal-memory-access we hit on sm90 today. To confirm, try stock
      torch 2.13 + release vLLM in a throwaway env.
  --model moonshotai/Kimi-K3
      2.8T params, native MXFP4 -> 1561GB of *already-4-bit* weights (dequant to
      bf16 happens per-tile in registers, so it costs nothing extra in HBM).
      768GB of HBM is ~2x short; needs >=16 cards, or ~800GB of --cpu-offload-gb
      against host RAM, which would measure PCIe rather than kernels.

GOTCHAS
  - --warmup must be >=2 for MoE models. The Triton fused_moe_kernel JITs on
    first use; with --warmup 1 that lands inside the measured run and cost
    DeepSeek V4 35.4s instead of 0.336s (105x).
  - TP is constrained by FP8 block quant: each shard's moe_intermediate_size
    must be divisible by 128. MiniMax has 1536, so TP=8 gives 192 and fails;
    TP=4 gives 384 and works.
  - TP=8 makes communication ~47% of CUDA time (measured on DeepSeek V4), which
    drowns out GEMMs. Prefer the smallest TP that fits.
  - MXFP4 kernels are sm100/sm120 only, so on H100 gpt-oss and Kimi K3
    dequantize 4-bit weights to bf16 inline instead of using FP4 tensor cores.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

SHARE_TRACE = os.path.expanduser(
    "~/fbsource/arvr/scripts/perfetto/share_trace.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
    p.add_argument("--prefill", type=int, default=4096,
                   help="Synthetic prompt length in tokens. Ignored with "
                        "--prompt.")
    p.add_argument("--prompt", default=None,
                   help="Real prompt text, for a coherence check.")
    p.add_argument("--batch", type=int, default=4, help="Concurrent requests.")
    p.add_argument("--decode", type=int, default=64, help="Tokens generated.")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--cpu-offload-gb", type=float, default=0,
                   help="Per-GPU weight offload to host RAM. Needed for models "
                        "larger than aggregate HBM.")
    p.add_argument("--kv-cache-dtype", default="auto",
                   help="e.g. fp8. DeepSeek V4's fp8_ds_mla layout requires it.")
    p.add_argument("--eager", action="store_true",
                   help="Disable compilation; also yields an empty torch trace.")
    p.add_argument("--spec-model", default=None,
                   help="Draft head for speculative decoding.")
    p.add_argument("--spec-tokens", type=int, default=3)
    p.add_argument("--profile", default=None, metavar="DIR",
                   help="Write a runtime torch profiler trace (CPU + CUDA "
                        "kernels) to DIR. This is what shows GEMM kernels; "
                        "TORCH_TRACE/tlparse only records compilation.")
    p.add_argument("--share", action="store_true",
                   help="Upload the --profile trace and print a Perfetto URL.")
    p.add_argument("--warmup", type=int, default=4,
                   help="Warmup iterations; 0 leaves compile and autotuning "
                        "in the numbers.")
    args = p.parse_args()
    if args.share and not args.profile:
        p.error("--share requires --profile DIR")
    return args


def share_traces(trace_dir: str, timeout: float = 180.0) -> None:
    """Upload every trace in ``trace_dir`` and print its Perfetto UI URL.

    ``stop_profile`` returns before the worker has finished writing, so poll
    until the set of files and their sizes stop changing.
    """
    pattern = os.path.join(trace_dir, "*.pt.trace.json.gz")
    deadline, previous = time.time() + timeout, None
    while time.time() < deadline:
        current = {p: os.path.getsize(p) for p in glob.glob(pattern)}
        if current and current == previous:
            break
        previous = current
        time.sleep(2)
    traces = sorted(previous or {})
    if not traces:
        print(f"!! no traces matched {pattern} within {timeout:.0f}s")
        return

    for trace in traces:
        # Run via the shebang (fbpython) rather than this interpreter.
        result = subprocess.run([SHARE_TRACE, trace], capture_output=True,
                                text=True)
        urls = [ln.strip() for ln in (result.stdout + result.stderr).splitlines()
                if ln.strip().startswith("https://")]
        if urls:
            print(f"perfetto: {urls[-1]}")
        else:
            print(f"!! upload failed for {trace} (rc={result.returncode})")
            print((result.stderr or result.stdout).strip()[-500:])


def run(llm: LLM, prompt, batch: int, decode: int) -> tuple[float, list]:
    """Wall-clock and outputs for one batch, ``decode`` tokens each."""
    prompts = [prompt] * batch
    # ignore_eos pins the output length; otherwise it varies run to run and
    # nothing is comparable.
    params = SamplingParams(temperature=0.0, max_tokens=decode, ignore_eos=True)
    start = time.perf_counter()
    outs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start
    produced = sum(len(o.outputs[0].token_ids) for o in outs)
    if produced != batch * decode:
        print(f"  !! produced {produced} tokens, expected {batch * decode}")
    return elapsed, outs


def main() -> None:
    args = parse_args()
    spec = {} if not args.spec_model else {
        "spec_method": "dflash",
        "spec_model": args.spec_model,
        "spec_tokens": args.spec_tokens,
    }

    profiler = {}
    if args.profile:
        profiler["profiler_config"] = {
            "profiler": "torch",
            # Must be absolute.
            "torch_profiler_dir": os.path.abspath(args.profile),
            # Both matter for GEMM attribution: shapes identify which GEMM,
            # flops give achieved throughput per kernel.
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_flops": True,
        }
        print(f"profiler trace -> {os.path.abspath(args.profile)}")

    print(f"TORCH_TRACE={os.environ.get('TORCH_TRACE', '(unset -- use tlp)')}")
    print(f"model={args.model} tp={args.tp} prefill={args.prefill} "
          f"batch={args.batch} decode={args.decode} eager={args.eager}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="auto",
        kv_cache_dtype=args.kv_cache_dtype,
        cpu_offload_gb=args.cpu_offload_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.eager,
        max_model_len=args.max_model_len or args.prefill + args.decode + 256,
        # Floor of 8 so the cudagraph capture set does not collapse to [1, 2]
        # at small --batch; it is derived from max_num_seqs.
        max_num_seqs=max(args.batch, 8),
        # Prefill the whole batch in one step, to keep prefill GEMMs large.
        max_num_batched_tokens=max(args.prefill * args.batch, 8192),
        # Off: the prompts are identical, so caching would serve every request
        # after the first and prefill would not be measured at all.
        enable_prefix_caching=False,
        **spec,
        **profiler,
    )

    # Token content does not affect GEMM cost, only the count does -- so for
    # timing runs a fixed in-vocab id beats tokenizing real text.
    prompt = args.prompt or TokensPrompt(prompt_token_ids=[100] * args.prefill)

    for i in range(args.warmup):
        print(f"warmup {i + 1}/{args.warmup}...")
        run(llm, prompt, args.batch, 8)

    if args.profile:
        llm.start_profile()
    elapsed, outs = run(llm, prompt, args.batch, args.decode)
    if args.profile:
        llm.stop_profile()

    out = outs[0]
    n_prompt = len(out.prompt_token_ids)
    total_prompt = args.batch * n_prompt
    total_out = args.batch * args.decode

    print("\n--- end to end ---")
    print(f"{elapsed:.3f} s for {total_prompt} prompt + {total_out} output "
          f"tokens ({total_out / elapsed:.1f} output tok/s)")
    print("per-phase: read the trace, execute_context_* is prefill and "
          "execute_generation_* is decode")
    print(f"sample: {out.outputs[0].text[:120]!r}")

    if args.share:
        print()
        share_traces(os.path.abspath(args.profile))


if __name__ == "__main__":
    main()
