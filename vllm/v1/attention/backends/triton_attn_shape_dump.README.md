# Triton attention shape dump

Records every attention batch shape that `TritonAttentionMetadataBuilder.build()`
sees during real serving and writes it as a
`micro-benchmarks/benchmark_paged_attention.py` **`capture_batches`** config, so
production traffic can be replayed as a micro-benchmark.

Implemented in `triton_attn_shape_dump.py`, hooked at the end of
`TritonAttentionMetadataBuilder.build()` in `triton_attn.py`. The hook is behind
an env check and the module is imported lazily, so a normal run pays nothing.

## Usage

```bash
# 1. Capture real traffic
VLLM_TRITON_ATTN_DUMP_SHAPES=1 \
VLLM_TRITON_ATTN_DUMP_PATH=/tmp/traffic.json \
  vllm serve Qwen/Qwen3-32B --attention-backend TRITON_ATTN
# ... send your workload, then shut the server down cleanly ...

# 2. Replay as a micro-benchmark
python micro-benchmarks/benchmark_paged_attention.py --config /tmp/traffic.json
```

The JSON is written once, atomically, from an `atexit` handler — the server must
exit cleanly (`SIGTERM`/`SIGINT`, not `SIGKILL`) for the file to appear.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_TRITON_ATTN_DUMP_SHAPES` | `0` | Master switch. `1` enables recording. |
| `VLLM_TRITON_ATTN_DUMP_PATH` | `triton_attn_shapes.json` | Output path. |
| `VLLM_TRITON_ATTN_DUMP_DEDUP` | `1` | `1` collapses identical batches into one entry with a `count`; `0` keeps every `build()` call in arrival order. |

Dedup is keyed on `(query_lens, seq_lens, max_query_len)`. Real traffic repeats
the same shapes constantly, so the default keeps the file small; use `0` when you
need the exact temporal sequence of batches.

## Output format

Matches the schema consumed by `run_capture_benchmarks` / `_expand_capture_entry`
(same as `micro-benchmarks/configs/helion_nvidia_qwen3_32b_fullcg.json`):

```json
{
  "capture_accurate": true,
  "num_query_heads": 64,
  "num_kv_heads": 8,
  "head_size": 128,
  "block_size": 16,
  "num_blocks": 8192,
  "dtype": "bfloat16",
  "warmup": 20,
  "rep": 100,
  "use_cuda_graphs": true,
  "variants": ["triton_auto", "triton_2d"],
  "capture_batches": [
    {
      "name": "batch_0",
      "query_lens_rle": [[1, 256]],
      "seq_lens_rle": [[1024, 256]],
      "capture_max_query_len": 256,
      "count": 1337
    }
  ]
}
```

Per-request lengths are run-length encoded as `[value, count]` pairs:
`query_lens[i] = query_start_loc[i+1] - query_start_loc[i]` and
`seq_lens[i]` straight from the metadata. `count` is extra provenance and is
ignored by the benchmark's parser.

The grid format (`micro-benchmarks/configs/p3hpc_paper.json`) is deliberately not
used: it is a Cartesian product of `batch_sizes × sequence_lengths ×
decode_shares` and cannot express arbitrary per-request shapes from real traffic.

## Caveats

- `num_blocks` is **not** the engine's real allocated KV block count (the builder
  does not expose it). It is a safe lower bound computed over all recorded
  batches, `max(num_reqs * ceil(max_seq_len / block_size))`, which is exactly
  what `build_attention_inputs_from_requests` needs to avoid its
  insufficient-blocks skip. Override it in the JSON if you want the real value.
- `dtype` is the model dtype, not the KV cache dtype; the benchmark generates
  Q/K/V in one dtype, so a quantized KV cache is not reproduced.
- `seq_lens` is copied to host on every recorded batch, which forces a
  device sync. Fine for tracing, but do not use this to measure serving
  throughput.
- Only `TRITON_ATTN` is instrumented. Batch shapes are backend-independent, so
  the resulting config is still valid input for the other benchmark variants.
- With tensor parallelism each rank runs its own builder and `atexit` handler.
  The temp file is pid-suffixed so writes do not interleave, but every rank then
  renames onto the same final path, so the last writer wins. Give each rank a
  distinct `VLLM_TRITON_ATTN_DUMP_PATH` if you want per-rank dumps. Ranks see
  the same batch shapes, so a single surviving dump is normally what you want.
