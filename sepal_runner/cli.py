"""`sepal serve MODEL`: fetch a Sepal model, look at the machine, pick settings that pass parity, and serve `POST /v1/systemone`.

    sepal serve BlossomsAI/Sepal-4B                  # a Hub name: adapter and pinned base are fetched on first use
    sepal serve E:\\models\\Sepal-4B                    # a local checkpoint directory (the one holding sepal.json)
    sepal serve BlossomsAI/Sepal-0.8B --dry-run      # print the plan and exit

Profiles chosen automatically: on a CUDA card of compute capability 8.0 or later (RTX 30xx and newer, A100), `gpu-f16` on 12 GB and up when
the plan fits, otherwise `cpu-bf16`. On a card below 8.0 (Pascal: GTX 10xx, Tesla P40; Volta and Turing too) the binary16 profiles `gpu-f16`
and `gpu-q8w` cannot run (they need tensor-core kernels) and are refused by name; `auto` takes `gpu-bf16` (bfloat16 weights, float32
arithmetic on ordinary CUDA cores) when its plan fits, else `gpu-w8g` (the same arithmetic over int8 weights with a float32 scale per group of
128 input values, about half the weight memory), else `cpu-bf16`. `gpu-bf16` and `gpu-w8g` can also be asked for with `--profile` on any CUDA
card, but on a card with tensor cores `gpu-f16` is about four times faster. The compute capability comes from `nvidia-smi`; a driver too old
to report it counts as unknown, and unknown is treated as a modern card.
`gpu-q8w` and `cpu-q8` (int8) exceed the q8 parity gate on this model (max |dp| on near-tie questions), so they are used only with `--allow-q8`,
with a printed warning. `gpu-w8g` is weight-only int8 as well; its parity is still being measured, it is not behind `--allow-q8`, and it
always prints a note saying that it stores int8 weights.
A model that fits neither is refused with the numbers and what would fit, before any weights are loaded.

Only public downloads, without login, unless $HF_TOKEN is set (the Sepal repositories are gated: accept the terms on the Hub first).
The memory model below is fitted to measurements of Sepal-0.8B and Sepal-4B on an RTX 3060 and a Ryzen 5 7600X (SUMMARY.md); it is an
estimate, and a plan that turns out too tight still fails with the engine's own message.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

GIB = 1 << 30
MIB = 1 << 20
HUB_ORG = "BlossomsAI"
WINDOWS = (24576, 16384, 12288, 8192, 4096)      # context windows tried, largest first
CALL_TOKENS = (12288, 8192, 4096, 2048, 1024)    # GPU max_call_tokens tried, largest first
SIMT_PROFILES = ("gpu-bf16", "gpu-w8g")          # float32 arithmetic on ordinary CUDA cores: any card from compute capability 6.1
TENSOR_PROFILES = ("gpu-f16", "gpu-q8w")         # binary16 tensor-core kernels: compute capability 8.0 and later
GPU_PROFILES = TENSOR_PROFILES + SIMT_PROFILES
SIMT_CALL_TOKENS = (2048, 1024)                  # the float32 path takes about 174 KB of device memory per call token: 2048 unless that does not fit
SIMT_WEIGHT_RATIO = {"gpu-bf16": 1.0, "gpu-w8g": 0.517}   # device weights / bf16 trunk bytes as stored (4B: 6.65 GiB measured; int8 + scales 3.69e9 computed)
SIMT_CALL_BYTES = 174 * 1024                     # activation buffers per call token at 4B (hidden 2560)


class Refusal(SystemExit):
    """A plan that cannot be served: the message names the numbers and what would fit."""


# ---------------------------------------------------------------- hardware

@dataclass
class Hardware:
    cores: int
    threads: int
    ram_total: int
    ram_free: int
    gpu_name: str | None = None
    vram_total: int = 0
    vram_free: int = 0
    compute_cap: float | None = None     # CUDA compute capability (6.1, 8.6, ...); None = unknown, which is treated as a modern card

    def describe(self) -> str:
        cc = f", compute capability {self.compute_cap:.1f}" if self.compute_cap is not None else ""
        gpu = f"{self.gpu_name} {self.vram_total / GIB:.1f} GiB ({self.vram_free / GIB:.1f} GiB free{cc})" if self.gpu_name else "no CUDA GPU"
        return f"{self.cores} cores / {self.threads} threads, RAM {self.ram_total / GIB:.1f} GiB ({self.ram_free / GIB:.1f} GiB free), {gpu}"


def _memory() -> tuple[int, int]:
    try:
        import psutil

        m = psutil.virtual_memory()
        return int(m.total), int(m.available)
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                        ("ave", ctypes.c_ulonglong)]

        s = Status()
        s.length = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return int(s.total), int(s.avail)
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        fields[k] = int(v.split()[0]) * 1024
    return fields["MemTotal"], fields.get("MemAvailable", fields["MemFree"])


def _physical_cores(threads: int) -> int:
    try:
        import psutil

        return psutil.cpu_count(logical=False) or threads
    except ImportError:
        return max(1, threads // 2) if threads > 2 else threads


def parse_gpu_line(line: str) -> tuple[str, int, int, float | None]:
    """(name, total MiB, free MiB, compute capability or None) from one `nvidia-smi --format=csv,noheader,nounits` line.

    The line has `name,total,free` or, from drivers that know the field, `name,total,free,compute_cap`. A compute capability that is
    missing or not a number ([N/A]) is None. Raises ValueError for a line that is neither.
    """
    parts = [x.strip() for x in line.split(",")]
    if len(parts) < 3:
        raise ValueError(f"not an nvidia-smi GPU line: {line!r}")
    cc: float | None = None
    if len(parts) >= 4:
        try:
            cc = float(parts[-1])
        except ValueError:
            cc = None
        else:
            cc = cc if cc > 0 else None
        parts = parts[:-1]
    name = ",".join(parts[:-2])
    return name, int(parts[-2]), int(parts[-1]), cc


def _query_gpu(smi: str) -> tuple[str, int, int, float | None] | None:
    """The first GPU, asking for compute_cap first and again without it when the driver does not know the field."""
    for fields in ("name,memory.total,memory.free,compute_cap", "name,memory.total,memory.free"):
        try:
            run = subprocess.run([smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        lines = run.stdout.strip().splitlines()
        if run.returncode != 0 or not lines:
            continue
        try:
            return parse_gpu_line(lines[0])
        except ValueError:
            continue
    return None


def detect_hardware() -> Hardware:
    threads = os.cpu_count() or 1
    total, free = _memory()
    hw = Hardware(cores=_physical_cores(threads), threads=threads, ram_total=total, ram_free=free)
    smi = shutil.which("nvidia-smi")
    if smi:
        gpu = _query_gpu(smi)
        if gpu:
            name, t, f, cc = gpu
            hw.gpu_name, hw.vram_total, hw.vram_free, hw.compute_cap = name, t * MIB, f * MIB, cc
    return hw


# ---------------------------------------------------------------- model files

def _headers(base: Path) -> dict[str, int]:
    """name -> byte size of every tensor in the base's safetensors files (headers only)."""
    sizes: dict[str, int] = {}
    for f in sorted(base.glob("*.safetensors")):
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                a, b = meta["data_offsets"]
                sizes[name] = b - a
    return sizes


@dataclass
class Shape:
    trunk_bytes: int        # layers + final norm, as stored (bf16)
    hidden: int
    kv_bytes_per_token: int
    slot_bytes: int


def model_shape(base: Path) -> Shape:
    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    t = cfg.get("text_config", cfg)
    sizes = _headers(base)
    trunk = sum(v for k, v in sizes.items() if ".layers." in k and "visual" not in k and "mtp" not in k) + \
        sum(v for k, v in sizes.items() if k.endswith("language_model.norm.weight") or k == "model.norm.weight")
    layers, interval = int(t["num_hidden_layers"]), int(t.get("full_attention_interval", 4))
    n_attn = layers // interval
    kv = n_attn * 2 * int(t["num_key_value_heads"]) * int(t.get("head_dim", 256)) * 2
    hidden = int(t["hidden_size"])
    return Shape(trunk, hidden, kv, int(50.3 * MIB * hidden / 2560))


def cpu_need_q8(shape: Shape, window: int) -> int:
    """Committed bytes of cpu-q8 (int8 weights with block scales; measured 5.7 GiB for 4B)."""
    scale = shape.trunk_bytes / 7.14e9
    return int(0.82 * shape.trunk_bytes + 0.9 * GIB + 0.8 * GIB * (window / 24576) * (scale + 0.3))


def cpu_need(shape: Shape, window: int) -> int:
    """Committed bytes of cpu-bf16: packed bf16 weights, thread workspace, per-request state and K/V (fit: 4B 8.2 GiB short, 9.7 at 24k; 0.8B 2.0 and 2.4)."""
    scale = shape.trunk_bytes / (7.14e9)
    return int(1.15 * shape.trunk_bytes + 0.8 * GIB + 1.0 * GIB * (window / 24576) * (scale + 0.3))


@dataclass
class GpuPlan:
    profile: str
    window: int
    max_call_tokens: int
    budget_mib: int
    prefix_cache: bool
    footprint: int           # expected device bytes beyond what Windows or other programs already hold


def gpu_need(shape: Shape, profile: str, window: int, call_tokens: int) -> tuple[int, int]:
    """(budget bytes the engine needs, bytes outside its budget: CUDA context and the Clef head).

    gpu-bf16 / gpu-w8g keep the embedding table in host memory, hold keys and values in float32 (twice the binary16 profiles' bytes) and
    their activation buffers are float32; the DeltaNet state slot is float32 in every profile.
    """
    simt = profile in SIMT_PROFILES
    ratio = SIMT_WEIGHT_RATIO[profile] if simt else 1.178 if profile == "gpu-f16" else 0.68
    weights = ratio * shape.trunk_bytes
    buffers = call_tokens * (SIMT_CALL_BYTES / MIB if simt else 0.179) * MIB * shape.hidden / 2560
    kv = 1.15 * window * shape.kv_bytes_per_token * (2 if simt else 1)
    need = weights + buffers + kv + 4 * shape.slot_bytes + 64 * MIB
    outside = 1.2 * GIB + 0.6 * GIB * shape.trunk_bytes / 7.14e9
    return int(need), int(outside)


def plan_gpu(shape: Shape, hw: Hardware, profile: str, model_window: int) -> GpuPlan | None:
    avail = hw.vram_free - int(0.3 * GIB)
    for window in [w for w in WINDOWS if w <= model_window] or [model_window]:
        for calls in (SIMT_CALL_TOKENS if profile in SIMT_PROFILES else CALL_TOKENS):
            if calls > 2 * window:
                continue
            need, outside = gpu_need(shape, profile, window, calls)
            if need + outside <= avail:
                budget = min(avail - outside, int(need * 1.12) + 4 * shape.slot_bytes)
                spare = budget - need
                return GpuPlan(profile, window, calls, budget // MIB, spare >= 6 * shape.slot_bytes, budget + outside)
        if window <= 4096:
            break
    return None


# ---------------------------------------------------------------- fetching

def _hub():
    try:
        import huggingface_hub
    except ImportError:
        raise Refusal("fetching from the Hub needs huggingface_hub (pip install huggingface_hub), or give a local directory")
    return huggingface_hub


def cache_dir(arg: str | None) -> Path:
    return Path(arg or os.environ.get("SEPAL_HOME") or Path.home() / ".cache" / "sepal-runner")


def _download(hub, repo: str, dest: Path, revision: str | None = None, patterns: list[str] | None = None) -> Path:
    token = os.environ.get("HF_TOKEN") or False        # never the stored login: public downloads, or the token the owner put in the environment
    try:
        return Path(hub.snapshot_download(repo, revision=revision, local_dir=str(dest), allow_patterns=patterns, token=token))
    except Exception as error:                          # noqa: BLE001
        kind = type(error).__name__
        if "Gated" in kind or "401" in str(error):
            raise Refusal(f"{repo} is gated: accept its terms on https://huggingface.co/{repo} and set HF_TOKEN to a read token, "
                          f"or download it yourself and pass the directory ({kind})")
        raise Refusal(f"could not fetch {repo}: {kind}: {str(error)[:200]}")


def resolve_model(spec: str, cache: Path) -> tuple[Path, str]:
    """(checkpoint directory holding sepal.json, served name)."""
    path = Path(spec)
    if path.is_dir():
        if (path / "sepal.json").is_file():
            return path, (path.parent.name if path.name == "final" else path.name)
        if (path / "final" / "sepal.json").is_file():
            return path / "final", path.parent.name or path.name
        raise Refusal(f"{path} has no sepal.json: pass the checkpoint directory (the one with adapter_model.safetensors)")
    repo = spec if "/" in spec else f"{HUB_ORG}/{spec}"
    dest = cache / "models" / repo.replace("/", "--")
    if not (dest / "sepal.json").is_file() or not (dest / "adapter_model.safetensors").is_file():
        print(f"fetching {repo} -> {dest}", flush=True)
        _download(_hub(), repo, dest)
    return dest, repo.split("/")[-1].lower()


def resolve_base(meta: dict, cache: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    base = meta["base"]
    repo, rev = base["id"], base["revision"]
    dest = cache / "bases" / f"{repo.replace('/', '--')}@{rev[:8]}"
    if not (dest / "config.json").is_file() or not list(dest.glob("*.safetensors")):
        print(f"fetching base {repo}@{rev[:8]} -> {dest}", flush=True)
        _download(_hub(), repo, dest, rev, ["config.json", "tokenizer*", "vocab.json", "merges.txt", "*.safetensors", "*.safetensors.index.json"])
    return dest


def find_library(kind: str, explicit: str | None) -> str | None:
    names = [f"sepal_runner_{kind}.dll", f"libsepal_runner_{kind}.so", f"sepal_runner_{kind}.so", f"libsepal_runner_{kind}.dylib"]
    if explicit and (kind == "cuda") == ("cuda" in Path(explicit).name):
        return explicit
    roots = [Path(os.environ.get("SEPAL_RUNNER_LIB_DIR", "")), Path(__file__).parent / "lib", Path(__file__).resolve().parents[1] / "runner" / "build" / "release",
             Path(__file__).resolve().parents[1] / "runner" / "cuda" / "build" / "release", Path("/opt/sepal/lib")]
    for root in roots:
        for n in names:
            if (root / n).is_file():
                return str(root / n)
    return None


# ---------------------------------------------------------------- the plan

@dataclass
class Plan:
    profile: str
    lib: str
    window: int
    engine_options: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    need: str = ""


def _cpu_options(args) -> dict:
    options: dict = {}
    if getattr(args, "threads", 0):
        options["threads"] = args.threads
    if getattr(args, "cpus", None):
        options["cpus"] = args.cpus
    if getattr(args, "cpu_isa", "auto") not in (None, "auto"):
        options["cpu_isa"] = args.cpu_isa
    return options


def make_plan(shape: Shape, hw: Hardware, model_window: int, args) -> Plan:
    notes: list[str] = []
    cpu_lib, gpu_lib = find_library("cpu", args.lib), find_library("cuda", args.lib)
    choice = args.profile
    wants_gpu = choice in ("auto",) + GPU_PROFILES and hw.gpu_name and gpu_lib
    # Below compute capability 8.0 (Pascal, Volta, Turing) the binary16 profiles have no kernels; an unknown capability counts as a modern card.
    pre_ampere = bool(hw.gpu_name) and hw.compute_cap is not None and hw.compute_cap < 8.0
    if choice in TENSOR_PROFILES and pre_ampere:
        raise Refusal(f"{choice} needs tensor cores (compute capability 8.0 or later) and {hw.gpu_name} is compute capability {hw.compute_cap:.1f}; "
                      f"use gpu-bf16 (bfloat16 weights, float32 arithmetic; about {sum(gpu_need(shape, 'gpu-bf16', 4096, 1024)) / GIB:.1f} GiB of VRAM for this model "
                      f"with a 4096-token window) or gpu-w8g (int8 weights with a float32 scale per 128 values; about {sum(gpu_need(shape, 'gpu-w8g', 4096, 1024)) / GIB:.1f} GiB), "
                      f"or leave --profile on auto")
    if choice in GPU_PROFILES and not wants_gpu:
        raise Refusal(f"{choice} needs a CUDA GPU with nvidia-smi and the sepal_runner_cuda library; this machine has: {hw.describe()}"
                      f"{'' if gpu_lib else ' (library not found)'}")
    if choice == "cpu-q8" and not args.allow_q8w:
        raise Refusal("cpu-q8 exceeds the q8 parity gate on this model (max |dp| 0.14 on 300 SEC-dev records against 0.05 allowed, near-tie questions); add --allow-q8 to serve it anyway")
    if choice == "gpu-q8w" and not args.allow_q8w:
        raise Refusal("gpu-q8w fails the q8 parity gate on this model (max |dp| 0.12 to 0.14 against 0.05, 1.1% flips on LONG-LOG); add --allow-q8 to serve it anyway")
    gpu_fail = ""
    if wants_gpu and choice in ("auto",) + GPU_PROFILES:
        tried = []
        if choice == "auto":
            candidates = list(SIMT_PROFILES) if pre_ampere else ["gpu-f16"] + (["gpu-q8w"] if args.allow_q8w else [])
        else:
            candidates = [choice] + (["gpu-q8w"] if (args.allow_q8w and choice == "gpu-f16") else [])
        for profile in candidates:
            if profile == "gpu-f16" and hw.vram_total < 11 * GIB:
                tried.append(f"gpu-f16 is only used on 12 GB cards and up (this one has {hw.vram_total / GIB:.1f} GiB)")
                continue
            plan = plan_gpu(shape, hw, profile, model_window)
            if plan is None:
                need, outside = gpu_need(shape, profile, 4096, 1024)
                tried.append(f"{profile} needs about {(need + outside) / GIB:.1f} GiB of free VRAM for a 4096-token window, {hw.vram_free / GIB:.1f} GiB are free")
                continue
            if profile == "gpu-q8w":
                notes.append("WARNING: gpu-q8w serves int8 weights; on this model it exceeds the documented q8 gate (max |dp| 0.12 to 0.14 on SEC-dev and LONG-LOG, "
                             "1.1% flips on LONG-LOG). Use it only where gpu-f16 does not fit and answers near 0.5 may flip.")
            if profile == "gpu-w8g":
                notes.append("gpu-w8g stores int8 weights (one float32 scale per group of 128 input values; activations, keys, values and states stay float32). "
                             "Its parity against the float profiles is still being measured; gpu-bf16 is the float profile and needs about "
                             f"{sum(gpu_need(shape, 'gpu-bf16', 4096, 1024)) / GIB:.1f} GiB of VRAM for this model at a 4096-token window.")
            if plan.window < model_window:
                notes.append(f"context limited to {plan.window} tokens by free VRAM (the model's window is {model_window}); longer requests get 422")
            opts = {"max_call_tokens": plan.max_call_tokens, "trunk_budget_mib": plan.budget_mib, "prefix_cache": plan.prefix_cache}
            return Plan(profile, gpu_lib, plan.window, {**opts, "threads": min(4, hw.cores)}, notes,
                        f"about {plan.footprint / GIB:.1f} GiB of VRAM and {(shape.trunk_bytes * 1.7) / GIB:.1f} GiB of RAM at load")
        gpu_fail = "; ".join(tried)
        if choice == "gpu-bf16":
            gpu_fail += f"; gpu-w8g (int8 weights) needs about {sum(gpu_need(shape, 'gpu-w8g', 4096, 1024)) / GIB:.1f} GiB"
        if choice != "auto":
            raise Refusal(f"{choice} does not fit: {gpu_fail}")
        notes.append(f"GPU not used: {gpu_fail}")
    if not cpu_lib:
        raise Refusal("no sepal_runner_cpu library found: build runner/ (see runner/README.md) or set $SEPAL_RUNNER_LIB_DIR / --lib")
    budget = hw.ram_free - 1 * GIB
    cpu_profiles = ["cpu-q8"] if choice == "cpu-q8" else ["cpu-bf16"] + (["cpu-q8"] if (args.allow_q8w and choice == "auto") else [])
    for cpu_profile in cpu_profiles:
        needs = cpu_need_q8 if cpu_profile == "cpu-q8" else cpu_need
        for window in [w for w in WINDOWS if w <= model_window] or [model_window]:
            need = needs(shape, window)
            if need <= budget:
                if window < model_window:
                    notes.append(f"context limited to {window} tokens by free RAM (the model's window is {model_window}); longer requests get 422")
                if cpu_profile == "cpu-q8":
                    notes.append("WARNING: cpu-q8 serves int8 weights and activations; on this model it exceeds the documented q8 gate (max |dp| 0.14 on 300 SEC-dev "
                                 "records, p99 0.034, 1 flip in 432). Answers near 0.5 can move by 0.1. It is about 1.5x faster than cpu-bf16 and needs 5.7 GiB for 4B.")
                # Threads: one per physical core of the CPUs this process may use (affinity, --cpus), capped by a cgroup CPU quota;
                # the engine counts them itself, so a container limited to 10 CPUs of an 80-thread host runs 10, not 40.
                options = _cpu_options(args)
                spare_mib = int((budget - need) * 0.4) // MIB
                if spare_mib >= 512:       # states of shared log prefixes: the same log asked with other questions only runs its tail
                    options["prefix_cache_mib"] = min(2048, spare_mib)
                return Plan(cpu_profile, cpu_lib, window, options, notes, f"about {need / GIB:.1f} GiB of RAM")
            if window <= 4096:
                break
    need = cpu_need(shape, 4096)
    small = ""
    if shape.trunk_bytes > 2e9:
        small = f" With --allow-q8 the int8 profile needs about {cpu_need_q8(shape, 4096) / GIB:.1f} GiB." if not args.allow_q8w else ""
        small += f" The smaller Sepal-0.8B needs about {cpu_need(Shape(int(1.0e9), 1024, 12288, 20 * MIB), 8192) / GIB:.1f} GiB."
    raise Refusal(f"this model does not fit on this machine: cpu-bf16 needs about {need / GIB:.1f} GiB of free RAM (4096-token window), "
                  f"{budget / GIB:.1f} GiB are available" + (f"; {gpu_fail}" if gpu_fail else "") + "." + small +
                  " What would fit: 16 GiB of RAM for 4B on CPU, a 12 GB CUDA card for gpu-f16, or an 8 GB CUDA card (GTX 10xx or later) for gpu-w8g.")


# ---------------------------------------------------------------- main

def serve(args) -> int:
    from sepal_runner import server

    cache = cache_dir(args.cache_dir)
    final, name = resolve_model(args.model, cache)
    meta = json.loads((final / "sepal.json").read_text(encoding="utf-8"))
    base = resolve_base(meta, cache, args.base)
    hw = detect_hardware()
    shape = model_shape(base)
    model_window = int(meta.get("max_len_score") or 16384)
    plan = make_plan(shape, hw, model_window, args)
    host = args.host
    if not server.is_loopback(host) and not args.api_key:
        raise Refusal(f"refusing to bind {host}: a non-loopback address needs an API key (--api-key or $SEPAL_API_KEY)")
    print(f"machine : {hw.describe()}")
    print(f"model   : {name} (base {meta['base']['id']}@{meta['base']['revision'][:8]}, LoRA r{meta.get('lora', {}).get('rank', '?')} + Clef head)")
    print(f"profile : {plan.profile}   window: {plan.window} tokens   needs {plan.need}")
    print(f"engine  : {json.dumps(plan.engine_options)}")
    for note in plan.notes:
        print(f"note    : {note}")
    if args.dry_run:
        return 0
    options = {"workers": args.workers, "cache_size": args.cache_size, "max_pending": args.max_pending, "request_timeout_s": args.request_timeout}
    service, engine = server.load_service(str(final), str(base), plan.lib, plan.profile, name, window=plan.window, engine_options=plan.engine_options, **options)
    runtime = getattr(engine, "info", {}).get("runtime") if hasattr(engine, "info") else None
    if runtime:
        print(f"runtime : {runtime}")
    info = {"profile": plan.profile, "window": plan.window, "base": meta["base"]["id"]}
    httpd = server.serve(service, name, host, args.port, args.api_key, info)
    print(f"serving : http://{host}:{args.port}/v1/systemone   (also /v1/models /healthz /readyz /metrics)"
          + ("   API key required" if args.api_key else ""), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        service.close()
        engine.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sepal", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve", help="fetch, plan and serve a model")
    p.add_argument("model", help="a Hub name (BlossomsAI/Sepal-4B, or Sepal-4B) or a checkpoint directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=os.environ.get("SEPAL_API_KEY"), help="required for a non-loopback --host")
    p.add_argument("--profile", default="auto", choices=["auto", "cpu-bf16", "gpu-f16", "gpu-q8w", "gpu-bf16", "gpu-w8g", "cpu-q8"],
                   help="auto: gpu-f16 on a 12 GB card of compute capability 8.0+, gpu-bf16 / gpu-w8g on older cards (Pascal), else cpu-bf16")
    p.add_argument("--allow-q8", "--allow-q8w", dest="allow_q8w", action="store_true",
                   help="allow the int8 profiles (gpu-q8w, cpu-q8; they exceed the parity gate on this model) when the float profile does not fit")
    p.add_argument("--base", help="a local base model directory instead of fetching the pinned one")
    p.add_argument("--lib", help="path of the sepal_runner_* library")
    p.add_argument("--cache-dir", help="where downloads go (default ~/.cache/sepal-runner or $SEPAL_HOME)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-size", type=int, default=4096)
    p.add_argument("--max-pending", type=int, default=512)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--threads", type=int, default=0,
                   help="CPU engine threads (default: one per physical core of the allowed CPUs, capped by a cgroup CPU quota)")
    p.add_argument("--cpus", default=os.environ.get("SEPAL_CPU_SET"),
                   help="CPU list for the CPU engine, e.g. 0-11 (Linux): its threads are pinned there, one per physical core, "
                        "so it stays off the cores of another engine on the same machine")
    p.add_argument("--cpu-isa", default=os.environ.get("SEPAL_CPU_ISA", "auto"), choices=["auto", "avx2", "avx2-tuned", "x86-64-v3", "avx512"],
                   help="CPU kernel set (default: the best one the CPU has; a set it lacks is refused with the missing features named)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)
    return serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
