#!/usr/bin/env python3
"""
gpu_ml_benchmark.py — Minimal PyTorch GPU benchmark
Requirements: torch (with CUDA). No other deps.

It measures:
  1) GEMM (matrix multiply)
  2) Conv2D (resnet-like 3x3)
  3) Toy self-attention
  4) A tiny training step (forward+backward+optimizer)

Usage examples:
  python gpu_ml_benchmark.py
  python gpu_ml_benchmark.py --precision fp16
  python gpu_ml_benchmark.py --precision bf16 --repeats 50
  python gpu_ml_benchmark.py --no-tf32
"""

import argparse, time, math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=30, help="Timed iters per test (after warmup)")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iters per test")
    p.add_argument("--precision", choices=["fp32","fp16","bf16","auto"], default="auto")
    p.add_argument("--no-tf32", action="store_true", help="Disable TF32 for matmul/conv")
    p.add_argument("--size", type=int, default=4096, help="Square size for GEMM (N x N)")
    p.add_argument("--batch", type=int, default=32, help="Batch size for conv/attn/training")
    p.add_argument("--seq", type=int, default=512, help="Sequence length for attention")
    p.add_argument("--dim", type=int, default=768, help="Hidden dim for attention and MLP")
    p.add_argument("--heads", type=int, default=12, help="Number of attention heads")
    return p.parse_args()

def env_info():
    lines = []
    lines.append(f"PyTorch: {torch.__version__}")
    cuda = torch.version.cuda
    lines.append(f"CUDA: {cuda}")
    lines.append(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        lines.append(f"[{i}] {props.name} | CC {props.major}.{props.minor} | {props.total_memory/1024**3:.1f} GB")
    return "\n".join(lines)

def set_precision(args):
    if args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        # Enable TF32 on supported GPUs (Ampere/Ada+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.precision == "auto":
        # Prefer bf16 if supported, else fp16, else fp32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif torch.cuda.is_available():
            return torch.float16
        else:
            return torch.float32
    elif args.precision == "fp32":
        return torch.float32
    elif args.precision == "fp16":
        return torch.float16
    elif args.precision == "bf16":
        return torch.bfloat16

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timer(fn, warmup, repeats):
    # Warmup
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync()
    t1 = time.perf_counter()
    return (t1 - t0) / repeats

def mbps(n_bytes, sec):
    return (n_bytes / 1e6) / sec

def bench_gemm(dtype, N, device, repeats, warmup):
    A = torch.randn(N, N, device=device, dtype=dtype)
    B = torch.randn(N, N, device=device, dtype=dtype)
    bytes_moved = (A.numel() + B.numel() + (N*N)) * A.element_size()
    def run():
        C = A @ B
        return C
    avg = timer(lambda: run(), warmup, repeats)
    # Rough FLOPs for GEMM: 2*N^3
    tflops = (2 * (N**3)) / avg / 1e12
    return {"time_s": avg, "throughput_TFLOP/s": tflops, "approx_mem_MBps": mbps(bytes_moved, avg)}

def bench_conv(dtype, B, device, repeats, warmup):
    # ResNet-like: Bx3x224x224, 64 filters, 3x3
    x = torch.randn(B, 3, 224, 224, device=device, dtype=dtype)
    conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device=device, dtype=dtype)
    # Ensure cuDNN autotune is on
    torch.backends.cudnn.benchmark = True
    def run():
        y = conv(x)
        return y
    avg = timer(lambda: run(), warmup, repeats)
    imgs_per_s = B / avg
    return {"time_s": avg, "images_per_s": imgs_per_s}

def bench_attention(dtype, B, H, S, D, device, repeats, warmup):
    # qkv proj
    model_dim = D
    qkv = nn.Linear(model_dim, 3*model_dim, bias=False).to(device=device, dtype=dtype)
    out = nn.Linear(model_dim, model_dim, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(B, S, model_dim, device=device, dtype=dtype)
    head_dim = model_dim // H
    scale = 1.0 / math.sqrt(head_dim)

    def run():
        qkv_proj = qkv(x)  # [B, S, 3*D]
        q, k, v = qkv_proj.chunk(3, dim=-1)
        # reshape to [B, H, S, Hd]
        q = q.view(B, S, H, head_dim).transpose(1, 2)
        k = k.view(B, S, H, head_dim).transpose(1, 2)
        v = v.view(B, S, H, head_dim).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, S, model_dim)
        y = out(y)
        return y

    avg = timer(lambda: run(), warmup, repeats)
    toks_per_s = (B * S) / avg
    return {"time_s": avg, "tokens_per_s": toks_per_s}

class TinyNet(nn.Module):
    def __init__(self, dim=768, classes=1000):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4*dim)
        self.fc2 = nn.Linear(4*dim, dim)
        self.fc3 = nn.Linear(dim, classes)
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x

def bench_training(dtype, B, D, device, repeats, warmup):
    torch.manual_seed(0)
    net = TinyNet(dim=D, classes=1000).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    x = torch.randn(B, D, device=device, dtype=dtype)
    target = torch.randint(0, 1000, (B,), device=device)
    loss_fn = nn.CrossEntropyLoss()

    def run():
        opt.zero_grad(set_to_none=True)
        logits = net(x)
        loss = loss_fn(logits, target)
        loss.backward()
        opt.step()
        return loss

    # Use autocast for mixed precision if dtype is fp16/bf16
    use_amp = dtype in (torch.float16, torch.bfloat16)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype==torch.float16))

    def run_amp():
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
            logits = net(x)
            loss = loss_fn(logits, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    fn = run_amp if use_amp else run
    avg = timer(lambda: fn(), warmup, repeats)
    steps_per_s = 1.0 / avg
    return {"time_s": avg, "steps_per_s": steps_per_s}

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please install a CUDA-enabled PyTorch build and run on a CUDA GPU.")
        sys.exit(1)

    args = parse_args()
    dtype = set_precision(args)
    device = torch.device("cuda")

    print("=== Environment ===")
    print(env_info())
    print(f"Precision: {str(dtype).split('.')[-1]} | TF32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"Args: repeats={args.repeats}, warmup={args.warmup}, size={args.size}, batch={args.batch}, seq={args.seq}, dim={args.dim}, heads={args.heads}")
    print("")

    results = {}

    # GEMM
    results["gemm"] = bench_gemm(dtype, args.size, device, args.repeats, args.warmup)
    print("[GEMM] N={}  avg_time={:.6f}s  throughput={:.2f} TFLOP/s  approx_mem={:.1f} MB/s".format(
        args.size, results["gemm"]["time_s"], results["gemm"]["throughput_TFLOP/s"], results["gemm"]["approx_mem_MBps"]
    ))

    # Conv2D
    results["conv2d"] = bench_conv(dtype, args.batch, device, args.repeats, args.warmup)
    print("[Conv2D] B={}, 224x224x3 -> 64@3x3  avg_time={:.6f}s  images/s={:.1f}".format(
        args.batch, results["conv2d"]["time_s"], results["conv2d"]["images_per_s"]
    ))

    # Attention
    results["attention"] = bench_attention(dtype, args.batch, args.heads, args.seq, args.dim, device, args.repeats, args.warmup)
    print("[Attention] B={}, S={}, D={}, H={}  avg_time={:.6f}s  tokens/s={:.1f}".format(
        args.batch, args.seq, args.dim, args.heads, results["attention"]["time_s"], results["attention"]["tokens_per_s"]
    ))

    # Training step
    results["training"] = bench_training(dtype, args.batch, args.dim, device, args.repeats, args.warmup)
    print("[TrainStep] B={}, D={}  avg_time={:.6f}s  steps/s={:.2f}".format(
        args.batch, args.dim, results["training"]["time_s"], results["training"]["steps_per_s"]
    ))

    print("")
    print("Tip: Run this script on both GPUs with the same args, then compare numbers.")
    print("     For apples-to-apples, keep driver/CUDA/torch as close as possible across machines.")
    print("     Try: --precision fp16 (or bf16) and --no-tf32 to see the impact of precision settings.")

if __name__ == "__main__":
    main()
