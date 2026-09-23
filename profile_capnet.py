import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from methods import CapNet
def parse_args():
    parser = argparse.ArgumentParser(description="Measure CapNet FLOPs and parameters")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--modalities",
        type=int,
        default=2,
        help="Number of images in data['imgs']; 2 means RGB + depth.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--top-ops",
        type=int,
        default=15,
        help="Number of operator rows shown in the profiler summary.",
    )
    return parser.parse_args()


def format_count(value):
    for suffix, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.3f} {suffix}"
    return f"{value:.0f}"


def main():
    args = parse_args()
    if args.modalities < 1:
        raise ValueError("--modalities must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Re-run with --device cpu.")

    device = torch.device(args.device)
    model = CapNet(pretrained=True).to(device).eval()
    images = [
        torch.randn(args.batch_size, 3, args.height, args.width, device=device)
        for _ in range(args.modalities)
    ]
    inputs = {"imgs": images, "train": False, "shape": (args.height, args.width)}

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with torch.inference_mode():
        # Warm up lazy kernels and verify that the requested input is valid.
        output = model(data=inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()

        with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
            output = model(data=inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

    events = prof.key_averages()
    total_flops = sum(event.flops or 0 for event in events)
    flops_by_op = defaultdict(int)
    for event in events:
        if event.flops:
            flops_by_op[event.key] += event.flops

    print("\nCapNet complexity")
    print(f"  Input             : {args.batch_size} x {args.modalities} x 3 x "
          f"{args.height} x {args.width}")
    print(f"  Output shape      : {tuple(output.shape)}")
    print(f"  Total parameters  : {format_count(total_params)} ({total_params:,})")
    print(f"  Trainable params  : {format_count(trainable_params)} ({trainable_params:,})")
    print(f"  Counted FLOPs     : {format_count(total_flops)} ({total_flops:,})")
    print(f"  Counted GFLOPs/sample: {total_flops / args.batch_size / 1e9:.3f}")
    print("\nFLOPs counted by operator:")
    for name, flops in sorted(flops_by_op.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name:<28} {format_count(flops)}")

    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print("\nProfiler summary:")
    print(events.table(sort_by=sort_by, row_limit=args.top_ops))
    attention_ops = [
        event.key
        for event in events
        if "attention" in event.key.lower() and not event.flops
    ]
    print("\nCounting convention: multiply-add = 2 FLOPs.")
    print("Only operators supported by PyTorch's FLOP formulas are included.")
    if attention_ops:
        print(
            "WARNING: PyTorch did not assign FLOPs to these attention operators; "
            "the reported result is therefore a lower bound:"
        )
        for name in sorted(set(attention_ops)):
            print(f"  - {name}")


if __name__ == "__main__":
    main()
