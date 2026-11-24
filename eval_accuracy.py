"""
Evaluation Benchmark for Alpha Algorithm (~2B Parameter GUI Vision-Action Model).
Measures:
1. Grounding Accuracy (Click-in-Bounding-Box rate %)
2. Euclidean Distance Error (Normalized [0..1000] and Screen Pixels)
3. Inference Latency & FPS Benchmark on NVIDIA GPU (RTX 4060 / T4)
"""

from typing import Optional, Dict, Any, List
import os
import sys
import time
import math
import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm

from model.alpha_model import AlphaVisionActionModel
from data.dataset import GUIDataset

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def point_in_bbox(px: float, py: float, bbox: List[float], tolerance: float = 0.0) -> bool:
    """
    Checks if normalized point (px, py) falls inside normalized bbox [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    # Ensure proper ordering
    min_x, max_x = min(x1, x2) - tolerance, max(x1, x2) + tolerance
    min_y, max_y = min(y1, y2) - tolerance, max(y1, y2) + tolerance
    return (min_x <= px <= max_x) and (min_y <= py <= max_y)


def evaluate(
    checkpoint_dir: Optional[str] = None,
    mode: str = "synthetic",
    data_path: Optional[str] = None,
    num_eval_samples: int = 50,
    image_size: int = 768,
    benchmark_iters: int = 20,
    tolerance_pct: float = 0.0
):
    console = Console() if HAS_RICH else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    if not torch.cuda.is_available():
        torch_dtype = torch.float32

    print("\n" + "=" * 70)
    print("      ALPHA ALGORITHM - GUI BENCHMARK & EVALUATION ENGINE")
    print("=" * 70)
    print(f"• Evaluation Mode:      {mode}")
    print(f"• Samples:              {num_eval_samples}")
    print(f"• Target Device:        {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"• Checkpoint:           {checkpoint_dir or 'Base / Initialized'}")
    print("=" * 70 + "\n")

    # 1. Initialize Model
    print("Loading Alpha Model for evaluation...")
    model = AlphaVisionActionModel(
        load_in_4bit=torch.cuda.is_available(),
        torch_dtype=torch_dtype
    )

    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"Loading trained weights from: {checkpoint_dir}")
        model.load_lightweight_checkpoint(checkpoint_dir)

    model.eval()

    # 2. Prepare Dataset
    dataset = GUIDataset(
        mode=mode,
        data_path=data_path,
        num_synthetic_samples=num_eval_samples,
        image_size=image_size,
        tokenizer=model.tokenizer,
        seed=1337  # Distinct evaluation seed
    )

    # 3. Grounding Accuracy Evaluation
    correct_clicks = 0
    euclidean_errors_norm = []
    euclidean_errors_px = []
    screen_w, screen_h = 1920, 1080  # Reference desktop resolution

    print(f"\nEvaluating click accuracy across {num_eval_samples} GUI test scenarios...\n")

    for i in tqdm(range(num_eval_samples), desc="Evaluating Accuracy"):
        sample = dataset[i]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device=device, dtype=torch_dtype)
        instruction = sample["instruction"]
        gt_bbox = sample["norm_bbox"]
        gt_click = sample["norm_click"]

        # Run model action prediction
        with torch.no_grad():
            output_str = model.generate_action(pixel_values, instruction)

        # Parse output JSON
        pred_x, pred_y = 500, 500
        try:
            start_idx = output_str.find("{")
            end_idx = output_str.rfind("}")
            if start_idx != -1 and end_idx != -1:
                data = json.loads(output_str[start_idx:end_idx + 1])
                pred_x = float(data.get("x", 500))
                pred_y = float(data.get("y", 500))
        except Exception:
            pass

        # Check in-box accuracy
        is_hit = point_in_bbox(pred_x, pred_y, gt_bbox, tolerance=tolerance_pct * 1000)
        if is_hit:
            correct_clicks += 1

        # Distance error
        gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2.0
        gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2.0
        err_norm = math.sqrt((pred_x - gt_cx) ** 2 + (pred_y - gt_cy) ** 2)
        euclidean_errors_norm.append(err_norm)

        # Map to reference screen pixels
        err_px = math.sqrt(((pred_x - gt_cx) / 1000.0 * screen_w) ** 2 + ((pred_y - gt_cy) / 1000.0 * screen_h) ** 2)
        euclidean_errors_px.append(err_px)

    accuracy_pct = (correct_clicks / num_eval_samples) * 100.0
    mean_err_norm = sum(euclidean_errors_norm) / len(euclidean_errors_norm)
    mean_err_px = sum(euclidean_errors_px) / len(euclidean_errors_px)

    # 4. Latency & FPS GPU Benchmark
    print(f"\nBenchmarking inference latency and FPS ({benchmark_iters} iterations)...")
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device, dtype=torch_dtype)

    # GPU warmup
    if torch.cuda.is_available():
        for _ in range(5):
            _ = model.encode_vision(dummy_input)
        torch.cuda.synchronize()

    # Benchmark Vision Encoder + Spatial Projector
    start_v = time.time()
    for _ in range(benchmark_iters):
        with torch.no_grad():
            _ = model.encode_vision(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    vision_latency_ms = ((time.time() - start_v) / benchmark_iters) * 1000.0

    # Benchmark full end-to-end action generation
    start_full = time.time()
    full_iters = min(10, benchmark_iters)
    for _ in range(full_iters):
        with torch.no_grad():
            _ = model.generate_action(dummy_input, "Click the Submit button", max_new_tokens=48)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    full_latency_ms = ((time.time() - start_full) / full_iters) * 1000.0
    fps = 1000.0 / full_latency_ms if full_latency_ms > 0 else 0.0

    # 5. Display Benchmark Results
    if HAS_RICH:
        table = Table(title="Alpha Algorithm Performance & Accuracy Benchmark", show_lines=True)
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Result", style="green bold", justify="right")
        table.add_column("Description / Standard", style="yellow", justify="left")

        table.add_row("Click-in-BBox Accuracy", f"{accuracy_pct:.1f}%", "Hit rate inside ground-truth element boundary")
        table.add_row("Mean Norm Distance Error", f"{mean_err_norm:.1f} / 1000", "Normalized coordinate Euclidean distance")
        table.add_row("Mean Pixel Distance Error", f"{mean_err_px:.1f} px", "Physical pixel offset on 1080p display")
        table.add_row("CNN + Projector Latency", f"{vision_latency_ms:.2f} ms", "Time to extract 2D spatial visual tokens")
        table.add_row("End-to-End Action Latency", f"{full_latency_ms:.2f} ms", "Perception + Autoregressive JSON generation")
        table.add_row("Throughput (Actions/Sec)", f"{fps:.2f} FPS", "Continuous interactive desktop control rate")

        console.print()
        console.print(table)
        console.print()
    else:
        print("\n" + "=" * 70)
        print("ALPHA ALGORITHM PERFORMANCE BENCHMARK RESULTS")
        print("=" * 70)
        print(f"• Click-in-BBox Accuracy:      {accuracy_pct:.1f}%")
        print(f"• Mean Normalized Distance:     {mean_err_norm:.1f} / 1000")
        print(f"• Mean Pixel Distance Error:    {mean_err_px:.1f} px (1080p)")
        print(f"• CNN + Projector Latency:      {vision_latency_ms:.2f} ms")
        print(f"• End-to-End Action Latency:    {full_latency_ms:.2f} ms")
        print(f"• Throughput:                   {fps:.2f} FPS")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Alpha Algorithm Grounding Accuracy & Latency")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--mode", type=str, choices=["synthetic", "huggingface"], default="synthetic")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--samples", type=int, default=30, help="Number of test samples")
    parser.add_argument("--benchmark-iters", type=int, default=15, help="Number of benchmark iterations")
    args = parser.parse_args()

    evaluate(
        checkpoint_dir=args.checkpoint,
        mode=args.mode,
        data_path=args.data_path,
        num_eval_samples=args.samples,
        benchmark_iters=args.benchmark_iters
    )


if __name__ == "__main__":
    main()
