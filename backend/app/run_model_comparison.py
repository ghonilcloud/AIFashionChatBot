#!/usr/bin/env python3
"""
CLI for running multi-model comparison benchmarks.

Usage:
    python run_model_comparison.py --sketch path/to/sketch.png --prompt "your prompt here"
    python run_model_comparison.py --sketch path/to/sketch.png --prompt "your prompt" --models sd-v1.5 sdxl ssd-1b
    python run_model_comparison.py --sketch path/to/sketch.png --prompt "your prompt" --steps 10 20 50 --trials 3
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.model_comparison import ModelComparisonRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive multi-model comparison for fashion design generation"
    )
    
    parser.add_argument(
        "--sketch",
        type=str,
        required=True,
        help="Path to the sketch image to use for all comparisons"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="A fashionable garment with vibrant colors, detailed seams, and professional tailoring. Realistic fabric textures. Solid neutral gray background.",
        help="Prompt to use for all generations (default: generic fashion prompt)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sd-v1.5", "sdxl", "ssd-1b"],
        help="Models to compare (default: sd-v1.5 sdxl ssd-1b)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[10, 20, 50],
        help="Inference steps to test (default: 10 20 50)"
    )
    
    parser.add_argument(
        "--guidance",
        type=float,
        nargs="+",
        default=[7.5, 10.0],
        help="Guidance scales to test (default: 7.5 10.0)"
    )
    
    parser.add_argument(
        "--conditioning",
        type=float,
        nargs="+",
        default=[0.7, 1.0],
        help="ControlNet conditioning scales to test (default: 0.7 1.0)"
    )
    
    parser.add_argument(
        "--trials",
        type=int,
        default=2,
        help="Number of trials per configuration (default: 2)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_results",
        help="Output directory for results (default: comparison_results)"
    )
    
    args = parser.parse_args()
    
    # Validate sketch path
    sketch_path = Path(args.sketch)
    if not sketch_path.exists():
        print(f"[ERROR] Sketch not found: {sketch_path}")
        sys.exit(1)
    
    # Create runner
    runner = ModelComparisonRunner(sketch_path=sketch_path, output_dir=args.output)
    
    # Run comparison
    runner.run_comparative_benchmark(
        prompt=args.prompt,
        models=args.models,
        num_inference_steps_list=args.steps,
        guidance_scales=args.guidance,
        controlnet_conditioning_scales=args.conditioning,
        trials=args.trials,
    )
    
    print(f"\n[SUCCESS] Comparison complete! Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()
