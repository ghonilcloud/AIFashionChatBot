#!/usr/bin/env python3
"""
Unified Benchmark Runner for Fashion Design Generation

Consolidates all benchmarking functionality into a single, easy-to-use script.

Features:
- Multi-model comparison (SD v1.5, SDXL, SSD-1B)
- Comprehensive metrics: CLIP, LPIPS, Sketch Fidelity (IoU, Edge Density)
- LLM-based prompt generation with Gemini
- Configurable parameters (steps, guidance, conditioning)
- CSV export for analysis

Usage Examples:
    # Quick single test
    python benchmark.py --sketch path/to/sketch.png --prompt "elegant dress"
    
    # Full model comparison
    python benchmark.py --sketch path/to/sketch.png --full-comparison
    
    # LLM-based benchmark (Gemini generates prompts from tones/Kansei)
    python benchmark.py --sketch path/to/sketch.png --llm-benchmark
    
    # Custom configuration
    python benchmark.py --sketch path/to/sketch.png --models sd-v1.5 sdxl --steps 20 50 --trials 3
"""

import argparse
import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.model_comparison import ModelComparisonRunner
from app.generator import build_gemini_instruction, call_gemini_for_prompt, refine_for_image_model


def run_single_test(sketch_path: str, prompt: str, output_dir: str = "benchmark_results"):
    """Run a single quick test with default parameters."""
    print("\n" + "="*70)
    print("SINGLE TEST MODE")
    print("="*70)
    
    runner = ModelComparisonRunner(sketch_path=sketch_path, output_dir=output_dir)
    
    result = runner.run_single_generation(
        model_name="sd-v1.5",
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.7,
        trial=1,
        seed=42
    )
    
    if result:
        runner.results.append(result)
        csv_path = runner.save_results_to_csv()
        print(f"\n✓ Test complete! Results saved to: {csv_path}")
    else:
        print("\n✗ Test failed")


def run_full_comparison(sketch_path: str, prompt: str, models: list, steps: list, 
                       guidance: list, conditioning: list, trials: int, output_dir: str):
    """Run comprehensive multi-model comparison."""
    print("\n" + "="*70)
    print("FULL MODEL COMPARISON MODE")
    print("="*70)
    
    runner = ModelComparisonRunner(sketch_path=sketch_path, output_dir=output_dir)
    
    runner.run_comparative_benchmark(
        prompt=prompt,
        models=models,
        num_inference_steps_list=steps,
        guidance_scales=guidance,
        controlnet_conditioning_scales=conditioning,
        trials=trials
    )
    
    print(f"\n✓ Comparison complete! Results saved to: {output_dir}/")


def run_llm_benchmark(sketch_path: str, output_dir: str = "comparison_results/clip_benchmark_llm"):
    """Run benchmark with LLM-generated prompts based on tones and Kansei words."""
    print("\n" + "="*70)
    print("LLM-BASED BENCHMARK MODE")
    print("="*70)
    print("Generating prompts using Gemini for 5 tone/Kansei combinations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 5 test cases: (Tone, Kansei Words)
    test_cases = [
        ("Elegant", ["Textured", "Tailored"]),
        ("Romantic", ["Fluid", "Technical"]),
        ("Street", ["Structured", "Vibrant"]),
        ("Playful", ["Contrast", "Soft"]),
        ("Luxurious", ["Organic"]),
    ]
    
    models = ["sd-v1.5", "sdxl", "ssd-1b"]
    steps = 20
    guidance = 7.5
    conditioning = 0.7
    seed = 42
    
    results = []
    runner = ModelComparisonRunner(sketch_path=sketch_path, output_dir=output_dir)
    
    for tone, kansei in test_cases:
        print(f"\n[{tone}] + {kansei}")
        
        # 1. Build Gemini instruction
        instruction = build_gemini_instruction([tone], kansei, sketch_path=sketch_path)
        
        # 2. Get Gemini LLM prompt
        gemini_prompt = call_gemini_for_prompt(instruction, sketch_path=sketch_path)
        
        # 3. Refine for image model
        llm_prompt = refine_for_image_model(gemini_prompt)
        print(f"   Generated prompt: {llm_prompt[:80]}...")
        
        for model in models:
            result = runner.run_single_generation(
                model_name=model,
                prompt=llm_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=conditioning,
                trial=1,
                seed=seed,
            )
            
            if result:
                result["tone"] = tone
                result["kansei"] = ", ".join(kansei)
                result["llm_prompt"] = llm_prompt
                results.append(result)
    
    # Save results
    import csv
    csv_path = output_path / "llm_benchmark_results.csv"
    
    fieldnames = ["tone", "kansei", "model", "llm_prompt", "clip_score", "lpips_score", 
                  "sketch_fidelity_iou", "total_time_seconds", "peak_memory_gb", "status", "error_message"]
    
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ LLM benchmark complete! Results saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for fashion design generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick single test
  python benchmark.py --sketch sketch.png --prompt "elegant dress"
  
  # Full model comparison (default: all models, steps 10/20/50, 2 trials)
  python benchmark.py --sketch sketch.png --full-comparison
  
  # LLM-based benchmark (Gemini generates prompts)
  python benchmark.py --sketch sketch.png --llm-benchmark
  
  # Custom configuration
  python benchmark.py --sketch sketch.png --models sd-v1.5 sdxl --steps 20 50 --trials 3
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--sketch",
        type=str,
        required=True,
        help="Path to the sketch image"
    )
    
    # Test modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--single",
        action="store_true",
        help="Run single quick test (default if no mode specified)"
    )
    mode_group.add_argument(
        "--full-comparison",
        action="store_true",
        help="Run comprehensive multi-model comparison"
    )
    mode_group.add_argument(
        "--llm-benchmark",
        action="store_true",
        help="Run benchmark with LLM-generated prompts (Gemini + tones/Kansei)"
    )
    
    # Configuration
    parser.add_argument(
        "--prompt",
        type=str,
        default="A fashionable garment with vibrant colors, detailed seams, and professional tailoring. Realistic fabric textures. Solid neutral gray background.",
        help="Text prompt for generation (not used in --llm-benchmark mode)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sd-v1.5", "sdxl", "ssd-1b"],
        choices=["sd-v1.5", "sdxl", "ssd-1b"],
        help="Models to compare (default: all three)"
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
        help="ControlNet conditioning scales (default: 0.7 1.0)"
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
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    
    args = parser.parse_args()
    
    # Validate sketch path
    sketch_path = Path(args.sketch)
    if not sketch_path.exists():
        print(f"[ERROR] Sketch not found: {sketch_path}")
        sys.exit(1)
    
    # Determine mode
    if args.llm_benchmark:
        run_llm_benchmark(str(sketch_path), args.output)
    elif args.full_comparison:
        run_full_comparison(
            sketch_path=str(sketch_path),
            prompt=args.prompt,
            models=args.models,
            steps=args.steps,
            guidance=args.guidance,
            conditioning=args.conditioning,
            trials=args.trials,
            output_dir=args.output
        )
    else:
        # Default: single test
        run_single_test(str(sketch_path), args.prompt, args.output)


if __name__ == "__main__":
    main()
