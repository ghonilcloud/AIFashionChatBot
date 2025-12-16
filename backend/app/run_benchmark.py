"""
Benchmark runner script.

Usage:
    python run_benchmark.py --sketch path/to/sketch.png --prompt "your prompt text"
    python run_benchmark.py --test-all  # Run full suite
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarking import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks on fashion design generation"
    )
    
    parser.add_argument(
        '--sketch',
        type=str,
        help='Path to sketch image for benchmarking',
        required=False
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        help='Text prompt for generation',
        required=False,
        default="A fashionable dress with elegant styling and rich colors, solid neutral gray background"
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save benchmark results',
        default='benchmark_results'
    )
    
    parser.add_argument(
        '--single',
        action='store_true',
        help='Run single benchmark test'
    )
    
    parser.add_argument(
        '--comparative',
        action='store_true',
        help='Run full comparative benchmark suite'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=2,
        help='Number of trials per test (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = BenchmarkRunner(output_dir=args.output_dir)
    
    # Check if sketch exists
    if args.sketch:
        sketch_path = args.sketch
        if not Path(sketch_path).exists():
            print(f"[ERROR] Sketch not found: {sketch_path}")
            sys.exit(1)
    else:
        print("[WARNING] No sketch provided. Using test mode.")
        sketch_path = "test_sketch.png"
    
    if args.single:
        print("\n[BENCHMARK] Running single test...")
        result = runner.benchmark_single_generation(
            sketch_path=sketch_path,
            prompt=args.prompt,
            output_path=str(Path(args.output_dir) / "single_test.png"),
            test_name="single_test"
        )
        runner.results.append(result)
        runner.save_results_to_csv()
    
    elif args.comparative:
        print("\n[BENCHMARK] Running comparative benchmark suite...")
        runner.run_comparative_benchmark(
            sketch_path=sketch_path,
            prompt=args.prompt,
            num_trials=args.trials
        )
    
    else:
        print("[INFO] No test specified. Use --single or --comparative")
        print("\nExample:")
        print("  python run_benchmark.py --sketch my_sketch.png --comparative --trials 3")


if __name__ == "__main__":
    main()
