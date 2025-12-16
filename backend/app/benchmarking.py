"""
Benchmarking module for fashion design generation.

Measures:
- Inference latency (total time, per-step breakdown)
- GPU/CPU memory usage
- CLIP score (text-image alignment)
- FID score (image quality)
- Generates CSV reports for research analysis
"""

import time
import torch
import psutil
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np


class PerformanceMetrics:
    """Tracks timing and memory metrics for generation."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.peak_memory = 0
    
    def start_timer(self):
        """Start overall timer."""
        self.start_time = time.time()
    
    def record_step(self, step_name: str):
        """Record time for a specific step."""
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        self.metrics[step_name] = elapsed
        print(f"[BENCH] {step_name}: {elapsed:.3f}s")
    
    def get_total_time(self) -> float:
        """Get total elapsed time."""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def record_memory(self):
        """Record peak memory usage."""
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB
            self.peak_memory = peak_memory
            print(f"[BENCH] Peak GPU Memory: {peak_memory:.2f} GB")
        else:
            process = psutil.Process()
            memory_info = process.memory_info()
            peak_memory_gb = memory_info.rss / 1e9
            self.peak_memory = peak_memory_gb
            print(f"[BENCH] Peak CPU Memory: {peak_memory_gb:.2f} GB")
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'total_time': self.get_total_time(),
            'peak_memory': self.peak_memory,
            'steps': self.metrics
        }


def compute_clip_score(image_path: str, prompt: str) -> float:
    """
    Compute CLIP score (text-image alignment).
    
    Range: 0-1 (higher is better)
    Measures how well the generated image matches the text prompt.
    """
    try:
        import open_clip
        from PIL import Image
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai',
            device=device
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        # Load and preprocess image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = tokenizer([prompt]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (image_features @ text_features.T).item()
        
        return similarity
    
    except ImportError:
        print("[WARNING] CLIP not installed. Install with: pip install open-clip-torch")
        return None
    except Exception as e:
        print(f"[ERROR] Computing CLIP score: {e}")
        return None


def compute_fid_score(real_image_path: str, generated_image_path: str) -> float:
    """
    Compute FID score (Fréchet Inception Distance).
    
    Range: 0-infinity (lower is better)
    Measures image quality and similarity to real images.
    
    Simplified version - compares pixel statistics.
    """
    try:
        from torchvision import models, transforms
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load pretrained InceptionV3
        inception = models.inception_v3(pretrained=True, transform_input=True)
        inception = inception.to(device)
        inception.eval()
        
        # Preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Load images
        real_img = Image.open(real_image_path).convert('RGB')
        gen_img = Image.open(generated_image_path).convert('RGB')
        
        real_tensor = preprocess(real_img).unsqueeze(0).to(device)
        gen_tensor = preprocess(gen_img).unsqueeze(0).to(device)
        
        # Get features
        with torch.no_grad():
            real_features = inception(real_tensor)[0]
            gen_features = inception(gen_tensor)[0]
        
        # Compute FID (simplified: L2 distance)
        fid = torch.norm(real_features - gen_features).item()
        
        return fid
    
    except ImportError:
        print("[WARNING] Computing FID requires torchvision")
        return None
    except Exception as e:
        print(f"[ERROR] Computing FID score: {e}")
        return None


class BenchmarkRunner:
    """Runs comprehensive benchmarks on image generation."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def benchmark_single_generation(
        self,
        sketch_path: str,
        prompt: str,
        output_path: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        model_name: str = "stable-diffusion-v1.5",
        test_name: str = "test_1",
    ) -> Dict:
        """
        Run single generation and collect all metrics.
        
        Returns:
            Dictionary with all metrics for CSV logging
        """
        metrics = PerformanceMetrics()
        metrics.start_timer()
        
        print(f"\n[BENCHMARK] Starting: {test_name}")
        print(f"[BENCHMARK] Model: {model_name}")
        print(f"[BENCHMARK] Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        
        # Import here to avoid circular imports
        try:
            from image_generator import generate_fashion_design
        except ImportError:
            from .image_generator import generate_fashion_design
        
        try:
            # Time the generation
            metrics.record_step("import_modules")
            
            gen_start = time.time()
            generated_image_path = generate_fashion_design(
                sketch_path=sketch_path,
                prompt=prompt,
                output_path=output_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            )
            gen_time = time.time() - gen_start
            metrics.metrics['generation'] = gen_time
            
            # Record memory
            metrics.record_memory()
            
            # Compute quality metrics
            print(f"[BENCHMARK] Computing quality metrics...")
            clip_score = compute_clip_score(generated_image_path, prompt)
            # FID requires original image, skip for now
            fid_score = None
            
            # Compile results
            result = {
                'timestamp': datetime.now().isoformat(),
                'test_name': test_name,
                'model': model_name,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'controlnet_conditioning_scale': controlnet_conditioning_scale,
                'total_time_seconds': metrics.get_total_time(),
                'generation_time_seconds': gen_time,
                'peak_memory_gb': metrics.peak_memory,
                'clip_score': clip_score if clip_score else 'N/A',
                'fid_score': fid_score if fid_score else 'N/A',
                'prompt_length': len(prompt.split()),
                'device': 'CUDA' if torch.cuda.is_available() else 'CPU',
                'status': 'success'
            }
            
            print(f"[BENCHMARK] RESULTS:")
            print(f"  Total time: {result['total_time_seconds']:.2f}s")
            print(f"  Generation time: {gen_time:.2f}s")
            print(f"  Peak memory: {metrics.peak_memory:.2f} GB")
            if clip_score:
                print(f"  CLIP score: {clip_score:.4f}")
            
            return result
        
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'test_name': test_name,
                'model': model_name,
                'status': 'failed',
                'error': str(e),
                'device': 'CUDA' if torch.cuda.is_available() else 'CPU',
            }
    
    def run_comparative_benchmark(
        self,
        sketch_path: str,
        prompt: str,
        num_trials: int = 3,
    ):
        """
        Run comparative benchmarks across different settings.
        
        Tests:
        - Different inference steps (10, 20, 50)
        - Different guidance scales (5.0, 7.5, 10.0)
        - Different ControlNet conditioning (0.5, 0.8, 1.0)
        """
        print("\n" + "="*70)
        print("COMPARATIVE BENCHMARK SUITE")
        print("="*70)
        
        inference_steps_list = [10, 20, 50]
        guidance_scales_list = [5.0, 7.5, 10.0]
        conditioning_scales_list = [0.5, 0.8, 1.0]
        
        test_counter = 0
        
        # Test different inference steps
        print("\n[SUITE] Testing different inference steps...")
        for steps in inference_steps_list:
            for trial in range(num_trials):
                test_counter += 1
                output_path = self.output_dir / f"bench_steps_{steps}_trial_{trial}.png"
                
                result = self.benchmark_single_generation(
                    sketch_path=sketch_path,
                    prompt=prompt,
                    output_path=str(output_path),
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=1.0,
                    test_name=f"steps_{steps}_trial_{trial}",
                )
                self.results.append(result)
        
        # Test different guidance scales
        print("\n[SUITE] Testing different guidance scales...")
        for guidance in guidance_scales_list:
            for trial in range(num_trials):
                test_counter += 1
                output_path = self.output_dir / f"bench_guidance_{guidance}_trial_{trial}.png"
                
                result = self.benchmark_single_generation(
                    sketch_path=sketch_path,
                    prompt=prompt,
                    output_path=str(output_path),
                    num_inference_steps=20,
                    guidance_scale=guidance,
                    controlnet_conditioning_scale=1.0,
                    test_name=f"guidance_{guidance}_trial_{trial}",
                )
                self.results.append(result)
        
        # Test different ControlNet conditioning
        print("\n[SUITE] Testing different ControlNet conditioning scales...")
        for conditioning in conditioning_scales_list:
            for trial in range(num_trials):
                test_counter += 1
                output_path = self.output_dir / f"bench_conditioning_{conditioning}_trial_{trial}.png"
                
                result = self.benchmark_single_generation(
                    sketch_path=sketch_path,
                    prompt=prompt,
                    output_path=str(output_path),
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=conditioning,
                    test_name=f"conditioning_{conditioning}_trial_{trial}",
                )
                self.results.append(result)
        
        print(f"\n[SUITE] Completed {test_counter} tests")
        self.save_results_to_csv()
    
    def save_results_to_csv(self):
        """Save all results to CSV file."""
        if not self.results:
            print("[WARNING] No results to save")
            return
        
        csv_path = self.output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Get all possible keys
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())
        
        fieldnames = sorted(list(all_keys))
        
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            
            print(f"\n[RESULTS] Saved to: {csv_path}")
            print(f"[RESULTS] Total tests: {len(self.results)}")
            
            # Print summary statistics
            self.print_summary_statistics()
        
        except Exception as e:
            print(f"[ERROR] Saving CSV: {e}")
    
    def print_summary_statistics(self):
        """Print summary statistics from results."""
        if not self.results:
            return
        
        # Filter successful results
        successful = [r for r in self.results if r.get('status') == 'success']
        
        if not successful:
            print("[WARNING] No successful results to summarize")
            return
        
        times = [r['total_time_seconds'] for r in successful]
        memories = [r['peak_memory_gb'] for r in successful]
        clip_scores = [float(r['clip_score']) for r in successful if isinstance(r['clip_score'], (int, float))]
        
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(f"Total successful runs: {len(successful)}")
        print(f"\nInference Time (seconds):")
        print(f"  Min: {min(times):.2f}s")
        print(f"  Max: {max(times):.2f}s")
        print(f"  Mean: {np.mean(times):.2f}s")
        print(f"  Std Dev: {np.std(times):.2f}s")
        
        print(f"\nMemory Usage (GB):")
        print(f"  Min: {min(memories):.2f} GB")
        print(f"  Max: {max(memories):.2f} GB")
        print(f"  Mean: {np.mean(memories):.2f} GB")
        
        if clip_scores:
            print(f"\nCLIP Score (text-image alignment):")
            print(f"  Min: {min(clip_scores):.4f}")
            print(f"  Max: {max(clip_scores):.4f}")
            print(f"  Mean: {np.mean(clip_scores):.4f}")
        
        print("="*70 + "\n")


# Quick test runner
if __name__ == "__main__":
    print("Benchmarking module loaded. Use in your code:")
    print("from benchmarking import BenchmarkRunner")
    print("runner = BenchmarkRunner()")
    print("runner.run_comparative_benchmark(sketch_path, prompt)")
