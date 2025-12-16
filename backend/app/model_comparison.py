"""
Comprehensive model comparison for fashion design generation.

Compares:
- Stable Diffusion v1.5
- Stable Diffusion XL
- Segmind SSD-1B

Measures:
- Inference time (per step and total)
- Peak memory usage
- CLIP score (text-image alignment)
- Sketch fidelity (edge overlap with original sketch)

Outputs results to CSV for analysis.
"""

import time
import csv
import psutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import torch
import numpy as np
from PIL import Image
import cv2

try:
    import open_clip
except ImportError:
    open_clip = None

try:
    import lpips
except ImportError:
    lpips = None


class SketchFidelityMetric:
    """Compute how well the generated image preserves the original sketch outline."""
    
    @staticmethod
    def compute_edge_overlap(sketch_path: str | Path, generated_path: str | Path) -> float:
        """
        Compute edge overlap between original sketch and generated image.
        
        Returns score from 0-1 where:
        - 1.0 = perfect outline match
        - 0.0 = no edge overlap
        
        Uses Canny edge detection on both images and computes intersection/union.
        """
        try:
            # Load images
            sketch = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)
            generated = cv2.imread(str(generated_path), cv2.IMREAD_GRAYSCALE)
            
            if sketch is None or generated is None:
                return 0.0
            
            # Resize to same dimensions
            h, w = sketch.shape
            generated = cv2.resize(generated, (w, h))
            
            # Apply Canny edge detection
            sketch_edges = cv2.Canny(sketch, 50, 150)
            generated_edges = cv2.Canny(generated, 50, 150)
            
            # Normalize to 0-1
            sketch_edges = sketch_edges.astype(np.float32) / 255.0
            generated_edges = generated_edges.astype(np.float32) / 255.0
            
            # Compute intersection and union
            intersection = np.sum(sketch_edges * generated_edges)
            union = np.sum(np.maximum(sketch_edges, generated_edges))
            
            if union == 0:
                return 0.0
            
            # Jaccard similarity (IoU)
            iou = intersection / union
            return float(iou)
        
        except Exception as e:
            print(f"[WARNING] Sketch fidelity computation failed: {e}")
            return 0.0
    
    @staticmethod
    def compute_edge_density_match(sketch_path: str | Path, generated_path: str | Path) -> float:
        """
        Compare edge density (number of edges) between sketch and generated image.
        
        Returns score from 0-1 where 1.0 means equal edge density.
        """
        try:
            sketch = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)
            generated = cv2.imread(str(generated_path), cv2.IMREAD_GRAYSCALE)
            
            if sketch is None or generated is None:
                return 0.0
            
            generated = cv2.resize(generated, (sketch.shape[1], sketch.shape[0]))
            
            sketch_edges = cv2.Canny(sketch, 50, 150)
            generated_edges = cv2.Canny(generated, 50, 150)
            
            sketch_edge_count = np.sum(sketch_edges > 0)
            generated_edge_count = np.sum(generated_edges > 0)
            
            if sketch_edge_count == 0:
                return 0.0
            
            # Ratio of edge counts (clamped to 0-1)
            ratio = min(generated_edge_count / sketch_edge_count, 1.0)
            return float(ratio)
        
        except Exception as e:
            print(f"[WARNING] Edge density computation failed: {e}")
            return 0.0


class ModelComparisonRunner:
    """Run comprehensive model comparison benchmarks."""
    
    def __init__(self, sketch_path: str | Path, output_dir: str | Path = "comparison_results"):
        self.sketch_path = Path(sketch_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.process = psutil.Process()
        
        # Initialize LPIPS model once for reuse
        self.lpips_model = None
        if lpips is not None:
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_model.eval()
            except Exception as e:
                print(f"[WARNING] Failed to initialize LPIPS model: {e}")
        
    def measure_memory_start(self) -> float:
        """Get initial memory usage."""
        torch.cuda.reset_peak_memory_stats() if self.device == "cuda" else None
        return self.process.memory_info().rss / (1024 ** 3)  # GB
    
    def measure_memory_end(self) -> float:
        """Get peak memory usage since start."""
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        else:
            return self.process.memory_info().rss / (1024 ** 3)  # GB
    
    def compute_clip_score(self, prompt: str, generated_path: str | Path) -> float:
        """Compute CLIP score between prompt and generated image."""
        if open_clip is None:
            print("[WARNING] open_clip not available, skipping CLIP score")
            return 0.0
        
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            model = model.to(self.device)
            model.eval()
            
            # Encode text
            text_tokens = open_clip.tokenize([prompt])
            with torch.no_grad():
                text_features = model.encode_text(text_tokens.to(self.device))
            
            # Encode image
            image = preprocess(Image.open(generated_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            score = (text_features @ image_features.T).item()
            return max(0.0, min(1.0, score))  # Clamp to 0-1
        
        except Exception as e:
            print(f"[WARNING] CLIP score computation failed: {e}")
            return 0.0
    
    def compute_lpips_score(self, sketch_path: str | Path, generated_path: str | Path) -> float:
        """Compute LPIPS perceptual similarity between sketch and generated image.
        
        Returns score from 0-1 where:
        - 0.0 = perceptually identical
        - 1.0 = perceptually very different
        
        Lower is better for sketch preservation.
        """
        if self.lpips_model is None:
            print("[WARNING] LPIPS model not available")
            return 0.0
        
        try:
            from torchvision import transforms
            
            # LPIPS expects [-1, 1] normalized RGB images
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # Load and preprocess images
            sketch_img = Image.open(sketch_path).convert('RGB')
            gen_img = Image.open(generated_path).convert('RGB')
            
            sketch_tensor = transform(sketch_img).unsqueeze(0).to(self.device)
            gen_tensor = transform(gen_img).unsqueeze(0).to(self.device)
            
            # Compute LPIPS distance
            with torch.no_grad():
                distance = self.lpips_model(sketch_tensor, gen_tensor)
            
            return float(distance.item())
        
        except Exception as e:
            print(f"[WARNING] LPIPS computation failed: {e}")
            return 0.0
    
    def run_single_generation(
        self,
        model_name: str,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        controlnet_conditioning_scale: float,
        trial: int,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a single generation and collect metrics.
        
        Returns dict with all measurements.
        """
        
        # Import here to avoid circular imports
        try:
            if model_name == "sd-v1.5":
                from .image_generator import load_pipeline, generate_fashion_design, preprocess_sketch
            elif model_name == "sdxl":
                from .image_generator import load_sdxl_pipeline as load_pipeline
                from .image_generator import generate_fashion_design, preprocess_sketch
            elif model_name == "ssd-1b":
                from .image_generator import load_segmind_pipeline as load_pipeline
                from .image_generator import generate_fashion_design, preprocess_sketch
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except ImportError as e:
            print(f"[ERROR] Import failed for {model_name}: {e}")
            return None
        
        # Generate output path
        output_filename = f"{model_name}_steps{num_inference_steps}_guidance{guidance_scale}_conditioning{controlnet_conditioning_scale}_trial{trial}.png"
        output_path = self.output_dir / output_filename
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "trial": trial,
            "seed": seed or 42,
            "device": self.device,
            "status": "failed",
            "error_message": "",
            "total_time_seconds": 0.0,
            "peak_memory_gb": 0.0,
            "clip_score": 0.0,
            "sketch_fidelity_iou": 0.0,
            "sketch_fidelity_edge_density": 0.0,
            "lpips_score": 0.0,
        }
        
        try:
            # Timing
            start_time = time.perf_counter()
            mem_start = self.measure_memory_start()
            
            # Load pipeline
            pipe = load_pipeline()

            # Generate image with the correct pipeline
            generate_fashion_design(
                sketch_path=self.sketch_path,
                prompt=prompt,
                output_path=output_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                seed=seed or 42,
                pipe=pipe,
            )
            
            end_time = time.perf_counter()
            mem_end = self.measure_memory_end()
            
            # Compute metrics
            clip_score = self.compute_clip_score(prompt, output_path)
            fidelity_iou = SketchFidelityMetric.compute_edge_overlap(self.sketch_path, output_path)
            fidelity_density = SketchFidelityMetric.compute_edge_density_match(self.sketch_path, output_path)
            lpips_score = self.compute_lpips_score(self.sketch_path, output_path)
            
            result.update({
                "status": "success",
                "total_time_seconds": end_time - start_time,
                "peak_memory_gb": mem_end,
                "clip_score": clip_score,
                "sketch_fidelity_iou": fidelity_iou,
                "sketch_fidelity_edge_density": fidelity_density,
                "lpips_score": lpips_score,
            })
            
            print(f"✓ {model_name} (steps={num_inference_steps}, guidance={guidance_scale}) - {end_time - start_time:.2f}s, {clip_score:.4f} CLIP, {fidelity_iou:.4f} fidelity, {lpips_score:.4f} LPIPS")
        
        except Exception as e:
            result["status"] = "failed"
            result["error_message"] = str(e)
            print(f"✗ {model_name} failed: {e}")
            traceback.print_exc()
        
        return result
    
    def run_comparative_benchmark(
        self,
        prompt: str,
        models: list[str] = None,
        num_inference_steps_list: list[int] = None,
        guidance_scales: list[float] = None,
        controlnet_conditioning_scales: list[float] = None,
        trials: int = 2,
    ):
        """
        Run comprehensive comparative benchmark across models and parameters.
        
        Default configuration tests 3 models × 3 steps × 2 guidance scales × 2 conditioning = 36 generations
        """
        
        if models is None:
            models = ["sd-v1.5", "sdxl", "ssd-1b"]
        
        if num_inference_steps_list is None:
            num_inference_steps_list = [10, 20, 50]
        
        if guidance_scales is None:
            guidance_scales = [7.5, 10.0]
        
        if controlnet_conditioning_scales is None:
            controlnet_conditioning_scales = [0.7, 1.0]
        
        total_tests = len(models) * len(num_inference_steps_list) * len(guidance_scales) * len(controlnet_conditioning_scales) * trials
        
        print(f"\n[COMPARISON] Starting multi-model benchmark")
        print(f"[COMPARISON] Models: {models}")
        print(f"[COMPARISON] Steps: {num_inference_steps_list}")
        print(f"[COMPARISON] Guidance: {guidance_scales}")
        print(f"[COMPARISON] ControlNet conditioning: {controlnet_conditioning_scales}")
        print(f"[COMPARISON] Trials per config: {trials}")
        print(f"[COMPARISON] Total generations: {total_tests}\n")
        
        test_count = 0
        
        for model in models:
            for steps in num_inference_steps_list:
                for guidance in guidance_scales:
                    for conditioning in controlnet_conditioning_scales:
                        for trial in range(1, trials + 1):
                            test_count += 1
                            print(f"[{test_count}/{total_tests}] {model} - steps={steps}, guidance={guidance}, conditioning={conditioning}, trial={trial}")
                            
                            result = self.run_single_generation(
                                model_name=model,
                                prompt=prompt,
                                num_inference_steps=steps,
                                guidance_scale=guidance,
                                controlnet_conditioning_scale=conditioning,
                                trial=trial,
                            )
                            
                            if result:
                                self.results.append(result)
        
        self.save_results_to_csv()
        self.print_summary_statistics()
    
    def save_results_to_csv(self):
        """Save all results to CSV file."""
        if not self.results:
            print("[WARNING] No results to save")
            return
        
        csv_path = self.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        fieldnames = [
            "timestamp", "model", "num_inference_steps", "guidance_scale", "controlnet_conditioning_scale",
            "trial", "seed", "device", "status", "total_time_seconds", "peak_memory_gb",
            "clip_score", "sketch_fidelity_iou", "sketch_fidelity_edge_density", "lpips_score", "error_message"
        ]
        
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            
            print(f"\n[RESULTS] Saved {len(self.results)} results to: {csv_path}")
            return csv_path
        
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
    
    def print_summary_statistics(self):
        """Print summary statistics across all results."""
        if not self.results:
            print("[WARNING] No results to summarize")
            return
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        # Group by model
        by_model = {}
        for r in self.results:
            model = r["model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(r)
        
        for model, results in by_model.items():
            successful = [r for r in results if r["status"] == "success"]
            
            if not successful:
                print(f"\n{model}: 0/{len(results)} successful")
                continue
            
            times = [r["total_time_seconds"] for r in successful]
            memories = [r["peak_memory_gb"] for r in successful]
            clip_scores = [r["clip_score"] for r in successful]
            fidelities = [r["sketch_fidelity_iou"] for r in successful]
            lpips_scores = [r["lpips_score"] for r in successful]
            
            print(f"\n{model.upper()} ({len(successful)}/{len(results)} successful)")
            print(f"  Inference Time:")
            print(f"    Min: {min(times):.2f}s, Max: {max(times):.2f}s, Mean: {np.mean(times):.2f}s, StdDev: {np.std(times):.2f}s")
            print(f"  Memory (GB):")
            print(f"    Min: {min(memories):.2f}, Max: {max(memories):.2f}, Mean: {np.mean(memories):.2f}")
            print(f"  CLIP Score:")
            print(f"    Min: {min(clip_scores):.4f}, Max: {max(clip_scores):.4f}, Mean: {np.mean(clip_scores):.4f}")
            print(f"  Sketch Fidelity (IoU):")
            print(f"    Min: {min(fidelities):.4f}, Max: {max(fidelities):.4f}, Mean: {np.mean(fidelities):.4f}")
            print(f"  LPIPS Score (lower=better):")
            print(f"    Min: {min(lpips_scores):.4f}, Max: {max(lpips_scores):.4f}, Mean: {np.mean(lpips_scores):.4f}")
        
        print("\n" + "="*80)
