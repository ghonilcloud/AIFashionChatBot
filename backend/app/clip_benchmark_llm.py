"""
Benchmark script: LLM prompt interpretation → GenAI generation → CLIP scoring
Generates images for 5 (Tone, Kansei) pairs using the same sketch, all 3 models, and logs CLIP scores.
"""
from pathlib import Path
from .generator import build_gemini_instruction, call_gemini_for_prompt, refine_for_image_model
from .model_comparison import ModelComparisonRunner
import csv

# --- Config ---
sketch_path = r"C:\Users\eraph\OneDrive\Desktop\5th Semester\AI\AIFashionChatBot\technical drawing - fashion\Top-32.jpg"
output_dir = Path("comparison_results/clip_benchmark_llm")
output_dir.mkdir(parents=True, exist_ok=True)

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

for tone, kansei in test_cases:
    # 1. Build Gemini instruction
    instruction = build_gemini_instruction([tone], kansei, sketch_path=sketch_path)
    # 2. Get Gemini LLM prompt
    gemini_prompt = call_gemini_for_prompt(instruction, sketch_path=sketch_path)
    # 3. Refine for image model
    llm_prompt = refine_for_image_model(gemini_prompt)
    for model in models:
        # 4. Run generation and scoring
        runner = ModelComparisonRunner(sketch_path=sketch_path, output_dir=output_dir)
        # Unique output filename for each (tone, kansei, model)
        safe_tone = tone.replace(' ', '_')
        safe_kansei = '_'.join([k.replace(' ', '_') for k in kansei])
        output_filename = f"{model}_{safe_tone}_{safe_kansei}.png"
        output_path = output_dir / output_filename
        error_message = None
        try:
            result = runner.run_single_generation(
                model_name=model,
                prompt=llm_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=conditioning,
                trial=1,
                seed=seed,
            )
            # If generation failed (e.g., due to safety filter), log error
            if result is None or result.get("status") != "success":
                error_message = result["error_message"] if result else "Unknown error"
                clip_score = None
            else:
                clip_score = result.get("clip_score")
        except Exception as e:
            error_message = str(e)
            clip_score = None
        # 5. Log results
        results.append({
            "tone": tone,
            "kansei": ", ".join(kansei),
            "model": model,
            "llm_prompt": llm_prompt,
            "clip_score": clip_score,
            "image_path": str(output_path),
            "error_message": error_message,
        })

# Write results to CSV

# Write results to CSV (with error_message column)
with open(output_dir / "clip_benchmark_llm_results.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["tone", "kansei", "model", "llm_prompt", "clip_score", "image_path", "error_message"])
    writer.writeheader()
    writer.writerows(results)

print("[DONE] Benchmark complete. Results saved to:", output_dir / "clip_benchmark_llm_results.csv")
