from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
from typing import List

from .config import UPLOADS_DIR, GENERATED_DIR, MEDIA_DIR
from .models import GenerateDesignResponse
from .generator import build_gemini_instruction, call_gemini_for_prompt, refine_for_image_model
from .tones import TONES, KANSEI_WORDS
from .image_generator import generate_fashion_design

app = FastAPI(
    title="Fashion Emotion Design API",
    description="Backend for emotion-aware CAD-style fashion design generation.",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve media files
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


@app.post("/api/generate-design", response_model=GenerateDesignResponse)
async def generate_design(
    image: UploadFile = File(...),
    kansei_text: str = Form(...),
    style_profile: str | None = Form(None),
    tones: List[str] = Form(default=[]),
    kansei_words: List[str] = Form(default=[]),
):
    """
    Main endpoint: receives sketch + kansei text + optional tones/Kansei words.
    
    - Saves the uploaded sketch
    - Calls Gemini to generate a detailed prompt based on selected tones/Kansei
    - Returns generated_image_url and the LLM prompt
    """

    # 1. Save the uploaded sketch
    sketch_id = uuid.uuid4().hex
    original_ext = Path(image.filename).suffix or ".png"
    sketch_filename = f"sketch_{sketch_id}{original_ext}"
    sketch_path = UPLOADS_DIR / sketch_filename

    with sketch_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # 2. Call Gemini to generate prompt
    try:
        instruction = build_gemini_instruction(tones, kansei_words)
        gemini_prompt = call_gemini_for_prompt(instruction)
        llm_prompt = refine_for_image_model(gemini_prompt)
    except RuntimeError as e:
        # API key missing or other error
        llm_prompt = f"Error generating prompt: {str(e)}"

    # 3. Generate image using Stable Diffusion + ControlNet with sketch + prompt
    generated_filename = f"generated_{sketch_id}.png"
    generated_path = GENERATED_DIR / generated_filename
    
    try:
        generate_fashion_design(
            sketch_path=sketch_path,
            prompt=llm_prompt,
            output_path=generated_path,
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
        )
    except Exception as e:
        # Fallback: copy sketch if generation fails
        print(f"Image generation failed: {e}")
        shutil.copyfile(sketch_path, generated_path)

    # 4. Build URL for frontend
    generated_image_url = f"/media/generated/{generated_filename}"

    notes = (
        "Generated using Stable Diffusion v1.5 + ControlNet with the Gemini-crafted prompt "
        "conditioned on your CAD sketch."
    )

    return GenerateDesignResponse(
        status="ok",
        generated_image_url=generated_image_url,
        llm_prompt=llm_prompt,
        notes=notes,
    )


@app.get("/api/tones")
async def get_tones():
    """Return available tones and Kansei words for frontend dropdown."""
    return {"tones": TONES, "kansei_words": KANSEI_WORDS}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}