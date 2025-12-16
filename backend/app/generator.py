"""Build prompts using preset tones/Kansei words and Gemini."""
from typing import List
import base64
from pathlib import Path

from .tones import TONES, KANSEI_WORDS
from .llm_client import get_genai_client


def build_gemini_instruction(
    selected_tones: List[str], 
    selected_kansei: List[str],
    sketch_path: str | Path = None,
) -> str:
    """Construct an instruction for Gemini to expand into a detailed image prompt.

    The instruction asks Gemini to produce a concise but highly-detailed image prompt
    suited for fashion sketches with realistic rendering while preserving the uploaded sketch silhouette.
    
    If sketch_path is provided, include it so Gemini can analyze the actual sketch shape.
    """
    tone_clause = ", ".join(selected_tones) if selected_tones else ""
    kansei_clause = ", ".join(selected_kansei) if selected_kansei else ""

    instructions = (
        f"You are a prompt engineer for fashion design generation. "
        f"Your task is to PRESERVE the silhouette and shape of the provided sketch while enhancing it with vibrant styling.\n\n"
        f"Selected tones/style: {tone_clause}\n"
        f"Selected Kansei (sensory/feeling) keywords: {kansei_clause}\n\n"
        f"CRITICAL - READ CAREFULLY:\n"
        f"1. ANALYZE THE CLOTHING OUTLINE: Examine the uploaded sketch. Identify EXACT clothing features:\n"
        f"   - Neckline shape (round, V-neck, square, etc.)\n"
        f"   - Sleeve type and length (sleeveless, short, long, puffed, etc.)\n"
        f"   - Hemline and length (mini, knee, midi, maxi, etc.)\n"
        f"   - Overall silhouette (A-line, straight, fitted, etc.)\n"
        f"   - Any unique clothing features (straps, buckles, pockets, belts, etc.)\n"
        f"\n2. FOCUS ON CLOTHING ONLY - Do NOT describe human body features (face, hair, skin, body shape, etc.)\n"
        f"   Only describe the garment itself and how it fits.\n"
        f"\n3. OUTPUT FORMAT - Generate a SHORT, DENSE prompt (MUST be under 75 words):\n"
        f"   - Start with: '[PRESERVE OUTLINE] <garment type>'\n"
        f"   - Specify EXACT neckline, sleeves, hemline from sketch\n"
        f"   - Add rich color palette matching tones/Kansei\n"
        f"   - Add fabric and texture details\n"
        f"   - NO human features or body description\n"
        f"\n4. STRICT RULE: Never change the clothing outline. Only enhance with colors, textures, and garment details.\n"
        f"   Your prompt MUST keep the exact same shape as the uploaded sketch.\n"
    )

    return instructions


def call_gemini_for_prompt(instruction: str, sketch_path: str | Path = None, model: str = "gemini-2.5-flash") -> str:
    """Call Gemini API with optional image attachment."""
    client = get_genai_client()
    
    # If sketch provided, send it with the instruction so Gemini can analyze it
    if sketch_path and Path(sketch_path).exists():
        with open(sketch_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        # Determine mime type
        suffix = Path(sketch_path).suffix.lower()
        mime_type = "image/png" if suffix == ".png" else "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
        
        # Send with image
        response = client.models.generate_content(
            model=model,
            contents=[
                {"text": instruction},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_data,
                    }
                },
            ],
        )
    else:
        response = client.models.generate_content(model=model, contents=instruction)
    
    return getattr(response, 'text', str(response))


def refine_for_image_model(gemini_text: str) -> str:
    """Transform Gemini output into a CLIP-compatible prompt (max 77 tokens).
    
    CLIP tokenizer max: 77 tokens. Prompt is truncated to fit while preserving
    essential silhouette and style information.
    
    Appends rendering and quality instructions to produce vibrant, high-quality fashion designs
    with solid background for clear clothing visibility.
    """
    try:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Start with the Gemini text
        base_prompt = gemini_text.strip()
        
        # Add strict silhouette preservation directive
        silhouette_directive = "[PRESERVE EXACT SKETCH SILHOUETTE]"
        
        # Add background instruction to differentiate clothing
        background_instruction = "solid neutral gray background"
        
        # Combine prompt
        full_prompt = f"{silhouette_directive} {base_prompt} {background_instruction}"
        
        # Tokenize and check length
        tokens = tokenizer.tokenize(full_prompt)
        
        # If over 77 tokens, truncate Gemini text while keeping directives
        if len(tokens) > 77:
            # Keep silhouette directive and background (always at end)
            directive_tokens = len(tokenizer.tokenize(f"{silhouette_directive} {background_instruction}"))
            remaining_tokens = 77 - directive_tokens - 2
            
            # Truncate Gemini text to fit
            gemini_tokens = tokenizer.tokenize(base_prompt)
            if len(gemini_tokens) > remaining_tokens:
                gemini_tokens = gemini_tokens[:remaining_tokens]
                base_prompt = tokenizer.convert_tokens_to_string(gemini_tokens)
        
        final_prompt = f"{silhouette_directive} {base_prompt} {background_instruction}"
        
        # Verify final token count
        final_tokens = tokenizer.tokenize(final_prompt)
        print(f"[PROMPT] Token count: {len(final_tokens)}/77 (CLIP compatible)")
        
        return final_prompt
        
    except ImportError:
        # Fallback if CLIPTokenizer not available - simple word-based truncation
        print("[WARNING] CLIPTokenizer not available, using word-based truncation")
        words = gemini_text.strip().split()
        if len(words) > 50:  # Conservative estimate for background instruction
            words = words[:50]
        
        final_prompt = "[PRESERVE EXACT SKETCH SILHOUETTE] " + " ".join(words) + " solid neutral gray background"
        return final_prompt
