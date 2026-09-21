import json
import re
import math
import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# ==========================================================
# HARDENED SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = """You are a strict, precise entity extraction system for bathroom design requirements.
Convert the user's input into a strictly valid JSON object. Do not forget commas between key-value pairs or array items.

Allowed Product Categories:
["sink", "vanity", "mirror", "toilet", "shower", "faucet", "bathtub", "cabinet"]

Rules:
1. "budget": Extract exact dollar budget as a number, or null if unmentioned.
2. "room_dimensions": Extract as [length, width] in feet. If user provides total square footage (e.g., "100 sq ft"), calculate the side lengths as [10, 10]. Return null if unmentioned.
3. "required_products": 
   - Extract ONLY items explicitly mentioned by the user that map to the Allowed Product Categories.
   - NEVER invent or assume products (do NOT add "toilet" or "shower" unless requested).
   - Strip all descriptive words (e.g. "black", "gold", "cute") from product names.
4. "preference_sequence": 
   - Extract all colors, materials, design styles, and modifiers (e.g. "black", "gold inlay", "minimalist", "modern") in exact order.
5. Return ONLY a single raw JSON object matching this schema:
{
    "budget": number or null,
    "room_dimensions": [number, number] or null,
    "required_products": [],
    "preference_sequence": []
}

### EXAMPLES:

User: "i need a black sink with gold inlay and a vanity desk with a mirror attached in my 100 sq ft room"
Output:
{
    "budget": null,
    "room_dimensions": [10, 10],
    "required_products": ["sink", "vanity", "mirror"],
    "preference_sequence": ["black", "gold inlay"]
}

User: "add a shower in minimalist style for a 10 by 12 room"
Output:
{
    "budget": null,
    "room_dimensions": [10, 12],
    "required_products": ["shower"],
    "preference_sequence": ["minimalist"]
}

User: "add a shower to it"
Output:
{
    "budget": null,
    "room_dimensions": null,
    "required_products": ["shower"],
    "preference_sequence": []
}
"""


# ==========================================================
# GGUF JSON EXTRACTOR
# ==========================================================

class SmolLMJSONExtractor:

    def __init__(self, model_path: str = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"):
        # Auto-download the GGUF model from Hugging Face if not present locally
        if not os.path.exists(model_path):
            print(f"[INFO] Model file '{model_path}' not found locally.")
            print("[INFO] Downloading Qwen2.5-1.5B GGUF file from Hugging Face...")
            
            # Extract target folder and filename from model_path
            target_dir = os.path.dirname(os.path.abspath(model_path))
            filename = os.path.basename(model_path)
            
            model_path = hf_hub_download(
                repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
                filename=filename,
                local_dir=target_dir if target_dir else ".",
                local_dir_use_symlinks=False
            )
            print(f"[SUCCESS] Download completed: {model_path}")
        else:
            print(f"[INFO] Found existing model file at: {model_path}")

        print(f"[INFO] Loading quantized GGUF model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=1024,      # Context window size
            n_threads=4,     # Adjust to your CPU cores if needed
            verbose=False
        )
        print("[INFO] GGUF Qwen model loaded successfully.")

    def _normalize_dimensions(self, parsed_data: dict) -> dict:
        """Fixes area vs side-length parsing errors (e.g. [100, 100] -> [10, 10])"""
        dims = parsed_data.get("room_dimensions")
        
        # Safely validate that dims is a 2-element list containing numbers
        if dims and isinstance(dims, list) and len(dims) == 2:
            length, width = dims[0], dims[1]
            
            # Ensure both values are valid numbers (not None or strings) before comparing
            if isinstance(length, (int, float)) and isinstance(width, (int, float)):
                if length == width and length >= 25:
                    side = round(math.sqrt(length), 1)
                    parsed_data["room_dimensions"] = [side, side]
            else:
                parsed_data["room_dimensions"] = None
        else:
            parsed_data["room_dimensions"] = None
            
        return parsed_data

    def generate_json(
        self,
        user_input: str,
        system_prompt: str = SYSTEM_PROMPT
    ) -> dict:
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

        output = self.llm(
            prompt,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
            stop=["<|im_end|>"]
        )

        response_str = output["choices"][0]["text"].strip()

        if "```json" in response_str:
            response_str = response_str.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in response_str:
            response_str = response_str.split("```", 1)[1].split("```", 1)[0].strip()

        safe_fallback = {
            "budget": None,
            "room_dimensions": None,
            "required_products": [],
            "preference_sequence": []
        }

        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not json_match:
            print("[ERROR] No JSON object found in LLM output.")
            print(f"Raw output: {response_str}")
            return safe_fallback

        clean_json_str = json_match.group(0)
        clean_json_str = re.sub(r',\s*([\]}])', r'\1', clean_json_str)

        try:
            parsed = json.loads(clean_json_str)
            return self._normalize_dimensions(parsed)
        except json.JSONDecodeError as e:
            print("[ERROR] Failed to decode extracted JSON substring. Returning fallback.")
            print(f"Extracted string: {clean_json_str}")
            print(f"Error: {e}")
            return safe_fallback
