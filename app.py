from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import os
import re

from config import (
    MODEL_PATH,
    MAX_NEW_TOKENS
)

print(f"Buscando modelo en: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"¡ERROR CRÍTICO!: El archivo del modelo no existe en {MODEL_PATH}")
    # Esto evitará el aborto misterioso y te dirá si la ruta está mal


from tokenizer import (
    encode,
    decode
)

from model import MiniGPT


# =========================================================
# DEVICE
# =========================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# =========================================================
# LOAD MODEL
# =========================================================

model = MiniGPT().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):
    prompt: str


# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post("/api/chat")
async def chat(request: ChatRequest):

    raw_prompt = request.prompt.strip().lower()

    if not raw_prompt:
        return {"reply": ""}

    # =========================================================
    # INTEGRATED CALCULATOR (TOOL USE)
    # =========================================================
    # This regex checks if the prompt contains numbers and math symbols
    # Clean up the prompt to look for purely algebraic structures
    math_cleaned = raw_prompt.replace("?", "").replace("x", "*").replace("÷", "/")
    math_cleaned = math_cleaned.strip()
    
    math_pattern = r'^[0-9\s\+\-\*\/\(\)\.]+$'
    
    if re.match(math_pattern, math_cleaned):
        try:
            # Safely evaluate the math string using a limited dictionary
            # Avoid using bare eval() on untrusted strings to prevent security issues
            allowed_chars = set("0123456789+-*/().¿? ")
            if all(char in allowed_chars for char in math_cleaned):
                result = eval(math_cleaned, {"__builtins__": None}, {})
                return {
                    "reply": f"🖩 {raw_prompt} = {result}." # output a message which tells the math result
                }
        except Exception:
            # If the math evaluation errors out (e.g., division by zero), 
            # we can fallback to the language model or return an error string
            return {"reply": "error: no se pudo calcular la operación."}

    # =========================================================
    # FALLBACK TO MINIGPT MODEL
    # =========================================================
    # Match the new template structure exactly
    formatted_prompt = f"user: {raw_prompt}\nagent: "

    input_tokens = encode(formatted_prompt)

    if len(input_tokens) == 0:
        return {"reply": "Input contains no known tokens."}

    context = torch.tensor([input_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=MAX_NEW_TOKENS
        )

    generated_text = decode(generated[0].tolist())

    # Extract only what the model generated after "agent: "
    completion = generated_text[len(formatted_prompt):]

    # Stop generation cleanly if it attempts to write the next '###' separator
    if "###" in completion:
        completion = completion.split("###")[0]

    return {"reply": completion.strip()}


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
async def root():

    return {
        "status": "online",
        "model": "MiniGPT"
    }