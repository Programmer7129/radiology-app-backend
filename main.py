from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import set_verbosity_error
import base64
import io
from PIL import Image
import os

set_verbosity_error()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Next.js app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and tokenizer
model_name = "StanfordAIMI/CheXagent-2-3b"
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
model = model.to(dtype)
model.eval()

class GenerateRequest(BaseModel):
    paths: List[str]  # List of base64 encoded images
    prompt: str

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    try:
        # Process the request using the CheXagent model
        query = tokenizer.from_list_format([*[{'image': path} for path in request.paths], {'text': request.prompt}])
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query}
        ]
        input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
        output = model.generate(
            input_ids.to(device),
            do_sample=False,
            num_beams=1,
            temperature=1.,
            top_p=1.,
            use_cache=True,
            max_new_tokens=512
        )[0]
        response = tokenizer.decode(output[input_ids.size(1):-1])
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 