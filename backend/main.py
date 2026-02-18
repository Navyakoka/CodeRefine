from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

app = FastAPI(title="CodeRefine – AI Code Review Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class CodeInput(BaseModel):
    code: str
    language: str

@app.get("/")
def root():
    return {"message": "CodeRefine backend is running"}

@app.post("/review")
def review_code(data: CodeInput):
    prompt = f"""
You are an expert software engineer.

Analyze the following {data.language} code and provide:
1. Bug detection
2. Performance improvement suggestions
3. Secure coding recommendations
4. A rewritten improved version of the code

Code:
{data.code}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )

    return {
        "analysis": response.choices[0].message.content
    }
