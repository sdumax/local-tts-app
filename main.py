import io
import uuid
import tempfile
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydub import AudioSegment

app = FastAPI(title="TTS App")

# Output directory for generated files
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Lazy-load pipeline to avoid startup delay
_pipeline = None

VOICES = {
    "af_heart": "American Female – Heart (warm)",
    "af_bella": "American Female – Bella (expressive)",
    "af_nicole": "American Female – Nicole (soft)",
    "am_adam":  "American Male – Adam (deep)",
    "am_michael": "American Male – Michael (clear)",
    "bf_emma":  "British Female – Emma",
    "bm_george": "British Male – George",
    "bm_lewis":  "British Male – Lewis",
}


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code="a")  # 'a' = American English
    return _pipeline


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0  # 0.5 – 2.0


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f:
        return f.read()


@app.get("/voices")
async def list_voices():
    return VOICES


@app.post("/generate")
async def generate(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if req.voice not in VOICES:
        raise HTTPException(status_code=400, detail=f"Unknown voice '{req.voice}'.")
    if not (0.5 <= req.speed <= 2.0):
        raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0.")

    try:
        pipeline = get_pipeline()
        audio_chunks = []

        for _, _, audio in pipeline(req.text, voice=req.voice, speed=req.speed):
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio was generated.")

        combined = np.concatenate(audio_chunks)
        sample_rate = 24000

        # Write WAV to temp buffer
        wav_buf = io.BytesIO()
        sf.write(wav_buf, combined, sample_rate, format="WAV")
        wav_buf.seek(0)

        # Convert WAV → MP3 via pydub
        segment = AudioSegment.from_wav(wav_buf)
        filename = f"{uuid.uuid4().hex}.mp3"
        out_path = OUTPUT_DIR / filename
        segment.export(str(out_path), format="mp3", bitrate="192k")

        return {"file": filename}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download(filename: str):
    # Sanitize: only allow hex filenames with .mp3
    if not filename.endswith(".mp3") or not filename[:-4].isalnum():
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=str(path),
        media_type="audio/mpeg",
        filename="tts_output.mp3",
    )
