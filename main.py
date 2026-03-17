import io
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from pydub import AudioSegment

app = FastAPI(title="TTS App")

# Output directory for generated files
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Lazy-load pipelines per lang_code
_pipelines: dict = {}

# voice_id -> (label, lang_code)
# Canadian English uses British pipeline (closest available phonology)
# Canadian French uses French pipeline (ff_siwis is the only French voice)
VOICES = {
    # American English
    "af_heart":   ("American Female – Heart (warm)",       "a"),
    "af_bella":   ("American Female – Bella (expressive)", "a"),
    "af_nicole":  ("American Female – Nicole (soft)",      "a"),
    "am_adam":    ("American Male – Adam (deep)",          "a"),
    "am_michael": ("American Male – Michael (clear)",      "a"),
    # British English
    "bf_emma":    ("British Female – Emma",                "b"),
    "bm_george":  ("British Male – George",                "b"),
    "bm_lewis":   ("British Male – Lewis",                 "b"),
    # Canadian English (British pipeline — closest available)
    "bf_alice":   ("Canadian English Female – Alice",      "b"),
    "bm_daniel":  ("Canadian English Male – Daniel",       "b"),
    # Canadian French (French pipeline)
    "ff_siwis":   ("Canadian French Female – Siwis",       "f"),
}


def get_pipeline(lang_code: str):
    if lang_code not in _pipelines:
        from kokoro import KPipeline, KModel
        # Share the model weights across pipelines to save memory
        if not _pipelines:
            _pipelines["_model"] = KModel().to("cpu").eval()
        _pipelines[lang_code] = KPipeline(lang_code=lang_code, model=_pipelines["_model"])
    return _pipelines[lang_code]


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
    return {k: v[0] for k, v in VOICES.items()}


@app.post("/generate")
async def generate(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if req.voice not in VOICES:
        raise HTTPException(status_code=400, detail=f"Unknown voice '{req.voice}'.")
    if not (0.5 <= req.speed <= 2.0):
        raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0.")

    try:
        _, lang_code = VOICES[req.voice]
        pipeline = get_pipeline(lang_code)
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
