# TTS App — Kokoro-82M

Text-to-speech web app using Kokoro-82M (free, local, no API key).

## Setup

```bash
# 1. Create virtual env
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install ffmpeg (needed for MP3 export)
brew install ffmpeg       # macOS
# sudo apt install ffmpeg # Ubuntu/Debian

# 4. Run
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000

## Voices

| ID | Description |
|----|-------------|
| `af_heart` | American Female – warm (default) |
| `af_bella` | American Female – expressive |
| `af_nicole` | American Female – soft |
| `am_adam` | American Male – deep |
| `am_michael` | American Male – clear |
| `bf_emma` | British Female |
| `bm_george` | British Male |
| `bm_lewis` | British Male |

## Notes

- First run downloads the model (~330 MB) from Hugging Face automatically.
- Generated MP3s are saved in `outputs/` — clean that folder periodically.
- Max text length: 5,000 characters per request.
- Speed range: 0.5× to 2.0×.
