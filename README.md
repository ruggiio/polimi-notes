# PoliMi Lecture Notes Pipeline

Automatically downloads Webex lecture recordings from Politecnico di Milano,
transcribes the audio with Whisper, extracts and OCR-analyses slide/blackboard
frames, and generates structured LaTeX notes using an LLM.

```
Webex URL → [Download] → .mp4 → [Whisper] → transcript
                                          ↘
                              [ffmpeg+OCR] → slide text
                                          ↘
                                    [LLM] → .tex → .pdf
```

---

## Requirements

- **Windows 10/11**, 64-bit  
- **Python 3.11+** ([python.org](https://python.org))  
- **NVIDIA GPU** with CUDA 11.8+ drivers  
- **ffmpeg** on PATH  
- **Git** (optional, for cloning)  
- **MiKTeX or TeX Live** (for PDF compilation — optional)

---

## Installation

### 1. Clone / download the project

```bat
git clone https://github.com/you/polimi-notes.git
cd polimi-notes
```

### 2. Create a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install PyTorch with CUDA support (do this FIRST)

Go to https://pytorch.org/get-started/locally/ and get the right command.  
For CUDA 12.1 (most common with recent drivers):

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install all other dependencies

```bat
pip install -r requirements.txt
```

### 5. Install Playwright browsers

```bat
playwright install chromium
```

### 6. Install ffmpeg

Download from https://ffmpeg.org/download.html (get a Windows build).  
Extract it and add the `bin\` folder to your PATH, or place `ffmpeg.exe` in the project root.

---

## Configuration

Edit `config/config.yaml`:

```yaml
auth:
  username: "10XXXXXX"    # Your PoliMi Person Code
  password: "yourpassword"
```

**Alternatively**, set environment variables (more secure):

```bat
set POLIMI_USER=10XXXXXX
set POLIMI_PASS=yourpassword
set ANTHROPIC_API_KEY=sk-ant-...
```

Or create a `.env` file in the project root:

```
POLIMI_USER=10XXXXXX
POLIMI_PASS=yourpassword
ANTHROPIC_API_KEY=sk-ant-...
```

### LLM Backend options

In `config/config.yaml` under `notes.backend`:

| Value | Requirements | Notes |
|-------|-------------|-------|
| `claude` | `ANTHROPIC_API_KEY` | Best quality. ~$0.01–0.05/lecture |
| `ollama` | Ollama installed + `ollama pull mistral` | Free, private, offline. Runs after Whisper frees VRAM |
| `openai` | `OPENAI_API_KEY` | Good alternative |

---

## Usage

### Full pipeline (most common)

```bat
python main.py run "https://politecnicomilano.webex.com/recordingservice/.../playback" ^
    --course "Analisi Matematica 2" ^
    --date "2024-03-15"
```

### Without OCR (faster, transcript only)

```bat
python main.py run "https://..." --course "Fisica" --no-ocr
```

### Use an already-downloaded video

```bat
python main.py run "https://..." --no-download --video "output/videos/lecture.mp4" --course "Chimica"
```

### Transcribe only

```bat
python main.py transcribe-only output/videos/lecture.mp4 --model large-v2
```

### Generate notes from existing transcript

```bat
python main.py notes-only output/transcripts/lecture.txt --course "Algebra" --backend claude
```

### Generate notes with OCR data

```bat
python main.py notes-only output/transcripts/lecture.txt ^
    --ocr output/frames/lecture_ocr.json ^
    --course "Algebra" --backend claude
```

### Debug SSO login (open browser window)

```bat
python main.py run "https://..." --headed
```

---

## Output Structure

```
output/
├── videos/
│   └── lecture_title.mp4
├── transcripts/
│   ├── lecture_title.txt          ← plain text transcript
│   └── lecture_title_segments.json ← timestamped segments
├── frames/
│   └── lecture_title/
│       ├── frame_0000100_10.0s.jpg
│       ├── ...
│       └── lecture_title_ocr.json  ← OCR results
└── latex/
    ├── lecture_title_notes.tex     ← LaTeX source
    └── lecture_title_notes.pdf     ← Compiled PDF (if pdflatex available)
```

---

## Whisper Model Selection

| Model | VRAM | Speed (A2000 4GB) | Quality |
|-------|------|-------------------|---------|
| `tiny` | ~1 GB | Very fast | Low |
| `base` | ~1 GB | Fast | Acceptable |
| `small` | ~2 GB | Good | Good |
| `medium` | ~3 GB | ~15 min/hr | Very good |
| `large-v2` | ~4 GB | ~30 min/hr | **Best** ← recommended |

For Italian lectures, `large-v2` is strongly recommended.

---

## Troubleshooting

### SSO login fails
- Run with `--headed` to see what's happening in the browser
- PoliMi may have changed their login page. Inspect the field selectors in `src/downloader/downloader.py` → `_do_polimi_sso()`
- If 2FA is required, the tool will pause — complete it manually in the headed browser

### Video URL not intercepted
- Some Webex recordings use DASH/HLS instead of plain MP4
- `yt-dlp` should handle both — the fallback in `download_lecture()` tries this automatically

### Whisper CUDA out of memory
- Lower the model: change `model: large-v2` → `model: medium` in config.yaml
- Or add `--device cpu` (slow but works)

### OCR produces garbage text
- Adjust `scene_change_threshold` (try 0.3 or 0.5)
- For handwritten blackboard content, EasyOCR works better than Tesseract; no change needed
- For very dark/low-contrast frames, the auto-enhancement in `ocr.py` should help

### pdflatex not found
- Install [MiKTeX](https://miktex.org/download) (Windows) or TeX Live
- Or set `compile_pdf: false` in config.yaml and compile manually

---

## Privacy & Security Notes

- Your PoliMi credentials are only used locally by Playwright to log in
- They are never sent anywhere except `auth.polimi.it`
- Prefer environment variables over storing credentials in `config.yaml`
- The `.env` file is gitignored by default

---

## License

MIT — personal use only. Respect PoliMi's terms of service regarding lecture recordings.
