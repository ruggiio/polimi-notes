# PoliMi Lecture Notes Pipeline

Automatically downloads Webex lecture recordings from Politecnico di Milano, transcribes the audio with Whisper, extracts and OCR-analyses slide/blackboard frames, selects important figures using Claude Vision, and generates structured LaTeX notes with an LLM — producing a complete PDF ready for studying.

```
Webex URL → [Playwright SSO] → .mp4 → [faster-whisper] → transcript
                                                        ↘
                                         [ffmpeg + EasyOCR] → slide text + figures
                                                        ↘
                                              [Claude API] → .tex → .pdf
```

---

## Requirements

- **Windows 10/11**, 64-bit
- **Python 3.11+** — [python.org](https://python.org)
- **NVIDIA GPU** with CUDA drivers (recommended — CPU works but is very slow)
- **ffmpeg** — [ffmpeg.org](https://ffmpeg.org/download.html)
- **MiKTeX or TeX Live** — for automatic PDF compilation (optional but recommended)
- **Git** — [git-scm.com](https://git-scm.com/download/win)

---

## Installation

### 1. Clone the repository

```bat

cd polimi-notes
```

### 2. Create a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install PyTorch with CUDA (do this FIRST)

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and select your CUDA version.
For CUDA 12.6 (most common with recent drivers):

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify GPU is detected:
```bat
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 4. Install all other dependencies

```bat
pip install -r requirements.txt
```

### 5. Install Playwright browser

```bat
playwright install chromium
```

### 6. Install ffmpeg

1. Download from [ffmpeg.org/download.html](https://ffmpeg.org/download.html) → Windows builds by BtbN
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH:
   - Win + S → "Environment Variables" → System Variables → Path → Edit → New → `C:\ffmpeg\bin`
4. Open a new terminal and verify: `ffmpeg -version`

### 7. Install MiKTeX (for PDF compilation)

Download from [miktex.org/download](https://miktex.org/download) and install.
Then add MiKTeX to PATH:
```bat
setx PATH "%PATH%;C:\Users\YOUR_USERNAME\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
```
Verify: `pdflatex --version`

---

## Configuration

### Step 1 — Create your `.env` file

Copy the example and fill in your credentials:

```bat
copy .env.example .env
notepad .env
```

Fill in:

```
POLIMI_USER=10XXXXXX          # Your PoliMi Person Code (numeric)
POLIMI_EMAIL=name.surname@mail.polimi.it   # Your PoliMi institutional email
POLIMI_PASS=your_password

ANTHROPIC_API_KEY=sk-ant-...  # Get from console.anthropic.com
```

> ⚠️ **Important:** `POLIMI_USER` is your numeric Person Code (e.g. `10812345`).
> `POLIMI_EMAIL` is your full institutional email used to log into Webex.
> These are two different values — both are required.

### Step 2 — Review `config/config.yaml`

The default configuration works out of the box. Key settings:

```yaml
transcription:
  model: "medium"        # faster-whisper model: tiny/base/small/medium/large-v2

notes:
  backend: "claude"      # LLM backend: claude / ollama / openai

figures:
  enabled: true          # Use Claude Vision to select and embed slide figures
  max_candidates: 30     # Max frames analyzed per lecture (cost control)

ocr:
  math_only_filter: false  # false = all slide text; true = formulas only
```

### LLM Backend options

| Backend | Requirements | Cost | Quality |
|---------|-------------|------|---------|
| `claude` | `ANTHROPIC_API_KEY` | ~$0.30–0.70/lecture | ⭐⭐⭐ Best |
| `openai` | `OPENAI_API_KEY` | ~$0.30–0.50/lecture | ⭐⭐⭐ Very good |
| `ollama` | Ollama installed locally | Free | ⭐⭐ Good |

For Ollama: install from [ollama.com](https://ollama.com) then run `ollama pull mistral`.

---

## ⚠️ Important: PoliMi 2FA Setup

The login flow goes through PoliMi's SSO. Two things to know:

### 1. Disable 2FA (or handle it manually)

If your PoliMi account has **two-factor authentication enabled**, the automated login will fail because it cannot enter a one-time code.

**Option A (recommended):** Temporarily disable 2FA in your PoliMi account settings at [aunicalogin.polimi.it](https://aunicalogin.polimi.it).

**Option B:** Run with `--headed` and complete 2FA manually in the browser window that opens. The pipeline will wait for you.

### 2. The "Continua" intermediate page

After entering your credentials, PoliMi sometimes shows a warning page titled **"L'autenticazione a due fattori è disattiva"** with a blue **"Continua"** button.

The pipeline detects this page and pauses, asking you to:
1. Click **"Continua"** manually in the browser
2. Press **Enter** in the terminal to continue

This only happens on the first login — subsequent runs reuse saved cookies.

---

## Usage

### Full pipeline (most common)

```bat
python main.py run "https://politecnicomilano.webex.com/recordingservice/sites/politecnicomilano/recording/RECORDING_ID/playback" --course "COURSE NAME"
```

The date is extracted automatically from the Webex page title. You can override it with `--date "2024-03-15"`.

### Skip download (use existing video)

```bat
python main.py run "https://..." --course "COURSE NAME" --no-download --video output\videos\lecture.mp4
```

### Skip download + transcription (use existing files)

```bat
python main.py run "https://..." --course "COURSE NAME" --no-download --video output\videos\lecture.mp4 --no-transcribe
```

### Generate notes only (from existing transcript)

```bat
python main.py notes-only output\transcripts\lecture.txt --course "COURSE NAME" --backend claude
```

### Generate notes with OCR data

```bat
python main.py notes-only output\transcripts\lecture.txt --ocr output\frames\lecture_ocr.json --course "COURSE NAME"
```

### Debug SSO login (open visible browser window)

```bat
python main.py run "https://..." --course "COURSE NAME" --headed
```

### Transcribe only

```bat
python main.py transcribe-only output\videos\lecture.mp4
```

### Skip automatic cleanup

```bat
python main.py run "https://..." --course "COURSE NAME" --no-cleanup
```

### Split lectures (professor records two separate sessions on the same day)

```bat
# First recording
python main.py run "https://...first_url..." --course "SMART MATERIALS" --suffix "Structural Steel"

# Second recording
python main.py run "https://...second_url..." --course "SMART MATERIALS" --suffix "Heat Treatment"
```

Produces:
```
output\notes\05-03-2026_SMART MATERIALS - Structural Steel.pdf
output\notes\05-03-2026_SMART MATERIALS - Heat Treatment.pdf
```

Without `--suffix`, the pipeline behaves normally and overwrites the previous run.

---

## Output Structure

```
output/
├── videos/
│   ├── lecture.mp4              ← downloaded video (deleted after run)
│   └── lecture_date.txt         ← extracted date (reused for --no-download runs)
├── transcripts/
│   ├── lecture.txt              ← plain text transcript
│   └── lecture_segments.json    ← timestamped segments
├── frames/
│   ├── lecture/                 ← extracted frames (JPG)
│   └── lecture_ocr.json         ← OCR results
├── latex/
│   ├── lecture_notes.tex        ← LaTeX source (overwritten each run)
│   └── figures/                 ← figures selected by Claude Vision
└── notes/                       ← FINAL OUTPUT — never auto-deleted
    ├── 08-10-2025_COURSE NAME.pdf
    ├── 15-10-2025_COURSE NAME.pdf
    └── ...
```

The `output/notes/` folder accumulates all your PDFs across lectures. Everything else is cleaned up automatically at the start of each new run.

---

## Whisper Model Selection

| Model | VRAM | Speed (RTX A2000 4GB) | Quality |
|-------|------|----------------------|---------|
| `tiny` | ~1 GB | Very fast | Low |
| `base` | ~1 GB | Fast | Acceptable |
| `small` | ~2 GB | Fast | Good |
| `medium` | ~3 GB | ~23 min/1.5h lecture | **Recommended** ✓ |
| `large-v2` | ~4 GB | ~90 min/1.5h lecture | Best accuracy |

`medium` with `faster-whisper` is the recommended balance of speed and quality.

---

## Cost Estimates

Per lecture (1.5h) with default settings (`claude` backend, figures enabled):

| Component | Cost |
|-----------|------|
| Whisper transcription | Free (local) |
| EasyOCR | Free (local) |
| Claude notes generation | ~$0.30 |
| Claude Vision (figures) | ~$0.16 |
| **Total per lecture** | **~$0.46** |
| **Total for 20 lectures** | **~$9.20** |

---

## Troubleshooting

### Login fails immediately
- Run with `--headed` to see what happens in the browser
- Make sure `POLIMI_EMAIL` is your full email (e.g. `10812345@mail.polimi.it`)
- Make sure `POLIMI_USER` is only the numeric code (e.g. `10812345`)
- Check that 2FA is disabled in your PoliMi account

### "Continua" button not clicked automatically
- Run with `--headed` and click it manually, then press Enter in the terminal
- This is a known limitation — the button is covered by a CSS overlay

### Video not downloading
- Delete cookies and retry: `del config\webex_cookies.json`
- Run with `--headed` to debug

### Whisper CUDA out of memory
- Switch to `medium` model in `config/config.yaml`
- Or set `device: "cpu"` (slow but always works)

### OCR crashes silently
- Restart the terminal to free GPU VRAM
- Reduce `scene_change_threshold` if too few frames are extracted

### pdflatex not found
- Make sure MiKTeX bin folder is in PATH (see Installation step 7)
- Open a fresh terminal after adding to PATH
- Or set `compile_pdf: false` in config.yaml and compile manually with Overleaf

### LaTeX compilation errors
- Open `output\latex\lecture_notes.tex` in Overleaf for detailed error messages
- Common issue: Unicode characters (σ, °) — update your `notes_gen.py` system prompt to rule 24

---

## Privacy & Security

- Your credentials are stored only in `.env` on your local machine
- They are only sent to `auth.polimi.it` and `aunicalogin.polimi.it` via the browser
- The `.env` file is gitignored and will never be committed to GitHub
- Cookies are saved locally in `config/webex_cookies.json` (also gitignored)

---

## License

MIT — personal use only. Respect Politecnico di Milano's terms of service regarding lecture recordings.