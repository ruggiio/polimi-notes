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

## Branches

| Branch | University | Platform | Authentication |
|--------|-----------|----------|----------------|
| [`main`](https://github.com/ruggiio/polimi-notes/tree/main) | Politecnico di Milano | Webex | PoliMi SSO |
| [`sharepoint`](https://github.com/ruggiio/polimi-notes/tree/sharepoint) | University of Parma (UniPR) | SharePoint / OneDrive | Microsoft SSO |

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
git clone https://github.com/ruggiio/polimi-notes.git
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
---

## Automazione notturna (Linux) — download, trascrizione e appunti senza interazione

Verificato su Ubuntu con RTX A2000 4 GB (settembre 2026). Quattro stadi idempotenti, ognuno fa solo ciò che manca su disco:

```
Archivio registrazioni PoliMi ──fetch──▶ output/videos/*.mp4 ──transcribe──▶ output/transcripts/*.txt ──notes──▶ output/notes/*.pdf ──cleanup
```

**Cleanup** (`auto.cleanup: true`): per ogni lezione che ha già il PDF cancella l'mp4 (`keep_videos: false`; fino al PDF il video resta, per poter ritrascrivere), i file di lavoro di pdflatex, i provini del triage, le cache di deck spostati/aggiornati in `output/slides/_decks/` e le conversioni pptx→pdf orfane. Restano transcript, sidecar `.json` del video (corso/data/argomento), `.tex` archiviato in `output/course/<corso>/` e PDF. `--no-cleanup` per saltarlo.

### Come funziona il login (nessuna password nello script)

- La 2FA PoliMi (CIE ID) non è automatizzabile, ma **non serve**: con "resta connesso" la sessione dura **10 giorni** e vive in un profilo Chromium persistente (`config/chrome_profile/`, gitignored).
- Con quella sessione il job apre l'archivio in headless; Webex chiede **solo l'email** (`POLIMI_EMAIL`, quella dichiarata dall'IdP, es. `nome.cognome@mail.polimi.it`) e l'IdP rilascia l'asserzione SAML senza password né 2FA.
- Quando la sessione scade il job esce con codice 3, manda una notifica desktop e salta solo lo stadio fetch. Rinnovo (una finestra, tu fai password + CIE):

```bash
.venv/bin/python tools/sso_login.py
```

### Setup

```bash
python3.12 -m venv .venv          # faster-whisper non richiede torch
.venv/bin/pip install playwright python-dotenv requests rich pyyaml typer faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12 chromadb
.venv/bin/playwright install chromium
cp .env.example .env               # POLIMI_USER, POLIMI_EMAIL, POLIMI_PASS
.venv/bin/python tools/sso_login.py
tools/install_timer.sh             # timer systemd --user, ogni notte alle 02:30 (recupera al risveglio)
```

Gli appunti vengono generati da **Claude Code** (`notes.backend: claude-code` → `claude -p`, abbonamento, nessuna API key), con lo stesso prompt del backend API. Per usare l'API imposta `backend: claude` e `ANTHROPIC_API_KEY`.

### Configurazione (`config/config.yaml`, sezione `auto`)

```yaml
auto:
  courses:
    - name: "BIOINSPIRED ROBOTICS"   # titolo note + scheda corso config/courses/<slug>.md
      match: "BIOINSPIRED"           # filtro "Corso" dell'archivio (nome o codice)
      # aa: 2025                     # A.A. di inizio; omesso = corrente
  kind: null                          # null = tutte le forme didattiche
  keep_videos: false                  # mp4 cancellato dal cleanup quando il PDF esiste
  cleanup: true
```

### Uso manuale

```bash
.venv/bin/python tools/fetch_lecture.py --aa 2025 --course BIOINSPIRED --list   # elenco
.venv/bin/python tools/fetch_lecture.py --aa 2025 --course BIOINSPIRED --pick oldest
.venv/bin/python tools/nightly.py --dry-run                                       # piano
.venv/bin/python tools/nightly.py --no-fetch --max-notes 1                        # solo trascrizione + 1 PDF
systemctl --user start polimi-notes-nightly.service; tail -f output/auto/nightly.log
```

Stato e fallimenti in `output/auto/state.json` (uno stadio che fallisce 3 volte viene saltato). Sonda diagnostica del flusso SSO: `tools/sso_probe.py`.

**Contesto di corso (RAG).** Con `rag.enabled: true` il job, prima di generare gli appunti, cerca in `output/rag` (ChromaDB; embedding all-MiniLM-L6-v2 in versione ONNX inclusa in chromadb, niente torch) i passaggi delle lezioni precedenti dello stesso corso più vicini a inizio/metà/fine della trascrizione e li passa nel prompt come "context from course material" (`rag.n_results` passaggi da `rag.chunk_size` parole, ≈3k token); dopo il PDF il transcript viene indicizzato (upsert per corso+data+chunk, quindi idempotente; la lezione in corso è esclusa dalla ricerca). I transcript con PDF ma non ancora indicizzati vengono recuperati al giro successivo.

**Lingua della lezione.** Il rilevatore di Whisper si fa ingannare dall'accento: un docente italiano che fa lezione in inglese viene rilevato `it` (p ≈ 0.8 su ogni finestra) e la decodifica produce una pseudo-traduzione. Con `transcription.language: null` la lingua è scelta tra `transcription.language_candidates` (default `[it, en]`) decodificando 3 finestre di 30 s con ciascuna e tenendo quella con la confidenza (avg_logprob) migliore; `auto.courses[].language: en` forza la lingua per un corso. La lingua degli appunti è `notes.language` (`en`, `it`, oppure `lecture` = quella rilevata; per corso `auto.courses[].notes_language`): viene dichiarata nel prompt e usata per i titoli dei box, altrimenti il modello oscilla tra un run e l'altro.

### Slide come supporto (WeBeep Sync)

Se `auto.slides: true` e in `auto.slides_dir` (default `~/Documenti/WeBeep Sync`, la cartella di [WeBeep Sync](https://github.com/toto04/webeep-sync)) esiste una cartella con il nome del corso, il job individua le slide della lezione, in ordine di affidabilità:

1. **dal video** (`slides_video: true`, `src/slides/video_match.py`): le registrazioni Webex sono lo schermo condiviso; un frame ogni 5 s, ritagliato alle bande nere e ridotto a miniatura 128×72, viene confrontato con le pagine di tutti i deck del corso (correlazione di intensità + gradiente; slide vere 0.85–0.95, false ≤ 0.6). Ne esce una **timeline** (`output/slides/<slug>/timeline.json`): quale deck, quale pagina, da quando a quando. ~1 min di CPU per ora di video, niente OCR né GPU. Se parte della lezione non ha slide riconoscibili (lavagna, deck non ancora caricato) il log lo dice.
2. **dalla trascrizione** (`select_decks`), se il video manca o non mostra slide: per ogni pagina di ogni deck una pertinenza (termini in comune pesati per idf sul corso × quanto sono detti); il deck con media più alta entra sempre, altri (fino a `slides_max_decks`) se hanno media ≥ 35 % del migliore e abbastanza pagine forti.

Con la timeline, al modello arrivano solo le pagine mostrate, con l'intervallo (`[09:25–14:20 · slide 6: …]`), le figure con "shown at mm:ss" e la trascrizione con un marcatore `[mm:ss]` al minuto: figure e definizioni finiscono dove il docente le ha mostrate. Scelta salvata (e modificabile) in `output/slides/<slug>/selection.json`; override con `"decks": ["nome.pdf", ...]` nel sidecar `.json` del video. Ogni deck è estratto una volta sola in `output/slides/_decks/<nome>__<hash>/`:

- **titoli e termini** → `initial_prompt` di Whisper (insieme al glossario della scheda corso);
- **testo pagina per pagina** → nel prompt degli appunti, come autorità per termini, nomi, simboli e formule (la trascrizione resta la fonte di ciò che è stato detto);
- **figure** (immagini raster sopra soglia, loghi ripetuti scartati, pagine vettoriali renderizzate) → elenco `[deck · slide N] latex_path` (le `slides_max_figures` più pertinenti di tutti i deck; mai vuoto se esiste una figura) che Claude inserisce dove servono, con didascalie contestuali; i riferimenti a file inesistenti vengono rimossi prima di compilare.

Manuale: `python main.py notes-only transcript.txt --course "X" --date 2025-09-29 --slides deck1.pdf --slides deck2.pdf [--video lezione.mp4] --suffix "Argomento"`.

Formati: PDF e PowerPoint (`.pptx/.ppt/.odp`, convertiti con LibreOffice in `output/slides/_converted/`, cache per data/dimensione).

**Slide linkate (OneDrive/SharePoint).** Molti docenti mettono su WeBeep solo un modulo `url` verso una cartella OneDrive, che WeBeep Sync ignora. Elenca quei link in `auto.courses[].slides_links`: ogni notte (o con `tools/sync_slides.py --all`) il job apre il link con la sessione PoliMi, elenca la cartella via REST API e scarica i file nuovi/modificati in `<slides_dir>/<CORSO>/_links/…` (stato in `.onedrive_sync.json`, resume, limite `slides_max_mb`).

**Triage delle figure.** Le immagini estratte da un pptx sono per metà sfondi, loghi, ritratti, screenshot. Con `slides_triage: true` un provino numerato viene mostrato a Haiku (tool Read, ~$0.05 per deck) che scarta le decorative e dà una didascalia descrittiva a ognuna delle altre; il modello che scrive gli appunti usa quelle didascalie per decidere dove inserirle.

WeBeep Sync senza root: `dpkg-deb -x webeep-sync-debian.deb ~/.local/opt/webeep-sync`, wrapper in `~/.local/bin/webeep-sync` con `--no-sandbox`, launcher in `~/.local/share/applications/`.

### Costi e modelli

Ogni chiamata `claude -p` è registrata in `output/auto/usage.jsonl` (modello, token, costo equivalente, durata). Per confrontare i modelli sulla stessa lezione:

```bash
.venv/bin/python tools/compare_models.py output/transcripts/LEZIONE.txt --course "X" --date 2025-09-29 \
    --topic "Argomento" --slides output/slides/<slug> --models sonnet opus haiku
```

Le chiamate batch usano `--strict-mcp-config` (nessun server MCP: ~14k token in meno per chiamata).

### Token per lezione: cosa è stato misurato (settembre 2026, lezione di 77 min, Sonnet 5)

| Voce | Token | Note |
|---|---|---|
| Ingresso | ~20.5k | system prompt 1.7k (in cache dalla seconda chiamata entro 1 h) + trascrizione ~13k + scheda corso + slide pertinenti |
| Uscita: LaTeX | ~18k | il preambolo (1.3k) lo aggiunge il codice, il modello scrive solo il corpo |
| Uscita: thinking | 13-22k | varia molto tra run; `--effort medium/low` lo azzera **ma condensa le note** (−22% / −48% di prosa, tabelle perse): non ammesso di default |
| Fix LaTeX (solo se serve) | ~450 | patch find/replace invece della riscrittura completa (20k) |

Ordine di grandezza a regime: **$0.43-0.50 eq per lezione**. Validazione deterministica di ogni modifica: `tools/validate_notes.py NOTES.tex --transcript T.txt --baseline OLD.tex` (copertura della trascrizione per blocchi di 3 min, frasi troncate, struttura, LaTeX). La variabilità tra run è alta (5960-8556 parole a parità di prompt): confrontare sempre almeno due campioni.
