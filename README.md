# a11y-agent – Voice-controlled Browser Agent

This project turns natural‐language, **voice** commands into fully‐automated browser actions using the open-source [browser-use](https://github.com/browser-use/browser-use) library.  
Hold <kbd>SPACE</kbd>, speak an instruction, release the key and the agent will:

1. Transcribe your speech with OpenAI Whisper
2. Launch a Chromium browser (Playwright) locally
3. Let the LLM (GPT-4o by default) reason about the task
4. Click, type and scroll until it fulfils the goal
5. Speak back the result ☺︎

```
$ python main.py --voice --start-url "https://bing.com"
```

## Features

- 🔊 **Push-to-talk** – hold <kbd>SPACE</kbd> to record, release to send
- 🖱️ **Autonomous web control** powered by _browser-use_ and Playwright
- 🦜 **OpenAI GPT-4o** by default (configurable via env)
- 💬 Speaks every step and the final answer (text fallback when `--voice` off)
- 🧩 Modular architecture – small files, easy to extend

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# one-time browser download (~120 MB)
playwright install
```

### Environment variables

Save your keys in a `.env` file or export them in the shell:

```
OPENAI_API_KEY=sk-…   # required for GPT-4o AND Whisper STT
# optional – use Anthropic, Gemini, etc. (see browser-use docs)
```

> The `voice_io` helper will also look for `OPENAI_API_KEY` to call the Whisper audio endpoint for transcription.

## Usage

```bash
python main.py [--voice] [--start-url URL] [--debug]
```

- `--voice` Enable microphone input + audio output (requires speakers)
- `--start-url` Load a page before the first step (default: Bing)
- `--debug` Raise exceptions instead of friendly messages

Example (text-only):

```
$ python main.py
Type your instructions (or 'exit' to quit):
› Find the release date of Windows 11 and read it back
```

## Project structure

```
a11y-agent/
├── main.py                # CLI & control loop
├── voice_io.py            # Speech-to-text + text-to-speech (push-to-talk added)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

Feel free to add more modules (e.g. custom _browser-use_ actions) under the same directory – the architecture is intentionally simple and modular.

## Troubleshooting

- First run is slow ➜ Playwright downloads the browser; subsequent runs are fast.
- "Playwright not installed" ➜ run `pip install playwright && playwright install`.
- Microphone not detected ➜ make sure `sounddevice` & `pynput` have the necessary OS permissions (macOS: System Settings ▸ Privacy & Security).
- GPT-4o too expensive? Replace the model via `browser_use.llm.ChatOpenAI(model="gpt-3.5-turbo")` inside `main.py`.

---

© 2025 — MIT licence
