# a11y-agent – Voice-controlled Browser Agent

A proof of concept for an AI based screen reader and browser assistant.

This project turns natural‐language, **voice** commands into fully‐automated browser actions using the open-source [browser-use](https://github.com/browser-use/browser-use) library.  
Hold <kbd>SPACE</kbd>, speak an instruction, release the key and the agent will:

1. Transcribe your speech with OpenAI Whisper
2. Launch a Chromium browser (Playwright) locally
3. Let the LLM (GPT-4o by default) reason about the task
4. Click, type and scroll until it fulfils the goal
5. Speak back the result ☺︎
6. Repeat the process until the user says "exit"

```
$ python main.py --voice --start-url "https://google.com"
```

## Features

- 🔊 **Push-to-talk** – hold <kbd>SPACE</kbd> to record, release to send
- 🖱️ **Autonomous web control** powered by _browser-use_ and Playwright
- 🦜 **OpenAI GPT-4.1** by default (configurable)
- 💬 Speaks every step and the final answer (text fallback when `--voice` off)
- 🔄 **Conversation history** – the agent remembers previous steps and uses them to reason about the current task
- Playback Caching – the agent will cache the playback of the same message to avoid repeated API calls
- Skip playback – press <kbd>ESC</kbd> to skip playback

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
› Open CNN.com and give me a brief summary of the latest news
```

## Project structure

```
a11y-agent/
├── main.py                # CLI & control loop
├── voice_io.py            # Speech-to-text + text-to-speech (push-to-talk added)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Troubleshooting

- First run is slow ➜ Playwright downloads the browser; subsequent runs are fast.
- "Playwright not installed" ➜ run `pip install playwright && playwright install`.
- **Voice I/O fails with API-key error** ➜ make sure the `OPENAI_API_KEY` is **exported in the shell** or present in a `.env` file in the project root. The key is loaded early via `python-dotenv`.
- **macOS "process is not trusted" warning** ➜ grant Accessibility permission:
  1. Keep the script running so macOS shows the prompt _or_ open **System Settings ▸ Privacy & Security ▸ Accessibility**.
  2. Click "+" and add the Terminal/iTerm/VScode app you use to run the program. Ensure the toggle is **enabled**.
  3. Re-launch the terminal and run the script again.
- Microphone not detected ➜ make sure `sounddevice` & `pynput` have the necessary OS permissions (see the same Privacy panel above).
- GPT-4o too expensive? Replace the model via `browser_use.llm.ChatOpenAI(model="gpt-3.5-turbo")` inside `main.py`.

## 🤝 Contributing

Ideas, improvements, PRs — all welcome. If you want to help make this ISO better, faster, or more flexible, open an issue or submit a pull request.

##📜 License

MIT — use freely, modify openly, and share widely. See the LICENSE file for details.

---

© 2025 — Jan Mittelman
