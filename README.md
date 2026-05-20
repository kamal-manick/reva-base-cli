# reva

A small Rust CLI for running a local chat with a GGUF language model on CPU, powered by [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2). On first run, `reva` downloads a small quantized model from Hugging Face into your local data directory and drops you into a streaming, multi-turn chat at the terminal.

No GPU required. No cloud calls at inference time.

## Features

- Local CPU inference against any GGUF model supported by `llama.cpp`
- Streaming token-by-token output
- Multi-turn chat with history and the model's built-in chat template
- Automatic first-run download of a default model ([LiquidAI LFM2.5-350M, Q4_K_M](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF))
- Reads the model's own `context_length` from the GGUF metadata
- Override the model at runtime by passing a path on the command line

> Speech-to-text (Whisper) assets are also fetched on first run. A voice-input mode is planned but not yet wired into the chat loop.

## Requirements

- Rust (stable, edition 2024) — install via [rustup](https://rustup.rs/)
- A C/C++ toolchain for building `llama.cpp` via `llama-cpp-sys-2`:
  - **Windows:** Visual Studio Build Tools with the "Desktop development with C++" workload, plus CMake
  - **macOS:** Xcode Command Line Tools (`xcode-select --install`) and CMake
  - **Linux:** `build-essential`, `cmake`, and `clang`
- Network access on first run (to download the model from Hugging Face)

## Install / Build

```sh
git clone https://github.com/kamal-manick/reva-base-cli.git
cd reva-base-cli
cargo build --release
```

The compiled binary will be at `target/release/reva` (or `reva.exe` on Windows).

## Usage

Run with the default model (downloaded on first launch):

```sh
cargo run --release
```

Or run the built binary directly:

```sh
./target/release/reva
```

Use a different GGUF model by passing its path as the first argument:

```sh
./target/release/reva /path/to/your-model.gguf
```

In the chat prompt, type your message and press Enter. Type `/exit` to quit.

```
════════════════════════════════════════
  Model  : LFM2.5-350M
  Context: 4096 tokens
════════════════════════════════════════
Type your message and press Enter. Type /exit to quit.

You: hello
Assistant: Hi! How can I help you today?
```

## Where files are stored

`reva` creates an app folder under your platform's local data directory:

- **Windows:** `%LOCALAPPDATA%\.reva`
- **macOS:** `~/Library/Application Support/.reva`
- **Linux:** `~/.local/share/.reva`

Layout:

```
.reva/
└── models/
    ├── llm/   # GGUF chat model
    ├── stt/   # Whisper speech-to-text assets (downloaded, not yet used)
    └── tts/   # reserved for future text-to-speech support
```

Delete a model file to force a re-download on next launch.

## Configuration

Sampling and chat defaults are currently set in code (see [src/chat.rs](src/chat.rs)):

- `temperature = 0.7`
- `top_p = 0.9`
- repetition penalty across the last 64 tokens
- context window taken from the model's GGUF `<arch>.context_length`, falling back to 2048
- system prompt: *"You are a helpful assistant. Answer clearly and concisely."*

## Project layout

```
src/
├── main.rs       # entry point: sets up folders, downloads, launches chat
├── settings.rs   # per-platform app folder setup
├── download.rs   # streaming downloader for model files
└── chat.rs       # llama.cpp session, chat template, sampling loop
```

## Roadmap

- Voice input via the already-downloaded Whisper assets
- TTS playback of assistant replies
- Configurable system prompt and sampling parameters via CLI flags or a config file

## License

Licensed under the [MIT License](LICENSE).
