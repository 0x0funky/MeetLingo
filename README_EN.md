<p align="center">
  <img src="Title.png" alt="MeetLingo" width="600">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-Windows-lightgrey.svg" alt="Platform">
</p>

<h1 align="center">🎙️ MeetLingo</h1>

<p align="center">
  <b>Real-time Voice Translation</b> — Open-source solution designed for online meetings
</p>

<p align="center">
  <a href="./README.md">🌐 中文</a> •
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

> 🚀 **Current TTS Engine**: Microsoft VibeVoice-Realtime-0.5B
> 
> More TTS solutions coming soon (Edge-TTS, Coqui, GPT-SoVITS, etc.) to build the most comprehensive open-source real-time translation tool!

---

## ✨ Features

- 🗣️ **Real-time Speech Recognition** — High-accuracy ASR using Faster-Whisper (large-v3)
- 🌐 **Streaming Translation** — Supports OpenAI / Gemini / Groq with live translation
- 🎭 **High-quality Speech Synthesis** — Currently using Microsoft VibeVoice-Realtime (more engines coming)
- ⚡ **Low Latency** — End-to-end latency < 1.5 seconds
- 🎯 **Meeting Integration** — Output to Zoom / Teams / Meet via VB-CABLE
- 🔓 **Fully Open Source** — MIT License, contributions welcome!

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Microphone │ → │  VAD + ASR  │ → │ LLM Translate│ → │  VibeVoice  │
│  (Chinese)  │    │  (Whisper)  │    │  (GPT-4o)   │    │  TTS Engine │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              ↓
                  ┌─────────────────────────────────────────────┐
                  │       VB-CABLE → Meeting Software Mic        │
                  │        (Your voice speaks English!)          │
                  └─────────────────────────────────────────────┘
```

## 📋 System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| **OS** | Windows 10 | Windows 11 |
| **GPU** | RTX 3060 (8GB VRAM) | RTX 4070+ (12GB VRAM) |
| **RAM** | 16GB | 32GB |
| **Python** | 3.10 | 3.11 |
| **CUDA** | 11.8 | 12.1 |

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/MeetLingo.git
cd MeetLingo

# Create Conda environment (recommended)
conda create -n meetlingo python=3.11 -y
conda activate meetlingo

# Install PyTorch (CUDA 12.1)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install VibeVoice
pip install git+https://github.com/microsoft/VibeVoice.git
```

### 2. Download Voice Prompts

VibeVoice requires pre-loaded voice prompt files (`.pt`):

```bash
# Create directory
mkdir -p voices/streaming_model

# Download voice files
# Method 1: Using Git LFS
git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/VibeVoice.git temp_vibe
cd temp_vibe
git sparse-checkout set demo/voices/streaming_model
git lfs pull
cp demo/voices/streaming_model/*.pt ../voices/streaming_model/
cd .. && rm -rf temp_vibe
```

**Available Voices:**
| Filename | Gender | Style |
|----------|--------|-------|
| `en-Carter_man.pt` | Male | Professional, steady |
| `en-Davis_man.pt` | Male | Young, energetic |
| `en-Emma_woman.pt` | Female | Warm, friendly |
| `en-Frank_man.pt` | Male | Mature, authoritative |
| `en-Grace_woman.pt` | Female | Professional, clear |
| `en-Mike_man.pt` | Male | Casual, friendly |

### 3. Install VB-CABLE (Virtual Audio Cable)

1. Download [VB-CABLE](https://vb-audio.com/Cable/)
2. Run installer **as Administrator**
3. Restart your computer

### 4. Set API Key

```bash
# Set environment variable (choose one)
set OPENAI_API_KEY=your_openai_key
# or
set GOOGLE_API_KEY=your_google_key
# or
set GROQ_API_KEY=your_groq_key
```

### 5. Run!

```bash
# Windows requires Hugging Face symlink settings
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DISABLE_SYMLINKS=1

python main.py
```

Open browser and visit http://localhost:7860

## 🎮 Usage

### Step 1: Configuration
1. Select **Input Device** (your microphone)
2. Select **Output Device** (`CABLE Input (VB-Audio Virtual Cable)`)
3. Select **LLM Provider** and enter API Key
4. Click **"Initialize System"**

### Step 2: Meeting Software Setup
In Zoom / Teams / Meet:
- Microphone: Select **`CABLE Output (VB-Audio Virtual Cable)`**

### Step 3: Start Translating
1. Switch to "Translation" tab
2. Click **"Start Translation"**
3. Speak into your microphone (in your language)
4. The system will automatically translate and speak in the target language!

## ⚙️ Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| ASR Model | `large-v3` | Whisper model size |
| Source Language | `zh` | Source language |
| Target Language | `en` | Target language |
| LLM Provider | `OpenAI` | Translation engine |
| LLM Model | `gpt-4o-mini` | Translation model |
| Voice | `en-Carter_man` | TTS voice |

## 📁 Project Structure

```
MeetLingo/
├── main.py                 # Main entry point
├── config.py               # Configuration management
├── requirements.txt        # Dependencies
├── voices/                 # VibeVoice voice prompts
│   └── streaming_model/
│       ├── en-Carter_man.pt
│       └── ...
├── modules/
│   ├── audio_io.py         # Audio I/O
│   ├── vad_asr.py          # VAD + Whisper ASR
│   ├── translator.py       # LLM streaming translation
│   ├── sentence_buffer.py  # Smart sentence buffering
│   └── tts_engine.py       # VibeVoice TTS
└── gui/
    └── gradio_app.py       # Gradio interface
```

## 🔧 Troubleshooting

### GPU Out of Memory
```bash
# Use a smaller Whisper model
# In config.py, change asr.model_size = "medium"
```

### Hugging Face Symlink Error (Windows)
```bash
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DISABLE_SYMLINKS=1
python main.py
```

### No Sound from VB-CABLE
1. Confirm VB-CABLE is properly installed
2. Check Windows Sound settings for CABLE Input/Output
3. Ensure correct output device is selected

## 🗺️ Roadmap

Our goal is to build the **most comprehensive open-source real-time voice translation tool** designed for online meetings.

### Current Features ✅
- [x] Faster-Whisper ASR (multilingual speech recognition)
- [x] LLM Streaming Translation (OpenAI / Gemini / Groq)
- [x] VibeVoice TTS (high-quality speech synthesis)
- [x] VB-CABLE meeting integration
- [x] Gradio Web UI

### Planned 🚧
- [ ] **More TTS Engine Support**
  - [ ] Edge-TTS (free, low latency)
  - [ ] Coqui TTS (open source, customizable)
  - [ ] GPT-SoVITS (voice cloning)
  - [ ] Fish-Speech
  - [ ] OpenVoice
- [ ] **Voice Cloning**
  - [ ] Upload reference audio
  - [ ] Real-time clone your voice
- [ ] **More Language Support**
  - [ ] Japanese ↔ English
  - [ ] Korean ↔ English
  - [ ] Multi-language auto-detection
- [ ] **Advanced Features**
  - [ ] Real-time subtitles
  - [ ] Meeting recording + translation
  - [ ] API Server mode
  - [ ] Docker deployment
- [ ] **Platform Support**
  - [ ] macOS support
  - [ ] Linux support

### Long-term Vision 🌟
- Become the go-to open-source solution for real-time meeting translation
- Support 10+ TTS engines
- Support 50+ languages
- One-click cloud deployment

---

## 🤝 Contributing

Contributions welcome! Whether it's:
- 🐛 Bug reports
- 💡 Feature suggestions
- 🔧 Pull Requests
- 📖 Documentation improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

- [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) - TTS Engine
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - ASR Engine
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection

---

<p align="center">
  Made with ❤️ for the open-source community
</p>

<p align="center">
  ⭐ If this project helps you, please give us a Star! ⭐
</p>

