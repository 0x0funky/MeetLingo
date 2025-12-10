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
  <b>即時語音翻譯</b> — 專為線上會議設計的開源解決方案
</p>

<p align="center">
  <a href="./README_EN.md">🌐 English</a> •
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

> 🚀 **目前 TTS 引擎**：Microsoft VibeVoice-Realtime-0.5B
> 
> 未來將支援更多 TTS 方案（Edge-TTS、Coqui、GPT-SoVITS 等），打造最完整的開源即時翻譯工具！

---

## ✨ Features

- 🗣️ **即時語音辨識** — 使用 Faster-Whisper (large-v3) 高精度 ASR
- 🌐 **串流翻譯** — 支援 OpenAI / Gemini / Groq，邊聽邊翻譯
- 🎭 **高品質語音合成** — 目前使用 Microsoft VibeVoice-Realtime（未來支援更多引擎）
- ⚡ **低延遲** — 端到端延遲 < 1.5 秒
- 🎯 **會議整合** — 透過 VB-CABLE 輸出到 Zoom / Teams / Meet
- 🔓 **完全開源** — MIT License，歡迎貢獻！

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  麥克風輸入  │ → │  VAD + ASR  │ → │  LLM 翻譯   │ → │  VibeVoice  │
│  (中文語音)  │    │  (Whisper)  │    │  (GPT-4o)   │    │  TTS 合成   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              ↓
                  ┌─────────────────────────────────────────────┐
                  │          VB-CABLE → 會議軟體麥克風            │
                  │         (你的聲音說英文！)                    │
                  └─────────────────────────────────────────────┘
```

## 📋 System Requirements

| 項目 | 最低需求 | 建議配置 |
|------|---------|---------|
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

# 建立 Conda 環境 (推薦)
conda create -n meetlingo python=3.11 -y
conda activate meetlingo

# 安裝 PyTorch (CUDA 12.1)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安裝依賴
pip install -r requirements.txt

# 安裝 VibeVoice
pip install git+https://github.com/microsoft/VibeVoice.git
```

### 2. Download Voice Prompts

VibeVoice 需要預先載入的語音提示檔 (`.pt`)：

```bash
# 建立目錄
mkdir -p voices/streaming_model

# 下載語音檔 (選擇一個或多個)
# 方法 1: 使用 Git LFS
git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/VibeVoice.git temp_vibe
cd temp_vibe
git sparse-checkout set demo/voices/streaming_model
git lfs pull
cp demo/voices/streaming_model/*.pt ../voices/streaming_model/
cd .. && rm -rf temp_vibe
```

**可用語音：**
| 檔案名稱 | 性別 | 風格 |
|---------|------|------|
| `en-Carter_man.pt` | 男 | 專業、穩重 |
| `en-Davis_man.pt` | 男 | 年輕、活力 |
| `en-Emma_woman.pt` | 女 | 溫暖、親切 |
| `en-Frank_man.pt` | 男 | 成熟、權威 |
| `en-Grace_woman.pt` | 女 | 專業、清晰 |
| `en-Mike_man.pt` | 男 | 輕鬆、友善 |

### 3. Install VB-CABLE (Virtual Audio Cable)

1. 下載 [VB-CABLE](https://vb-audio.com/Cable/)
2. **以系統管理員身份** 執行安裝
3. 重新啟動電腦

### 4. Set API Key

```bash
# 設定環境變數 (選擇一個)
set OPENAI_API_KEY=your_openai_key
# 或
set GOOGLE_API_KEY=your_google_key
# 或
set GROQ_API_KEY=your_groq_key
```

### 5. Run!

```bash
# Windows 需要設定 Hugging Face 符號連結
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DISABLE_SYMLINKS=1

python main.py
```

開啟瀏覽器訪問 http://localhost:7860

## 🎮 Usage

### Step 1: 設定
1. 選擇 **輸入設備**（你的麥克風）
2. 選擇 **輸出設備**（`CABLE Input (VB-Audio Virtual Cable)`）
3. 選擇 **LLM 提供者** 並輸入 API Key
4. 點擊 **「初始化系統」**

### Step 2: 會議軟體設定
在 Zoom / Teams / Meet 中：
- 麥克風：選擇 **`CABLE Output (VB-Audio Virtual Cable)`**

### Step 3: 開始翻譯
1. 切換到「翻譯」分頁
2. 點擊 **「開始翻譯」**
3. 對著麥克風說中文
4. 系統會自動翻譯並用語音說英文！

## ⚙️ Configuration

| 設定項 | 預設值 | 說明 |
|--------|--------|------|
| ASR Model | `large-v3` | Whisper 模型大小 |
| Source Language | `zh` | 來源語言 |
| Target Language | `en` | 目標語言 |
| LLM Provider | `OpenAI` | 翻譯引擎 |
| LLM Model | `gpt-4o-mini` | 翻譯模型 |
| Voice | `en-Carter_man` | TTS 語音 |

## 📁 Project Structure

```
MeetLingo/
├── main.py                 # 主程式入口
├── config.py               # 配置管理
├── requirements.txt        # 依賴清單
├── voices/                 # VibeVoice 語音提示檔
│   └── streaming_model/
│       ├── en-Carter_man.pt
│       └── ...
├── modules/
│   ├── audio_io.py         # 音訊 I/O
│   ├── vad_asr.py          # VAD + Whisper ASR
│   ├── translator.py       # LLM 串流翻譯
│   ├── sentence_buffer.py  # 智慧斷句
│   └── tts_engine.py       # VibeVoice TTS
└── gui/
    └── gradio_app.py       # Gradio 介面
```

## 🔧 Troubleshooting

### GPU 記憶體不足
```bash
# 使用較小的 Whisper 模型
# 在 config.py 中修改 asr.model_size = "medium"
```

### Hugging Face 符號連結錯誤 (Windows)
```bash
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DISABLE_SYMLINKS=1
python main.py
```

### VB-CABLE 沒有聲音
1. 確認 VB-CABLE 已正確安裝
2. 在 Windows 音效設定中確認 CABLE Input/Output 存在
3. 確認選擇了正確的輸出設備

## 🗺️ Roadmap

我們的目標是打造**最完整的開源即時語音翻譯工具**，專為線上會議設計。

### 現有功能 ✅
- [x] Faster-Whisper ASR（多語言語音辨識）
- [x] LLM 串流翻譯（OpenAI / Gemini / Groq）
- [x] VibeVoice TTS（高品質語音合成）
- [x] VB-CABLE 會議整合
- [x] Gradio Web UI

### 計劃中 🚧
- [ ] **更多 TTS 引擎支援**
  - [ ] Edge-TTS（免費、低延遲）
  - [ ] Coqui TTS（開源、可自訂）
  - [ ] GPT-SoVITS（語音克隆）
  - [ ] Fish-Speech
  - [ ] OpenVoice
- [ ] **語音克隆功能**
  - [ ] 上傳參考音檔
  - [ ] 即時克隆你的聲音
- [ ] **更多語言支援**
  - [ ] 日文 ↔ 英文
  - [ ] 韓文 ↔ 英文
  - [ ] 多語言自動偵測
- [ ] **進階功能**
  - [ ] 即時字幕顯示
  - [ ] 會議錄音 + 翻譯
  - [ ] API Server 模式
  - [ ] Docker 部署
- [ ] **平台支援**
  - [ ] macOS 支援
  - [ ] Linux 支援

### 長期願景 🌟
- 成為線上會議即時翻譯的首選開源方案
- 支援 10+ 種 TTS 引擎
- 支援 50+ 種語言
- 一鍵部署到雲端

---

## 🤝 Contributing

歡迎貢獻！無論是：
- 🐛 Bug 回報
- 💡 功能建議
- 🔧 Pull Request
- 📖 文檔改進

請查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解詳情。

## 📄 License

MIT License - 詳見 [LICENSE](LICENSE)

## 🙏 Acknowledgments

- [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) - TTS 引擎
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - ASR 引擎
- [Silero VAD](https://github.com/snakers4/silero-vad) - 語音活動檢測

---

<p align="center">
  Made with ❤️ for the open-source community
</p>

<p align="center">
  ⭐ 如果這個專案對你有幫助，請給我們一個 Star！⭐
</p>
