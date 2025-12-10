"""
Configuration management for Real-time AI Voice Translator
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AudioConfig:
    """Audio I/O configuration"""
    sample_rate: int = 16000  # 16kHz for ASR
    output_sample_rate: int = 24000  # 24kHz for TTS output
    channels: int = 1
    chunk_size: int = 512  # samples per chunk
    dtype: str = "float32"
    
    # Device indices (None = default)
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    monitor_device: Optional[int] = None  # For self-monitoring


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""
    # Silero VAD settings
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 300  # Silence threshold for sentence end
    max_speech_duration_s: float = 30.0
    speech_pad_ms: int = 100
    

@dataclass
class ASRConfig:
    """Automatic Speech Recognition configuration"""
    model_size: str = "large-v3"  # Faster-Whisper model
    device: str = "cuda"
    compute_type: str = "float16"  # or "int8" for lower VRAM
    language: str = "zh"  # Source language
    beam_size: int = 5
    vad_filter: bool = True


@dataclass
class TranslatorConfig:
    """LLM Translation configuration"""
    provider: Literal["openai", "gemini", "groq"] = "openai"
    model: str = "gpt-4o-mini"  # Fast and cost-effective
    
    # API Keys from environment
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    
    # Translation settings
    source_language: str = "Chinese"
    target_language: str = "English"
    temperature: float = 0.3
    max_tokens: int = 500
    
    # Model mappings
    openai_models: dict = field(default_factory=lambda: {
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "gpt-4-turbo": "gpt-4-turbo",
    })
    gemini_models: dict = field(default_factory=lambda: {
        "gemini-pro": "gemini-pro",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.5-pro": "gemini-1.5-pro",
    })


@dataclass
class SentenceBufferConfig:
    """Smart sentence buffering configuration"""
    # Punctuation that triggers immediate TTS
    sentence_end_punctuation: tuple = (".", "!", "?", "。", "！", "？")
    clause_punctuation: tuple = (",", ";", ":", "，", "；", "：")
    
    # Word count threshold for clause-level cuts
    # 調高這些值可以減少語音卡頓，讓每段語音更完整
    min_words_for_clause_cut: int = 10   # 原本 5 → 提高到 10
    max_words_before_force_cut: int = 25  # 原本 15 → 提高到 25
    
    # Time-based thresholds
    max_buffer_time_ms: int = 3000  # Force flush after 3 seconds (原本 2秒)


@dataclass
class TTSConfig:
    """Text-to-Speech (Microsoft VibeVoice-Realtime-0.5B) configuration
    
    Reference: https://github.com/microsoft/VibeVoice
    
    Features:
    - ~300ms latency for first speech chunk
    - Streaming text input support
    - Multiple embedded speaker voices
    """
    model_type: str = "vibevoice"  # Microsoft VibeVoice-Realtime
    model_path: Optional[str] = None  # Will download from HuggingFace
    
    # Speaker selection (VibeVoice uses embedded voice prompts)
    speaker: str = "default"  # Available: default, speaker_1, speaker_2, speaker_3, speaker_4
    
    # Voice cloning (note: VibeVoice uses embedded prompts for safety)
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None
    
    # Generation settings
    device: str = "cuda"
    dtype: str = "float16"  # or "float32", "bfloat16"
    
    # Streaming settings (VibeVoice supports true streaming)
    use_streaming: bool = True
    use_websocket: bool = False  # For server-based streaming
    websocket_url: str = "ws://localhost:8765"
    
    # Audio output
    sample_rate: int = 24000
    speed: float = 1.0


@dataclass
class AppConfig:
    """Main application configuration"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    buffer: SentenceBufferConfig = field(default_factory=SentenceBufferConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    
    # Paths
    models_dir: Path = field(default_factory=lambda: Path("./models"))
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    
    # Logging
    log_level: str = "INFO"
    show_latency: bool = True
    
    def __post_init__(self):
        """Ensure directories exist"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> AppConfig:
    """Get default application configuration"""
    return AppConfig()


def load_config_from_yaml(path: str) -> AppConfig:
    """Load configuration from YAML file"""
    import yaml
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    config = AppConfig()
    
    # Update nested configs
    if 'audio' in data:
        config.audio = AudioConfig(**data['audio'])
    if 'vad' in data:
        config.vad = VADConfig(**data['vad'])
    if 'asr' in data:
        config.asr = ASRConfig(**data['asr'])
    if 'translator' in data:
        config.translator = TranslatorConfig(**data['translator'])
    if 'buffer' in data:
        config.buffer = SentenceBufferConfig(**data['buffer'])
    if 'tts' in data:
        config.tts = TTSConfig(**data['tts'])
    
    return config

