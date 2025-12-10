"""
Gradio GUI for Real-time AI Voice Translator
Provides a web-based interface for configuration and monitoring
"""
import gradio as gr
import numpy as np
import time
import os
from typing import Optional, Tuple, List
from pathlib import Path
from threading import Thread, Event
from queue import Queue
from rich.console import Console

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig, get_default_config
from modules.audio_io import AudioIO, AudioDevice
from modules.vad_asr import VADASR, ASRResult
from modules.translator import StreamingTranslator, TranslationChunk
from modules.sentence_buffer import SentenceBuffer, BufferChunk
from modules.tts_engine import TTSEngine, TTSChunk

console = Console()


class TranslatorApp:
    """Main application class that orchestrates all modules"""
    
    def __init__(self):
        self.config = get_default_config()
        
        # Module instances (lazy loaded)
        self._audio_io: Optional[AudioIO] = None
        self._vad_asr: Optional[VADASR] = None
        self._translator: Optional[StreamingTranslator] = None
        self._sentence_buffer: Optional[SentenceBuffer] = None
        self._tts_engine: Optional[TTSEngine] = None
        
        # State
        self._is_running = False
        self._stop_event = Event()
        self._processing_thread: Optional[Thread] = None
        
        # Logs and status
        self._asr_log: List[str] = []
        self._translation_log: List[str] = []
        self._latency_log: List[float] = []
    
    def get_audio_devices(self) -> Tuple[List[str], List[str]]:
        """Get lists of input and output audio devices"""
        input_devices = ["Default"] + [
            f"[{d.index}] {d.name}" for d in AudioIO.get_input_devices()
        ]
        output_devices = ["Default"] + [
            f"[{d.index}] {d.name}" for d in AudioIO.get_output_devices()
        ]
        return input_devices, output_devices
    
    def _parse_device_index(self, device_str: str) -> Optional[int]:
        """Parse device index from dropdown string"""
        if device_str == "Default" or not device_str:
            return None
        try:
            # Extract index from "[index] name"
            return int(device_str.split("]")[0].replace("[", ""))
        except:
            return None
    
    def initialize_modules(
        self,
        input_device: str,
        output_device: str,
        monitor_device: str,
        llm_provider: str,
        api_key: str,
        voice_name: str,
    ) -> str:
        """Initialize all modules with given configuration"""
        try:
            # Parse device indices
            input_idx = self._parse_device_index(input_device)
            output_idx = self._parse_device_index(output_device)
            monitor_idx = self._parse_device_index(monitor_device)
            
            # Initialize Audio I/O
            self._audio_io = AudioIO(
                input_sample_rate=16000,
                output_sample_rate=24000,
            )
            self._audio_io.set_input_device(input_idx)
            self._audio_io.set_output_device(output_idx)
            if monitor_idx is not None:
                self._audio_io.set_monitor_device(monitor_idx)
            
            # Initialize VAD + ASR
            self._vad_asr = VADASR(
                asr_model_size=self.config.asr.model_size,
                asr_device=self.config.asr.device,
                asr_compute_type=self.config.asr.compute_type,
                asr_language=self.config.asr.language,
            )
            
            # Initialize Translator
            if not api_key:
                return "❌ Error: API Key is required"
            
            self._translator = StreamingTranslator(
                provider=llm_provider.lower(),
                api_key=api_key,
                source_language=self.config.translator.source_language,
                target_language=self.config.translator.target_language,
            )
            
            # Initialize Sentence Buffer
            self._sentence_buffer = SentenceBuffer(
                min_words_for_clause_cut=self.config.buffer.min_words_for_clause_cut,
                max_words_before_force_cut=self.config.buffer.max_words_before_force_cut,
                target_language="en",
            )
            
            # Initialize TTS Engine with selected voice
            self._tts_engine = TTSEngine(
                device=self.config.tts.device,
                sample_rate=self.config.tts.sample_rate,
                speaker=voice_name,
            )
            
            return f"✅ 初始化完成！語音: {voice_name}"
            
        except Exception as e:
            return f"❌ Error initializing modules: {str(e)}"
    
    def _pipeline_loop(self):
        """Main processing pipeline loop"""
        console.print("[green]Pipeline started[/green]")
        
        while not self._stop_event.is_set():
            try:
                # Get ASR result
                asr_result = self._vad_asr.get_result(timeout=0.1)
                
                if asr_result and asr_result.text.strip():
                    start_time = time.time()
                    
                    # DEBUG: Log ASR result
                    console.print(f"[cyan][DEBUG ASR] 識別結果: '{asr_result.text}'[/cyan]")
                    
                    # Log ASR result
                    self._asr_log.append(f"[{time.strftime('%H:%M:%S')}] {asr_result.text}")
                    
                    # Translate with streaming
                    full_translation = ""
                    try:
                        console.print(f"[yellow][DEBUG 翻譯] 開始翻譯...[/yellow]")
                        for trans_chunk in self._translator.translate_stream(asr_result.text):
                            if not trans_chunk.is_complete:
                                full_translation += trans_chunk.text
                                console.print(f"[yellow][DEBUG 翻譯] 收到片段: '{trans_chunk.text}'[/yellow]")
                                
                                # Feed to sentence buffer
                                buffer_chunks = self._sentence_buffer.feed(trans_chunk.text)
                                
                                # Send buffer chunks to TTS (only if text is not empty)
                                for buf_chunk in buffer_chunks:
                                    if buf_chunk.text and buf_chunk.text.strip():
                                        console.print(f"[magenta][DEBUG TTS] 排入合成佇列: '{buf_chunk.text}'[/magenta]")
                                        self._tts_engine.queue_synthesis(
                                            buf_chunk.text,
                                            buf_chunk.chunk_index,
                                            buf_chunk.is_final,
                                        )
                        
                        console.print(f"[green][DEBUG 翻譯] 完整翻譯: '{full_translation}'[/green]")
                        
                    except Exception as trans_error:
                        console.print(f"[red][DEBUG 翻譯] 翻譯錯誤: {trans_error}[/red]")
                        continue
                    
                    # Flush remaining buffer (only if we got a translation)
                    if full_translation.strip():
                        final_chunk = self._sentence_buffer.flush()
                        if final_chunk and final_chunk.text and final_chunk.text.strip():
                            console.print(f"[magenta][DEBUG TTS] 排入最終片段: '{final_chunk.text}'[/magenta]")
                            self._tts_engine.queue_synthesis(
                                final_chunk.text,
                                final_chunk.chunk_index,
                                True,
                            )
                        
                        # Log translation
                        self._translation_log.append(
                            f"[{time.strftime('%H:%M:%S')}] {full_translation}"
                        )
                        
                        # Calculate latency
                        latency = (time.time() - start_time) * 1000
                        self._latency_log.append(latency)
                        console.print(f"[blue][DEBUG 延遲] 總延遲: {latency:.0f}ms[/blue]")
                    else:
                        console.print(f"[red][DEBUG 翻譯] 翻譯結果為空![/red]")
                
                # Get TTS audio and play
                tts_chunk = self._tts_engine.get_audio_chunk(timeout=0.05)
                if tts_chunk:
                    audio = tts_chunk.audio
                    # Ensure 1D array
                    if audio.ndim > 1:
                        audio = audio.flatten()
                    
                    if len(audio) > 0:
                        duration_sec = len(audio) / tts_chunk.sample_rate
                        console.print(f"[green][DEBUG TTS] 播放音訊: {len(audio)} samples ({duration_sec:.2f}秒), text='{tts_chunk.text[:30]}...'[/green]")
                        self._audio_io.play_audio(audio)
                    else:
                        console.print(f"[red][DEBUG TTS] 收到空音訊! text='{tts_chunk.text}'[/red]")
                
            except Exception as e:
                console.print(f"[red]Pipeline error: {e}[/red]")
        
        console.print("[yellow]Pipeline stopped[/yellow]")
    
    def start(self) -> str:
        """Start the translation pipeline"""
        if self._is_running:
            return "⚠️ Already running"
        
        if not all([self._audio_io, self._vad_asr, self._translator, self._tts_engine]):
            return "❌ Please initialize modules first"
        
        try:
            # Clear logs
            self._asr_log = []
            self._translation_log = []
            self._latency_log = []
            
            # Set up audio input callback BEFORE starting streams
            def on_audio_input(audio_chunk):
                self._vad_asr.feed_audio(audio_chunk)
            
            self._audio_io.set_on_audio_input(on_audio_input)
            
            # Start all modules
            self._audio_io.start()
            self._vad_asr.start()
            self._tts_engine.start()
            
            # Start pipeline thread
            self._stop_event.clear()
            self._processing_thread = Thread(target=self._pipeline_loop, daemon=True)
            self._processing_thread.start()
            
            self._is_running = True
            return "🎙️ Translation started! Speak into your microphone..."
            
        except Exception as e:
            return f"❌ Error starting: {str(e)}"
    
    def stop(self) -> str:
        """Stop the translation pipeline"""
        if not self._is_running:
            return "⚠️ Not running"
        
        try:
            self._stop_event.set()
            
            if self._audio_io:
                self._audio_io.stop()
            if self._vad_asr:
                self._vad_asr.stop()
            if self._tts_engine:
                self._tts_engine.stop()
            
            if self._processing_thread:
                self._processing_thread.join(timeout=2.0)
            
            self._is_running = False
            return "⏹️ Translation stopped"
            
        except Exception as e:
            return f"❌ Error stopping: {str(e)}"
    
    def get_status(self) -> Tuple[str, str, str]:
        """Get current status for display"""
        # ASR log - show recent entries
        if self._asr_log:
            asr_text = "\n".join(self._asr_log[-15:])
        else:
            asr_text = "🎤 等待語音輸入..."
        
        # Translation log - show recent entries
        if self._translation_log:
            trans_text = "\n".join(self._translation_log[-15:])
        else:
            trans_text = "🌐 等待翻譯結果..."
        
        # Latency info with more detail
        if self._latency_log:
            recent = self._latency_log[-10:]
            avg_latency = sum(recent) / len(recent)
            min_latency = min(recent)
            max_latency = max(recent)
            latency_text = f"平均: {avg_latency:.0f}ms | 最小: {min_latency:.0f}ms | 最大: {max_latency:.0f}ms | 樣本數: {len(self._latency_log)}"
        else:
            latency_text = "⏳ 等待數據..."
        
        return asr_text, trans_text, latency_text


# Global app instance
app = TranslatorApp()


def create_app() -> gr.Blocks:
    """Create the Gradio interface"""
    
    # Get device lists
    input_devices, output_devices = app.get_audio_devices()
    
    with gr.Blocks() as demo:
        
        gr.Markdown("""
        # 🎙️ MeetLingo
        
        **即時語音翻譯** — 專為線上會議設計的開源解決方案
        
        `Whisper ASR` → `LLM 翻譯` → `VibeVoice TTS` | 延遲 < 1.5 秒
        
        **使用流程**: ⚙️ 設定 → 🚀 初始化 → 🎤 開始翻譯
        """)
        
        with gr.Tabs():
            # Tab 1: Configuration
            with gr.TabItem("⚙️ 設定"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 音訊設備")
                        input_device = gr.Dropdown(
                            choices=input_devices,
                            value="Default",
                            label="輸入設備 (麥克風)",
                        )
                        output_device = gr.Dropdown(
                            choices=output_devices,
                            value="Default",
                            label="輸出設備 (VB-CABLE Input)",
                        )
                        monitor_device = gr.Dropdown(
                            choices=["None"] + output_devices,
                            value="None",
                            label="監聽設備 (耳機，可選)",
                        )
                        refresh_btn = gr.Button("🔄 重新整理設備")
                    
                    with gr.Column():
                        gr.Markdown("### LLM 翻譯設定")
                        llm_provider = gr.Dropdown(
                            choices=["OpenAI", "Gemini", "Groq"],
                            value="OpenAI",
                            label="LLM 提供者",
                        )
                        api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="輸入你的 API Key",
                        )
                
                gr.Markdown("### 🎭 語音設定 (VibeVoice)")
                with gr.Row():
                    voice_select = gr.Dropdown(
                        choices=[
                            ("Carter (男, 專業)", "en-Carter_man"),
                            ("Davis (男, 年輕)", "en-Davis_man"),
                            ("Emma (女, 溫暖)", "en-Emma_woman"),
                            ("Frank (男, 成熟)", "en-Frank_man"),
                            ("Grace (女, 專業)", "en-Grace_woman"),
                            ("Mike (男, 輕鬆)", "en-Mike_man"),
                            ("Samuel (男, 印度腔)", "in-Samuel_man"),
                        ],
                        value="en-Carter_man",
                        label="選擇語音",
                    )
                
                with gr.Row():
                    init_btn = gr.Button("🚀 初始化系統", variant="primary", size="lg")
                    init_status = gr.Textbox(label="初始化狀態", interactive=False)
            
            # Tab 2: Translation
            with gr.TabItem("🎤 翻譯"):
                with gr.Row():
                    start_btn = gr.Button("▶️ 開始翻譯", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ 停止翻譯", variant="stop", size="lg")
                
                status_text = gr.Textbox(
                    label="狀態",
                    interactive=False,
                    value="請先初始化系統",
                )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📝 語音識別 (ASR)")
                        asr_output = gr.Textbox(
                            label="識別結果",
                            lines=8,
                            interactive=False,
                            elem_classes=["log-box"],
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 🌐 翻譯結果")
                        translation_output = gr.Textbox(
                            label="翻譯文字",
                            lines=8,
                            interactive=False,
                            elem_classes=["log-box"],
                        )
                
                with gr.Row():
                    latency_display = gr.Textbox(
                        label="⏱️ 延遲監控",
                        interactive=False,
                    )
                    refresh_status_btn = gr.Button("🔄 更新顯示")
            
            # Tab 3: Help
            with gr.TabItem("❓ 說明"):
                gr.Markdown("""
                ## 🎙️ MeetLingo
                
                即時語音翻譯，專為線上會議設計的開源解決方案。
                
                ---
                
                ### 📋 前置需求
                
                | 項目 | 說明 |
                |------|------|
                | **VB-CABLE** | 虛擬音源線，[下載連結](https://vb-audio.com/Cable/) |
                | **API Key** | OpenAI / Gemini / Groq 任選一個 |
                | **GPU** | NVIDIA RTX 3060+ (8GB VRAM) |
                | **VibeVoice 語音檔** | `voices/streaming_model/*.pt` |
                
                ---
                
                ### ⚙️ 設定步驟
                
                1. **選擇輸入設備** — 你的麥克風
                2. **選擇輸出設備** — `CABLE Input (VB-Audio Virtual Cable)`
                3. **輸入 API Key** — 選擇 LLM 提供者並填入 Key
                4. **點擊「初始化系統」** — 等待模型載入完成
                
                ---
                
                ### 🎯 會議軟體設定
                
                在 **Zoom / Teams / Meet** 中：
                - 麥克風：選擇 **`CABLE Output (VB-Audio Virtual Cable)`**
                
                這樣會議軟體會接收翻譯後的英文語音！
                
                ---
                
                ### 🚀 開始使用
                
                1. 切換到「翻譯」分頁
                2. 點擊 **「開始翻譯」**
                3. 對著麥克風說中文
                4. 系統會即時翻譯並輸出英文語音
                
                ---
                
                ### ⚡ 延遲優化建議
                
                | 方法 | 效果 |
                |------|------|
                | 使用 **Groq API** | 翻譯速度最快 (~300ms) |
                | 確保 **GPU 可用** | ASR + TTS 加速 |
                | 說話時**停頓清楚** | 幫助 VAD 切分 |
                
                ---
                
                ### 🔧 常見問題
                
                **Q: 沒有聲音輸出？**  
                A: 確認選擇了正確的輸出設備 (CABLE Input)
                
                **Q: GPU 記憶體不足？**  
                A: 關閉其他 GPU 程式，或使用較小的 Whisper 模型
                
                **Q: 翻譯延遲太高？**  
                A: 嘗試使用 Groq API，速度最快
                """)
        
        # Event handlers
        def refresh_devices():
            input_devs, output_devs = app.get_audio_devices()
            return (
                gr.update(choices=input_devs),
                gr.update(choices=output_devs),
                gr.update(choices=["None"] + output_devs),
            )
        
        refresh_btn.click(
            refresh_devices,
            outputs=[input_device, output_device, monitor_device],
        )
        
        def init_system(input_dev, output_dev, monitor_dev, provider, key, voice):
            return app.initialize_modules(
                input_dev, output_dev, monitor_dev,
                provider, key, voice,
            )
        
        init_btn.click(
            init_system,
            inputs=[
                input_device, output_device, monitor_device,
                llm_provider, api_key, voice_select,
            ],
            outputs=[init_status],
        )
        
        start_btn.click(app.start, outputs=[status_text])
        stop_btn.click(app.stop, outputs=[status_text])
        
        def update_status():
            asr, trans, latency = app.get_status()
            return asr, trans, latency
        
        refresh_status_btn.click(
            update_status,
            outputs=[asr_output, translation_output, latency_display],
        )
        
        # Auto-refresh using Timer (Gradio 4.x+)
        try:
            timer = gr.Timer(value=0.5, active=True)
            timer.tick(
                update_status,
                outputs=[asr_output, translation_output, latency_display],
            )
        except Exception:
            # Fallback for older Gradio: user needs to click refresh button
            console.print("[yellow]Auto-refresh not available, use manual refresh button[/yellow]")
        
    return demo


def launch_app(share: bool = False, server_port: int = 7860):
    """Launch the Gradio app"""
    demo = create_app()
    demo.launch(
        share=share,
        server_port=server_port,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    launch_app()

