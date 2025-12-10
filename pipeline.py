"""
Complete Pipeline Integration for Real-time AI Voice Translator

This module orchestrates the entire translation pipeline:
Mic → VAD → ASR → LLM Translation → Sentence Buffer → TTS → Virtual Audio Output

Key Features:
- Full streaming support for low latency
- Thread-safe queue-based processing
- Latency monitoring and optimization
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from threading import Thread, Event, Lock
from queue import Queue, Empty
from rich.console import Console

from config import AppConfig, get_default_config
from modules.audio_io import AudioIO
from modules.vad_asr import VADASR, ASRResult
from modules.translator import StreamingTranslator, TranslationChunk
from modules.sentence_buffer import SentenceBuffer, BufferChunk
from modules.tts_engine import TTSEngine, TTSChunk
from utils.helpers import LatencyTracker

console = Console()


@dataclass
class PipelineStats:
    """Statistics for pipeline monitoring"""
    total_utterances: int = 0
    total_translations: int = 0
    total_tts_chunks: int = 0
    total_audio_seconds: float = 0.0
    
    asr_latency_tracker: LatencyTracker = field(default_factory=LatencyTracker)
    translation_latency_tracker: LatencyTracker = field(default_factory=LatencyTracker)
    tts_latency_tracker: LatencyTracker = field(default_factory=LatencyTracker)
    e2e_latency_tracker: LatencyTracker = field(default_factory=LatencyTracker)
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "utterances": self.total_utterances,
            "translations": self.total_translations,
            "tts_chunks": self.total_tts_chunks,
            "audio_seconds": self.total_audio_seconds,
            "asr_latency": self.asr_latency_tracker.get_stats(),
            "translation_latency": self.translation_latency_tracker.get_stats(),
            "tts_latency": self.tts_latency_tracker.get_stats(),
            "e2e_latency": self.e2e_latency_tracker.get_stats(),
        }


class TranslationPipeline:
    """
    Complete real-time translation pipeline.
    
    Data Flow:
    1. AudioIO captures microphone input
    2. VADASR detects speech and transcribes
    3. StreamingTranslator translates with LLM
    4. SentenceBuffer segments for TTS
    5. TTSEngine generates speech
    6. AudioIO outputs to virtual cable
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_default_config()
        
        # Module instances
        self._audio_io: Optional[AudioIO] = None
        self._vad_asr: Optional[VADASR] = None
        self._translator: Optional[StreamingTranslator] = None
        self._sentence_buffer: Optional[SentenceBuffer] = None
        self._tts_engine: Optional[TTSEngine] = None
        
        # State
        self._is_initialized = False
        self._is_running = False
        self._stop_event = Event()
        
        # Processing threads
        self._asr_thread: Optional[Thread] = None
        self._translation_thread: Optional[Thread] = None
        self._tts_thread: Optional[Thread] = None
        self._playback_thread: Optional[Thread] = None
        
        # Inter-thread queues
        self._asr_queue: Queue = Queue(maxsize=50)
        self._translation_queue: Queue = Queue(maxsize=100)
        self._tts_queue: Queue = Queue(maxsize=50)
        
        # Statistics
        self.stats = PipelineStats()
        
        # Callbacks
        self._on_asr_result: Optional[Callable[[ASRResult], None]] = None
        self._on_translation: Optional[Callable[[str, str], None]] = None
        self._on_tts_audio: Optional[Callable[[TTSChunk], None]] = None
        self._on_error: Optional[Callable[[str, Exception], None]] = None
    
    def set_callbacks(
        self,
        on_asr_result: Optional[Callable[[ASRResult], None]] = None,
        on_translation: Optional[Callable[[str, str], None]] = None,
        on_tts_audio: Optional[Callable[[TTSChunk], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ):
        """Set event callbacks for monitoring"""
        self._on_asr_result = on_asr_result
        self._on_translation = on_translation
        self._on_tts_audio = on_tts_audio
        self._on_error = on_error
    
    def initialize(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        monitor_device: Optional[int] = None,
        llm_provider: str = "openai",
        api_key: str = "",
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> bool:
        """
        Initialize all pipeline modules.
        
        Args:
            input_device: Microphone device index
            output_device: Virtual cable device index
            monitor_device: Headphone device index (optional)
            llm_provider: LLM provider (openai, gemini, groq)
            api_key: API key for LLM
            reference_audio_path: Path to reference audio for voice cloning
            reference_text: Transcript of reference audio
        
        Returns:
            True if initialization successful
        """
        try:
            console.print("[yellow]Initializing pipeline...[/yellow]")
            
            # 1. Initialize Audio I/O
            self._audio_io = AudioIO(
                input_sample_rate=self.config.audio.sample_rate,
                output_sample_rate=self.config.audio.output_sample_rate,
                channels=self.config.audio.channels,
                chunk_size=self.config.audio.chunk_size,
            )
            self._audio_io.set_input_device(input_device)
            self._audio_io.set_output_device(output_device)
            if monitor_device is not None:
                self._audio_io.set_monitor_device(monitor_device)
            
            console.print("[green]✓ Audio I/O initialized[/green]")
            
            # 2. Initialize VAD + ASR
            self._vad_asr = VADASR(
                vad_threshold=self.config.vad.threshold,
                min_speech_duration_ms=self.config.vad.min_speech_duration_ms,
                min_silence_duration_ms=self.config.vad.min_silence_duration_ms,
                asr_model_size=self.config.asr.model_size,
                asr_device=self.config.asr.device,
                asr_compute_type=self.config.asr.compute_type,
                asr_language=self.config.asr.language,
                sample_rate=self.config.audio.sample_rate,
            )
            console.print("[green]✓ VAD + ASR initialized[/green]")
            
            # 3. Initialize Translator
            if not api_key:
                raise ValueError("API key is required")
            
            self._translator = StreamingTranslator(
                provider=llm_provider,
                api_key=api_key,
                source_language=self.config.translator.source_language,
                target_language=self.config.translator.target_language,
                temperature=self.config.translator.temperature,
            )
            console.print("[green]✓ Translator initialized[/green]")
            
            # 4. Initialize Sentence Buffer
            self._sentence_buffer = SentenceBuffer(
                min_words_for_clause_cut=self.config.buffer.min_words_for_clause_cut,
                max_words_before_force_cut=self.config.buffer.max_words_before_force_cut,
                max_buffer_time_ms=self.config.buffer.max_buffer_time_ms,
                target_language="en",
            )
            console.print("[green]✓ Sentence buffer initialized[/green]")
            
            # 5. Initialize TTS Engine
            self._tts_engine = TTSEngine(
                model_path=self.config.tts.model_path,
                device=self.config.tts.device,
                sample_rate=self.config.tts.sample_rate,
            )
            
            # Load reference voice if provided
            if reference_audio_path:
                success = self._tts_engine.load_reference_voice(
                    reference_audio_path,
                    reference_text,
                )
                if success:
                    console.print("[green]✓ Voice clone loaded[/green]")
                else:
                    console.print("[yellow]⚠ Voice clone failed, using default voice[/yellow]")
            
            console.print("[green]✓ TTS engine initialized[/green]")
            
            self._is_initialized = True
            console.print("[bold green]✓ Pipeline initialization complete![/bold green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Pipeline initialization failed: {e}[/red]")
            if self._on_error:
                self._on_error("initialization", e)
            return False
    
    def _audio_input_callback(self, audio_chunk: np.ndarray):
        """Callback for audio input - feeds to VAD/ASR"""
        self._vad_asr.feed_audio(audio_chunk)
    
    def _asr_processing_loop(self):
        """Thread loop for ASR result processing"""
        console.print("[cyan]ASR processing thread started[/cyan]")
        
        while not self._stop_event.is_set():
            try:
                result = self._vad_asr.get_result(timeout=0.1)
                
                if result and result.text.strip():
                    # Record latency
                    asr_latency = (result.end_time - result.start_time) * 1000
                    self.stats.asr_latency_tracker.record(asr_latency)
                    self.stats.total_utterances += 1
                    
                    # Callback
                    if self._on_asr_result:
                        self._on_asr_result(result)
                    
                    # Queue for translation
                    self._asr_queue.put((result.text, time.time()))
                    
            except Empty:
                continue
            except Exception as e:
                console.print(f"[red]ASR error: {e}[/red]")
                if self._on_error:
                    self._on_error("asr", e)
    
    def _translation_processing_loop(self):
        """Thread loop for translation processing"""
        console.print("[cyan]Translation processing thread started[/cyan]")
        
        while not self._stop_event.is_set():
            try:
                item = self._asr_queue.get(timeout=0.1)
                source_text, start_time = item
                
                # Stream translation
                full_translation = ""
                chunk_index = 0
                
                for trans_chunk in self._translator.translate_stream(source_text):
                    if trans_chunk.is_complete:
                        break
                    
                    full_translation += trans_chunk.text
                    
                    # Feed to sentence buffer
                    buffer_chunks = self._sentence_buffer.feed(trans_chunk.text)
                    
                    # Queue buffer chunks for TTS
                    for buf_chunk in buffer_chunks:
                        self._translation_queue.put((
                            buf_chunk.text,
                            chunk_index,
                            buf_chunk.is_final,
                            start_time,
                        ))
                        chunk_index += 1
                
                # Flush remaining buffer
                final_chunk = self._sentence_buffer.flush()
                if final_chunk:
                    self._translation_queue.put((
                        final_chunk.text,
                        chunk_index,
                        True,
                        start_time,
                    ))
                
                # Record latency
                trans_latency = (time.time() - start_time) * 1000
                self.stats.translation_latency_tracker.record(trans_latency)
                self.stats.total_translations += 1
                
                # Callback
                if self._on_translation:
                    self._on_translation(source_text, full_translation)
                    
            except Empty:
                continue
            except Exception as e:
                console.print(f"[red]Translation error: {e}[/red]")
                if self._on_error:
                    self._on_error("translation", e)
    
    def _tts_processing_loop(self):
        """Thread loop for TTS processing"""
        console.print("[cyan]TTS processing thread started[/cyan]")
        
        while not self._stop_event.is_set():
            try:
                item = self._translation_queue.get(timeout=0.1)
                text, chunk_index, is_final, pipeline_start_time = item
                
                if not text.strip():
                    continue
                
                # Generate TTS
                tts_start = time.time()
                tts_chunk = self._tts_engine.synthesize_to_chunk(text, chunk_index)
                tts_time = (time.time() - tts_start) * 1000
                
                if tts_chunk and len(tts_chunk.audio) > 0:
                    # Record latencies
                    self.stats.tts_latency_tracker.record(tts_time)
                    
                    e2e_latency = (time.time() - pipeline_start_time) * 1000
                    self.stats.e2e_latency_tracker.record(e2e_latency)
                    
                    self.stats.total_tts_chunks += 1
                    self.stats.total_audio_seconds += len(tts_chunk.audio) / tts_chunk.sample_rate
                    
                    # Queue for playback
                    self._tts_queue.put(tts_chunk)
                    
                    # Callback
                    if self._on_tts_audio:
                        self._on_tts_audio(tts_chunk)
                    
            except Empty:
                continue
            except Exception as e:
                console.print(f"[red]TTS error: {e}[/red]")
                if self._on_error:
                    self._on_error("tts", e)
    
    def _playback_loop(self):
        """Thread loop for audio playback"""
        console.print("[cyan]Playback thread started[/cyan]")
        
        while not self._stop_event.is_set():
            try:
                tts_chunk = self._tts_queue.get(timeout=0.1)
                
                if tts_chunk and len(tts_chunk.audio) > 0:
                    # Play audio
                    self._audio_io.play_audio(tts_chunk.audio)
                    
            except Empty:
                continue
            except Exception as e:
                console.print(f"[red]Playback error: {e}[/red]")
    
    def start(self) -> bool:
        """Start the translation pipeline"""
        if not self._is_initialized:
            console.print("[red]Pipeline not initialized. Call initialize() first.[/red]")
            return False
        
        if self._is_running:
            console.print("[yellow]Pipeline already running[/yellow]")
            return True
        
        try:
            console.print("[yellow]Starting pipeline...[/yellow]")
            
            self._stop_event.clear()
            
            # Start audio I/O
            self._audio_io.set_on_audio_input(self._audio_input_callback)
            self._audio_io.start()
            
            # Start VAD/ASR
            self._vad_asr.start()
            
            # Start processing threads
            self._asr_thread = Thread(target=self._asr_processing_loop, daemon=True)
            self._translation_thread = Thread(target=self._translation_processing_loop, daemon=True)
            self._tts_thread = Thread(target=self._tts_processing_loop, daemon=True)
            self._playback_thread = Thread(target=self._playback_loop, daemon=True)
            
            self._asr_thread.start()
            self._translation_thread.start()
            self._tts_thread.start()
            self._playback_thread.start()
            
            self._is_running = True
            console.print("[bold green]✓ Pipeline started! Speak into your microphone...[/bold green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to start pipeline: {e}[/red]")
            return False
    
    def stop(self):
        """Stop the translation pipeline"""
        if not self._is_running:
            return
        
        console.print("[yellow]Stopping pipeline...[/yellow]")
        
        self._stop_event.set()
        
        # Stop modules
        if self._audio_io:
            self._audio_io.stop()
        if self._vad_asr:
            self._vad_asr.stop()
        
        # Wait for threads
        for thread in [self._asr_thread, self._translation_thread, 
                       self._tts_thread, self._playback_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        # Clear queues
        for queue in [self._asr_queue, self._translation_queue, self._tts_queue]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    break
        
        self._is_running = False
        console.print("[yellow]Pipeline stopped[/yellow]")
    
    def is_running(self) -> bool:
        """Check if pipeline is running"""
        return self._is_running
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.stats.get_summary()
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = PipelineStats()


def run_pipeline_demo():
    """Demo function to test the pipeline"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    console.print("\n[bold]Pipeline Demo[/bold]")
    console.print("=" * 60)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[red]No API key found. Set OPENAI_API_KEY or GOOGLE_API_KEY[/red]")
        return
    
    provider = "openai" if os.getenv("OPENAI_API_KEY") else "gemini"
    
    # Create pipeline
    pipeline = TranslationPipeline()
    
    # Set callbacks
    def on_asr(result):
        console.print(f"[cyan]ASR: {result.text}[/cyan]")
    
    def on_translation(source, target):
        console.print(f"[green]Translation: {target}[/green]")
    
    pipeline.set_callbacks(
        on_asr_result=on_asr,
        on_translation=on_translation,
    )
    
    # Initialize
    success = pipeline.initialize(
        llm_provider=provider,
        api_key=api_key,
    )
    
    if not success:
        return
    
    # Start
    pipeline.start()
    
    console.print("\n[yellow]Press Ctrl+C to stop...[/yellow]")
    
    try:
        while True:
            time.sleep(1)
            stats = pipeline.get_stats()
            e2e_avg = stats["e2e_latency"]["avg_ms"]
            if e2e_avg > 0:
                console.print(f"[dim]E2E Latency: {e2e_avg:.0f}ms[/dim]")
    except KeyboardInterrupt:
        pass
    
    pipeline.stop()
    
    # Print final stats
    console.print("\n[bold]Final Statistics:[/bold]")
    stats = pipeline.get_stats()
    console.print(f"  Utterances: {stats['utterances']}")
    console.print(f"  Translations: {stats['translations']}")
    console.print(f"  TTS Chunks: {stats['tts_chunks']}")
    console.print(f"  Audio Generated: {stats['audio_seconds']:.1f}s")
    console.print(f"  Avg E2E Latency: {stats['e2e_latency']['avg_ms']:.0f}ms")


if __name__ == "__main__":
    run_pipeline_demo()

