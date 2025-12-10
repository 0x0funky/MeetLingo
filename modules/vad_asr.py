"""
VAD + ASR Module - Voice Activity Detection and Speech Recognition
Uses Silero VAD + Faster-Whisper for low-latency streaming recognition
"""
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, List, Generator, Callable
from queue import Queue, Empty
from threading import Thread, Event
import time
from rich.console import Console

console = Console()


@dataclass
class ASRResult:
    """ASR recognition result"""
    text: str
    language: str
    confidence: float
    start_time: float
    end_time: float
    is_final: bool  # Whether this is a final result or partial
    
    def __str__(self):
        status = "✓" if self.is_final else "..."
        return f"[{status}] {self.text} (conf: {self.confidence:.2f})"


class SileroVAD:
    """
    Silero Voice Activity Detection
    Detects speech segments in audio stream
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 100,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate
        
        # Load Silero VAD model
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )
        
        # Get utility functions
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils
        
        # State for streaming
        self._speech_buffer: List[np.ndarray] = []
        self._is_speaking = False
        self._silence_samples = 0
        self._speech_samples = 0
        
        console.print("[green]✓ Silero VAD loaded[/green]")
    
    def reset(self):
        """Reset VAD state"""
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_samples = 0
        self._speech_samples = 0
        self.model.reset_states()
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process audio chunk and return speech segment if detected.
        Returns None if still accumulating or in silence.
        """
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()
        
        # Get speech probability
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        samples_per_ms = self.sample_rate / 1000
        min_speech_samples = int(self.min_speech_duration_ms * samples_per_ms)
        min_silence_samples = int(self.min_silence_duration_ms * samples_per_ms)
        
        if speech_prob >= self.threshold:
            # Speech detected
            self._speech_buffer.append(audio_chunk)
            self._speech_samples += len(audio_chunk)
            self._silence_samples = 0
            
            if not self._is_speaking and self._speech_samples >= min_speech_samples:
                self._is_speaking = True
        else:
            # Silence detected
            if self._is_speaking:
                self._silence_samples += len(audio_chunk)
                self._speech_buffer.append(audio_chunk)  # Include trailing silence
                
                if self._silence_samples >= min_silence_samples:
                    # End of speech segment
                    speech_audio = np.concatenate(self._speech_buffer)
                    self._speech_buffer = []
                    self._is_speaking = False
                    self._speech_samples = 0
                    self._silence_samples = 0
                    return speech_audio
            else:
                # Not speaking, reset buffer
                self._speech_buffer = []
                self._speech_samples = 0
        
        return None
    
    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """Get speech probability for a chunk"""
        audio_tensor = torch.from_numpy(audio_chunk).float()
        return self.model(audio_tensor, self.sample_rate).item()


class FasterWhisperASR:
    """
    Faster-Whisper based ASR with streaming support
    """
    
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "zh",
        beam_size: int = 5,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        
        # Import and load model
        from faster_whisper import WhisperModel
        
        console.print(f"[yellow]Loading Whisper model: {model_size}...[/yellow]")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        console.print(f"[green]✓ Whisper model loaded ({device}, {compute_type})[/green]")
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> ASRResult:
        """
        Transcribe audio segment to text
        """
        start_time = time.time()
        
        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        # Transcribe
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
            ),
        )
        
        # Collect all segments
        text_parts = []
        total_confidence = 0.0
        segment_count = 0
        
        for segment in segments:
            text_parts.append(segment.text.strip())
            total_confidence += segment.avg_logprob
            segment_count += 1
        
        end_time = time.time()
        
        text = " ".join(text_parts)
        avg_confidence = np.exp(total_confidence / max(segment_count, 1))
        
        return ASRResult(
            text=text,
            language=info.language,
            confidence=avg_confidence,
            start_time=start_time,
            end_time=end_time,
            is_final=True,
        )
    
    def transcribe_streaming(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Generator[ASRResult, None, None]:
        """
        Streaming transcription - yields partial results
        """
        # For Faster-Whisper, we simulate streaming by processing in chunks
        # Real streaming would require a different approach
        
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
        )
        
        for segment in segments:
            yield ASRResult(
                text=segment.text.strip(),
                language=info.language,
                confidence=np.exp(segment.avg_logprob),
                start_time=segment.start,
                end_time=segment.end,
                is_final=True,
            )


class VADASR:
    """
    Combined VAD + ASR pipeline for real-time speech recognition
    """
    
    def __init__(
        self,
        vad_threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        asr_model_size: str = "large-v3",
        asr_device: str = "cuda",
        asr_compute_type: str = "float16",
        asr_language: str = "zh",
        sample_rate: int = 16000,
    ):
        self.sample_rate = sample_rate
        
        # Initialize VAD
        self.vad = SileroVAD(
            threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            sample_rate=sample_rate,
        )
        
        # Initialize ASR
        self.asr = FasterWhisperASR(
            model_size=asr_model_size,
            device=asr_device,
            compute_type=asr_compute_type,
            language=asr_language,
        )
        
        # Processing queue
        self._audio_queue: Queue = Queue()
        self._result_queue: Queue = Queue()
        self._stop_event = Event()
        self._processing_thread: Optional[Thread] = None
        
        # Callbacks
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None
        self._on_transcription: Optional[Callable[[ASRResult], None]] = None
        
        # Statistics
        self._total_audio_processed = 0.0
        self._total_transcriptions = 0
    
    def set_callbacks(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
        on_transcription: Optional[Callable[[ASRResult], None]] = None,
    ):
        """Set event callbacks"""
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_transcription = on_transcription
    
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        audio_buffer = []
        chunk_count = 0
        
        while not self._stop_event.is_set():
            try:
                # Get audio chunk from queue
                audio_chunk = self._audio_queue.get(timeout=0.1)
                chunk_count += 1
                
                # DEBUG: Log every 50 chunks (approx every 5 seconds at typical chunk sizes)
                if chunk_count % 50 == 0:
                    console.print(f"[dim][DEBUG VAD] 已處理 {chunk_count} 個音訊片段, 佇列大小: {self._audio_queue.qsize()}[/dim]")
                
                # Process through VAD
                speech_prob = self.vad.get_speech_probability(audio_chunk)
                speech_segment = self.vad.process_chunk(audio_chunk)
                
                # DEBUG: Log when speech is detected
                if speech_prob > 0.3:
                    console.print(f"[cyan][DEBUG VAD] 語音機率: {speech_prob:.2f}, 正在說話: {self.vad._is_speaking}[/cyan]")
                
                if speech_segment is not None:
                    # Complete speech segment detected
                    duration_sec = len(speech_segment) / self.sample_rate
                    console.print(f"[green][DEBUG VAD] 偵測到完整語音片段! 長度: {duration_sec:.2f}秒[/green]")
                    
                    if self._on_speech_end:
                        self._on_speech_end()
                    
                    # Transcribe
                    console.print(f"[yellow][DEBUG ASR] 開始語音辨識...[/yellow]")
                    result = self.asr.transcribe(speech_segment, self.sample_rate)
                    console.print(f"[green][DEBUG ASR] 辨識結果: '{result.text}' (信心度: {result.confidence:.2f})[/green]")
                    
                    if result.text.strip():
                        self._result_queue.put(result)
                        self._total_transcriptions += 1
                        
                        if self._on_transcription:
                            self._on_transcription(result)
                    else:
                        console.print(f"[red][DEBUG ASR] 辨識結果為空![/red]")
                    
                    self._total_audio_processed += len(speech_segment) / self.sample_rate
                
            except Empty:
                continue
            except Exception as e:
                console.print(f"[red]ASR Error: {e}[/red]")
    
    def start(self):
        """Start the VAD+ASR processing pipeline"""
        self._stop_event.clear()
        self._processing_thread = Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        console.print("[green]✓ VAD+ASR pipeline started[/green]")
    
    def stop(self):
        """Stop the processing pipeline"""
        self._stop_event.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        self.vad.reset()
        console.print("[yellow]VAD+ASR pipeline stopped[/yellow]")
    
    def feed_audio(self, audio_chunk: np.ndarray):
        """Feed audio chunk into the pipeline"""
        try:
            # DEBUG: Log audio input stats occasionally
            if not hasattr(self, '_feed_count'):
                self._feed_count = 0
            self._feed_count += 1
            
            if self._feed_count == 1:
                console.print(f"[green][DEBUG 音訊輸入] 開始接收音訊, chunk大小: {len(audio_chunk)}, 最大值: {np.abs(audio_chunk).max():.4f}[/green]")
            elif self._feed_count % 100 == 0:
                console.print(f"[dim][DEBUG 音訊輸入] 已接收 {self._feed_count} 個音訊片段, 最大音量: {np.abs(audio_chunk).max():.4f}[/dim]")
            
            self._audio_queue.put_nowait(audio_chunk)
        except:
            pass  # Queue full
    
    def get_result(self, timeout: float = 0.1) -> Optional[ASRResult]:
        """Get transcription result from queue"""
        try:
            return self._result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def stream_results(self) -> Generator[ASRResult, None, None]:
        """Generator that yields transcription results"""
        while not self._stop_event.is_set():
            result = self.get_result(timeout=0.1)
            if result is not None:
                yield result
    
    def transcribe_audio(self, audio: np.ndarray) -> ASRResult:
        """Direct transcription without VAD (for testing)"""
        return self.asr.transcribe(audio, self.sample_rate)
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return {
            "total_audio_processed_seconds": self._total_audio_processed,
            "total_transcriptions": self._total_transcriptions,
            "audio_queue_size": self._audio_queue.qsize(),
            "result_queue_size": self._result_queue.qsize(),
        }


if __name__ == "__main__":
    # Test VAD
    console.print("\n[bold]Testing Silero VAD...[/bold]")
    vad = SileroVAD()
    
    # Generate test audio (silence + tone + silence)
    sample_rate = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Silence
    silence = np.zeros(int(sample_rate * 0.5))
    # Tone (simulating speech)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    test_audio = np.concatenate([silence, tone, silence])
    
    # Process in chunks
    chunk_size = 512
    for i in range(0, len(test_audio), chunk_size):
        chunk = test_audio[i:i + chunk_size]
        if len(chunk) == chunk_size:
            prob = vad.get_speech_probability(chunk.astype(np.float32))
            console.print(f"  Chunk {i // chunk_size}: Speech prob = {prob:.3f}")
    
    console.print("\n[green]✓ VAD test complete[/green]")

