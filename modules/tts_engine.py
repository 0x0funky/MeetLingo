"""
TTS Engine Module - Microsoft VibeVoice Integration
https://github.com/microsoft/VibeVoice

VibeVoice-Realtime-0.5B Features:
- Streaming text input support
- ~300ms latency for first speech chunk
- Real-time speech generation
- Voice prompts in embedded format

This module handles:
1. Loading VibeVoice model
2. Converting text to speech with streaming support
3. WebSocket-based real-time generation
"""
import numpy as np
import torch
import torchaudio
from dataclasses import dataclass
from typing import Optional, List, Generator, Callable, Union
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event, Lock
import time
import asyncio
import copy
import os
from rich.console import Console

console = Console()


@dataclass
class TTSChunk:
    """A chunk of generated audio"""
    audio: np.ndarray
    sample_rate: int
    chunk_index: int
    text: str
    is_final: bool
    generation_time_ms: float
    
    def __str__(self):
        duration_ms = len(self.audio) / self.sample_rate * 1000
        return f"[{self.chunk_index}] {duration_ms:.0f}ms audio for: '{self.text[:30]}...'"


class VoiceCloner:
    """
    Handles voice embedding for VibeVoice.
    
    Note: VibeVoice-Realtime uses embedded voice prompts to mitigate deepfake risks.
    For custom voice cloning, contact Microsoft team as mentioned in their docs.
    """
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._reference_audio: Optional[np.ndarray] = None
        self._reference_text: Optional[str] = None
        self._voice_embedding: Optional[torch.Tensor] = None
    
    def load_reference(
        self, 
        audio_path: str,
        reference_text: Optional[str] = None,
    ) -> bool:
        """
        Load reference audio for voice cloning.
        
        Note: VibeVoice-Realtime uses pre-defined speaker embeddings.
        This method stores the reference for potential future use.
        """
        try:
            import soundfile as sf
            
            audio, sr = sf.read(audio_path)
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
            
            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max() * 0.95
            
            self._reference_audio = audio
            self._reference_text = reference_text
            
            duration = len(audio) / self.sample_rate
            console.print(f"[green]✓ Reference audio loaded: {duration:.1f}s[/green]")
            console.print("[yellow]Note: VibeVoice-Realtime uses embedded voice prompts.[/yellow]")
            console.print("[yellow]Contact Microsoft for custom voice customization.[/yellow]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error loading reference audio: {e}[/red]")
            return False
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled = resampler(audio_tensor)
        return resampled.squeeze(0).numpy()
    
    def get_reference_audio(self) -> Optional[np.ndarray]:
        return self._reference_audio
    
    def get_reference_text(self) -> Optional[str]:
        return self._reference_text
    
    def is_loaded(self) -> bool:
        return self._reference_audio is not None


class VibeVoiceEngine:
    """
    Microsoft VibeVoice-Realtime-0.5B TTS Engine
    
    Features:
    - ~300ms latency for first speech chunk
    - Streaming text input support
    - Real-time speech generation
    
    Reference: https://github.com/microsoft/VibeVoice
    """
    
    # Available speakers (voice prompt files in voices/streaming_model/)
    AVAILABLE_SPEAKERS = [
        "en-Carter_man",      # Male, professional
        "en-Davis_man",       # Male, young
        "en-Emma_woman",      # Female, warm
        "en-Frank_man",       # Male, mature
        "en-Grace_woman",     # Female, professional
        "en-Mike_man",        # Male, casual
        "in-Samuel_man",      # Male, Indian accent
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "float16",
        sample_rate: int = 24000,
        speaker: str = "default",
    ):
        self.model_path = model_path or "microsoft/VibeVoice-Realtime-0.5B"
        self.device = device
        self.dtype = dtype
        self.sample_rate = sample_rate
        self.speaker = speaker
        
        self._model = None
        self._processor = None
        self._model_type = None
        self._cached_prompt = None  # Cached voice prompt for VibeVoice
        self._voice_cloner = VoiceCloner(sample_rate=sample_rate)
        self._is_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the VibeVoice model"""
        try:
            # Try to import and load VibeVoice
            try:
                from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
                from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
                
                console.print("[yellow]Loading VibeVoice-Realtime-0.5B...[/yellow]")
                
                # Determine dtype
                if self.device == "cuda":
                    torch_dtype = torch.bfloat16
                    attn_impl = "flash_attention_2"
                else:
                    torch_dtype = torch.float32
                    attn_impl = "sdpa"
                
                # Load processor
                self._processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)
                
                # Load model
                try:
                    self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype,
                        device_map=self.device,
                        attn_implementation=attn_impl,
                    )
                except Exception as e:
                    console.print(f"[yellow]Flash attention failed, using SDPA: {e}[/yellow]")
                    self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype,
                        device_map=self.device,
                        attn_implementation="sdpa",
                    )
                
                self._model.eval()
                self._model.set_ddpm_inference_steps(num_steps=5)
                
                # Try to load default voice prompt
                self._load_default_voice_prompt()
                
                self._model_type = "vibevoice"
                self._is_loaded = True
                console.print("[green]✓ VibeVoice-Realtime-0.5B loaded[/green]")
                console.print(f"[green]  Device: {self.device}[/green]")
                
                if self._cached_prompt is None:
                    console.print("[yellow]⚠ No voice prompt loaded. Will fall back to edge-tts.[/yellow]")
                    console.print("[yellow]  Download voice files from: https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model[/yellow]")
                
                return
                
            except ImportError as e:
                console.print(f"[yellow]VibeVoice import error: {e}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]VibeVoice load error: {e}[/yellow]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            
            # Fallback to edge-tts
            try:
                import edge_tts
                console.print("[yellow]⚠ Using edge-tts as fallback (no voice cloning)[/yellow]")
                self._model = "edge-tts"
                self._model_type = "edge-tts"
                self._is_loaded = True
                return
            except ImportError:
                pass
            
            console.print("[red]✗ No TTS engine available[/red]")
            self._is_loaded = False
            
        except Exception as e:
            console.print(f"[red]Error loading TTS model: {e}[/red]")
            self._is_loaded = False
    
    def _load_default_voice_prompt(self):
        """Try to load the selected voice prompt file"""
        # Check common locations for voice files
        voice_dirs = [
            Path("./voices/streaming_model"),
            Path("./demo/voices/streaming_model"),
            Path(__file__).parent.parent / "voices" / "streaming_model",
        ]
        
        # Build the expected filename from speaker parameter
        voice_filename = f"{self.speaker}.pt"
        
        for voice_dir in voice_dirs:
            if voice_dir.exists():
                # First try to load the selected speaker
                voice_file = voice_dir / voice_filename
                if voice_file.exists():
                    try:
                        console.print(f"[cyan]Loading voice prompt: {voice_file}[/cyan]")
                        self._cached_prompt = torch.load(
                            voice_file, 
                            map_location=self.device, 
                            weights_only=False
                        )
                        console.print(f"[green]✓ Voice prompt loaded: {voice_file.name}[/green]")
                        return
                    except Exception as e:
                        console.print(f"[yellow]Failed to load voice prompt: {e}[/yellow]")
                
                # Fall back to first available voice file
                pt_files = list(voice_dir.glob("*.pt"))
                if pt_files:
                    try:
                        voice_file = pt_files[0]
                        console.print(f"[cyan]Loading fallback voice prompt: {voice_file}[/cyan]")
                        self._cached_prompt = torch.load(
                            voice_file, 
                            map_location=self.device, 
                            weights_only=False
                        )
                        console.print(f"[green]✓ Voice prompt loaded: {voice_file.name}[/green]")
                        return
                    except Exception as e:
                        console.print(f"[yellow]Failed to load voice prompt: {e}[/yellow]")
        
        console.print("[yellow]No voice prompt files found[/yellow]")
        console.print("[yellow]Download from: https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model[/yellow]")
    
    def load_voice_prompt(self, voice_file: str) -> bool:
        """Load a specific voice prompt file"""
        try:
            self._cached_prompt = torch.load(
                voice_file,
                map_location=self.device,
                weights_only=False
            )
            console.print(f"[green]✓ Voice prompt loaded: {voice_file}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to load voice prompt: {e}[/red]")
            return False
    
    def set_speaker(self, speaker: str):
        """Set the speaker voice (from available embedded speakers)"""
        if speaker in self.AVAILABLE_SPEAKERS:
            self.speaker = speaker
            console.print(f"[green]Speaker set to: {speaker}[/green]")
        else:
            console.print(f"[yellow]Unknown speaker: {speaker}. Available: {self.AVAILABLE_SPEAKERS}[/yellow]")
    
    def load_reference_voice(
        self, 
        audio_path: str,
        reference_text: Optional[str] = None,
    ) -> bool:
        """Load reference audio (note: VibeVoice uses embedded prompts)"""
        return self._voice_cloner.load_reference(audio_path, reference_text)
    
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize speech from text.
        """
        if not self._is_loaded:
            console.print("[red]TTS model not loaded[/red]")
            return None
        
        if not text.strip():
            return np.array([], dtype=np.float32)
        
        try:
            # VibeVoice synthesis - requires cached voice prompt
            if self._model_type == "vibevoice" and self._cached_prompt is not None:
                return self._synthesize_vibevoice(text)
            
            # Edge-TTS fallback
            elif self._model_type == "edge-tts" or self._cached_prompt is None:
                return self._synthesize_edge_tts(text)
            
        except Exception as e:
            console.print(f"[red]TTS synthesis error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            # Try edge-tts as fallback
            try:
                return self._synthesize_edge_tts(text)
            except:
                return None
    
    def _synthesize_vibevoice(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using VibeVoice-Realtime"""
        # Prepare inputs using process_input_with_cached_prompt
        inputs = self._processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=self._cached_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Move tensors to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        
        # Generate audio
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.5,
            tokenizer=self._processor.tokenizer,
            generation_config={'do_sample': False},
            verbose=False,
            show_progress_bar=False,
            all_prefilled_outputs=copy.deepcopy(self._cached_prompt),
        )
        
        # Extract audio
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            if isinstance(audio, torch.Tensor):
                # Convert to float32 first (BFloat16 is not supported by numpy)
                audio = audio.to(torch.float32).cpu().numpy()
            
            # Ensure 1D array and correct dtype
            audio = np.asarray(audio, dtype=np.float32).flatten()
            console.print(f"[dim][DEBUG VibeVoice] 生成音訊: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}秒)[/dim]")
            return audio
        
        return np.array([], dtype=np.float32)
    
    def synthesize_streaming(
        self, 
        text: str,
    ) -> Generator[np.ndarray, None, None]:
        """
        Streaming synthesis - yields audio chunks as they're generated.
        
        VibeVoice-Realtime supports true streaming with ~300ms first chunk latency.
        
        優化：不再切成小片段，直接輸出完整音訊減少卡頓
        """
        if not self._is_loaded:
            return
        
        if not text.strip():
            return
        
        try:
            audio = self.synthesize(text)
            if audio is not None and len(audio) > 0:
                # 直接輸出完整音訊，不再切成小片段
                # 這樣可以讓 audio_io 的 crossfade 更有效
                yield audio
                        
        except Exception as e:
            console.print(f"[red]Streaming synthesis error: {e}[/red]")
    
    async def synthesize_streaming_async(
        self,
        text: str,
    ):
        """Async streaming synthesis for WebSocket integration"""
        if not self._is_loaded:
            return
        
        for chunk in self.synthesize_streaming(text):
            yield chunk
            await asyncio.sleep(0)  # Allow other tasks to run
    
    def _synthesize_edge_tts(self, text: str) -> Optional[np.ndarray]:
        """Fallback synthesis using edge-tts"""
        import asyncio
        import edge_tts
        import io
        import soundfile as sf
        
        # Skip empty or whitespace-only text
        if not text or not text.strip():
            return np.array([], dtype=np.float32)
        
        # Clean the text
        text = text.strip()
        
        # Edge-TTS needs minimum text length to work properly
        if len(text) < 5:
            if not hasattr(self, '_edge_tts_buffer'):
                self._edge_tts_buffer = ""
            self._edge_tts_buffer += text + " "
            return np.array([], dtype=np.float32)
        
        # Check if we have buffered text to prepend
        if hasattr(self, '_edge_tts_buffer') and self._edge_tts_buffer:
            text = self._edge_tts_buffer + text
            self._edge_tts_buffer = ""
        
        async def generate():
            voice = "en-US-JennyNeural"
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data
        
        try:
            # Create new event loop for thread safety
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                audio_bytes = loop.run_until_complete(generate())
            finally:
                try:
                    loop.close()
                except:
                    pass
            
            if not audio_bytes or len(audio_bytes) < 100:
                return np.array([], dtype=np.float32)
            
            audio_io = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_io)
            
            if sr != self.sample_rate:
                audio = self._voice_cloner._resample(audio, sr, self.sample_rate)
            
            return audio.astype(np.float32)
            
        except Exception as e:
            console.print(f"[red]Edge-TTS error: {e}[/red]")
            return np.array([], dtype=np.float32)
    
    def is_ready(self) -> bool:
        return self._is_loaded
    
    def has_voice_clone(self) -> bool:
        return self._cached_prompt is not None
    
    def get_model_info(self) -> dict:
        """Get model information"""
        model_name = "edge-tts (fallback)"
        if self._model_type == "vibevoice":
            if self._cached_prompt is not None:
                model_name = "VibeVoice-Realtime-0.5B"
            else:
                model_name = "VibeVoice-Realtime-0.5B (no voice prompt)"
        return {
            "model": model_name,
            "device": self.device,
            "dtype": self.dtype,
            "sample_rate": self.sample_rate,
            "speaker": self.speaker,
            "is_loaded": self._is_loaded,
            "has_voice_prompt": self._cached_prompt is not None,
            "available_speakers": self.AVAILABLE_SPEAKERS,
        }


class TTSEngine:
    """
    High-level TTS engine with queue-based processing.
    Wraps VibeVoice for integration with the translation pipeline.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        sample_rate: int = 24000,
        speaker: str = "default",
        use_websocket: bool = False,
        ws_url: str = "ws://localhost:8765",
    ):
        self.sample_rate = sample_rate
        self.use_websocket = use_websocket
        
        # Initialize VibeVoice engine
        self._engine = VibeVoiceEngine(
            model_path=model_path,
            device=device,
            sample_rate=sample_rate,
            speaker=speaker,
        )
        
        # Processing queues
        self._input_queue: Queue = Queue()
        self._output_queue: Queue = Queue()
        self._stop_event = Event()
        self._processing_thread: Optional[Thread] = None
        
        # Callbacks
        self._on_audio_chunk: Optional[Callable[[TTSChunk], None]] = None
        
        # Statistics
        self._total_chunks = 0
        self._total_audio_seconds = 0.0
        self._total_processing_time = 0.0
    
    def set_callback(self, on_audio_chunk: Callable[[TTSChunk], None]):
        """Set callback for generated audio chunks"""
        self._on_audio_chunk = on_audio_chunk
    
    def set_speaker(self, speaker: str):
        """Set speaker voice"""
        self._engine.set_speaker(speaker)
    
    def load_reference_voice(
        self, 
        audio_path: str,
        reference_text: Optional[str] = None,
    ) -> bool:
        """Load reference audio"""
        return self._engine.load_reference_voice(audio_path, reference_text)
    
    def load_voice_prompt(self, voice_file: str) -> bool:
        """Load a VibeVoice voice prompt file (.pt)"""
        return self._engine.load_voice_prompt(voice_file)
    
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synchronous synthesis"""
        return self._engine.synthesize(text)
    
    def synthesize_streaming(self, text: str) -> Generator[np.ndarray, None, None]:
        """Streaming synthesis"""
        return self._engine.synthesize_streaming(text)
    
    def synthesize_to_chunk(self, text: str, chunk_index: int = 0) -> Optional[TTSChunk]:
        """Synthesize and return as TTSChunk"""
        start_time = time.time()
        audio = self._engine.synthesize(text)
        gen_time = (time.time() - start_time) * 1000
        
        if audio is None:
            return None
        
        return TTSChunk(
            audio=audio,
            sample_rate=self.sample_rate,
            chunk_index=chunk_index,
            text=text,
            is_final=True,
            generation_time_ms=gen_time,
        )
    
    def _processing_loop(self):
        """Background processing loop"""
        while not self._stop_event.is_set():
            try:
                item = self._input_queue.get(timeout=0.1)
                
                if item is None:
                    continue
                
                text, chunk_index, is_final = item
                
                # Use streaming for lower latency
                start_time = time.time()
                audio_chunks = []
                
                for audio_chunk in self._engine.synthesize_streaming(text):
                    if len(audio_chunk) > 0:
                        audio_chunks.append(audio_chunk)
                        
                        # Emit partial chunk for immediate playback
                        partial_chunk = TTSChunk(
                            audio=audio_chunk,
                            sample_rate=self.sample_rate,
                            chunk_index=chunk_index,
                            text=text,
                            is_final=False,
                            generation_time_ms=(time.time() - start_time) * 1000,
                        )
                        self._output_queue.put(partial_chunk)
                        
                        if self._on_audio_chunk:
                            self._on_audio_chunk(partial_chunk)
                
                gen_time = (time.time() - start_time) * 1000
                
                if audio_chunks:
                    full_audio = np.concatenate(audio_chunks)
                    self._total_chunks += 1
                    self._total_audio_seconds += len(full_audio) / self.sample_rate
                    self._total_processing_time += gen_time
                
            except Empty:
                continue
            except Exception as e:
                console.print(f"[red]TTS processing error: {e}[/red]")
    
    def start(self):
        """Start background processing"""
        if not self._engine.is_ready():
            console.print("[yellow]⚠ TTS engine not fully ready[/yellow]")
        
        self._stop_event.clear()
        self._processing_thread = Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        
        model_info = self._engine.get_model_info()
        console.print(f"[green]✓ TTS engine started ({model_info['model']})[/green]")
    
    def stop(self):
        """Stop background processing"""
        self._stop_event.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        console.print("[yellow]TTS engine stopped[/yellow]")
    
    def queue_synthesis(self, text: str, chunk_index: int = 0, is_final: bool = False):
        """Queue text for background synthesis"""
        self._input_queue.put((text, chunk_index, is_final))
    
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[TTSChunk]:
        """Get generated audio chunk from output queue"""
        try:
            return self._output_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def stream_audio(self) -> Generator[TTSChunk, None, None]:
        """Generator that yields audio chunks"""
        while not self._stop_event.is_set():
            chunk = self.get_audio_chunk(timeout=0.1)
            if chunk is not None:
                yield chunk
    
    def is_ready(self) -> bool:
        return self._engine.is_ready()
    
    def has_voice_clone(self) -> bool:
        return self._engine.has_voice_clone()
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        avg_time = (
            self._total_processing_time / self._total_chunks 
            if self._total_chunks > 0 else 0
        )
        return {
            "total_chunks": self._total_chunks,
            "total_audio_seconds": self._total_audio_seconds,
            "average_generation_time_ms": avg_time,
            "input_queue_size": self._input_queue.qsize(),
            "output_queue_size": self._output_queue.qsize(),
            "is_ready": self.is_ready(),
            "has_voice_clone": self.has_voice_clone(),
            "model_info": self._engine.get_model_info(),
        }
    
    def clear_queues(self):
        """Clear all queues"""
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except:
                break
        
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except:
                break


if __name__ == "__main__":
    console.print("\n[bold]Testing VibeVoice TTS Engine...[/bold]")
    console.print("Reference: https://github.com/microsoft/VibeVoice")
    console.print("")
    
    # Initialize engine
    engine = TTSEngine(device="cuda" if torch.cuda.is_available() else "cpu")
    
    console.print(f"\n[bold]Model Info:[/bold]")
    info = engine.get_stats()["model_info"]
    for key, value in info.items():
        console.print(f"  {key}: {value}")
    
    if engine.is_ready():
        console.print("\n[green]✓ TTS engine ready[/green]")
        
        # Test synthesis
        test_text = "Hello, this is a test of Microsoft VibeVoice real-time text to speech."
        console.print(f"\n[yellow]Synthesizing: '{test_text}'[/yellow]")
        
        chunk = engine.synthesize_to_chunk(test_text)
        
        if chunk:
            console.print(f"[green]✓ Generated {len(chunk.audio) / chunk.sample_rate:.2f}s audio[/green]")
            console.print(f"[green]  Generation time: {chunk.generation_time_ms:.0f}ms[/green]")
        else:
            console.print("[red]✗ Synthesis failed[/red]")
    else:
        console.print("\n[yellow]⚠ TTS engine not ready[/yellow]")
        console.print("[yellow]Install VibeVoice:[/yellow]")
        console.print("[white]  pip install git+https://github.com/microsoft/VibeVoice.git[/white]")
