"""
Audio I/O Module - Handles microphone input, virtual cable output, and monitoring
"""
import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import Optional, Callable, List, Generator
from threading import Thread, Event
from queue import Queue, Empty
import time
from rich.console import Console

console = Console()


@dataclass
class AudioDevice:
    """Audio device information"""
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_input: bool
    is_output: bool
    
    def __str__(self):
        device_type = []
        if self.is_input:
            device_type.append("Input")
        if self.is_output:
            device_type.append("Output")
        return f"[{self.index}] {self.name} ({'/'.join(device_type)})"


class AudioIO:
    """
    Handles all audio input/output operations with low-latency streaming.
    
    Features:
    - Microphone input streaming
    - Virtual audio cable output
    - Optional monitoring output
    - Thread-safe queue-based processing
    """
    
    def __init__(
        self,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        channels: int = 1,
        chunk_size: int = 512,
        dtype: str = "float32",
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.dtype = dtype
        
        # Queues for audio data
        self.input_queue: Queue = Queue(maxsize=100)
        self.output_queue: Queue = Queue(maxsize=100)
        self.monitor_queue: Queue = Queue(maxsize=100)
        
        # Control events
        self._stop_event = Event()
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None
        self._monitor_stream: Optional[sd.OutputStream] = None
        
        # Device indices
        self._input_device: Optional[int] = None
        self._output_device: Optional[int] = None
        self._monitor_device: Optional[int] = None
        
        # Callbacks
        self._on_audio_input: Optional[Callable[[np.ndarray], None]] = None
        
        # Statistics
        self._input_chunks_processed = 0
        self._output_chunks_processed = 0
        
        # Audio smoothing - 用於 crossfade 減少卡頓
        self._last_audio_tail: Optional[np.ndarray] = None
        self._crossfade_samples = 256  # crossfade 長度 (約 10ms @ 24kHz)
        
    @staticmethod
    def list_devices() -> List[AudioDevice]:
        """List all available audio devices"""
        devices = []
        device_list = sd.query_devices()
        
        for i, device in enumerate(device_list):
            devices.append(AudioDevice(
                index=i,
                name=device['name'],
                max_input_channels=device['max_input_channels'],
                max_output_channels=device['max_output_channels'],
                default_sample_rate=device['default_samplerate'],
                is_input=device['max_input_channels'] > 0,
                is_output=device['max_output_channels'] > 0,
            ))
        
        return devices
    
    @staticmethod
    def get_input_devices() -> List[AudioDevice]:
        """Get only input devices (microphones)"""
        return [d for d in AudioIO.list_devices() if d.is_input]
    
    @staticmethod
    def get_output_devices() -> List[AudioDevice]:
        """Get only output devices (speakers, virtual cables)"""
        return [d for d in AudioIO.list_devices() if d.is_output]
    
    @staticmethod
    def find_vb_cable_device() -> Optional[AudioDevice]:
        """Find VB-CABLE virtual audio device"""
        for device in AudioIO.get_output_devices():
            if "CABLE" in device.name.upper() and "INPUT" in device.name.upper():
                return device
        return None
    
    def set_input_device(self, device_index: Optional[int]):
        """Set the input device (microphone)"""
        self._input_device = device_index
        console.print(f"[green]Input device set to: {device_index}[/green]")
    
    def set_output_device(self, device_index: Optional[int]):
        """Set the output device (virtual cable)"""
        self._output_device = device_index
        console.print(f"[green]Output device set to: {device_index}[/green]")
    
    def set_monitor_device(self, device_index: Optional[int]):
        """Set the monitor device (headphones for self-listening)"""
        self._monitor_device = device_index
        console.print(f"[green]Monitor device set to: {device_index}[/green]")
    
    def set_on_audio_input(self, callback: Callable[[np.ndarray], None]):
        """Set callback for audio input chunks"""
        self._on_audio_input = callback
    
    def _input_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for input stream"""
        if status:
            console.print(f"[yellow]Input status: {status}[/yellow]")
        
        # Copy data to avoid reference issues
        audio_chunk = indata.copy().flatten()
        self._input_chunks_processed += 1
        
        # Debug: show audio level periodically
        if self._input_chunks_processed == 1:
            console.print(f"[green][DEBUG 音訊] 開始接收麥克風輸入![/green]")
        if self._input_chunks_processed % 100 == 0:
            max_level = np.abs(audio_chunk).max()
            console.print(f"[dim][DEBUG 音訊] 已接收 {self._input_chunks_processed} chunks, 音量: {max_level:.4f}[/dim]")
        
        # Put in queue for processing
        try:
            self.input_queue.put_nowait(audio_chunk)
        except:
            pass  # Queue full, drop frame
        
        # Call user callback if set
        if self._on_audio_input:
            self._on_audio_input(audio_chunk)
    
    def _output_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Callback for output stream"""
        if status and "underflow" not in str(status).lower():
            # Ignore underflow warnings (normal when waiting for audio)
            console.print(f"[yellow]Output status: {status}[/yellow]")
        
        try:
            # Get audio from queue
            audio_chunk = self.output_queue.get_nowait()
            
            # Ensure correct shape
            if len(audio_chunk) < frames:
                # Pad with zeros and apply fade out to reduce clicks
                padded = np.zeros(frames, dtype=np.float32)
                padded[:len(audio_chunk)] = audio_chunk
                # Fade out at the end to reduce click
                fade_len = min(64, len(audio_chunk))
                if fade_len > 0:
                    fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                    padded[len(audio_chunk)-fade_len:len(audio_chunk)] *= fade
                audio_chunk = padded
            elif len(audio_chunk) > frames:
                # Truncate if too long
                audio_chunk = audio_chunk[:frames]
            
            outdata[:] = audio_chunk.reshape(-1, 1)
            self._output_chunks_processed += 1
            
            # Also send to monitor if enabled
            if self._monitor_stream is not None:
                try:
                    self.monitor_queue.put_nowait(audio_chunk.copy())
                except:
                    pass
                    
        except Empty:
            # No audio available, output silence (fade in from 0)
            outdata.fill(0)
    
    def _monitor_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Callback for monitor stream"""
        try:
            audio_chunk = self.monitor_queue.get_nowait()
            
            if len(audio_chunk) < frames:
                audio_chunk = np.pad(audio_chunk, (0, frames - len(audio_chunk)))
            elif len(audio_chunk) > frames:
                audio_chunk = audio_chunk[:frames]
            
            outdata[:] = audio_chunk.reshape(-1, 1)
        except Empty:
            outdata.fill(0)
    
    def start_input_stream(self):
        """Start the input (microphone) stream"""
        if self._input_stream is not None:
            return
        
        try:
            # Use default device if None
            device = self._input_device
            if device is None:
                # Get the default input device
                default_input = sd.default.device[0]
                if default_input is not None and default_input >= 0:
                    device = default_input
                    console.print(f"[dim]Using default input device: {device}[/dim]")
            
            self._input_stream = sd.InputStream(
                device=device,
                samplerate=self.input_sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.chunk_size,
                callback=self._input_callback,
            )
            self._input_stream.start()
            console.print("[green]✓ Input stream started[/green]")
        except Exception as e:
            console.print(f"[red]Failed to start input stream: {e}[/red]")
            console.print("[yellow]Please select a specific input device in settings.[/yellow]")
    
    def start_output_stream(self):
        """Start the output (virtual cable) stream"""
        if self._output_stream is not None:
            return
        
        try:
            # Use default device if None
            device = self._output_device
            if device is None:
                # Get the default output device
                default_output = sd.default.device[1]
                if default_output is not None and default_output >= 0:
                    device = default_output
                    console.print(f"[dim]Using default output device: {device}[/dim]")
            
            self._output_stream = sd.OutputStream(
                device=device,
                samplerate=self.output_sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=2048,  # Larger buffer for smoother playback
                latency='high',  # 使用較高延遲設定，減少卡頓
                callback=self._output_callback,
            )
            self._output_stream.start()
            console.print("[green]✓ Output stream started[/green]")
        except Exception as e:
            console.print(f"[red]Failed to start output stream: {e}[/red]")
            console.print("[yellow]Please select a specific output device in settings.[/yellow]")
    
    def start_monitor_stream(self):
        """Start the monitor (headphones) stream"""
        if self._monitor_device is None or self._monitor_stream is not None:
            return
        
        self._monitor_stream = sd.OutputStream(
            device=self._monitor_device,
            samplerate=self.output_sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=2048,  # 與 output stream 一致
            latency='high',  # 減少卡頓
            callback=self._monitor_callback,
        )
        self._monitor_stream.start()
        console.print("[green]✓ Monitor stream started[/green]")
    
    def start(self):
        """Start all configured streams"""
        self._stop_event.clear()
        self.start_input_stream()
        self.start_output_stream()
        self.start_monitor_stream()
    
    def stop(self):
        """Stop all streams"""
        self._stop_event.set()
        
        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
        
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None
        
        if self._monitor_stream is not None:
            self._monitor_stream.stop()
            self._monitor_stream.close()
            self._monitor_stream = None
        
        console.print("[yellow]Audio streams stopped[/yellow]")
    
    def get_input_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get audio chunk from input queue"""
        try:
            return self.input_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def stream_input_audio(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields input audio chunks"""
        while not self._stop_event.is_set():
            audio = self.get_input_audio(timeout=0.1)
            if audio is not None:
                yield audio
    
    def _apply_crossfade(self, audio: np.ndarray) -> np.ndarray:
        """Apply crossfade with previous audio to reduce clicks/pops"""
        if self._last_audio_tail is None or len(audio) < self._crossfade_samples:
            return audio
        
        # Apply fade in to the beginning
        fade_len = min(self._crossfade_samples, len(self._last_audio_tail), len(audio))
        if fade_len > 0:
            # Create fade curves
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            
            # Crossfade: blend end of previous with start of current
            audio[:fade_len] = (
                self._last_audio_tail[-fade_len:] * fade_out + 
                audio[:fade_len] * fade_in
            )
        
        return audio
    
    def play_audio(self, audio: np.ndarray):
        """Queue audio for playback on output device with crossfade smoothing"""
        if len(audio) == 0:
            return
        
        # Ensure float32 and flatten
        audio = np.asarray(audio, dtype=np.float32).flatten()
        
        # Apply crossfade with previous chunk to reduce pops/clicks
        audio = self._apply_crossfade(audio)
        
        # Store tail for next crossfade
        if len(audio) >= self._crossfade_samples:
            self._last_audio_tail = audio[-self._crossfade_samples:].copy()
        
        # Use larger chunks for smoother playback (減少切換次數)
        chunk_size = 2048  # 從 1024 提高到 2048 (約 85ms @ 24kHz)
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            try:
                self.output_queue.put(chunk, timeout=0.5)
            except:
                console.print("[yellow]Output queue full, dropping audio chunk[/yellow]")
                break
    
    def play_audio_blocking(self, audio: np.ndarray, sample_rate: Optional[int] = None):
        """Play audio directly (blocking)"""
        sr = sample_rate or self.output_sample_rate
        sd.play(audio, sr, device=self._output_device)
        sd.wait()
    
    def get_stats(self) -> dict:
        """Get audio processing statistics"""
        return {
            "input_chunks_processed": self._input_chunks_processed,
            "output_chunks_processed": self._output_chunks_processed,
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
        }
    
    def clear_queues(self):
        """Clear all audio queues"""
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except:
                break
        
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except:
                break
        
        while not self.monitor_queue.empty():
            try:
                self.monitor_queue.get_nowait()
            except:
                break


class AudioRecorder:
    """Simple audio recorder for capturing reference audio"""
    
    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
        self._audio_data: List[np.ndarray] = []
    
    def start_recording(self, device: Optional[int] = None):
        """Start recording audio"""
        self._audio_data = []
        self._recording = True
        
        def callback(indata, frames, time_info, status):
            if self._recording:
                self._audio_data.append(indata.copy())
        
        self._stream = sd.InputStream(
            device=device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        console.print("[red]● Recording started...[/red]")
    
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return audio data"""
        self._recording = False
        self._stream.stop()
        self._stream.close()
        
        if self._audio_data:
            audio = np.concatenate(self._audio_data).flatten()
            duration = len(audio) / self.sample_rate
            console.print(f"[green]✓ Recording stopped. Duration: {duration:.2f}s[/green]")
            return audio
        
        return np.array([])
    
    def save_recording(self, audio: np.ndarray, path: str):
        """Save recorded audio to file"""
        import soundfile as sf
        sf.write(path, audio, self.sample_rate)
        console.print(f"[green]✓ Audio saved to: {path}[/green]")


if __name__ == "__main__":
    # Test audio devices
    console.print("\n[bold]Available Audio Devices:[/bold]")
    for device in AudioIO.list_devices():
        console.print(f"  {device}")
    
    console.print("\n[bold]Input Devices (Microphones):[/bold]")
    for device in AudioIO.get_input_devices():
        console.print(f"  {device}")
    
    console.print("\n[bold]Output Devices:[/bold]")
    for device in AudioIO.get_output_devices():
        console.print(f"  {device}")
    
    vb_cable = AudioIO.find_vb_cable_device()
    if vb_cable:
        console.print(f"\n[green]VB-CABLE found: {vb_cable}[/green]")
    else:
        console.print("\n[yellow]VB-CABLE not found. Please install VB-Audio Virtual Cable.[/yellow]")

