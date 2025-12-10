"""
Helper utilities for Real-time AI Voice Translator
"""
import logging
import sys
from typing import Dict, Any, Optional
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging with rich output.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )
    
    logger = logging.getLogger("realtime_tts")
    return logger


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    import torch
    
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info


def check_cuda_available() -> bool:
    """Check if CUDA is available for GPU acceleration"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def estimate_audio_duration(num_samples: int, sample_rate: int) -> float:
    """
    Estimate audio duration in seconds.
    
    Args:
        num_samples: Number of audio samples
        sample_rate: Sample rate in Hz
    
    Returns:
        Duration in seconds
    """
    return num_samples / sample_rate


def format_latency(latency_ms: float) -> str:
    """
    Format latency value for display.
    
    Args:
        latency_ms: Latency in milliseconds
    
    Returns:
        Formatted string
    """
    if latency_ms < 1000:
        return f"{latency_ms:.0f}ms"
    else:
        return f"{latency_ms / 1000:.2f}s"


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are installed.
    
    Returns:
        Dictionary mapping package names to availability
    """
    dependencies = {
        "torch": False,
        "torchaudio": False,
        "sounddevice": False,
        "faster_whisper": False,
        "openai": False,
        "google.generativeai": False,
        "gradio": False,
        "phonemizer": False,
    }
    
    for package in dependencies:
        try:
            __import__(package.replace(".", "_"))
            dependencies[package] = True
        except ImportError:
            pass
    
    return dependencies


def print_system_info():
    """Print system information to console"""
    console.print("\n[bold]System Information[/bold]")
    console.print("=" * 50)
    
    # Device info
    info = get_device_info()
    console.print(f"Python: {info['python_version'].split()[0]}")
    console.print(f"PyTorch: {info['torch_version']}")
    
    if info['cuda_available']:
        console.print(f"[green]CUDA: {info['cuda_version']}[/green]")
        console.print(f"[green]GPU: {info['gpu_name']}[/green]")
        console.print(f"[green]VRAM: {info['gpu_memory_gb']:.1f} GB[/green]")
    else:
        console.print("[yellow]CUDA: Not available[/yellow]")
    
    # Dependencies
    console.print("\n[bold]Dependencies[/bold]")
    deps = check_dependencies()
    for package, available in deps.items():
        status = "[green]✓[/green]" if available else "[red]✗[/red]"
        console.print(f"  {status} {package}")
    
    console.print("=" * 50)


class LatencyTracker:
    """Track and report latency statistics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._latencies: list = []
        self._timestamps: list = []
    
    def record(self, latency_ms: float):
        """Record a latency measurement"""
        import time
        self._latencies.append(latency_ms)
        self._timestamps.append(time.time())
        
        # Keep only recent measurements
        if len(self._latencies) > self.window_size:
            self._latencies.pop(0)
            self._timestamps.pop(0)
    
    def get_average(self) -> float:
        """Get average latency"""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)
    
    def get_min(self) -> float:
        """Get minimum latency"""
        return min(self._latencies) if self._latencies else 0.0
    
    def get_max(self) -> float:
        """Get maximum latency"""
        return max(self._latencies) if self._latencies else 0.0
    
    def get_p95(self) -> float:
        """Get 95th percentile latency"""
        if not self._latencies:
            return 0.0
        sorted_latencies = sorted(self._latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    def get_stats(self) -> Dict[str, float]:
        """Get all statistics"""
        return {
            "avg_ms": self.get_average(),
            "min_ms": self.get_min(),
            "max_ms": self.get_max(),
            "p95_ms": self.get_p95(),
            "count": len(self._latencies),
        }
    
    def reset(self):
        """Reset all measurements"""
        self._latencies = []
        self._timestamps = []


if __name__ == "__main__":
    print_system_info()

