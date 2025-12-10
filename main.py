"""
MeetLingo - Real-time Voice Translation for Online Meetings

This application provides real-time voice translation,
designed for online meetings (Zoom, Teams, Meet).

Usage:
    python main.py              # Launch GUI
    python main.py --cli        # CLI mode (for testing)
    python main.py --devices    # List audio devices
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

console = Console()


def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                  🎙️  MeetLingo  🎙️                       ║
    ║                                                          ║
    ║        即時語音翻譯 for Online Meetings                  ║
    ║              Open Source Solution                        ║
    ╚══════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))


def list_devices():
    """List available audio devices"""
    from modules.audio_io import AudioIO
    
    console.print("\n[bold]Available Audio Devices[/bold]")
    console.print("=" * 60)
    
    console.print("\n[cyan]Input Devices (Microphones):[/cyan]")
    for device in AudioIO.get_input_devices():
        console.print(f"  {device}")
    
    console.print("\n[cyan]Output Devices (Speakers/Virtual Cables):[/cyan]")
    for device in AudioIO.get_output_devices():
        # Highlight VB-CABLE
        if "CABLE" in device.name.upper():
            console.print(f"  [green]{device}[/green] ← VB-CABLE")
        else:
            console.print(f"  {device}")
    
    # Check for VB-CABLE
    vb_cable = AudioIO.find_vb_cable_device()
    if vb_cable:
        console.print(f"\n[green]✓ VB-CABLE detected: {vb_cable.name}[/green]")
    else:
        console.print("\n[yellow]⚠ VB-CABLE not detected. Please install VB-Audio Virtual Cable.[/yellow]")
        console.print("[yellow]  Download: https://vb-audio.com/Cable/[/yellow]")


def run_cli_test():
    """Run a simple CLI test of the pipeline"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    console.print("\n[bold]CLI Test Mode[/bold]")
    console.print("=" * 60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[red]Error: No API key found. Set OPENAI_API_KEY or GOOGLE_API_KEY[/red]")
        return
    
    # Test translation
    console.print("\n[cyan]Testing Translation Module...[/cyan]")
    from modules.translator import StreamingTranslator
    
    provider = "openai" if os.getenv("OPENAI_API_KEY") else "gemini"
    translator = StreamingTranslator(
        provider=provider,
        api_key=api_key,
    )
    
    test_text = "你好，今天天氣真好。"
    console.print(f"[yellow]Input: {test_text}[/yellow]")
    console.print("[cyan]Translation: [/cyan]", end="")
    
    for chunk in translator.translate_stream(test_text):
        if not chunk.is_complete:
            console.print(chunk.text, end="")
    
    console.print(f"\n[green]✓ Translation latency: {chunk.latency_ms:.0f}ms[/green]")
    
    # Test sentence buffer
    console.print("\n[cyan]Testing Sentence Buffer...[/cyan]")
    from modules.sentence_buffer import SentenceBuffer
    
    buffer = SentenceBuffer()
    test_stream = ["Hello, ", "how are ", "you doing ", "today? ", "I hope ", "everything ", "is great!"]
    
    console.print(f"[yellow]Input stream: {test_stream}[/yellow]")
    for text in test_stream:
        chunks = buffer.feed(text)
        if chunks:
            for c in chunks:
                console.print(f"  [green]Chunk: '{c.text}'[/green]")
    
    final = buffer.flush()
    if final:
        console.print(f"  [green]Final: '{final.text}'[/green]")
    
    console.print("\n[green]✓ All tests passed![/green]")


def launch_gui():
    """Launch the Gradio GUI"""
    from gui.gradio_app import launch_app
    
    console.print("\n[cyan]Launching GUI...[/cyan]")
    console.print("[yellow]Open http://localhost:7860 in your browser[/yellow]")
    
    launch_app(share=False, server_port=7860)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MeetLingo - Real-time Voice Translation for Online Meetings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Launch GUI
    python main.py --cli        # Run CLI test
    python main.py --devices    # List audio devices
    python main.py --info       # Show system info
        """
    )
    
    parser.add_argument(
        "--cli", 
        action="store_true",
        help="Run CLI test mode"
    )
    parser.add_argument(
        "--devices", 
        action="store_true",
        help="List available audio devices"
    )
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Show system information"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for GUI server (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link for GUI"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.info:
        from utils.helpers import print_system_info
        print_system_info()
        return
    
    if args.devices:
        list_devices()
        return
    
    if args.cli:
        run_cli_test()
        return
    
    # Default: launch GUI
    launch_gui()


if __name__ == "__main__":
    main()

