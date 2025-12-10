"""
Streaming Translation Module - LLM-based translation with streaming output
Supports OpenAI GPT, Google Gemini, and Groq APIs
"""
import asyncio
from dataclasses import dataclass
from typing import Optional, AsyncGenerator, Generator, Literal, Callable
from queue import Queue, Empty
from threading import Thread, Event
import time
from rich.console import Console

console = Console()


@dataclass
class TranslationChunk:
    """A chunk of translated text"""
    text: str
    is_complete: bool  # Whether this is the final chunk
    source_text: str
    latency_ms: float  # Time since translation started
    
    def __str__(self):
        status = "✓" if self.is_complete else "..."
        return f"[{status}] {self.text}"


class OpenAITranslator:
    """OpenAI GPT-based streaming translator"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        source_language: str = "Chinese",
        target_language: str = "English",
        temperature: float = 0.3,
    ):
        from openai import OpenAI, AsyncOpenAI
        
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self.temperature = temperature
        
        self.system_prompt = f"""You are a professional real-time translator. 
Translate the following {source_language} text to {target_language}.
Rules:
1. Translate naturally and fluently, maintaining the original meaning
2. Keep the translation concise and suitable for speech
3. Do not add explanations or notes
4. Output ONLY the translated text, nothing else
5. Preserve the tone and emotion of the original"""
        
        console.print(f"[green]✓ OpenAI Translator initialized ({model})[/green]")
    
    def translate(self, text: str) -> str:
        """Synchronous translation (non-streaming)"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()
    
    def translate_stream(self, text: str) -> Generator[TranslationChunk, None, None]:
        """Streaming translation - yields chunks as they're generated"""
        start_time = time.time()
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
            stream=True,
        )
        
        accumulated_text = ""
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated_text += content
                
                yield TranslationChunk(
                    text=content,
                    is_complete=False,
                    source_text=text,
                    latency_ms=(time.time() - start_time) * 1000,
                )
        
        # Final chunk
        yield TranslationChunk(
            text="",
            is_complete=True,
            source_text=text,
            latency_ms=(time.time() - start_time) * 1000,
        )
    
    async def translate_stream_async(
        self, 
        text: str
    ) -> AsyncGenerator[TranslationChunk, None]:
        """Async streaming translation"""
        start_time = time.time()
        
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                
                yield TranslationChunk(
                    text=content,
                    is_complete=False,
                    source_text=text,
                    latency_ms=(time.time() - start_time) * 1000,
                )
        
        yield TranslationChunk(
            text="",
            is_complete=True,
            source_text=text,
            latency_ms=(time.time() - start_time) * 1000,
        )


class GeminiTranslator:
    """Google Gemini-based streaming translator"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        source_language: str = "Chinese",
        target_language: str = "English",
        temperature: float = 0.3,
    ):
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        
        # Try different model names for compatibility
        model_candidates = [model, "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-pro"]
        self.model = None
        
        for candidate in model_candidates:
            try:
                self.model = genai.GenerativeModel(candidate)
                model = candidate
                break
            except Exception:
                continue
        
        if self.model is None:
            self.model = genai.GenerativeModel(model)
        self.model_name = model
        self.source_language = source_language
        self.target_language = target_language
        self.temperature = temperature
        
        self.prompt_template = f"""Translate the following {source_language} text to {target_language}.
Rules:
1. Translate naturally and fluently
2. Keep it concise for speech
3. Output ONLY the translation
4. Preserve tone and emotion

Text to translate: {{text}}"""
        
        console.print(f"[green]✓ Gemini Translator initialized ({model})[/green]")
    
    def translate(self, text: str) -> str:
        """Synchronous translation"""
        prompt = self.prompt_template.format(text=text)
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature}
        )
        return response.text.strip()
    
    def translate_stream(self, text: str) -> Generator[TranslationChunk, None, None]:
        """Streaming translation"""
        start_time = time.time()
        prompt = self.prompt_template.format(text=text)
        
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature},
            stream=True,
        )
        
        for chunk in response:
            if chunk.text:
                yield TranslationChunk(
                    text=chunk.text,
                    is_complete=False,
                    source_text=text,
                    latency_ms=(time.time() - start_time) * 1000,
                )
        
        yield TranslationChunk(
            text="",
            is_complete=True,
            source_text=text,
            latency_ms=(time.time() - start_time) * 1000,
        )


class GroqTranslator:
    """Groq-based ultra-fast streaming translator"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-70b-versatile",
        source_language: str = "Chinese",
        target_language: str = "English",
        temperature: float = 0.3,
    ):
        from groq import Groq
        
        self.client = Groq(api_key=api_key)
        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self.temperature = temperature
        
        self.system_prompt = f"""You are a professional real-time translator.
Translate {source_language} to {target_language}.
Output ONLY the translation, nothing else.
Keep it natural and suitable for speech."""
        
        console.print(f"[green]✓ Groq Translator initialized ({model})[/green]")
    
    def translate(self, text: str) -> str:
        """Synchronous translation"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()
    
    def translate_stream(self, text: str) -> Generator[TranslationChunk, None, None]:
        """Streaming translation"""
        start_time = time.time()
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                
                yield TranslationChunk(
                    text=content,
                    is_complete=False,
                    source_text=text,
                    latency_ms=(time.time() - start_time) * 1000,
                )
        
        yield TranslationChunk(
            text="",
            is_complete=True,
            source_text=text,
            latency_ms=(time.time() - start_time) * 1000,
        )


class StreamingTranslator:
    """
    Unified streaming translator interface
    Supports multiple LLM providers with consistent API
    """
    
    def __init__(
        self,
        provider: Literal["openai", "gemini", "groq"] = "openai",
        api_key: str = "",
        model: Optional[str] = None,
        source_language: str = "Chinese",
        target_language: str = "English",
        temperature: float = 0.3,
    ):
        self.provider = provider
        self.source_language = source_language
        self.target_language = target_language
        
        # Default models for each provider
        default_models = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash",
            "groq": "llama-3.1-70b-versatile",
        }
        
        model = model or default_models.get(provider, "gpt-4o-mini")
        
        # Initialize the appropriate translator
        if provider == "openai":
            self._translator = OpenAITranslator(
                api_key=api_key,
                model=model,
                source_language=source_language,
                target_language=target_language,
                temperature=temperature,
            )
        elif provider == "gemini":
            self._translator = GeminiTranslator(
                api_key=api_key,
                model=model,
                source_language=source_language,
                target_language=target_language,
                temperature=temperature,
            )
        elif provider == "groq":
            self._translator = GroqTranslator(
                api_key=api_key,
                model=model,
                source_language=source_language,
                target_language=target_language,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Processing queue for async operation
        self._input_queue: Queue = Queue()
        self._output_queue: Queue = Queue()
        self._stop_event = Event()
        self._processing_thread: Optional[Thread] = None
        
        # Callbacks
        self._on_chunk: Optional[Callable[[TranslationChunk], None]] = None
        self._on_complete: Optional[Callable[[str, str], None]] = None
        
        # Statistics
        self._total_translations = 0
        self._total_latency_ms = 0.0
    
    def set_callbacks(
        self,
        on_chunk: Optional[Callable[[TranslationChunk], None]] = None,
        on_complete: Optional[Callable[[str, str], None]] = None,
    ):
        """Set event callbacks"""
        self._on_chunk = on_chunk
        self._on_complete = on_complete
    
    def translate(self, text: str) -> str:
        """Synchronous translation"""
        return self._translator.translate(text)
    
    def translate_stream(self, text: str) -> Generator[TranslationChunk, None, None]:
        """Streaming translation - yields chunks"""
        full_translation = ""
        
        for chunk in self._translator.translate_stream(text):
            if not chunk.is_complete:
                full_translation += chunk.text
            
            if self._on_chunk:
                self._on_chunk(chunk)
            
            yield chunk
        
        if self._on_complete:
            self._on_complete(text, full_translation)
        
        self._total_translations += 1
    
    def translate_stream_to_queue(self, text: str) -> Queue:
        """
        Start streaming translation and return queue of chunks.
        Useful for integrating with other async processes.
        """
        output_queue = Queue()
        
        def process():
            full_translation = ""
            for chunk in self._translator.translate_stream(text):
                if not chunk.is_complete:
                    full_translation += chunk.text
                output_queue.put(chunk)
            
            # Signal completion
            output_queue.put(None)
        
        thread = Thread(target=process, daemon=True)
        thread.start()
        
        return output_queue
    
    def _processing_loop(self):
        """Background processing loop"""
        while not self._stop_event.is_set():
            try:
                text = self._input_queue.get(timeout=0.1)
                
                full_translation = ""
                for chunk in self._translator.translate_stream(text):
                    if not chunk.is_complete:
                        full_translation += chunk.text
                    self._output_queue.put(chunk)
                    
                    if self._on_chunk:
                        self._on_chunk(chunk)
                
                if self._on_complete:
                    self._on_complete(text, full_translation)
                
                self._total_translations += 1
                
            except Empty:
                continue
            except Exception as e:
                console.print(f"[red]Translation error: {e}[/red]")
    
    def start(self):
        """Start background processing"""
        self._stop_event.clear()
        self._processing_thread = Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        console.print("[green]✓ Translator background processing started[/green]")
    
    def stop(self):
        """Stop background processing"""
        self._stop_event.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        console.print("[yellow]Translator stopped[/yellow]")
    
    def queue_translation(self, text: str):
        """Queue text for background translation"""
        self._input_queue.put(text)
    
    def get_result(self, timeout: float = 0.1) -> Optional[TranslationChunk]:
        """Get translation chunk from output queue"""
        try:
            return self._output_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def stream_results(self) -> Generator[TranslationChunk, None, None]:
        """Generator that yields translation chunks"""
        while not self._stop_event.is_set():
            chunk = self.get_result(timeout=0.1)
            if chunk is not None:
                yield chunk
    
    def get_stats(self) -> dict:
        """Get translation statistics"""
        avg_latency = (
            self._total_latency_ms / self._total_translations 
            if self._total_translations > 0 else 0
        )
        return {
            "total_translations": self._total_translations,
            "average_latency_ms": avg_latency,
            "input_queue_size": self._input_queue.qsize(),
            "output_queue_size": self._output_queue.qsize(),
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Test with OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        console.print("\n[bold]Testing OpenAI Streaming Translation...[/bold]")
        
        translator = StreamingTranslator(
            provider="openai",
            api_key=api_key,
            model="gpt-4o-mini",
        )
        
        test_text = "你好，今天天氣真好，我們去公園散步吧。"
        console.print(f"[yellow]Source: {test_text}[/yellow]")
        console.print("[cyan]Translation: [/cyan]", end="")
        
        for chunk in translator.translate_stream(test_text):
            if not chunk.is_complete:
                console.print(chunk.text, end="")
        
        console.print(f"\n[green]✓ Latency: {chunk.latency_ms:.0f}ms[/green]")
    else:
        console.print("[yellow]OPENAI_API_KEY not set, skipping test[/yellow]")

