"""
Smart Sentence Buffer Module
Implements intelligent sentence segmentation for low-latency TTS pipeline

Key Strategy:
- Don't wait for complete LLM output
- Cut on punctuation (comma, period) or when word count > threshold
- Send segments to TTS immediately for parallel processing
"""
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Generator, Callable
from queue import Queue, Empty
from threading import Thread, Event, Lock
from rich.console import Console

console = Console()


@dataclass
class BufferChunk:
    """A chunk ready for TTS processing"""
    text: str
    chunk_index: int
    is_sentence_end: bool  # True if ends with sentence-ending punctuation
    is_final: bool  # True if this is the last chunk
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self):
        status = "█" if self.is_sentence_end else "▪"
        return f"[{self.chunk_index}]{status} {self.text}"


class SentenceBuffer:
    """
    Smart sentence buffer that segments streaming text for TTS.
    
    Strategy:
    1. Accumulate text from LLM stream
    2. When punctuation or word threshold is reached, emit chunk
    3. Handle both English and Chinese punctuation
    4. Minimize latency while maintaining natural speech breaks
    """
    
    # Punctuation patterns
    SENTENCE_END_EN = r'[.!?]'
    SENTENCE_END_ZH = r'[。！？]'
    CLAUSE_PUNCT_EN = r'[,;:]'
    CLAUSE_PUNCT_ZH = r'[，；：、]'
    
    def __init__(
        self,
        min_words_for_clause_cut: int = 10,    # 提高到 10 減少卡頓
        max_words_before_force_cut: int = 25,  # 提高到 25 讓句子更完整
        max_buffer_time_ms: int = 3000,        # 3秒超時
        target_language: str = "en",  # "en" or "zh"
    ):
        self.min_words_for_clause_cut = min_words_for_clause_cut
        self.max_words_before_force_cut = max_words_before_force_cut
        self.max_buffer_time_ms = max_buffer_time_ms
        self.target_language = target_language
        
        # Buffer state
        self._buffer = ""
        self._chunk_index = 0
        self._buffer_start_time = time.time()
        self._lock = Lock()
        
        # Output queue
        self._output_queue: Queue = Queue()
        
        # Callbacks
        self._on_chunk: Optional[Callable[[BufferChunk], None]] = None
        
        # Compile regex patterns
        if target_language == "zh":
            self._sentence_end_pattern = re.compile(
                f'({self.SENTENCE_END_EN}|{self.SENTENCE_END_ZH})'
            )
            self._clause_pattern = re.compile(
                f'({self.CLAUSE_PUNCT_EN}|{self.CLAUSE_PUNCT_ZH})'
            )
        else:
            self._sentence_end_pattern = re.compile(self.SENTENCE_END_EN)
            self._clause_pattern = re.compile(self.CLAUSE_PUNCT_EN)
    
    def set_callback(self, on_chunk: Callable[[BufferChunk], None]):
        """Set callback for emitted chunks"""
        self._on_chunk = on_chunk
    
    def _count_words(self, text: str) -> int:
        """Count words in text (handles both English and Chinese)"""
        if self.target_language == "zh":
            # For Chinese, count characters as "words"
            # Remove spaces and punctuation
            cleaned = re.sub(r'[\s\p{P}]+', '', text, flags=re.UNICODE)
            return len(cleaned)
        else:
            # For English, split by whitespace
            return len(text.split())
    
    def _should_cut(self, text: str) -> tuple[bool, bool, int]:
        """
        Determine if we should cut the buffer.
        Returns: (should_cut, is_sentence_end, cut_position)
        """
        # Check for sentence-ending punctuation
        sentence_match = None
        for match in self._sentence_end_pattern.finditer(text):
            sentence_match = match
        
        if sentence_match:
            return True, True, sentence_match.end()
        
        # Check word count
        word_count = self._count_words(text)
        
        # Force cut if too many words
        if word_count >= self.max_words_before_force_cut:
            # Try to find a clause punctuation to cut at
            clause_match = None
            for match in self._clause_pattern.finditer(text):
                clause_match = match
            
            if clause_match:
                return True, False, clause_match.end()
            
            # No punctuation, cut at word boundary
            if self.target_language == "en":
                # Find last space
                last_space = text.rfind(' ')
                if last_space > 0:
                    return True, False, last_space + 1
            
            # Force cut at current position
            return True, False, len(text)
        
        # Check for clause punctuation with minimum word count
        if word_count >= self.min_words_for_clause_cut:
            clause_match = None
            for match in self._clause_pattern.finditer(text):
                clause_match = match
            
            if clause_match:
                return True, False, clause_match.end()
        
        # Check time-based flush
        elapsed_ms = (time.time() - self._buffer_start_time) * 1000
        if elapsed_ms >= self.max_buffer_time_ms and len(text.strip()) > 0:
            return True, False, len(text)
        
        return False, False, 0
    
    def _emit_chunk(self, text: str, is_sentence_end: bool, is_final: bool = False):
        """Emit a chunk for TTS processing"""
        text = text.strip()
        if not text:
            return
        
        chunk = BufferChunk(
            text=text,
            chunk_index=self._chunk_index,
            is_sentence_end=is_sentence_end,
            is_final=is_final,
        )
        
        self._chunk_index += 1
        self._output_queue.put(chunk)
        
        if self._on_chunk:
            self._on_chunk(chunk)
    
    def feed(self, text: str) -> List[BufferChunk]:
        """
        Feed text into the buffer.
        Returns list of chunks that were emitted.
        """
        emitted_chunks = []
        
        with self._lock:
            self._buffer += text
            
            # Process buffer until no more cuts needed
            while True:
                should_cut, is_sentence_end, cut_pos = self._should_cut(self._buffer)
                
                if not should_cut:
                    break
                
                # Extract and emit the chunk
                chunk_text = self._buffer[:cut_pos]
                self._buffer = self._buffer[cut_pos:].lstrip()
                self._buffer_start_time = time.time()
                
                chunk = BufferChunk(
                    text=chunk_text.strip(),
                    chunk_index=self._chunk_index,
                    is_sentence_end=is_sentence_end,
                    is_final=False,
                )
                
                if chunk.text:
                    self._chunk_index += 1
                    self._output_queue.put(chunk)
                    emitted_chunks.append(chunk)
                    
                    if self._on_chunk:
                        self._on_chunk(chunk)
        
        return emitted_chunks
    
    def flush(self) -> Optional[BufferChunk]:
        """Flush remaining buffer content"""
        with self._lock:
            if self._buffer.strip():
                chunk = BufferChunk(
                    text=self._buffer.strip(),
                    chunk_index=self._chunk_index,
                    is_sentence_end=True,
                    is_final=True,
                )
                
                self._chunk_index += 1
                self._buffer = ""
                self._output_queue.put(chunk)
                
                if self._on_chunk:
                    self._on_chunk(chunk)
                
                return chunk
        
        return None
    
    def reset(self):
        """Reset buffer state"""
        with self._lock:
            self._buffer = ""
            self._chunk_index = 0
            self._buffer_start_time = time.time()
            
            # Clear queue
            while not self._output_queue.empty():
                try:
                    self._output_queue.get_nowait()
                except:
                    break
    
    def get_chunk(self, timeout: float = 0.1) -> Optional[BufferChunk]:
        """Get next chunk from output queue"""
        try:
            return self._output_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def stream_chunks(self) -> Generator[BufferChunk, None, None]:
        """Generator that yields chunks"""
        while True:
            chunk = self.get_chunk(timeout=0.1)
            if chunk is not None:
                yield chunk
                if chunk.is_final:
                    break
    
    def get_buffer_content(self) -> str:
        """Get current buffer content (for debugging)"""
        with self._lock:
            return self._buffer
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        with self._lock:
            return {
                "buffer_length": len(self._buffer),
                "buffer_words": self._count_words(self._buffer),
                "chunks_emitted": self._chunk_index,
                "queue_size": self._output_queue.qsize(),
            }


class StreamingBufferProcessor:
    """
    Processes streaming translation output through sentence buffer.
    Connects translator output to TTS input.
    """
    
    def __init__(
        self,
        buffer: SentenceBuffer,
        on_chunk: Optional[Callable[[BufferChunk], None]] = None,
    ):
        self.buffer = buffer
        self._on_chunk = on_chunk
        self._stop_event = Event()
        self._processing_thread: Optional[Thread] = None
        
        # Input queue for translation chunks
        self._input_queue: Queue = Queue()
    
    def process_translation_stream(
        self, 
        translation_chunks: Generator,
    ) -> Generator[BufferChunk, None, None]:
        """
        Process translation stream and yield TTS-ready chunks.
        This is the main "glue" between LLM and TTS.
        """
        self.buffer.reset()
        
        for trans_chunk in translation_chunks:
            if trans_chunk.is_complete:
                # Flush remaining buffer on completion
                final_chunk = self.buffer.flush()
                if final_chunk:
                    yield final_chunk
                break
            
            # Feed translation text to buffer
            emitted = self.buffer.feed(trans_chunk.text)
            
            for chunk in emitted:
                if self._on_chunk:
                    self._on_chunk(chunk)
                yield chunk
    
    def start_background_processing(self):
        """Start background processing thread"""
        self._stop_event.clear()
        
        def process_loop():
            while not self._stop_event.is_set():
                try:
                    text = self._input_queue.get(timeout=0.1)
                    if text is None:
                        # Signal to flush
                        chunk = self.buffer.flush()
                        if chunk and self._on_chunk:
                            self._on_chunk(chunk)
                    else:
                        emitted = self.buffer.feed(text)
                        for chunk in emitted:
                            if self._on_chunk:
                                self._on_chunk(chunk)
                except Empty:
                    continue
        
        self._processing_thread = Thread(target=process_loop, daemon=True)
        self._processing_thread.start()
    
    def stop(self):
        """Stop background processing"""
        self._stop_event.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
    
    def feed_text(self, text: str):
        """Feed text to background processor"""
        self._input_queue.put(text)
    
    def signal_complete(self):
        """Signal that translation is complete"""
        self._input_queue.put(None)


if __name__ == "__main__":
    # Test sentence buffer
    console.print("\n[bold]Testing Sentence Buffer...[/bold]")
    
    buffer = SentenceBuffer(
        min_words_for_clause_cut=5,
        max_words_before_force_cut=15,
        target_language="en",
    )
    
    # Simulate streaming translation
    test_stream = [
        "Hello, ",
        "how are ",
        "you doing ",
        "today? ",
        "I hope ",
        "you're having ",
        "a wonderful ",
        "day, ",
        "and everything ",
        "is going ",
        "well for ",
        "you. ",
        "Let me ",
        "know if ",
        "you need ",
        "anything!",
    ]
    
    console.print("[yellow]Simulating streaming input...[/yellow]")
    
    all_chunks = []
    for text in test_stream:
        console.print(f"  Feed: '{text}'", end="")
        chunks = buffer.feed(text)
        if chunks:
            console.print(f" -> Emitted: {[c.text for c in chunks]}")
            all_chunks.extend(chunks)
        else:
            console.print()
    
    # Flush remaining
    final = buffer.flush()
    if final:
        console.print(f"  Flush -> Emitted: '{final.text}'")
        all_chunks.append(final)
    
    console.print("\n[bold]All emitted chunks:[/bold]")
    for chunk in all_chunks:
        console.print(f"  {chunk}")
    
    console.print(f"\n[green]✓ Total chunks: {len(all_chunks)}[/green]")

