"""
Token-aware text chunking for LLM training.
"""
from typing import List, Dict, Any
import re


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences (simple implementation).
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    """
    # Simple sentence splitting (can be improved with NLTK/spaCy)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    max_length: int = 512,
    overlap: int = 50,
    preserve_sentences: bool = True
) -> List[Dict[str, Any]]:
    """
    Chunk text into smaller pieces with optional overlap.
    
    Args:
        text: Input text
        max_length: Maximum chunk length (in characters, approximate)
        overlap: Number of characters to overlap between chunks
        preserve_sentences: If True, try to preserve sentence boundaries
    
    Returns:
        List of chunk dictionaries with metadata
    """
    if preserve_sentences:
        return _chunk_by_sentences(text, max_length, overlap)
    else:
        return _chunk_by_characters(text, max_length, overlap)


def _chunk_by_sentences(text: str, max_length: int, overlap: int) -> List[Dict[str, Any]]:
    """Chunk text while preserving sentence boundaries."""
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_id = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence exceeds max_length, save current chunk
        if current_length + sentence_length > max_length and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "length": len(chunk_text),
                "sentence_count": len(current_chunk)
            })
            chunk_id += 1
            
            # Start new chunk with overlap
            if overlap > 0:
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = overlap_length
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "length": len(chunk_text),
            "sentence_count": len(current_chunk)
        })
    
    return chunks


def _chunk_by_characters(text: str, max_length: int, overlap: int) -> List[Dict[str, Any]]:
    """Chunk text by character count."""
    chunks = []
    chunk_id = 0
    start = 0
    
    while start < len(text):
        end = start + max_length
        chunk_text = text[start:end]
        
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "length": len(chunk_text),
            "start_pos": start,
            "end_pos": end
        })
        
        chunk_id += 1
        start = end - overlap
    
    return chunks


def chunk_batch(
    texts: List[str],
    max_length: int = 512,
    overlap: int = 50,
    preserve_sentences: bool = True
) -> List[List[Dict[str, Any]]]:
    """
    Chunk a batch of texts.
    
    Args:
        texts: List of input texts
        max_length: Maximum chunk length
        overlap: Overlap between chunks
        preserve_sentences: Preserve sentence boundaries
    
    Returns:
        List of chunk lists (one per input text)
    """
    return [
        chunk_text(text, max_length, overlap, preserve_sentences)
        for text in texts
    ]
