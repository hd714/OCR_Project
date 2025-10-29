import re
from typing import List, Dict, Any

class DocumentChunker:
    """Split large documents into smaller chunks for Milvus"""
    
    def __init__(self, chunk_size: int = 60000, overlap: int = 500):
        """
        Initialize chunker

        Args:
            chunk_size: Maximum characters per chunk (keep under 65535)
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks preserving sentence boundaries

        Args:
            text: Full document text
            source_info: Metadata about the document

        Returns:
            List of chunk dictionaries
        """
        if len(text) <= self.chunk_size:
            # Document is small enough, return as is
            return [{
                'text': text,
                'chunk_index': 0,
                'total_chunks': 1,
                **source_info
            }]

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'chunk_index': chunk_index,
                        **source_info
                    })

                    # Start new chunk with overlap
                    overlap_sentences = []
                    overlap_length = 0
                    for sent in reversed(current_chunk):
                        if overlap_length + len(sent) <= self.overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_length += len(sent)
                        else:
                            break

                    current_chunk = overlap_sentences + [sentence]
                    current_length = overlap_length + sentence_length
                    chunk_index += 1
                else:
                    # Single sentence too long, split it
                    words = sentence.split()
                    word_chunk = []
                    word_length = 0

                    for word in words:
                        if word_length + len(word) + 1 > self.chunk_size:
                            if word_chunk:
                                chunks.append({
                                    'text': ' '.join(word_chunk),
                                    'chunk_index': chunk_index,
                                    **source_info
                                })
                                chunk_index += 1
                            word_chunk = [word]
                            word_length = len(word)
                        else:
                            word_chunk.append(word)
                            word_length += len(word) + 1

                    if word_chunk:
                        current_chunk = word_chunk
                        current_length = word_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1

        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'chunk_index': chunk_index,
                **source_info
            })

        # Add total chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total_chunks

        return chunks
