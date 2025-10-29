def insert_document_chunked(self, text, drug_name="", efficacy=""):
    """Insert a document, splitting into chunks if necessary"""

    # Check text length
    if len(text) <= 60000:
        # Small enough to insert directly
        self.insert_document(text, drug_name, efficacy)
        return

    # Need to chunk
    print(f"Document too large ({len(text)} chars), splitting into chunks...") 

    from milvus_chunking_fix import DocumentChunker
    chunker = DocumentChunker(chunk_size=60000, overlap=500)
    chunks = chunker.chunk_text(text, {
        'drug_name': drug_name,
        'efficacy': efficacy
    })

    print(f"Created {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        chunk_text = chunk['text']

        # For first chunk, use extracted metadata
        if i == 0:
            chunk_drug = drug_name
            chunk_efficacy = efficacy
        else:
            # Try to extract from chunk
            chunk_drug = drug_name or self._extract_drug_name(chunk_text)
            chunk_efficacy = efficacy or self._extract_efficacy(chunk_text)

        # Add chunk identifier to text for retrieval
        chunk_id = f"[Chunk {chunk['chunk_index']+1}/{chunk['total_chunks']}] "
        chunk_text = chunk_id + chunk_text[:60000]  # Ensure under limit

        # Insert chunk
        self.insert_document(chunk_text, chunk_drug, chunk_efficacy)
        print(f"  Inserted chunk {i+1}/{len(chunks)}")
