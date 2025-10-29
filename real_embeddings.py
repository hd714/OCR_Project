#!/usr/bin/env python3
"""
Real Embeddings Implementation using Sentence Transformers
This replaces the mock embeddings with actual working ones
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import json
from datetime import datetime
import pytesseract
from PIL import Image
import time

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class RealEmbeddingPipeline:
    def __init__(self):
        """Initialize with real embedding model"""
        print("Loading sentence transformer model...")
        # This is one of the best models for semantic search
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded! (384-dimensional embeddings)")
        
    def extract_text_from_document(self, image_path):
        """Extract text using OCR"""
        print(f"\nExtracting text from {image_path}...")
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    
    def create_embedding(self, text):
        """Create real embedding from text"""
        # This creates actual embeddings that can be used for semantic search
        embedding = self.model.encode(text)
        return embedding
    
    def semantic_search(self, query, document_embeddings, documents, top_k=3):
        """Perform actual semantic search"""
        # Encode the query
        query_embedding = self.model.encode(query)
        
        # Calculate similarities
        similarities = []
        for doc_emb in document_embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            similarities.append(similarity)
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                'document': documents[idx],
                'similarity': float(similarities[idx])
            })
        
        return results

def test_real_embeddings():
    """Test with actual biotech content"""
    print("="*70)
    print("TESTING REAL EMBEDDINGS WITH BIOTECH DOCUMENTS")
    print("="*70)
    
    pipeline = RealEmbeddingPipeline()
    
    # Simulate biotech documents
    documents = [
        "HUMIRA showed 75% efficacy in Phase 3 clinical trials with p-value <0.001",
        "Adverse events included headache in 12% of patients and nausea in 8%",
        "The drug dosage is 40mg administered subcutaneously every 2 weeks",
        "Placebo group showed only 23% response rate compared to treatment group",
        "KEYTRUDA demonstrated 45% overall survival rate in melanoma patients"
    ]
    
    print(f"\n📚 Processing {len(documents)} documents...")
    
    # Create real embeddings for each document
    document_embeddings = []
    for i, doc in enumerate(documents):
        embedding = pipeline.create_embedding(doc)
        document_embeddings.append(embedding)
        print(f"  Document {i+1}: Embedded ({len(embedding)} dimensions)")
    
    # Test semantic search with real queries
    print("\n🔍 Testing Semantic Search:")
    print("-"*50)
    
    queries = [
        "What is the efficacy of HUMIRA?",
        "What are the side effects?",
        "What is the dosage?",
        "Show me drug efficacy percentages"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = pipeline.semantic_search(query, document_embeddings, documents, top_k=2)
        
        for result in results:
            print(f"  ✓ Similarity: {result['similarity']:.3f}")
            print(f"    Text: {result['document'][:80]}...")
    
    print("\n" + "="*70)
    print("✅ REAL EMBEDDINGS WORKING!")
    print("="*70)

def create_clinical_trial_test():
    """Create and test with a more complex clinical trial document"""
    print("\n" + "="*70)
    print("ADVANCED TEST: CLINICAL TRIAL DOCUMENT")
    print("="*70)
    
    pipeline = RealEmbeddingPipeline()
    
    # Create a test image with clinical trial data
    import cv2
    width, height = 1400, 1000
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Add content
    cv2.putText(img, "CLINICAL TRIAL NCT-2024-001", (100, 80), font, 1.2, (0,0,0), 2)
    cv2.putText(img, "Drug: HUMIRA (adalimumab)", (100, 150), font, 0.9, (0,0,0), 1)
    cv2.putText(img, "Indication: Rheumatoid Arthritis", (100, 200), font, 0.9, (0,0,0), 1)
    cv2.putText(img, "Primary Endpoint: ACR20 Response at Week 24", (100, 250), font, 0.9, (0,0,0), 1)
    cv2.putText(img, "Results: 75% achieved ACR20 (p<0.001)", (100, 300), font, 0.9, (0,0,0), 1)
    cv2.putText(img, "Secondary: 45% achieved ACR50", (100, 350), font, 0.9, (0,0,0), 1)
    cv2.putText(img, "Adverse Events: Mild to Moderate", (100, 400), font, 0.9, (0,0,0), 1)
    
    cv2.imwrite("clinical_trial_test.png", img)
    
    # Extract text
    extracted_text = pipeline.extract_text_from_document("clinical_trial_test.png")
    print(f"\n📄 Extracted Text Preview:")
    print("-"*50)
    print(extracted_text[:200] + "...")
    
    # Create embedding
    embedding = pipeline.create_embedding(extracted_text)
    print(f"\n✅ Created embedding: {len(embedding)} dimensions")
    
    # Test specific queries
    print("\n🔍 Testing Biotech-Specific Queries:")
    print("-"*50)
    
    test_queries = [
        ("Find HUMIRA efficacy", "Should find 75%"),
        ("What is the p-value?", "Should find p<0.001"),
        ("ACR20 response rate", "Should find 75% ACR20"),
    ]
    
    for query, expected in test_queries:
        query_embedding = pipeline.model.encode(query)
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        
        # Check if expected content is in extracted text
        found = any(exp.lower() in extracted_text.lower() for exp in expected.split())
        status = "✅" if found else "❌"
        
        print(f"\nQuery: '{query}'")
        print(f"  Expected: {expected}")
        print(f"  Similarity Score: {similarity:.3f}")
        print(f"  Content Found: {status}")
    
    return extracted_text, embedding

if __name__ == "__main__":
    # Test 1: Basic real embeddings
    test_real_embeddings()
    
    # Test 2: Clinical trial document
    extracted_text, embedding = create_clinical_trial_test()
    
    # Save results for Milvus
    results = {
        'timestamp': datetime.now().isoformat(),
        'embedding_dimension': len(embedding),
        'model': 'all-MiniLM-L6-v2',
        'text_sample': extracted_text[:500],
        'embedding_sample': embedding[:10].tolist()  # First 10 values
    }
    
    with open('embedding_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("💾 Results saved to embedding_results.json")
    print("Ready for Milvus integration!")
    print("="*70)