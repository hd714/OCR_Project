#!/usr/bin/env python3
"""
Production Milvus Integration for Biotech Document Pipeline
Migrates from simple vector DB to Milvus Docker container
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, 
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType, 
    utility,
    MilvusException
)

class MilvusBiotechPipeline:
    """Production-ready Milvus integration for biotech documents"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: str = "19530",
                 embedding_dim: int = 384):
        """
        Initialize Milvus connection for biotech document storage
        
        Args:
            host: Milvus server host (Docker container)
            port: Milvus server port
            embedding_dim: Dimension of embeddings (384 for all-MiniLM-L6-v2)
        """
        self.host = host
        self.port = port
        self.embedding_dim = embedding_dim
        self.collection_name = "biotech_documents"
        self.collection = None
        self.model = None
        
        # Initialize embedding model
        self._init_embedding_model()
        
    def _init_embedding_model(self):
        """Initialize sentence transformer model"""
        print("🔄 Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
        
    def connect_to_milvus(self, retry_attempts: int = 5, retry_delay: int = 5):
        """
        Connect to Milvus Docker container with retry logic
        
        Args:
            retry_attempts: Number of connection attempts
            retry_delay: Seconds between attempts
        """
        for attempt in range(retry_attempts):
            try:
                print(f"\n🔌 Connecting to Milvus at {self.host}:{self.port} (attempt {attempt + 1}/{retry_attempts})...")
                
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    timeout=30
                )
                
                # Test connection
                server_version = utility.get_server_version()
                print(f"✅ Connected to Milvus server version: {server_version}")
                return True
                
            except Exception as e:
                print(f"❌ Connection attempt {attempt + 1} failed: {e}")
                if attempt < retry_attempts - 1:
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    print("\n❌ Failed to connect to Milvus after all attempts")
                    print("\n📋 Please ensure Docker containers are running:")
                    print("   docker-compose up -d")
                    return False
        
        return False
    
    def create_biotech_collection(self):
        """Create optimized collection for biotech documents"""
        print(f"\n📚 Setting up collection: {self.collection_name}")
        
        # Check if collection exists
        if utility.has_collection(self.collection_name):
            print(f"⚠️  Collection '{self.collection_name}' already exists")
            self.collection = Collection(self.collection_name)
            
            # Drop and recreate for clean setup (remove this in production)
            if input("Drop and recreate collection? (y/n): ").lower() == 'y':
                self.collection.drop()
                print("🗑️  Dropped existing collection")
            else:
                return self.collection
        
        # Define schema for biotech documents
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_text", dtype=DataType.VARCHAR, max_length=65535,
                       description="Full text extracted from document"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim,
                       description="Sentence transformer embedding"),
            FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=100,
                       description="Type: clinical_trial, earnings_call, news, poster"),
            FieldSchema(name="drug_name", dtype=DataType.VARCHAR, max_length=200,
                       description="Primary drug mentioned"),
            FieldSchema(name="efficacy", dtype=DataType.VARCHAR, max_length=100,
                       description="Efficacy percentage if mentioned"),
            FieldSchema(name="p_value", dtype=DataType.VARCHAR, max_length=50,
                       description="Statistical p-value if present"),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500,
                       description="Original file path"),
            FieldSchema(name="extraction_method", dtype=DataType.VARCHAR, max_length=50,
                       description="OCR method used: tesseract, easyocr, pdfplumber"),
            FieldSchema(name="processing_time", dtype=DataType.FLOAT,
                       description="Time taken to process in seconds"),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50,
                       description="When document was processed")
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Biotech documents with clinical trial data, drug efficacy, and medical information"
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            consistency_level="Strong"  # Ensure consistency for production
        )
        
        print(f"✅ Created collection: {self.collection_name}")
        
        # Create index for vector similarity search
        self._create_index()
        
        return self.collection
    
    def _create_index(self):
        """Create optimized index for semantic search"""
        print("\n🔍 Creating search index...")
        
        index_params = {
            "metric_type": "IP",  # Inner Product for normalized embeddings
            "index_type": "IVF_FLAT",  # Good balance of speed and accuracy
            "params": {"nlist": 128}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        print("✅ Index created for fast similarity search")
        
    def insert_biotech_document(self, 
                                text: str,
                                document_type: str,
                                source_file: str,
                                extraction_method: str,
                                processing_time: float,
                                drug_name: str = "",
                                efficacy: str = "",
                                p_value: str = "") -> int:
        """
        Insert a single biotech document with metadata
        
        Returns:
            Document ID
        """
        # Generate embedding
        embedding = self.model.encode(text)
        
        # Extract drug and efficacy info if not provided
        if not drug_name:
            drug_name = self._extract_drug_name(text)
        if not efficacy:
            efficacy = self._extract_efficacy(text)
        if not p_value:
            p_value = self._extract_p_value(text)
        
        # Prepare data for insertion
        data = {
            "document_text": text[:65535],  # Truncate if needed
            "embedding": embedding.tolist(),
            "document_type": document_type,
            "drug_name": drug_name,
            "efficacy": efficacy,
            "p_value": p_value,
            "source_file": source_file,
            "extraction_method": extraction_method,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Insert
        result = self.collection.insert([data])
        self.collection.flush()
        
        doc_id = result.primary_keys[0]
        print(f"✅ Inserted document ID: {doc_id} (Drug: {drug_name}, Efficacy: {efficacy})")
        
        return doc_id
    
    def batch_insert_documents(self, documents: List[Dict[str, Any]]) -> List[int]:
        """Insert multiple documents efficiently"""
        print(f"\n📦 Batch inserting {len(documents)} documents...")
        
        # Prepare batch data
        texts = []
        embeddings = []
        document_types = []
        drug_names = []
        efficacies = []
        p_values = []
        source_files = []
        extraction_methods = []
        processing_times = []
        timestamps = []
        
        for doc in documents:
            text = doc['text']
            texts.append(text[:65535])
            embeddings.append(self.model.encode(text).tolist())
            document_types.append(doc.get('document_type', 'unknown'))
            drug_names.append(doc.get('drug_name', self._extract_drug_name(text)))
            efficacies.append(doc.get('efficacy', self._extract_efficacy(text)))
            p_values.append(doc.get('p_value', self._extract_p_value(text)))
            source_files.append(doc.get('source_file', ''))
            extraction_methods.append(doc.get('extraction_method', 'unknown'))
            processing_times.append(doc.get('processing_time', 0.0))
            timestamps.append(datetime.now().isoformat())
        
        # Batch insert
        data = [
            texts,
            embeddings,
            document_types,
            drug_names,
            efficacies,
            p_values,
            source_files,
            extraction_methods,
            processing_times,
            timestamps
        ]
        
        result = self.collection.insert(data)
        self.collection.flush()
        
        print(f"✅ Inserted {len(result.primary_keys)} documents")
        return result.primary_keys
    
    def semantic_search(self, 
                       query: str,
                       top_k: int = 5,
                       filter_expression: str = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search for biotech information
        
        Args:
            query: Search query (e.g., "HUMIRA efficacy")
            top_k: Number of results to return
            filter_expression: Optional filter (e.g., "document_type == 'clinical_trial'")
        
        Returns:
            List of search results with metadata
        """
        print(f"\n🔍 Searching: '{query}'")
        
        # Load collection
        self.collection.load()
        
        # Generate query embedding
        query_embedding = self.model.encode(query)
        
        # Search parameters
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        
        # Perform search
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expression,
            output_fields=["document_text", "drug_name", "efficacy", "p_value", 
                         "document_type", "source_file", "extraction_method"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    'score': hit.score,
                    'text': hit.entity.get('document_text', '')[:500],
                    'drug_name': hit.entity.get('drug_name', ''),
                    'efficacy': hit.entity.get('efficacy', ''),
                    'p_value': hit.entity.get('p_value', ''),
                    'document_type': hit.entity.get('document_type', ''),
                    'source_file': hit.entity.get('source_file', ''),
                    'extraction_method': hit.entity.get('extraction_method', '')
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def _extract_drug_name(self, text: str) -> str:
        """Extract drug names from text"""
        import re
        # Common drug name patterns
        drug_patterns = [
            r'HUMIRA', r'KEYTRUDA', r'OPDIVO', r'ENBREL', r'REMICADE',
            r'adalimumab', r'pembrolizumab', r'nivolumab'
        ]
        
        for pattern in drug_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return pattern.upper()
        
        return ""
    
    def _extract_efficacy(self, text: str) -> str:
        """Extract efficacy percentages from text"""
        import re
        # Look for percentage patterns near efficacy keywords
        patterns = [
            r'(\d+\.?\d*)\s*%\s*(?:efficacy|response|achieved|ACR20|ACR50)',
            r'(?:efficacy|response|achieved|ACR20|ACR50)\s*(?:of|:)?\s*(\d+\.?\d*)\s*%'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)}%"
        
        return ""
    
    def _extract_p_value(self, text: str) -> str:
        """Extract p-values from text"""
        import re
        patterns = [
            r'p\s*[<=>]\s*[\d.]+',
            r'p-value\s*[<=>]\s*[\d.]+'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return ""
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        self.collection.load()
        
        stats = {
            'total_documents': self.collection.num_entities,
            'collection_name': self.collection_name,
            'embedding_dimension': self.embedding_dim,
            'index_type': 'IVF_FLAT',
            'metric_type': 'IP'
        }
        
        return stats


def test_with_clinical_trials():
    """Test Milvus with clinical trial documents"""
    print("\n" + "="*70)
    print("TESTING MILVUS WITH CLINICAL TRIAL DATA")
    print("="*70)
    
    pipeline = MilvusBiotechPipeline()
    
    # Connect to Milvus
    if not pipeline.connect_to_milvus():
        print("❌ Cannot proceed without Milvus connection")
        return
    
    # Create collection
    pipeline.create_biotech_collection()
    
    # Insert test documents
    test_documents = [
        {
            'text': "CLINICAL TRIAL NCT-2024-001: HUMIRA demonstrated 75% efficacy in Phase 3 trials for rheumatoid arthritis. Primary endpoint ACR20 response achieved with p-value <0.001. Secondary endpoint showed 45% achieved ACR50.",
            'document_type': 'clinical_trial',
            'drug_name': 'HUMIRA',
            'efficacy': '75%',
            'p_value': 'p<0.001',
            'source_file': 'clinical_trial_001.pdf',
            'extraction_method': 'tesseract',
            'processing_time': 0.6
        },
        {
            'text': "Adverse Events Table: Headache occurred in 12% of HUMIRA patients vs 10% placebo. Injection site reactions in 8% vs 2%. Upper respiratory infection 15% vs 14%. All events were mild to moderate.",
            'document_type': 'clinical_trial',
            'drug_name': 'HUMIRA',
            'source_file': 'clinical_trial_001.pdf',
            'extraction_method': 'tesseract',
            'processing_time': 0.5
        },
        {
            'text': "KEYTRUDA Phase 2 results: Overall response rate 45% in advanced melanoma. Median progression-free survival 6.2 months. Statistical significance p=0.002.",
            'document_type': 'clinical_trial',
            'drug_name': 'KEYTRUDA',
            'efficacy': '45%',
            'p_value': 'p=0.002',
            'source_file': 'keytruda_trial.pdf',
            'extraction_method': 'pdfplumber',
            'processing_time': 0.3
        },
        {
            'text': "PharmaVision 2024 Poster: Novel monoclonal antibody BIO-X showed promising results. Table 1: Efficacy Results by Dosage - 40mg Q2W: 82% response, 40mg QW: 85% response, 80mg Q2W: 88% response.",
            'document_type': 'poster',
            'drug_name': 'BIO-X',
            'efficacy': '82-88%',
            'source_file': 'pharmavision_poster.png',
            'extraction_method': 'easyocr',
            'processing_time': 24.0
        }
    ]
    
    # Insert documents
    doc_ids = pipeline.batch_insert_documents(test_documents)
    
    # Test semantic search
    print("\n" + "="*70)
    print("TESTING SEMANTIC SEARCH QUERIES")
    print("="*70)
    
    test_queries = [
        "What is HUMIRA's efficacy?",
        "Find adverse events table",
        "Show me drugs with efficacy over 80%",
        "What are the p-values in clinical trials?",
        "Find PharmaVision poster information"
    ]
    
    for query in test_queries:
        results = pipeline.semantic_search(query, top_k=2)
        
        print(f"\n📍 Query: '{query}'")
        print("-" * 50)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.3f}):")
            print(f"  Drug: {result['drug_name']}")
            print(f"  Efficacy: {result['efficacy']}")
            print(f"  P-value: {result['p_value']}")
            print(f"  Type: {result['document_type']}")
            print(f"  Text: {result['text'][:150]}...")
    
    # Get collection stats
    stats = pipeline.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    
    print("\n✅ Milvus integration test complete!")


def migrate_from_simple_db():
    """Migrate existing embeddings to Milvus"""
    print("\n" + "="*70)
    print("MIGRATING FROM SIMPLE VECTOR DB TO MILVUS")
    print("="*70)
    
    # Check for existing embedding results
    embedding_file = Path("embedding_results.json")
    
    if embedding_file.exists():
        print(f"📂 Found existing embeddings: {embedding_file}")
        
        with open(embedding_file, 'r') as f:
            existing_data = json.load(f)
        
        print(f"  Model: {existing_data.get('model')}")
        print(f"  Dimension: {existing_data.get('embedding_dimension')}")
        
        # Connect to Milvus
        pipeline = MilvusBiotechPipeline()
        
        if pipeline.connect_to_milvus():
            pipeline.create_biotech_collection()
            
            # Insert the existing data
            doc = {
                'text': existing_data.get('text_sample', ''),
                'document_type': 'clinical_trial',
                'source_file': 'migrated_from_simple_db.txt',
                'extraction_method': 'tesseract',
                'processing_time': 0.6
            }
            
            pipeline.insert_biotech_document(**doc)
            print("✅ Migrated existing data to Milvus")
    else:
        print("ℹ️  No existing embeddings found to migrate")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Milvus Integration for Biotech Pipeline")
    parser.add_argument('--test', action='store_true', help='Run test with clinical trial data')
    parser.add_argument('--migrate', action='store_true', help='Migrate from simple DB')
    parser.add_argument('--host', default='localhost', help='Milvus host')
    parser.add_argument('--port', default='19530', help='Milvus port')
    
    args = parser.parse_args()
    
    if args.test:
        test_with_clinical_trials()
    elif args.migrate:
        migrate_from_simple_db()
    else:
        print("\n📋 Milvus Biotech Pipeline")
        print("Options:")
        print("  --test     Run test with clinical trial data")
        print("  --migrate  Migrate from simple vector DB")
        print("\nExample:")
        print("  python milvus_production.py --test")
