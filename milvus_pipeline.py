#!/usr/bin/env python3
"""Production Milvus Integration for Biotech Documents"""

import time
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import re

class MilvusBiotechPipeline:
    def __init__(self):
        self.collection_name = "biotech_documents"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def connect_to_milvus(self):
        """Connect to Milvus"""
        print("Connecting to Milvus...")
        for attempt in range(5):
            try:
                connections.connect(host="localhost", port="19530", timeout=30)
                print(f"Connected to Milvus!")
                return True
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 4:
                    time.sleep(5)
        return False
    
    def create_collection(self):
        """Create biotech collection"""
        if utility.has_collection(self.collection_name):
            print(f"Collection {self.collection_name} exists")
            return Collection(self.collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="drug_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="efficacy", dtype=DataType.VARCHAR, max_length=100),
        ]
        
        schema = CollectionSchema(fields, description="Biotech documents")
        collection = Collection(self.collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Created collection: {self.collection_name}")
        return collection
    
    def insert_document(self, text, drug_name="", efficacy=""):
        """Insert a document"""
        # Extract info if not provided
        if not drug_name:
            drugs = re.findall(r'HUMIRA|KEYTRUDA|adalimumab', text, re.IGNORECASE)
            drug_name = drugs[0].upper() if drugs else ""
        
        if not efficacy:
            eff = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:efficacy|response|achieved)', text, re.IGNORECASE)
            efficacy = f"{eff.group(1)}%" if eff else ""
        
        # Generate embedding
        embedding = self.model.encode(text)
        
        # Insert
        collection = Collection(self.collection_name)
        data = [[text[:65535]], [embedding.tolist()], [drug_name], [efficacy]]
        collection.insert(data)
        collection.flush()
        print(f"Inserted: Drug={drug_name}, Efficacy={efficacy}")
    
    def search(self, query, top_k=3):
        """Search for documents"""
        collection = Collection(self.collection_name)
        collection.load()
        
        query_embedding = self.model.encode(query)
        
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text", "drug_name", "efficacy"]
        )
        
        return results

def test_pipeline():
    """Test with sample data"""
    pipeline = MilvusBiotechPipeline()
    
    if not pipeline.connect_to_milvus():
        print("Failed to connect. Make sure Docker is running.")
        return
    
    collection = pipeline.create_collection()
    
    # Insert test documents
    docs = [
        "HUMIRA demonstrated 75% efficacy in Phase 3 trials with p<0.001",
        "Adverse events: Headache 12%, Injection site 8%",
        "KEYTRUDA showed 45% response rate in melanoma patients"
    ]
    
    for doc in docs:
        pipeline.insert_document(doc)
    
    # Test searches
    print("\nTesting searches:")
    queries = ["HUMIRA efficacy", "adverse events", "response rate"]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = pipeline.search(query)
        for hits in results:
            for hit in hits:
                print(f"  Score: {hit.score:.3f}")
                print(f"  Drug: {hit.entity.get('drug_name')}")
                print(f"  Efficacy: {hit.entity.get('efficacy')}")

if __name__ == "__main__":
    test_pipeline()
