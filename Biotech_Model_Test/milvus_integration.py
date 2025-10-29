"""
Milvus Vector Database Integration
Handles storage and retrieval of text, image, and multimodal embeddings
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException
)

class MilvusManager:
    """Manages Milvus collections for multimodal document storage"""
    
    def __init__(self,
                 host: str = "localhost",
                 port: str = "19530",
                 collection_prefix: str = "biotech_docs"):
        """
        Initialize Milvus connection
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_prefix: Prefix for collection names
        """
        self.host = host
        self.port = port
        self.collection_prefix = collection_prefix
        self.collections = {}
        
        # Connect to Milvus
        self._connect()
        
        # Initialize collections
        self._init_collections()
    
    def _connect(self):
        """Establish connection to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            print(f"✅ Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {e}")
            raise
    
    def _init_collections(self):
        """Initialize different collections for different embedding types"""
        
        # Collection 1: Text-only embeddings
        self._create_text_collection()
        
        # Collection 2: Image-only embeddings  
        self._create_image_collection()
        
        # Collection 3: Multimodal (combined) embeddings
        self._create_multimodal_collection()
        
        # Collection 4: Hybrid approach (separate text + image)
        self._create_hybrid_collection()
    
    def _create_text_collection(self):
        """Create collection for text embeddings"""
        collection_name = f"{self.collection_prefix}_text"
        
        if utility.has_collection(collection_name):
            self.collections['text'] = Collection(collection_name)
            print(f"✅ Loaded existing text collection: {collection_name}")
            return
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # Adjustable
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
        ]
        
        schema = CollectionSchema(fields, description="Text embeddings from documents")
        collection = Collection(name=collection_name, schema=schema)
        
        # Create index for similarity search
        index_params = {
            "metric_type": "IP",  # Inner Product for similarity
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        self.collections['text'] = collection
        print(f"✅ Created text collection: {collection_name}")
    
    def _create_image_collection(self):
        """Create collection for image embeddings (CLIP)"""
        collection_name = f"{self.collection_prefix}_image"
        
        if utility.has_collection(collection_name):
            self.collections['image'] = Collection(collection_name)
            print(f"✅ Loaded existing image collection: {collection_name}")
            return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),  # CLIP dimension
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="image_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
        ]
        
        schema = CollectionSchema(fields, description="Image embeddings from documents")
        collection = Collection(name=collection_name, schema=schema)
        
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        self.collections['image'] = collection
        print(f"✅ Created image collection: {collection_name}")
    
    def _create_multimodal_collection(self):
        """Create collection for joint multimodal embeddings"""
        collection_name = f"{self.collection_prefix}_multimodal"
        
        if utility.has_collection(collection_name):
            self.collections['multimodal'] = Collection(collection_name)
            print(f"✅ Loaded existing multimodal collection: {collection_name}")
            return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # Combined dimension
            FieldSchema(name="fusion_method", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text_weight", dtype=DataType.FLOAT),
            FieldSchema(name="image_weight", dtype=DataType.FLOAT),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
        ]
        
        schema = CollectionSchema(fields, description="Multimodal embeddings")
        collection = Collection(name=collection_name, schema=schema)
        
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        self.collections['multimodal'] = collection
        print(f"✅ Created multimodal collection: {collection_name}")
    
    def _create_hybrid_collection(self):
        """Create collection for hybrid search (stores both text and image embeddings)"""
        collection_name = f"{self.collection_prefix}_hybrid"
        
        if utility.has_collection(collection_name):
            self.collections['hybrid'] = Collection(collection_name)
            print(f"✅ Loaded existing hybrid collection: {collection_name}")
            return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="has_table", dtype=DataType.BOOL),
            FieldSchema(name="has_figure", dtype=DataType.BOOL),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
        ]
        
        schema = CollectionSchema(fields, description="Hybrid embeddings for dual search")
        collection = Collection(name=collection_name, schema=schema)
        
        # Create indices for both embedding fields
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="text_embedding", index_params=index_params)
        collection.create_index(field_name="image_embedding", index_params=index_params)
        
        self.collections['hybrid'] = collection
        print(f"✅ Created hybrid collection: {collection_name}")
    
    def insert_text_embeddings(self, 
                               documents: List[Dict[str, Any]],
                               embeddings: List[np.ndarray]) -> List[int]:
        """
        Insert text embeddings into Milvus
        
        Args:
            documents: List of document metadata
            embeddings: List of embedding vectors
            
        Returns:
            List of inserted IDs
        """
        collection = self.collections['text']
        
        # Prepare data
        data = []
        for doc, emb in zip(documents, embeddings):
            data.append({
                "document_id": doc.get("document_id", ""),
                "text": doc.get("text", "")[:65535],  # Truncate if needed
                "embedding": emb.tolist() if isinstance(emb, np.ndarray) else emb,
                "source_type": doc.get("source_type", "ocr"),
                "page_num": doc.get("page_num", 0),
                "metadata": json.dumps(doc.get("metadata", {}))[:65535]
            })
        
        # Insert data
        result = collection.insert(data)
        collection.flush()
        
        print(f"✅ Inserted {len(result.primary_keys)} text embeddings")
        return result.primary_keys
    
    def insert_image_embeddings(self,
                                images: List[Dict[str, Any]],
                                embeddings: List[np.ndarray]) -> List[int]:
        """Insert image embeddings into Milvus"""
        collection = self.collections['image']
        
        data = []
        for img, emb in zip(images, embeddings):
            data.append({
                "document_id": img.get("document_id", ""),
                "image_path": img.get("image_path", ""),
                "embedding": emb.tolist() if isinstance(emb, np.ndarray) else emb,
                "page_num": img.get("page_num", 0),
                "image_type": img.get("image_type", "figure"),
                "metadata": json.dumps(img.get("metadata", {}))[:65535]
            })
        
        result = collection.insert(data)
        collection.flush()
        
        print(f"✅ Inserted {len(result.primary_keys)} image embeddings")
        return result.primary_keys
    
    def search_text(self,
                    query_embedding: np.ndarray,
                    top_k: int = 10,
                    filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar text embeddings
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional filter expression
            
        Returns:
            List of search results with metadata
        """
        collection = self.collections['text']
        collection.load()
        
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filters,
            output_fields=["document_id", "text", "page_num", "metadata"]
        )
        
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "id": hit.id,
                "score": hit.score,
                "document_id": hit.entity.get("document_id"),
                "text": hit.entity.get("text"),
                "page_num": hit.entity.get("page_num"),
                "metadata": json.loads(hit.entity.get("metadata", "{}"))
            })
        
        return formatted_results
    
    def search_hybrid(self,
                     text_embedding: Optional[np.ndarray] = None,
                     image_embedding: Optional[np.ndarray] = None,
                     top_k: int = 10,
                     alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both text and image embeddings
        
        Args:
            text_embedding: Text query vector
            image_embedding: Image query vector  
            top_k: Number of results
            alpha: Weight for text (1-alpha for image)
            
        Returns:
            Combined and ranked results
        """
        results = {}
        
        # Search text embeddings
        if text_embedding is not None:
            text_results = self.search_text(text_embedding, top_k * 2)
            for res in text_results:
                doc_id = res['document_id']
                if doc_id not in results:
                    results[doc_id] = {'text_score': 0, 'image_score': 0, 'data': res}
                results[doc_id]['text_score'] = res['score']
        
        # Search image embeddings
        if image_embedding is not None:
            collection = self.collections['image']
            collection.load()
            
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            image_results = collection.search(
                data=[image_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k * 2,
                output_fields=["document_id", "image_path", "metadata"]
            )
            
            for hit in image_results[0]:
                doc_id = hit.entity.get("document_id")
                if doc_id not in results:
                    results[doc_id] = {
                        'text_score': 0,
                        'image_score': 0,
                        'data': {
                            'document_id': doc_id,
                            'metadata': json.loads(hit.entity.get("metadata", "{}"))
                        }
                    }
                results[doc_id]['image_score'] = hit.score
        
        # Combine scores
        for doc_id in results:
            results[doc_id]['combined_score'] = (
                alpha * results[doc_id]['text_score'] + 
                (1 - alpha) * results[doc_id]['image_score']
            )
        
        # Sort by combined score
        sorted_results = sorted(
            results.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]
        
        return sorted_results
    
    def evaluate_search_quality(self,
                              test_queries: List[Dict[str, Any]],
                              expected_results: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate search quality metrics
        
        Args:
            test_queries: List of test query embeddings with metadata
            expected_results: List of expected document IDs for each query
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'precision_at_k': [],
            'recall_at_k': [],
            'mrr': [],  # Mean Reciprocal Rank
            'ndcg': []  # Normalized Discounted Cumulative Gain
        }
        
        for query, expected in zip(test_queries, expected_results):
            # Perform search
            results = self.search_text(query['embedding'], top_k=10)
            retrieved_ids = [r['document_id'] for r in results]
            
            # Calculate precision@k
            relevant_retrieved = len(set(retrieved_ids) & set(expected))
            precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
            metrics['precision_at_k'].append(precision)
            
            # Calculate recall@k
            recall = relevant_retrieved / len(expected) if expected else 0
            metrics['recall_at_k'].append(recall)
            
            # Calculate MRR
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in expected:
                    metrics['mrr'].append(1 / rank)
                    break
            else:
                metrics['mrr'].append(0)
        
        # Average metrics
        return {
            'avg_precision': np.mean(metrics['precision_at_k']),
            'avg_recall': np.mean(metrics['recall_at_k']),
            'avg_mrr': np.mean(metrics['mrr']),
            'precision_std': np.std(metrics['precision_at_k']),
            'recall_std': np.std(metrics['recall_at_k'])
        }
    
    def cleanup(self):
        """Clean up connections"""
        connections.disconnect("default")
        print("✅ Disconnected from Milvus")


# Example usage function
def example_usage():
    """Demonstrate how to use MilvusManager"""
    
    # Initialize manager
    manager = MilvusManager()
    
    # Example: Insert text embeddings
    documents = [
        {
            "document_id": "doc001",
            "text": "HUMIRA showed 75% efficacy in Phase 3 trials",
            "page_num": 1,
            "metadata": {"drug": "HUMIRA", "efficacy": "75%"}
        }
    ]
    
    # Mock embedding (replace with actual embedding model)
    embeddings = [np.random.randn(768)]
    
    # Insert
    ids = manager.insert_text_embeddings(documents, embeddings)
    print(f"Inserted IDs: {ids}")
    
    # Search
    query_embedding = np.random.randn(768)
    results = manager.search_text(query_embedding, top_k=5)
    print(f"Search results: {results}")
    
    # Cleanup
    manager.cleanup()


if __name__ == "__main__":
    example_usage()