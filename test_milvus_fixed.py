import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from pymilvus import connections, utility
    print("Testing Milvus connection...")
    
    # Try to connect
    connections.connect(
        alias="default",
        host="localhost", 
        port="19530",
        timeout=10
    )
    
    print("SUCCESS: Connected to Milvus!")
    
    # Check server version
    version = utility.get_server_version()
    print(f"Milvus version: {version}")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Try: pip install pymilvus==2.3.3")
except Exception as e:
    print(f"Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Is Docker Desktop running?")
    print("2. Did you run: docker-compose up -d")
    print("3. Did you wait 45 seconds after starting?")
