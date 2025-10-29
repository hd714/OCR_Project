from pymilvus import connections
import time

print("Testing Milvus connection...")
time.sleep(2)

try:
    connections.connect(host='localhost', port='19530', timeout=10)
    print("SUCCESS: Connected to Milvus!")
except Exception as e:
    print(f"FAILED: {e}")
    print("\nMake sure you ran: milvus.bat start")
