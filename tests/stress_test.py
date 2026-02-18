
import ctypes
import threading
import time
import random
import os

# Define Telemetry Struct
class SynapseTelemetry(ctypes.Structure):
    _fields_ = [
        ("total_requests", ctypes.c_uint64),
        ("l1_hits", ctypes.c_uint64),
        ("l2_hits", ctypes.c_uint64),
        ("l3_misses", ctypes.c_uint64),
        ("current_quantization_error", ctypes.c_double),
        ("avg_migration_latency_us", ctypes.c_double)
    ]

# Load Library
# Assuming built in build/Release or similar. Adjust path as needed.
LIB_PATH = os.path.abspath("./build/Release/synapse_vllm_adapter.dll")
if not os.path.exists(LIB_PATH):
    # Try Linux path
    LIB_PATH = os.path.abspath("./build/libsynapse_vllm_adapter.so")

print(f"[StressTest] Loading Library: {LIB_PATH}")
lib = ctypes.CDLL(LIB_PATH)

# Bindings
lib.synapse_init.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
lib.synapse_allocate.argtypes = [ctypes.c_size_t, ctypes.c_int]
lib.synapse_allocate.restype = ctypes.c_uint64
lib.synapse_get_telemetry.argtypes = [ctypes.POINTER(SynapseTelemetry)]

# Constants
NUM_USERS = 50
BLOCK_SIZE = 2 * 1024 * 1024 # 2MB
HBM_LIMIT = 4 * 1024 * 1024 * 1024 # 4GB (Small for test)
HOST_LIMIT = 8 * 1024 * 1024 * 1024 # 8GB
DURATION_SEC = 10

stop_event = threading.Event()
backpressure_count = 0
backpressure_lock = threading.Lock()

def user_worker(user_id):
    global backpressure_count
    allocated_blocks = []
    
    print(f"[User {user_id}] Started.")
    
    while not stop_event.is_set():
        # Simulate variable token length (4k to 512k tokens -> variable blocks)
        # Here we just allocate fixed blocks rapidly
        
        # 0 = HBM (Preferred)
        block_id = lib.synapse_allocate(BLOCK_SIZE, 0)
        
        if block_id == 0:
            # Backpressure!
            with backpressure_lock:
                backpressure_count += 1
            # Backoff
            time.sleep(0.1)
        else:
            allocated_blocks.append(block_id)
            # Simulate usage
            time.sleep(random.uniform(0.001, 0.01))
            
    print(f"[User {user_id}] Finished. Allocated {len(allocated_blocks)} blocks.")

def monitor_telemetry():
    metrics = SynapseTelemetry()
    while not stop_event.is_set():
        lib.synapse_get_telemetry(ctypes.byref(metrics))
        print(f"[Telemetry] Reqs: {metrics.total_requests} | L1 Hits: {metrics.l1_hits} | "
              f"Mig Latency: {metrics.avg_migration_latency_us:.2f} us | "
              f"Backpressure Events: {backpressure_count}")
        time.sleep(1)

def run_memory_tsunami():
    print("[StressTest] Initializing Engine...")
    lib.synapse_init(HBM_LIMIT, HOST_LIMIT)
    
    threads = []
    
    # Start Monitor
    monitor_thread = threading.Thread(target=monitor_telemetry)
    monitor_thread.start()
    
    print(f"[StressTest] Spawning {NUM_USERS} concurrent users...")
    for i in range(NUM_USERS):
        t = threading.Thread(target=user_worker, args=(i,))
        threads.append(t)
        t.start()
        
    # Run for Duration
    time.sleep(DURATION_SEC)
    
    print("[StressTest] Stopping...")
    stop_event.set()
    
    for t in threads:
        t.join()
    monitor_thread.join()
    
    lib.synapse_shutdown()
    
    print("\n[StressTest] Result:")
    print(f"Total Backpressure Events: {backpressure_count}")
    
    status = "FAIL"
    if backpressure_count > 0:
        print("[PASS] System correctly triggered Backpressure on OOM.")
        status = "PASS"
    else:
        print("[WARN] System did not saturate. Increase load or decrease memory limits.")
        # If we didn't trigger it, we didn't test it.
        status = "WARN"

    # JSON Output
    result = {
        "backpressure_events": backpressure_count,
        "status": status
    }
    print(f"JSON_START{json.dumps(result)}JSON_END")

if __name__ == "__main__":
    import json
    run_memory_tsunami()
