# SynapseVM vLLM Configuration Plugin
# Place this in your vLLM project structure or PYTHONPATH

import os
import ctypes
from typing import Optional

class SynapseVMConfig:
    """
    Configuration for the SynapseVM Memory Backend.
    """
    def __init__(
        self,
        hbm_limit_bytes: int = 6 * 1024**3,  # 6GB Limit (RTX 4070 Safe)
        host_pool_bytes: int = 16 * 1024**3, # 16GB Host RAM
        library_path: Optional[str] = None
    ):
        self.hbm_limit_bytes = hbm_limit_bytes
        self.host_pool_bytes = host_pool_bytes
        
        # Auto-detect library
        if library_path is None:
            # Assume it's in the build directory relative to this script or in system path
            # For development:
            self.library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/libsynapse_vllm_adapter.so"))
        else:
            self.library_path = library_path

    def load_backend(self):
        """
        Loads the SynapseVM Shared Library and initializes the allocator.
        """
        if not os.path.exists(self.library_path):
            raise FileNotFoundError(f"SynapseVM Library not found at: {self.library_path}")
            
        lib = ctypes.CDLL(self.library_path)
        
        # Initialize
        lib.synapse_init.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        lib.synapse_init(self.hbm_limit_bytes, self.host_pool_bytes)
        
        print(f"[SynapseVM] Backend Initialized with HBM Limit: {self.hbm_limit_bytes / 1024**3:.2f} GB")
        return lib

# Usage Example:
# config = SynapseVMConfig()
# backend = config.load_backend()
# ptr = backend.synapse_allocate(2*1024*1024, 0)
