import torch

def default_device():
    """
    Provides a default device for any machine considering this priority order:
    1. CUDA
    2. MPS
    3. CPU
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device = "mps"

    return device