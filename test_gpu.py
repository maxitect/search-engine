import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Get GPU device count
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Get current GPU
    current_device = torch.cuda.current_device()
    print(f"Current GPU: {torch.cuda.get_device_name(current_device)}")
    
    # Get GPU memory info
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Get GPU properties
    print("\nGPU Properties:")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**2:.2f} MB")
else:
    print("No GPU available. Using CPU.") 