import torch

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"{len(gpus)} GPU(s) available: {[gpu.name for gpu in gpus]}")
else:
    print("No GPUs available.")


if torch.cuda.is_available():
    print("GPU is available.")
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("GPU not available.")
