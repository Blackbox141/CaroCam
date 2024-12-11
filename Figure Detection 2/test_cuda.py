
import torch
print(torch.cuda.is_available())  # Sollte 'True' zurückgeben, wenn CUDA verfügbar ist
print(torch.version.cuda)         # Zeigt die verwendete CUDA-Version an

