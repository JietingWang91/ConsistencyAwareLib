import torch
from evaluation_index.metrics import PSED

# 4 samples, each represented by a 2-dimensional vector
# The first two samples belong to Class 0, and the last two samples belong to Class 1.
features = torch.tensor([
    [1.0, 0.9],
    [0.9, 1.1],
    [-1.0, -0.8],
    [-0.9, -1.2]
], dtype=torch.float32)

labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

loss = PSED(features, labels)
print("PSED:", loss)