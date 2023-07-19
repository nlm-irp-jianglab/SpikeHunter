import torch
import sys

path=sys.argv[1]
torch.hub.set_dir(path)
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
print("Setup done!")