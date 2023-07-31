import os, torch

esm_folder = os.getenv('esm_folder')
torch.hub.set_dir(esm_folder)
_, __ = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
