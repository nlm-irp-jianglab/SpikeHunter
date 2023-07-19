import esm
import torch
torch.hub.set_dir("#ESM_PATH#")

class ESM1bTokenize(object):
    """ Tokenizes a sequence for ESM1b model input """

    def __init__(self):
        alphabet = esm.Alphabet.from_architecture("ESM-1b")
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, x):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(list(zip(*x)))
        return batch_tokens

class ESM2Tokenize(object):
    """ Tokenizes a sequence for ESM2 model input """
    def __init__(self):
        model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, x):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(list(zip(*x)))
        return batch_tokens
