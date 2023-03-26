import esm

class ESM1bTokenize(object):
    """ Tokenizes a sequence for ESM1b model input """

    def __init__(self):
        alphabet = esm.Alphabet.from_architecture("ESM-1b")
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, x):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(list(zip(*x)))
        return batch_tokens
