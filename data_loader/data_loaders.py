from torchvision import datasets, transforms
from base import BaseDataLoader
import pandas as pd

class TsvDataLoader(BaseDataLoader):
    """
    Load tsv file and create BaseDataLoader class
    """
    def __init__(self, path_to_tsv, batch_size, shuffle=True, validation_split=0.0, num_workers=1, prediction=False):
        dataset = pd.read_csv(path_to_tsv,sep="\t")
        if prediction:
            dataset = dataset[["ID", "aa_seq"]] # select columns
            dataset = [ [[(i[0], i[1]), len(i[1])], 0. ] for i in dataset.values]
            self.dataset = dataset
            super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)

        else:
            dataset = dataset[["ID", "aa_seq", "label"]] # select columns
            dataset = [ [[(i[0], i[1]), len(i[1])], i[2] ] for i in dataset.values]
            self.dataset = dataset
            super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)