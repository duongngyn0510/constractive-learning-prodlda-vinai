import torch
from torch.utils.data import Dataset


class ConstractiveProLDADataset(Dataset):
    def __init__(self, bow: torch.Tensor(), tfidf: torch.Tensor()):
        super(ConstractiveProLDADataset, self).__init__()
        self.bow = bow
        self.tfidf = tfidf
        self.n_data = bow.shape[0]
        
    def __len__(self):
        return self.n_data
    
    def __getitem__(self, idx):
        bow = self.bow[idx]
        tfidf = self.tfidf[idx]
        return bow, tfidf