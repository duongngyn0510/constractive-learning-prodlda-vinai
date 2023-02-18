import torch
import pickle
import os
import numpy as np
from config.config_clprolda import Config
from model.constractive_prodlda import ConstractiveProdLDA
from train.train import compute_beta, loss, train
from train.infer import get_latent_representation
from data.datareader import ConstractiveProLDADataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from classification import classification


class CLProdLDA:
    def __init__(self, config, infer=False):
        self.config = config
        self.cs_model = ConstractiveProdLDA(config)
        self.cs_model.to(config.device)
        self.optimizer = Adam(self.cs_model.parameters(), lr=config.lr)

        if infer:
            checkpoint = torch.load(config.checkpoint_path)
            self.cs_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def fit(self, train_dataloader):
        training_loss = train(self.cs_model, train_dataloader, self.config, self.optimizer)
        return training_loss

    def infer(self, train_dataloader):
        latent_vectors = get_latent_representation(self.cs_model, train_dataloader, self.config)
        return latent_vectors


def get_data(config):
    # get data
    with open('./data/bow.pkl', 'rb') as f:
        bow = pickle.load(f)
    bow = torch.tensor(bow.toarray(), dtype=torch.float32)

    with open('./data/tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    tfidf = torch.tensor(tfidf.toarray(), dtype=torch.float32)

    with open('./data/label.pkl', 'rb') as f:
        label = pickle.load(f)
    
    bow = bow.to(config.device)
    tfidf = tfidf.to(config.device)

    # get dataloader
    dataset = ConstractiveProLDADataset(bow, tfidf)
    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    return train_dataloader, label


def inference():
    config = Config
    train_dataloader, label = get_data(config)

    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')

    model = CLProdLDA(config)
    best_training_loss = float('inf')

    for t in range(config.epochs):
        print(f"Epoch {t+1}")
        training_loss = model.fit(train_dataloader)
        print('-------------------------')

    latent_vectors = model.infer(train_dataloader)
    latent_vectors = latent_vectors.detach().cpu().numpy()

    with open('./checkpoint/latent_vectors.pkl', 'wb') as f:
        pickle.dump(latent_vectors, f)


def classify():
    config = Config
    with open('./checkpoint/latent_vectors.pkl', 'rb') as f:
        latent_vectors = pickle.load(f)
    with open('./data/tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
        tfidf = tfidf.toarray()
    
    train_dataloader, label = get_data(config)

    # truncate because of drop_last=True in DataLoader for compitable shape
    label = label[: len(train_dataloader) * config.batch_size]
    tfidf = tfidf[: len(train_dataloader) * config.batch_size]

    split = int(len(label) * 0.8)

    tfidf_array_train = tfidf[: split]
    tfidf_array_test = tfidf[split: ]

    latent_vectors_train = latent_vectors[: split]
    latent_vectors_test = latent_vectors[split: ]

    y_train = label[: split]
    y_test = label[split: ]

    # tf-idf vector
    mean_tfidf, std_tfidf = classification(tfidf_array_train, y_train, tfidf_array_test, y_test)
    print(f'Using tfidf vector: {mean_tfidf:.3f} +- {std_tfidf:.3f}')

    # latent vector
    mean_clprodlda, std_clprodlda = classification(latent_vectors_train, y_train, latent_vectors_test, y_test)
    print(f'Using latent vector: {mean_clprodlda:.3f} +- {std_clprodlda:.3f}')

if __name__ == '__main__':
    inference()
    classify()
