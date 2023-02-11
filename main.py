import torch
import pickle
from config.config_clprolda import Config
from model.constractive_prodlda import ConstractiveProdLDA
from train.train import compute_beta, loss, train, get_latent_representation
from data.datareader import ConstractiveProLDADataset
from torch.optim import Adam
from torch.utils.data import DataLoader


config = Config()

# get data
with open('./data/bow.pkl', 'rb') as f:
    bow = pickle.load(f)
bow = torch.tensor(bow.toarray(), dtype=torch.float32)

with open('./data/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
tfidf = torch.tensor(tfidf.toarray(), dtype=torch.float32)

bow = bow.to(config.device)
tfidf = tfidf.to(config.device)

# get model
cs_model = ConstractiveProdLDA(config)
cs_model.to(config.device)

# get optimizer
optimizer = Adam(cs_model.parameters(), lr=config.lr)

# get dataloader
dataset = ConstractiveProLDADataset(bow, tfidf)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
infer_dataloader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)

if __name__ == '__main__':
    for t in range(config.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(cs_model, train_dataloader, config, optimizer)
    print("Done!")
