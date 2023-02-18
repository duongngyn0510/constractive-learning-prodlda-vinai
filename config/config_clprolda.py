import torch
import pickle

with open('./data/bow.pkl', 'rb') as f:
    bow = pickle.load(f)

class Config():
    hidden_size = 200
    num_topics = 50
    drop_lda = 0.2
    vocab_size = bow.shape[1]
    k = 50
    learn_prior = False   
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    lr = 2e-3
    train_cl = True 
    batch_size = 32
    epochs = 80