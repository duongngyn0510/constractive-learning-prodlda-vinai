import torch
import pickle

with open('./data/bow.pkl', 'rb') as f:
    bow = pickle.load(f)

class Config():
    hidden_size = 100
    num_topics = 20
    drop_lda = 0.2
    vocab_size = bow.shape[1]
    k = 50
    learn_prior = False # nếu train thì loss hay ra nan  
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    lr = 1e-4
    train_cl = True # dùng constractive hay không dùng thì kết quả vẫn chán na ná nhau (khả năng prodlda hoạt động kém)
    batch_size = 16
    epochs = 10