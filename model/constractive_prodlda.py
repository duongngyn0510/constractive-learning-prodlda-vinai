import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import Encoder
from model.decoder import Decoder



class ConstractiveProdLDA(nn.Module):
    def __init__(self, config):
        super(ConstractiveProdLDA, self).__init__()
        self.config = config
        self.encode = Encoder(config)
        self.decode = Decoder(config)
        self.drop_lda = nn.Dropout(config.drop_lda)
        
        # prior mean
        topic_mean_prior = 0.
        mean_prior = torch.Tensor(1, config.num_topics).fill_(topic_mean_prior)
        
        # prior variance
        topic_var_prior = 1 - (1. / config.num_topics)
        var_prior = torch.Tensor(1, config.num_topics).fill_(topic_var_prior)
        
        # prior log variance
        log_var_prior = var_prior.log()
        
        # training prior ?
        if not config.learn_prior:
            self.register_buffer('mean_prior', mean_prior)
            self.register_buffer('var_prior', var_prior)
            self.register_buffer('log_var_prior', log_var_prior)
        else:
            self.register_parameter('mean_prior', nn.Parameter(mean_prior))
            self.register_parameter('var_prior', nn.Parameter(var_prior))
            self.register_parameter('log_var_prior', nn.Parameter(log_var_prior))
    
    @staticmethod
    def sampling(x, x_recon, tfidf, k, ids):  # Data sampling in Contrastive Learning for ProdLDA - VinAI
        """
        Args:
            x: array_like, shape (batch_size, vocab_size)
            x_recon: array_like, shape (batch_size, vocab_size)
            tfidf: array_like, shape (batch_size, vocab_size)
            k: int, top-k scores
        Return:
            x_negative: Negative samples, shape (batch_size, vocab_size)
            x_positive: Positive samples, shape (batch_size, vocab_size)
        """
        b, n = x.shape
        x_recon = x_recon.clone().reshape(-1)
        x_pos = x.clone().reshape(-1)
        x_neg = x.clone().reshape(-1)

        ids_sort = torch.argsort(tfidf, dim=1, descending=True)
        ids_sort += ids

        # get top-k highest, lowest tfidf score
        top_max_ids = ids_sort[:, :k].reshape(-1)
        top_min_ids = ids_sort[:, -k:].reshape(-1)

        x_neg[top_max_ids] = x_recon[top_max_ids]
        x_pos[top_min_ids] = x_recon[top_min_ids]

        x_neg = x_neg.reshape(b, -1)
        x_pos = x_pos.reshape(b, -1)

        return x_neg.clone().detach(), x_pos.clone().detach()  
    
    def forward(self, x, tfidf, ids):
        """
        Args:
            x: bag-of-word input, shape (batch_size, vocab_size)
            tfidf: tfidf scores, shape (batch_size, vocab_size)
            ids: shape (batch_size, 1)
        Returns:
            mean_prior: shape (1, num_topics)
            var_prior: shape (1, num_topics)
            log_var_prior: shape (1, num_topics) 
            mean_pos: shape (batch_size, num_topics)
            var_pos: shape (batch_size, num_topics)
            log_var_pos: shape (batch_size, num_topics)
            x_recon: Reconstructed documents, shape (batch_size, vocab_size)
            z: News representation, shape (batch_size, num_topics)
            z_neg: Negative representation, shape (batch_size, num_topics) if training else None
            z_pos: Positive representation, shape (batch_size, num_topics) if training else None
        """
        
        # z: news latent vector representation
        z, mean_pos, log_var_pos = self.encode(x) 
        var_pos = log_var_pos.exp()
        
        # reconstruct document    
        x_recon = self.decode(z)                                            
        
        if self.training:
            x_neg, x_pos = self.sampling(x, x_recon, tfidf, self.config.k, ids)

            # latent vector representation for negative samples
            z_neg, _, _ = self.encode(x_neg)
        
            # latent vector representation for positive samples
            z_pos, _, _ = self.encode(x_pos)
        else:
            z_neg = None
            z_pos = None
        
        return self.mean_prior, self.var_prior, self.log_var_prior, \
                mean_pos, var_pos, log_var_pos, x_recon, z, z_neg, z_pos

