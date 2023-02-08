import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder


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
    def sampling(x, x_recon, tfidf, k):  # Data sampling in Contrastive Learning for ProdLDA - VinAI
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
        x_recon = x_recon.clone()
        x_pos = x.clone()
        x_neg = x.clone()

        _, top_max_ids = torch.topk(tfidf, dim=1, k=k)                 # get top-k highest tfidf score
        x_neg[:, top_max_ids] = x_recon[:, top_max_ids]

        _, top_min_ids = torch.topk(tfidf, dim=1, k=k, largest=False)  # get top-k lowest tfidf score
        x_pos[:, top_min_ids] = x_recon[:, top_min_ids]

        return x_neg.clone().detach(), x_pos.clone().detach()   
    
    def forward(self, x, tfidf):
        """
        Args:
            x: bag-of-word input, shape (batch_size, vocab_size)
            tfidf: tfidf scores, shape (batch_size, vocab_size)
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
            x_neg, x_pos = self.sampling(x, x_recon, tfidf, self.config.k)

            # latent vector representation for negative samples
            z_neg, _, _ = self.encode(x_neg)
        
            # latent vector representation for positive samples
            z_pos, _, _ = self.encode(x_pos)
        else:
            z_neg = None
            z_pos = None
        
        return self.mean_prior, self.var_prior, self.log_var_prior, \
                mean_pos, var_pos, log_var_pos, x_recon, z, z_neg, z_pos
    

def compute_beta(z, z_pos, z_neg):
    """
    Args:
        z: News representation, shape (batch_size, num_topics)
        z_neg: Negative representation, shape (batch_size, num_topics)
        z_pos: Positive representation, shape (batch_size, num_topics)
    Returns:
        beta: Tensor.float
    """
    positive_product = torch.einsum('b n, b n -> b', z, z_pos)    # z . z+
    negative_product = torch.einsum('b n, b n -> b', z, z_neg)    # z . z-
    gamma = positive_product / negative_product
    beta = gamma.mean()
    return beta


def loss(config, x, mean_prior, var_prior, log_var_prior, 
         mean_pos, var_pos, log_var_pos, x_recon, 
         z, z_neg, z_pos, beta):
    # NL
    NL = -(x * (x_recon + 1e-10).log()).sum(dim=1)
    
    # KLD
    var_division = var_pos / var_prior
    diff = mean_pos - mean_prior
    diff_term = diff * diff / var_prior
    logvar_division = log_var_prior - log_var_pos
    KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - config.num_topics)
    
    # Constractive
    if config.train_cl:
        positive_product = torch.einsum('b n, b n -> b', z, z_pos)    # z . z+
        negative_product = torch.einsum('b n, b n -> b', z, z_neg)    # z . z-
        CL = - (positive_product / (positive_product + beta * negative_product)).log()
        return NL + KLD + CL
    else:
        return NL + KLD 