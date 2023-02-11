import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(config.vocab_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.drop_lda = nn.Dropout(config.drop_lda)
        
        self.mean_fc = nn.Linear(config.hidden_size, config.num_topics)
        self.mean_bn = nn.BatchNorm1d(config.num_topics, affine=False)
        
        self.log_var_fc = nn.Linear(config.hidden_size, config.num_topics)
        self.log_var_bn = nn.BatchNorm1d(config.num_topics, affine=False)

    @staticmethod
    def reparameterize(mean_pos, log_var_pos):
        std_pos = (0.5 * log_var_pos).exp()
        eps = torch.rand_like(std_pos)
        return eps.mul(std_pos).add(mean_pos)
    
    def forward(self, x):
        """
        Args:
            x: bag-of-word input, shape (batch_size, vocab_size)
        Returns:
            z: latent representation, shape (batch_size, num_topics)
            mean_posterior: shape (batch_size, num_topics)
            log_var_posterior: shape (batch_size, num_topics)
        """
        # compute posterior
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = self.drop_lda(x)
        
        mean_posterior = self.mean_bn(self.mean_fc(x))             # posterior mean
        log_var_posterior = self.log_var_bn(self.log_var_fc(x))    # posterior log variance
        
        # latent representation
        z = F.softmax(
            self.reparameterize(mean_posterior, log_var_posterior),
            dim=1
        ) 
        z = self.drop_lda(z)
        
        return z, mean_posterior, log_var_posterior